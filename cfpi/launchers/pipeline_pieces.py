import copy
import inspect
import os.path as osp

import eztils.torch as ptu
import torch

from cfpi import conf
from cfpi.core.logging import logger
from cfpi.data_management.env_replay_buffer import (
    EnvReplayBuffer,
    EnvReplayBufferNextAction,
    EnvReplayBufferNextActionNewAction,
)
from cfpi.data_management.hdf5_path_loader import (
    d4rl_qlearning_dataset_with_next_actions,
    load_hdf5,
    load_hdf5_next_actions,
    load_hdf5_next_actions_new_actions,
)
from cfpi.envs import NormalizedBoxEnv, env_producer
from cfpi.launchers.pipeline import PipelineCtx
from cfpi.policies.gaussian_policy import GaussianPolicy, UnnormalizeGaussianPolicy
from cfpi.policies.stochastic import MakeCfpiDeterministic, MakeDeterministic
from cfpi.pytorch.networks.mlp import ConcatMlp, QuantileMlp
from cfpi.pytorch.torch_rl_algorithm import OfflineTorchBatchRLAlgorithm
from cfpi.samplers.path_collector import MdpPathCollector


def offline_init(_):
    logger.set_offline_rl()


def pac_sanity_check(ctx):
    assert ctx.variant["d4rl"]
    # assert ctx.variant["algorithm_kwargs"]["zero_step"]
    # assert ctx.variant["normalize_env"]
    # assert ctx.variant["IQN"]
    assert (
        ctx.variant["checkpoint_params"]
        in user_defined_attrs_dict(conf.CheckpointParams).keys()
    )
    assert ctx.variant["checkpoint_params"] != "Q"

    params: conf.CheckpointParams.CheckpointParam = getattr(conf.CheckpointParams, ctx.variant["checkpoint_params"])()
    assert ctx.variant["seed"] in params.seeds
    assert ctx.variant["env_id"] in params.envs, ctx.variant["env_id"]
    assert ctx.variant["seed"] in conf.CheckpointParams.Q_IQN().seeds
    assert ctx.variant["env_id"] in conf.CheckpointParams.Q_IQN().envs


def create_eval_env(ctx: PipelineCtx):
    d4rl = ctx.variant["d4rl"]

    ctx.eval_env = env_producer(
        env_id=ctx.variant["env_id"],
        d4rl=d4rl,
        seed=ctx.variant["seed"],
        normalize_env=False,
    )


def create_dataset(ctx: PipelineCtx):
    ctx.dataset = ctx.eval_env.get_dataset()


def create_dataset_next_actions(ctx: PipelineCtx):
    ctx.dataset = d4rl_qlearning_dataset_with_next_actions(ctx.eval_env)
    if "antmaze" in ctx.variant["env_id"]:
        ctx.dataset["rewards"] -= 1


def optionally_normalize_dataset(ctx: PipelineCtx):
    ctx.obs_mean = ctx.dataset["observations"].mean(0)
    ctx.obs_std = ctx.dataset["observations"].std(0)

    if not ctx.variant["normalize_env"]:
        return

    ctx.eval_env = NormalizedBoxEnv(
        ctx.eval_env,
        obs_mean=ctx.dataset["observations"].mean(0),
        obs_std=ctx.dataset["observations"].std(0),
    )

    ctx.dataset["observations"] = (ctx.dataset["observations"] - ctx.obs_mean) / (
        ctx.obs_std + 1e-8
    )
    ctx.dataset["next_observations"] = (
        ctx.dataset["next_observations"] - ctx.obs_mean
    ) / (ctx.obs_std + 1e-8)
    action_space = ctx.eval_env._wrapped_env.action_space

    rg = (action_space.high - action_space.low) / 2
    center = (action_space.high + action_space.low) / 2
    ctx.dataset["actions"] = (ctx.dataset["actions"] - center) / rg

    if "next_actions" in ctx.dataset:
        ctx.dataset["next_actions"] = (ctx.dataset["next_actions"] - center) / rg
    if "new_next_actions" in ctx.dataset:
        ctx.dataset["new_next_actions"] = (
            ctx.dataset["new_next_actions"] - center
        ) / rg


def create_q(ctx: PipelineCtx):
    obs_dim = ctx.eval_env.observation_space.low.size
    action_dim = ctx.eval_env.action_space.low.size

    qf1 = ctx.variant["qf_class"](
        input_size=obs_dim + action_dim, output_size=1, **ctx.variant["qf_kwargs"]
    )
    qf2 = ctx.variant["qf_class"](
        input_size=obs_dim + action_dim, output_size=1, **ctx.variant["qf_kwargs"]
    )

    target_qf1 = ctx.variant["qf_class"](
        input_size=obs_dim + action_dim, output_size=1, **ctx.variant["qf_kwargs"]
    )
    target_qf2 = ctx.variant["qf_class"](
        input_size=obs_dim + action_dim, output_size=1, **ctx.variant["qf_kwargs"]
    )

    ctx.qfs = [qf1, qf2]
    ctx.target_qfs = [target_qf1, target_qf2]


def create_policy(ctx: PipelineCtx):
    obs_dim = ctx.eval_env.observation_space.low.size
    action_dim = ctx.eval_env.action_space.low.size

    ctx.policy = ctx.variant["policy_class"](
        obs_dim=obs_dim, action_dim=action_dim, **ctx.variant["policy_kwargs"]
    )


def create_trainer(ctx: PipelineCtx):
    arg_names = inspect.getfullargspec(ctx.trainer_cls.__init__).args
    arg_names.remove("self")

    passed_args = {}
    for arg in arg_names:
        try:
            passed_args[arg] = getattr(ctx, arg)
        except AttributeError:
            if ctx.variant["trainer_kwargs"].get(arg) is not None:
                passed_args[arg] = ctx.variant["trainer_kwargs"][arg]
    ctx.trainer = ctx.trainer_cls(**passed_args)


def create_eval_policy(ctx: PipelineCtx):
    ctx.eval_policy = MakeDeterministic(ctx.policy)


def create_pac_eval_policy(ctx: PipelineCtx):
    ctx.eval_policy = MakeCfpiDeterministic(ctx.trainer)


def create_eval_path_collector(ctx: PipelineCtx):
    ctx.eval_path_collector = MdpPathCollector(
        ctx.eval_env,
        ctx.eval_policy,
        rollout_fn=ctx.variant["rollout_fn"],
    )


def create_replay_buffer(ctx: PipelineCtx):
    ctx.replay_buffer = ctx.variant["replay_buffer_class"](
        ctx.variant["replay_buffer_size"],
        ctx.eval_env,
    )


def create_algorithm(ctx: PipelineCtx):
    ctx.algorithm = OfflineTorchBatchRLAlgorithm(
        trainer=ctx.trainer,
        evaluation_env=ctx.eval_env,
        evaluation_data_collector=ctx.eval_path_collector,
        replay_buffer=ctx.replay_buffer,
        **ctx.variant["algorithm_kwargs"],
    )


def load_demos(ctx: PipelineCtx):
    if isinstance(ctx.replay_buffer, EnvReplayBufferNextActionNewAction):
        load_hdf5_next_actions_new_actions(ctx.dataset, ctx.replay_buffer)
        assert torch.equal(
            ctx.replay_buffer._next_actions[: ctx.replay_buffer._size - 1],
            ctx.replay_buffer._actions[1 : ctx.replay_buffer._size],
        )
        assert torch.equal(
            ctx.replay_buffer._new_next_actions[: ctx.replay_buffer._size - 1],
            ctx.replay_buffer._actions[1 : ctx.replay_buffer._size],
        )
    elif isinstance(ctx.replay_buffer, EnvReplayBufferNextAction):
        load_hdf5_next_actions(ctx.dataset, ctx.replay_buffer)
        assert torch.equal(
            ctx.replay_buffer._next_actions[: ctx.replay_buffer._size - 1],
            ctx.replay_buffer._actions[1 : ctx.replay_buffer._size],
        )

    elif isinstance(ctx.replay_buffer, EnvReplayBuffer):
        # Off policy
        load_hdf5(ctx.dataset, ctx.replay_buffer)
    else:
        raise Exception("Unsupported replay buffer class", type(ctx.replay_buffer))


def user_defined_attrs_dict(cls):
    return {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}


def load_checkpoint_iqn_q(ctx: PipelineCtx):
    q_params = conf.CheckpointParams.Q_IQN()

    q_epoch = q_params.validation_optimal_epochs[ctx.variant["env_id"]]
    params = torch.load(
        osp.join(
            conf.CHECKPOINT_PATH,
            q_params.path,
            ctx.variant["env_id"],
            str(ctx.variant["seed"]),
            f"itr_{q_epoch}.pt",
        ),
        map_location=ptu.DEVICE,
    )
    assert isinstance(params["trainer/qf1"], QuantileMlp)
    assert isinstance(params["trainer/qf2"], QuantileMlp)

    ctx.qfs = [params["trainer/qf1"], params["trainer/qf2"]]
    ctx.target_qfs = [params["trainer/target_qf1"], params["trainer/target_qf2"]]


def load_checkpoint_iql_q(ctx: PipelineCtx):
    class UnnormalizeIQL(ConcatMlp):
        def __init__(self, state_mean, state_std, mlp: ConcatMlp):
            self.__dict__ = copy.deepcopy(mlp.__dict__)
            self.state_mean = ptu.torch_ify(state_mean)
            self.state_std = ptu.torch_ify(state_std)

        def forward(self, obs, act, **kwargs):
            unnormalized_obs = self.unnormalize(obs)
            return super().forward(unnormalized_obs, act, **kwargs)

        def unnormalize(self, obs):
            return (obs * self.state_std) + self.state_mean

    q_params = conf.CheckpointParams.Q_IQL()

    params = torch.load(
        osp.join(
            conf.CHECKPOINT_PATH,
            q_params.path,
            ctx.variant["env_id"],
            str(ctx.variant["seed"]),
            "params.pt",
        ),
        map_location=ptu.DEVICE,
    )

    ctx.qfs[0].load_state_dict(params["trainer/qf1"])
    ctx.qfs[1].load_state_dict(params["trainer/qf2"])

    ctx.qfs[0] = UnnormalizeIQL(ctx.obs_mean, ctx.obs_std, ctx.qfs[0])
    ctx.qfs[1] = UnnormalizeIQL(ctx.obs_mean, ctx.obs_std, ctx.qfs[1])


def load_checkpoint_iql_policy(ctx: PipelineCtx):
    iql_params = conf.CheckpointParams.Q_IQL()

    params = torch.load(
        osp.join(
            conf.CHECKPOINT_PATH,
            iql_params.path,
            ctx.variant["env_id"],
            str(ctx.variant["seed"]),
            "params.pt",
        ),
        map_location=ptu.DEVICE,
    )

    obs_dim = ctx.eval_env.observation_space.low.size
    action_dim = ctx.eval_env.action_space.low.size

    ctx.policy = GaussianPolicy(
        [256, 256],
        obs_dim,
        action_dim,
        std_architecture="values",
        max_log_std=0,
        min_log_std=-6,
    )
    ctx.policy.load_state_dict(params["trainer/policy"])
    ctx.policy = UnnormalizeGaussianPolicy(ctx.obs_mean, ctx.obs_std, ctx.policy)


def load_checkpoint_policy(ctx: PipelineCtx):
    params = getattr(conf.CheckpointParams, ctx.variant["checkpoint_params"])()

    policy_path = ""
    base = osp.join(
        conf.CHECKPOINT_PATH,
        params.path,
        ctx.variant["env_id"],
        str(ctx.variant["seed"]),
    )
    # policy_path = osp.join(base, "itr_1000.pt")

    if osp.exists(osp.join(base, "itr_500.pt")):
        policy_path = osp.join(base, "itr_500.pt")
    else:
        policy_path = osp.join(base, "itr_-500.pt")

    ctx.policy = torch.load(policy_path, map_location="cpu")[params.key]


def train(ctx: PipelineCtx):
    ctx.algorithm.to(ptu.DEVICE)
    ctx.algorithm.train()
