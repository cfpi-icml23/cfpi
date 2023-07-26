import math
import os.path as osp

import eztils.torch as ptu
import numpy as np
import torch
import torch.optim as optim
from eztils.torch import TanhDelta, TanhGaussianMixture
from eztils import red
from cfpi import conf
from cfpi.data_management.hdf5_path_loader import (
    d4rl_qlearning_dataset_with_next_actions_new_actions,
)
from cfpi.launchers.pipeline import Pipeline, PipelineCtx, Pipelines
from cfpi.launchers.pipeline_pieces import (
    create_eval_policy,
    create_q,
    load_checkpoint_iql_policy,
    load_checkpoint_iql_q,
    user_defined_attrs_dict,
)
from cfpi.pytorch.algorithms.cfpi.single_gaussian import SG_CFPI_Trainer
from cfpi.pytorch.algorithms.q_training.sarsa_iqn import create_q_iqn


class MG_CFPI_Trainer(SG_CFPI_Trainer):
    def __init__(
        self,
        policy,
        qfs,
        target_qfs,
        discount=0.99,
        reward_scale=1,
        policy_lr=0.001,
        qf_lr=0.001,
        optimizer_class=optim.Adam,
        soft_target_tau=0.01,
        target_update_period=1,
        num_quantiles=8,
        plotter=None,
        render_eval_paths=False,
        beta_LB=0.5,
        trivial_threshold=0.05,
        delta_range=None,
        num_delta=None,
        num_candidate_actions=10,
        # new params
        action_selection_mode="max_from_both",
        use_max_lambda=False,
        IQN=True,
    ):
        if delta_range is None:
            delta_range = [0.0, 0.0]
        super().__init__(
            policy=policy,
            qfs=qfs,
            target_qfs=target_qfs,
            discount=discount,
            reward_scale=reward_scale,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            optimizer_class=optimizer_class,
            soft_target_tau=soft_target_tau,
            target_update_period=target_update_period,
            num_quantiles=num_quantiles,
            plotter=plotter,
            render_eval_paths=render_eval_paths,
            delta_range=delta_range,
            num_delta=num_delta,
            beta_LB=beta_LB,
            IQN=IQN,
        )

        assert action_selection_mode in ["jensen", "max", "max_from_both", "easy_bcq"]
        self.action_selection_mode = action_selection_mode
        self.use_max_lambda = use_max_lambda
        self.num_candidate_actions = num_candidate_actions
        self.trivial_threshold = trivial_threshold
        print("action_selection mode:", action_selection_mode)

    def get_cfpi_action(self, obs) -> TanhDelta:
        dist: TanhGaussianMixture = self.get_action_dist(obs)

        if self.easy_bcq or self.action_selection_mode == "easy_bcq":
            return self.get_easy_bcq_action(obs, dist)

        if self.delta_range == [0.0, 0.0]:
            batch_size = obs.shape[0]
            obs_exp = obs.repeat_interleave(dist.num_gaussians, dim=0)
            qfs = self.calc_q_LB(
                obs_exp, dist.mean.reshape(batch_size * dist.num_gaussians, -1)
            ).reshape(batch_size, dist.num_gaussians)
            mask = dist.weights < self.trivial_threshold
            qfs[mask] = -torch.inf

            idx = torch.argmax(qfs, dim=1)
            selected_actions = dist.normal_mean[torch.arange(len(idx)), idx]
            return TanhDelta(selected_actions)

        self.sample_delta()

        if self.action_selection_mode == "jensen":
            jensen_proposal, jensen_value = self.compute_jensen_proposal(obs, dist)
            return TanhDelta(jensen_proposal)
        elif self.action_selection_mode == "max":
            max_proposal, max_value = self.compute_max_proposal(obs, dist)
            return TanhDelta(max_proposal)
        elif self.action_selection_mode == "max_from_both":
            jensen_proposal, jensen_value = self.compute_jensen_proposal(obs, dist)
            max_proposal, max_value = self.compute_max_proposal(obs, dist)

            with torch.no_grad():
                # [batch_size, 2, act_dim]
                proposal = torch.cat(
                    [max_proposal.unsqueeze(1), jensen_proposal.unsqueeze(1)], dim=1
                )
                # [batch_size, 2]
                value = torch.cat(
                    [max_value.unsqueeze(1), jensen_value.unsqueeze(1)], dim=1
                )

                idx = torch.argmax(value, dim=1)
                selected_actions = proposal[torch.arange(len(idx)), idx]
                if torch.any(torch.isnan(selected_actions)):
                    red("not good, found nan actions")
                    raise Exception("Action selection is NaN!")
            return TanhDelta(selected_actions)
        else:
            raise NotImplementedError

    def calc_log_p_mu(self, Sigma_beta):
        return -0.5 * (2 * math.pi * Sigma_beta).prod(-1).log()

    def compute_max_proposal(self, obs, dist: TanhGaussianMixture) -> torch.tensor:
        """
        Max proposals
        """
        # * preliminaries
        pre_tanh_mu_beta = dist.normal_mean
        weights = dist.weights

        num_gaussians = dist.num_gaussians
        batch_size = obs.shape[0]

        # * calculate delta. this is the m distance constraint. we require the Mahalanobis (m) distance to be <= this value.
        # [batch_size, num_gaussian, act_dim]
        Sigma_beta = torch.pow(dist.stddev, 2)
        # [batch_size, num_gaussian]
        log_weights = weights.log()
        # [batch_size, num_gaussian]
        log_p_mu = self.calc_log_p_mu(Sigma_beta)

        if self.use_max_lambda:
            pseudo_log_p_mu = (log_weights + log_p_mu).max(-1, keepdim=True)[0]
        else:
            pseudo_log_p_mu = (log_weights + log_p_mu).sum(-1, keepdim=True)

        # [batch_size, num_delta, num_gaussian]
        max_delta = 2 * (  #! refer to appendix in paper
            self.delta + (log_weights - pseudo_log_p_mu + log_p_mu).unsqueeze(1)
        ).clamp(min=0.0)

        # * calculate gradient of q lower bound w.r.t action
        pre_tanh_mu_beta.requires_grad_()
        # [batch_size * num_gaussian, obs_dim]
        obs_exp = obs.repeat_interleave(num_gaussians, dim=0)
        # [batch_size * num_gaussian, act_dim]
        mu_beta = torch.tanh(pre_tanh_mu_beta.reshape(-1, pre_tanh_mu_beta.shape[-1]))

        # Get the lower bound of the Q estimate
        # [batch_size * num_gaussian, 1, ensemble_size]
        q_LB = self.calc_q_LB(obs_exp, mu_beta)
        # [batch_size * num_gaussian, 1]
        q_LB = q_LB.reshape(-1, num_gaussians)

        # Obtain the gradient of q_LB wrt to a
        # with a evaluated at mu_proposal
        grad = torch.autograd.grad(
            q_LB.sum(), pre_tanh_mu_beta
        )  #! this returns a tuple!!
        # [batch_size, num_gaussian, act_dim]
        grad = grad[0]

        assert grad is not None
        assert pre_tanh_mu_beta.shape == grad.shape

        # * calculate proposals
        denom = self.get_shift_denominator(grad, Sigma_beta)
        # [batch_size, num_gaussians, action_dim]
        direction = (torch.mul(Sigma_beta, grad) / denom).unsqueeze(1)

        # [batch_size, num_delta, num_gaussians, action_dim]
        delta_mu = torch.sqrt(2 * max_delta).unsqueeze(-1) * direction

        mu_proposal = (pre_tanh_mu_beta.unsqueeze(1) + delta_mu).reshape(
            batch_size, self.num_delta * num_gaussians, -1
        )
        # [batch_size * num_gaussians * num_delta, action_dim]
        tanh_mu_proposal = torch.tanh(mu_proposal).reshape(
            batch_size * self.num_delta * num_gaussians, -1
        )

        # * get the lower bounded q
        assert self.num_delta == 1
        obs_exp = obs.repeat_interleave(self.num_delta * num_gaussians, dim=0)
        # obs_exp = obs.repeat_(self.num_delta * num_gaussians, 1)
        q_LB = self.calc_q_LB(obs_exp, tanh_mu_proposal)
        q_LB = q_LB.reshape(batch_size, num_gaussians * self.num_delta)
        # mask low probabilities
        q_LB[(weights.repeat(1, self.num_delta) < self.trivial_threshold)] = -torch.inf

        # * argmax the proposals
        max_value, idx = torch.max(q_LB, dim=1)
        select_mu_proposal = mu_proposal[torch.arange(len(idx)), idx]

        return select_mu_proposal, max_value

    def compute_jensen_proposal(self, obs, dist: TanhGaussianMixture) -> torch.tensor:
        # * preliminaries
        mean_per_comp = dist.normal_mean
        weights = dist.weights
        batch_size = obs.shape[0]
        Sigma_beta = torch.pow(dist.stddev, 2) + 1e-6
        normalized_factor = (weights.unsqueeze(-1) / Sigma_beta).sum(
            1
        )  #! this is "A" in the paper
        pre_tanh_mu_bar = (weights.unsqueeze(-1) / Sigma_beta * mean_per_comp).sum(
            1
        ) / normalized_factor

        # * calculate delta. this is the m distance constraint. we require the Mahalanobis (m) distance to be <= this value.
        # [batch_size, num_gaussian]
        # jensen_delta = -2 * self.tau + (weights * log_p_mu).sum(-1)
        jensen_delta = self.delta  # this is flexible

        # Obtain the change in mu
        pseudo_delta = (
            2 * jensen_delta
            - (weights * (torch.pow(mean_per_comp, 2) / Sigma_beta).sum(-1))
            .sum(1, keepdim=True)
            .unsqueeze(1)
            + (torch.pow(pre_tanh_mu_bar, 2) * normalized_factor)
            .sum(1, keepdim=True)
            .unsqueeze(1)
        )
        pre_tanh_mu_bar.requires_grad_()
        mu_bar = torch.tanh(pre_tanh_mu_bar)

        if torch.all(pseudo_delta < 0):
            return mu_bar, ptu.ones(mu_bar.shape[0]) * -torch.inf

        # * calculate gradient of q lower bound w.r.t action
        q_LB = self.calc_q_LB(obs, mu_bar)
        # Obtain the gradient of q_LB wrt to a
        # with a evaluated at mu_proposal
        grad = torch.autograd.grad(q_LB.sum(), pre_tanh_mu_bar)[0]

        assert grad is not None
        assert pre_tanh_mu_bar.shape == grad.shape

        denom = self.get_shift_denominator(grad, 1 / normalized_factor)

        numerator = torch.sqrt((pseudo_delta).clamp(min=0.0))
        delta_mu = numerator * (
            torch.mul(1 / normalized_factor, grad) / denom
        ).unsqueeze(1)

        # * calculate proposals
        mu_proposal = pre_tanh_mu_bar.unsqueeze(1) + delta_mu
        jensen_value = (delta_mu * grad).sum(-1) + q_LB.squeeze(-1)

        # * get the lower bounded q
        obs_exp = obs.repeat(self.num_delta, 1)
        q_LB = self.calc_q_LB(
            obs_exp, torch.tanh(mu_proposal).reshape(batch_size * self.num_delta, -1)
        )
        q_LB = q_LB.reshape(batch_size, self.num_delta)
        q_LB[(pseudo_delta <= -1e-10).squeeze(-1)] = -torch.inf

        # * argmax the proposals
        jensen_value, idx = torch.max(q_LB, dim=1)
        select_mu_proposal = mu_proposal[torch.arange(len(idx)), idx]

        # # * optionally check correctness
        if False:
            for i in range(self.num_delta):
                if q_LB[0, i] == -torch.inf:
                    continue
                self.check_jensen_correctness(
                    grad,
                    mean_per_comp,
                    Sigma_beta,
                    weights,
                    dist,
                    mu_proposal[0, i],
                    2 * jensen_delta[0, i],
                    # -2 * self.delta[0, i] - (weights * self.calc_log_p_mu(Sigma_beta)).sum(-1),
                )

        return select_mu_proposal, jensen_value

    def check_jensen_correctness(
        self,
        grad,
        mean_per_comp,
        Sigma_beta,
        weights,
        dist,
        predicted_mu_proposal,
        delta,
    ):
        print("checking jensen correctness...")
        import cvxpy as cp

        # Construct the problem.
        grad_np = ptu.to_np(grad.squeeze())
        mean_per_comp_np = ptu.to_np(mean_per_comp.squeeze())
        Sigma_beta_np = ptu.to_np(Sigma_beta.squeeze())
        weights_np = ptu.to_np(weights.squeeze())
        num_comp = dist.num_gaussians

        x = cp.Variable(grad_np.shape[0])
        objective = cp.Minimize(-(grad_np @ x))
        constraints = [
            sum(
                [
                    sum((x - mean_per_comp_np[i]) ** 2 / Sigma_beta_np[i])
                    * weights_np[i]
                    for i in range(num_comp)
                ]
            )
            <= delta.item()
        ]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        prob.solve()
        # # The optimal value for x is stored in `x.value`.
        if x.value is not None:  #! why is this none sometimes?
            assert np.allclose(
                ptu.to_np(predicted_mu_proposal), x.value, atol=1e-1
            ), f"{predicted_mu_proposal} != {x.value}"
        # The optimal Lagrange multiplier for a constraint is stored in
        # `constraint.dual_value`.
        # print(constraints[0].dual_value


def mg_sanity_check(ctx: PipelineCtx):
    assert ctx.variant["checkpoint_params"] != "SG"


MGBasePipeline = Pipeline.from_(
    Pipelines.offline_zerostep_pac_pipeline, "MGBasePipeline"
)
MGBasePipeline.pipeline.insert(0, mg_sanity_check)

# * --------------------------------------------------


def create_stochastic_eval_policy(ctx: PipelineCtx):
    ctx.eval_policy = ctx.policy


MGEvalBCPipeline = Pipeline.from_(MGBasePipeline, "MGEvalBCPipeline")
MGEvalBCPipeline.replace("create_pac_eval_policy", create_stochastic_eval_policy)

# * --------------------------------------------------

MGIQLPipeline = Pipeline.from_(MGBasePipeline, "MGIQLPipeline")


def iql_sanity_check(ctx):
    assert ctx.variant["d4rl"]
    assert ctx.variant["algorithm_kwargs"]["zero_step"]
    # if 'antmaze' in ctx.variant['env_id']:
    #     assert ctx.variant["normalize_env"] == False
    # else:
    #     assert ctx.variant["normalize_env"] == True

    assert not ctx.variant["IQN"]
    assert (
        ctx.variant["checkpoint_params"]        
        in user_defined_attrs_dict(conf.CheckpointParams).keys()
    )
    assert ctx.variant["checkpoint_params"] != "Q"

    params: conf.CheckpointParams.CheckpointParam = getattr(conf.CheckpointParams, ctx.variant["checkpoint_params"])()    
    assert ctx.variant["seed"] in params.seeds
    assert ctx.variant["env_id"] in params.envs, ctx.variant["env_id"]
    assert ctx.variant["seed"] in conf.CheckpointParams.Q_IQL.seeds
    assert ctx.variant["env_id"] in conf.CheckpointParams.Q_IQL.envs


MGIQLPipeline.pipeline.insert(6, create_q)

MGIQLPipeline.replace("pac_sanity_check", iql_sanity_check)
MGIQLPipeline.replace("load_checkpoint_iqn_q", load_checkpoint_iql_q)


AllIQLPipeline = Pipeline.from_(MGIQLPipeline, "AllIQLPipeline")
AllIQLPipeline.delete("mg_sanity_check")
AllIQLPipeline.replace("load_checkpoint_policy", load_checkpoint_iql_policy)

# * --------------------------------------------------
# MGPAC + IQL for ICLR rebuttal
# Q functions are trained by the IQL with normalize_env = False
# Policies are trained in our repos with normalize_env = False
MGIQLAntMazePipeline = Pipeline.from_(MGIQLPipeline, "MGIQLAntMazePipeline")


def mg_iql_antmaze_sanity_check(ctx):
    assert ctx.variant["d4rl"]
    assert ctx.variant["algorithm_kwargs"]["zero_step"]
    assert "antmaze" in ctx.variant["env_id"]
    assert ctx.variant["normalize_env"] is False

    assert not ctx.variant["IQN"]
    assert not ctx.variant["trainer_kwargs"]["IQN"]
    assert (
        ctx.variant["checkpoint_params"]
        in user_defined_attrs_dict(conf.CheckpointParams).keys()
    )
    assert ctx.variant["checkpoint_params"] != "Q"

    params = getattr(conf.CheckpointParams, ctx.variant["checkpoint_params"])
    assert ctx.variant["seed"] in params.seeds
    assert ctx.variant["env_id"] in params.envs, ctx.variant["env_id"]
    assert ctx.variant["seed"] in conf.CheckpointParams.Q_IQL.seeds
    assert ctx.variant["env_id"] in conf.CheckpointParams.Q_IQL.envs


def load_checkpoint_antmaze_iql_q(ctx: PipelineCtx):
    q_params = conf.CheckpointParams.Q_IQL

    params = torch.load(
        osp.join(
            conf.CHECKPOINT_PATH,
            q_params.path,
            ctx.variant["env_id"],
            str(ctx.variant["seed"]),
            "params.pt",
        ),
        map_location=ptu.device,
    )

    ctx.qfs[0].load_state_dict(params["trainer/qf1"])
    ctx.qfs[1].load_state_dict(params["trainer/qf2"])


def load_checkpoint_antmaze_iql_policy(ctx: PipelineCtx):
    params = getattr(conf.CheckpointParams, ctx.variant["checkpoint_params"])

    policy_path = ""
    base = osp.join(
        conf.CHECKPOINT_PATH,
        params.path,
        ctx.variant["env_id"],
        str(ctx.variant["seed"]),
    )

    policy_path = osp.join(base, "itr_500.pt")
    ctx.policy = torch.load(policy_path, map_location="cpu")[params.key]


MGIQLAntMazePipeline.replace("iql_sanity_check", mg_iql_antmaze_sanity_check)
MGIQLAntMazePipeline.replace("load_checkpoint_iql_q", load_checkpoint_antmaze_iql_q)
MGIQLAntMazePipeline.replace(
    "load_checkpoint_policy", load_checkpoint_antmaze_iql_policy
)

# BC via Sampling
MGIQLAntMazeEvalBCPipeline = Pipeline.from_(
    MGIQLAntMazePipeline, "MGIQLAntMazeEvalBCPipeline"
)
MGIQLAntMazeEvalBCPipeline.replace(
    "create_pac_eval_policy", create_stochastic_eval_policy
)

MGIQLAntMazeMleEvalBCPipeline = Pipeline.from_(
    MGIQLAntMazePipeline, "MGIQLAntMazeMleEvalBCPipeline"
)
MGIQLAntMazeMleEvalBCPipeline.replace("create_pac_eval_policy", create_eval_policy)


def load_checkpoint_antmaze_ensemble_iql_q(ctx: PipelineCtx):
    q_params = conf.CheckpointParams.Q_IQL_ENSEMBLE

    params = torch.load(
        osp.join(
            conf.CHECKPOINT_PATH,
            q_params.path,
            ctx.variant["env_id"],
            str(ctx.variant["seed"]),
            "itr_700.pt",
        ),
        map_location=ptu.device,
    )
    ctx.qfs = params["trainer/qfs"]
    ctx.target_qfs = params["trainer/target_qfs"]


MGEnsembleIQLAntMazePipeline = Pipeline.from_(
    MGIQLAntMazePipeline, "MGEnsembleIQLAntMazePipeline"
)
MGEnsembleIQLAntMazePipeline.replace(
    "load_checkpoint_antmaze_iql_q", load_checkpoint_antmaze_ensemble_iql_q
)


# * --------------------------------------------------
# Multi-step MGPAC on AntMaze for ICLR rebuttal
# normalize_env = False
# Policies are trained in our repos with normalize_env = False


def multi_step_mg_antmaze_sanity_check(ctx):
    assert ctx.variant["d4rl"]
    assert "antmaze" in ctx.variant["env_id"]
    assert ctx.variant["normalize_env"] is False

    assert ctx.variant["trainer_kwargs"]["IQN"]
    assert (
        ctx.variant["checkpoint_params"]
        in user_defined_attrs_dict(conf.CheckpointParams).keys()
    )

    params = getattr(conf.CheckpointParams, ctx.variant["checkpoint_params"])
    assert ctx.variant["seed"] in params.seeds
    assert ctx.variant["env_id"] in params.envs, ctx.variant["env_id"]
    assert ctx.variant["seed"] in conf.CheckpointParams.Q_IQL.seeds
    assert ctx.variant["env_id"] in conf.CheckpointParams.Q_IQL.envs


IterativepMGAntMazePipeline = Pipeline.from_(
    MGBasePipeline, "IterativepMGAntMazePipeline"
)
IterativepMGAntMazePipeline.replace(
    "pac_sanity_check", multi_step_mg_antmaze_sanity_check
)
IterativepMGAntMazePipeline.replace("load_checkpoint_iqn_q", create_q_iqn)
IterativepMGAntMazePipeline.replace(
    "load_checkpoint_policy", load_checkpoint_antmaze_iql_policy
)


def create_dataset_next_actions_new_actions(ctx: PipelineCtx):
    ctx.dataset = d4rl_qlearning_dataset_with_next_actions_new_actions(ctx.eval_env)
    if "antmaze" in ctx.variant["env_id"]:
        ctx.dataset["rewards"] -= 1


MultiStepMGAntMazePipeline = Pipeline.from_(
    MGBasePipeline, "MultiStepMGAntMazePipeline"
)
MultiStepMGAntMazePipeline.replace(
    "create_dataset_next_actions", create_dataset_next_actions_new_actions
)
