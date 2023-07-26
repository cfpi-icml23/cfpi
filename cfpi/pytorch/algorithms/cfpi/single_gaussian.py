""" 
PAC with a Q Lower Bound, with pretrained q and pi beta from one step repository.
"""

from typing import List, Union

import pickle
from collections import OrderedDict
from os import path as osp

import eztils.torch as ptu
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from eztils import create_stats_ordered_dict
from eztils.torch import (
    Delta,
    MultivariateDiagonalNormal,
    TanhDelta,
    TanhGaussianMixture,
    TanhNormal,
)

from cfpi import conf
from cfpi.launchers.pipeline import Pipeline, PipelineCtx, Pipelines
from cfpi.policies.gaussian_policy import (
    TanhGaussianPolicy,
    UnnormalizeTanhGaussianPolicy,
)
from cfpi.pytorch.algorithms.q_training.sarsa_iqn import (
    get_tau,
    quantile_regression_loss,
)
from cfpi.pytorch.networks.mlp import ParallelMlp, QuantileMlp
from cfpi.pytorch.torch_rl_algorithm import TorchTrainer


class SG_CFPI_Trainer(TorchTrainer):
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
        target_policy_noise=0.1,
        target_policy_noise_clip=0.5,
        num_quantiles=8,
        plotter=None,
        render_eval_paths=False,
        # NEW PARAMS
        beta_LB=0.5,
        easy_bcq=False,
        num_candidate_actions=10,
        delta_range=None,
        num_delta=None,
        IQN=True,
    ):
        super().__init__()
        if delta_range is None:
            delta_range = [0.0, 0.0]

        self.iqn = IQN
        self.policy = policy
        self.qfs: Union[List[QuantileMlp], ParallelMlp] = qfs
        self.target_qfs = target_qfs
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.num_quantiles = num_quantiles
        if IQN:
            self.qf_criterion = quantile_regression_loss
        else:
            self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        if isinstance(qfs, nn.Module):
            self.qfs_optimizer = optimizer_class(
                self.qfs.parameters(),
                lr=qf_lr,
            )
        elif isinstance(qfs, list):
            self.qf1_optimizer = optimizer_class(
                self.qfs[0].parameters(),
                lr=qf_lr,
            )
            self.qf2_optimizer = optimizer_class(
                self.qfs[1].parameters(),
                lr=qf_lr,
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

        # New params
        self.beta_LB = beta_LB
        # [1(for batch_size), num_delta, 1(for action_dim)]
        self.delta_range = delta_range

        assert (
            len(self.delta_range) == 2
        ), f"Delta range ({self.delta_range}) should be in the form of [lower_range, upper_range]!"

        if self.delta_range != [0.0, 0.0]:
            assert (
                self.delta_range[0] <= self.delta_range[1]
            ), f"Delta range ({self.delta_range}) should be in the form of [lower_range, upper_range]!"

        if num_delta is None:
            num_delta = int((self.delta_range[1] - self.delta_range[0]) * 10)

        if self.delta_range[1] == self.delta_range[0]:
            num_delta = 1

        self.num_delta = num_delta
        self.easy_bcq = easy_bcq
        self.num_candidate_actions = num_candidate_actions
        self.behavior_policy = self.policy  # for naming's sake
        self.max_action = 1.0

        print("-----------------")
        print("Delta:", delta_range)
        print("Num delta:", self.num_delta)
        print("beta_LB:", self.beta_LB)
        print("-----------------")

    def get_action_dist(self, obs):
        return self.behavior_policy(obs)

    def get_iqn_outputs(self, obs, act, use_target=False):
        qfs = self.target_qfs if use_target else self.qfs
        res = ptu.zeros(obs.shape[0], len(qfs))
        for i, iqn in enumerate(qfs):
            res[:, i] = iqn.get_mean(obs, act)

        return res

    def get_shift_denominator(self, grad, sigma):
        # The dividor is (g^T Sigma g) ** 0.5
        # Sigma is diagonal, so this works out to be
        # ( sum_{i=1}^k (g^(i))^2 (sigma^(i))^2 ) ** 0.5
        return (
            torch.sqrt(
                torch.sum(torch.mul(torch.pow(grad, 2), sigma), dim=1, keepdim=True)
            )
            + 1e-7
        )

    def sample_delta(self):
        """
        Sample and set delta for this range.
        """
        self.delta = (
            ptu.rand(self.num_delta) * (self.delta_range[1] - self.delta_range[0])
            + self.delta_range[0]
        ).reshape(1, self.num_delta, 1)

    def get_easy_bcq_action(self, obs, action_dist):
        batch_size = obs.shape[0]
        if isinstance(action_dist, TanhGaussianMixture):
            proposals = action_dist.sample(self.num_candidate_actions).squeeze()
        else:
            proposals = action_dist.sample_n(self.num_candidate_actions).reshape(
                batch_size * self.num_candidate_actions, -1
            )
        obs_exp = obs.repeat_interleave(self.num_candidate_actions, dim=0)
        assert batch_size == 1
        qfs = self.calc_q_LB(obs_exp, proposals).reshape(batch_size, -1)
        idx = torch.argmax(qfs, dim=1)
        selected_actions = proposals[idx].squeeze()
        return Delta(selected_actions)

    def get_cfpi_action(self, obs) -> TanhDelta:
        dist: TanhNormal = self.get_action_dist(obs)
        is_train = obs.shape[0] > 1

        if self.easy_bcq:
            return self.get_easy_bcq_action(obs, dist)

        if self.delta_range == [0.0, 0.0]:
            if self.iqn:
                return TanhDelta(dist.normal_mean)
            else:
                return Delta(dist.mean)

        self.sample_delta()
        if self.iqn:
            return self.compute_cfpi_action(obs, dist, is_train)
        else:
            return self.compute_cfpi_action_iql(obs, dist)

    def calc_q_LB(self, obs, act, use_target=False):
        assert not use_target
        if isinstance(self.qfs, list):
            q = self.get_iqn_outputs(obs, act, use_target)

            mu_q = q.mean(-1)
            sigma_q = q.std(-1)
            q_LB = mu_q - self.beta_LB * sigma_q
            return q_LB
        else:
            qfs = self.target_qfs if use_target else self.qfs
            sample_idxs = np.random.choice(10, 2, replace=False)
            q = qfs(obs, act)
            q_LB = q[..., sample_idxs].min(-1)
            return qfs(obs, act).mean(-1)

    def compute_cfpi_action(self, obs, dist: TanhNormal, is_train: bool):
        # * preliminaries

        pre_tanh_mu_beta = dist.normal_mean
        batch_size = obs.shape[0]
        pre_tanh_mu_beta.requires_grad_()
        mu_beta = torch.tanh(pre_tanh_mu_beta)

        # * calculate gradient of q lower bound w.r.t action
        # Get the lower bound of the Q estimate
        q_LB = self.calc_q_LB(obs, mu_beta, use_target=is_train)
        # Obtain the gradient of q_LB wrt to action
        # with action evaluated at mu_proposal
        grad = torch.autograd.grad(q_LB.sum(), pre_tanh_mu_beta)[
            0
        ]  #! note: this doesn't work on cpu.

        assert grad is not None
        assert pre_tanh_mu_beta.shape == grad.shape

        # * cacluate proposals
        # Obtain Sigma_T (the covariance matrix of the normal distribution)
        Sigma_beta = torch.pow(dist.stddev, 2)

        denom = self.get_shift_denominator(grad, Sigma_beta)

        # [batch_size, num_deltas, action_dim]
        delta_mu = torch.sqrt(2 * self.delta) * (
            torch.mul(Sigma_beta, grad) / denom
        ).unsqueeze(1)

        mu_proposal = pre_tanh_mu_beta + delta_mu
        tanh_mu_proposal = torch.tanh(mu_proposal).reshape(
            batch_size * self.num_delta, -1
        )

        # * get the lower bounded q
        obs_exp = obs.repeat_interleave(self.num_delta, dim=0)
        q_LB = self.calc_q_LB(obs_exp, tanh_mu_proposal)
        q_LB = q_LB.reshape(batch_size, self.num_delta)

        # * argmax the proposals
        select_idx = q_LB.argmax(1)
        selected = mu_proposal[torch.arange(len(select_idx)), select_idx]
        return TanhDelta(selected)

    def compute_cfpi_action_iql(self, obs, dist: MultivariateDiagonalNormal):
        # * preliminaries

        mu_beta = dist.mean
        batch_size = obs.shape[0]
        mu_beta.requires_grad_()

        # * calculate gradient of q lower bound w.r.t action
        # Get the lower bound of the Q estimate
        q_LB = self.calc_q_LB(obs, mu_beta)
        # Obtain the gradient of q_LB wrt to a
        # with a evaluated at mu_proposal
        grad = torch.autograd.grad(q_LB.sum(), mu_beta)[
            0
        ]  #! note: this doesn't work on cpu.

        assert grad is not None
        assert mu_beta.shape == grad.shape

        # * cacluate proposals
        # Obtain Sigma_T (the covariance matrix of the normal distribution)
        Sigma_beta = torch.pow(dist.stddev, 2)

        denom = self.get_shift_denominator(grad, Sigma_beta)

        # [batch_size, num_deltas, action_dim]
        delta_mu = torch.sqrt(2 * self.delta) * (
            torch.mul(Sigma_beta, grad) / denom
        ).unsqueeze(1)

        mu_proposal = torch.clamp(mu_beta + delta_mu, -1, 1)
        mu_proposal_reshaped = torch.clamp(mu_proposal, -1, 1).reshape(
            batch_size * self.num_delta, -1
        )

        # * get the lower bounded q
        obs_exp = obs.repeat_interleave(self.num_delta, dim=0)
        q_LB = self.calc_q_LB(obs_exp, mu_proposal_reshaped)
        q_LB = q_LB.reshape(batch_size, self.num_delta)

        # * argmax the proposals
        select_idx = q_LB.argmax(1)
        selected = mu_proposal[torch.arange(len(select_idx)), select_idx]
        return Delta(selected)

    def train_from_torch(self, batch):
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        losses.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.qfs[0], self.target_qfs[0], self.soft_target_tau)
        ptu.soft_update_from_to(self.qfs[1], self.target_qfs[1], self.soft_target_tau)

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        next_actions = batch["next_actions"]

        """
        QF Loss
        """
        assert isinstance(self.qfs[0], QuantileMlp)
        assert isinstance(self.qfs[1], QuantileMlp)

        batch_size = obs.shape[0]
        tau_hat_samples, presum_tau_samples = get_tau(
            batch_size * 2, self.num_quantiles
        )
        tau_hat, next_tau_hat = tau_hat_samples.reshape(2, batch_size, -1)
        presum_tau, next_presum_tau = presum_tau_samples.reshape(2, batch_size, -1)

        z1_pred = self.qfs[0](obs, actions, tau_hat)
        z2_pred = self.qfs[1](obs, actions, tau_hat)

        if self._n_train_steps_total < 20 * 1000:
            new_next_actions = next_actions
        else:
            try:
                new_next_actions = batch["new_next_actions"]
            except KeyError:
                next_dist = self.get_cfpi_action(next_obs)
                new_next_actions = next_dist.mean
            noise = ptu.randn(new_next_actions.shape) * self.target_policy_noise
            noise = torch.clamp(
                noise, -self.target_policy_noise_clip, self.target_policy_noise_clip
            )
            new_next_actions = torch.clamp(
                new_next_actions + noise, -self.max_action, self.max_action
            )

        with torch.no_grad():
            target_z1_value = self.target_qfs[0](
                next_obs, new_next_actions, next_tau_hat
            )
            target_z2_value = self.target_qfs[1](
                next_obs, new_next_actions, next_tau_hat
            )

            target_z_value = torch.min(target_z1_value, target_z2_value)
            z_target = (
                self.reward_scale * rewards
                + (1.0 - terminals) * self.discount * target_z_value
            ).detach()

        qf1_loss = self.qf_criterion(z1_pred, z_target, tau_hat, next_presum_tau)
        qf2_loss = self.qf_criterion(z2_pred, z_target, tau_hat, next_presum_tau)
        qfs_loss = qf1_loss + qf2_loss

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            with torch.no_grad():
                eval_statistics["QF1 Loss"] = ptu.to_np(qf1_loss)
                eval_statistics["QF2 Loss"] = ptu.to_np(qf2_loss)

                q1_pred = (z1_pred * presum_tau).sum(-1)
                q2_pred = (z2_pred * presum_tau).sum(-1)
                q_preds = (q1_pred + q2_pred) / 2

            target_q_value = (target_z_value * next_presum_tau).sum(-1)

            eval_statistics.update(
                create_stats_ordered_dict(
                    "Mean Q Predictions",
                    ptu.to_np(q_preds),
                )
            )
            eval_statistics.update(
                create_stats_ordered_dict(
                    "Mean Target Q Predictions",
                    ptu.to_np(target_q_value),
                )
            )

        return qfs_loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        if isinstance(self.qfs, list):
            return self.qfs + self.target_qfs + [self.policy]
        else:
            return [
                self.policy,
                self.qfs,
                self.target_qfs,
            ]

    @property
    def optimizers(self):
        return [
            self.qfs_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.target_qfs,
        )


def sg_sanity_check(ctx: PipelineCtx):
    assert ctx.variant["checkpoint_params"] == "SG"


SGBasePipeline = Pipeline.from_(
    Pipelines.offline_zerostep_pac_pipeline, "SGBasePipeline"
)
SGBasePipeline.pipeline.insert(0, sg_sanity_check)

# * --------------------------------------------------




def variable_epoch_load_checkpoint_policy(ctx: PipelineCtx):
    params = getattr(conf.CheckpointParams, ctx.variant["checkpoint_params"])

    ctx.policy = torch.load(
        osp.join(
            conf.CHECKPOINT_PATH,
            params.path,
            ctx.variant["env_id"],
            str(ctx.variant["seed"]),
            f'itr_+{ctx.variant["epoch_no"]}'.pt,
        ),
        map_location="cpu",
    )[params.key]

    if params.unnormalize:
        if (
            ctx.variant["env_id"] == "halfcheetah-medium-expert-v2"
            and ctx.variant["seed"] < 4
        ):
            pass
        else:
            ctx.policy = UnnormalizeTanhGaussianPolicy(
                ctx.obs_mean, ctx.obs_std, ctx.policy
            )


EpochBCExperiment = Pipeline.from_(SGBasePipeline, "EpochBCExperiment")
EpochBCExperiment.replace(
    "load_checkpoint_policy", variable_epoch_load_checkpoint_policy
)


# * --------------------------------------------------
def load_gt_policy(d4rl_datset):
    policy_dict = {k: v for k, v in d4rl_datset.items() if "bias" in k or "weight" in k}
    policy_dict = {
        ".".join(k.split("/")[2:]): ptu.torch_ify(v) for k, v in policy_dict.items()
    }
    assert len(policy_dict) == 8

    hidden_sz, obs_dim = policy_dict["fc0.weight"].shape
    act_dim, hidden_sz = policy_dict["last_fc.weight"].shape

    pi = TanhGaussianPolicy([256, 256], obs_dim, act_dim)
    pi.load_state_dict(policy_dict)
    return pi


def load_ground_truth_policy(ctx: PipelineCtx):
    ctx.policy = load_gt_policy(ctx.dataset)


GTExperiment = Pipeline.from_(SGBasePipeline, "GroundTruthExperiment")
GTExperiment.replace("load_checkpoint_policy", load_ground_truth_policy)


# * --------------------------------------------------
def load_checkpoint_cql(ctx: PipelineCtx):
    ctx.CQL = pickle.load(
        open(
            osp.join(
                conf.CHECKPOINT_PATH,
                "cql-models",
                ctx.variant["env_id"],
                str(ctx.variant["seed"]),
                "model_989.pkl",
            ),
            "rb",
        )
    )["sac"]


def load_cql_policy(ctx: PipelineCtx):
    class TanhNormalWrapper(nn.Module):
        def __init__(self, policy) -> None:
            super().__init__()
            self.policy = policy

        def forward(self, *args, **kwargs):
            mu, std = self.policy(*args, **kwargs)
            return TanhNormal(mu, std)

    ctx.policy = TanhNormalWrapper(ctx.CQL.policy)


def load_cql_q(ctx: PipelineCtx):
    load_checkpoint_cql(ctx)

    ctx.qfs.append(ctx.CQL.qf1)
    ctx.qfs.append(ctx.CQL.qf2)


CQLExperiment = Pipeline.from_(SGBasePipeline, "CQLExperiment")
CQLExperiment.replace("load_checkpoint_policy", load_cql_policy)
CQLExperiment.replace("load_checkpoint_iqn_q", load_cql_q)


# def create_q(ctx: PipelineCtx):
#     obs_dim = ctx.eval_env.observation_space.low.size
#     action_dim = ctx.eval_env.action_space.low.size

#     qf1 = ctx.variant["qf_class"](
#         input_size=obs_dim + action_dim, output_size=1, **ctx.variant["qf_kwargs"]
#     )
#     qf2 = ctx.variant["qf_class"](
#         input_size=obs_dim + action_dim, output_size=1, **ctx.variant["qf_kwargs"]
#     )

#     target_qf1 = ctx.variant["qf_class"](
#         input_size=obs_dim + action_dim, output_size=1, **ctx.variant["qf_kwargs"]
#     )
#     target_qf2 = ctx.variant["qf_class"](
#         input_size=obs_dim + action_dim, output_size=1, **ctx.variant["qf_kwargs"]
#     )

#     ctx.qfs = [qf1, qf2]
#     ctx.target_qfs = [target_qf1, target_qf2]

# SGAntMazePipeline = Pipeline.from_(SGBasePipeline, "SGAntMazePipeline")
# SGAntMazePipeline.pipeline.insert(4, create_q)
# SGAntMazePipeline.replace("load_checkpoint_iqn_q", load_checkpoint_iql_q)
