"""Torch implementation of Ensemble Implicit Q-Learning (IQL) based off
https://github.com/ikostrikov/implicit_q_learning
"""

from collections import OrderedDict

import eztils.torch as ptu
import numpy as np
import torch
import torch.optim as optim
from eztils import create_stats_ordered_dict
from torch import nn as nn

from cfpi.core.logging import add_prefix
from cfpi.launchers.pipeline import Pipeline, PipelineCtx
from cfpi.launchers.pipeline_pieces import (
    create_algorithm,
    create_dataset_next_actions,
    create_eval_env,
    create_eval_path_collector,
    create_eval_policy,
    create_policy,
    create_replay_buffer,
    create_trainer,
    load_demos,
    offline_init,
    optionally_normalize_dataset,
    train,
)
from cfpi.policies.gaussian_policy import GaussianPolicy
from cfpi.pytorch.networks import LinearTransform
from cfpi.pytorch.networks.mlp import ParallelMlp
from cfpi.pytorch.torch_rl_algorithm import TorchTrainer


class EnsembleIQLTrainer(TorchTrainer):
    def __init__(
        self,
        policy,
        qfs,
        target_qfs,
        vfs,
        quantile=0.5,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        policy_weight_decay=0,
        q_weight_decay=0,
        optimizer_class=optim.Adam,
        policy_update_period=1,
        q_update_period=1,
        reward_transform_class=None,
        reward_transform_kwargs=None,
        terminal_transform_class=None,
        terminal_transform_kwargs=None,
        clip_score=None,
        soft_target_tau=1e-2,
        target_update_period=1,
        beta=1.0,
    ):
        super().__init__()

        # https://github.com/rail-berkeley/rlkit/blob/master/examples/iql/antmaze_finetune.py
        assert reward_scale == 1.0
        assert policy_weight_decay == 0
        assert q_weight_decay == 0
        # NOTE: We set b = 0, as we already do the minus 1 in create_dataset_next_actions
        assert reward_transform_kwargs == dict(m=1, b=0)
        assert terminal_transform_kwargs is None
        assert quantile == 0.9
        assert clip_score == 100
        assert beta == 0.1

        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.vfs = vfs

        self.optimizers = {}

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            weight_decay=policy_weight_decay,
            lr=policy_lr,
        )
        self.optimizers[self.policy] = self.policy_optimizer
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr,
        )
        self.vfs_optimizer = optimizer_class(
            self.vfs.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period

        self.reward_transform_class = reward_transform_class or LinearTransform
        self.reward_transform_kwargs = reward_transform_kwargs or dict(m=1, b=0)
        self.terminal_transform_class = terminal_transform_class or LinearTransform
        self.terminal_transform_kwargs = terminal_transform_kwargs or dict(m=1, b=0)
        self.reward_transform = self.reward_transform_class(
            **self.reward_transform_kwargs
        )
        self.terminal_transform = self.terminal_transform_class(
            **self.terminal_transform_kwargs
        )

        self.clip_score = clip_score
        self.beta = beta
        self.quantile = quantile

    def train_from_torch(
        self,
        batch,
        # train=True,
        # pretrain=False,
    ):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        batch_size = rewards.shape[0]
        if self.reward_transform:
            rewards = self.reward_transform(rewards)

        if self.terminal_transform:
            terminals = self.terminal_transform(terminals)
        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)

        """
        QF Loss
        """
        # [batch_size, 1, num_heads]
        q_preds = self.qfs(obs, actions)
        # [batch_size, 1, num_heads // 2]
        target_vf_preds = self.vfs(next_obs).detach().repeat_interleave(2, dim=-1)
        assert torch.equal(target_vf_preds[..., 0], target_vf_preds[..., 1])

        with torch.no_grad():
            terminals = terminals.unsqueeze(-1).expand(-1, -1, self.qfs.num_heads)
            rewards = rewards.unsqueeze(-1).expand(-1, -1, self.qfs.num_heads)
            q_target = (
                self.reward_scale * rewards
                + (1.0 - terminals) * self.discount * target_vf_preds
            )

        qfs_loss = torch.sum(
            torch.mean(
                (q_preds - q_target.detach()) ** 2,
                dim=0,
            )
        )

        """
        VF Loss
        """
        # [batch_size, 1, num_heads]
        target_q_preds = self.target_qfs(obs, actions)
        # [batch_size, 1, num_heads // 2, 2]
        target_q_preds_reshape = target_q_preds.reshape(batch_size, 1, -1, 2)
        assert torch.equal(target_q_preds_reshape[:, :, 0], target_q_preds[..., :2])
        # [batch_size, 1, num_heads // 2]
        q_preds = target_q_preds_reshape.min(dim=-1)[0].detach()
        # # In-distribution minimization trick from the RedQ
        # sample_idxs = np.random.choice(10, 2, replace=False)
        # q_pred = torch.min(target_q_preds[:, :, sample_idxs], dim=-1, keepdim=True)[
        #     0
        # ].detach()

        # [batch_size, 1, num_heads // 2]
        vf_preds = self.vfs(obs)
        # [batch_size, 1, num_heads // 2]
        vf_err = vf_preds - q_preds
        # [batch_size, 1, num_heads // 2]
        vf_sign = (vf_err > 0).float()
        # [batch_size, 1, num_heads // 2]
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        # Take the mean over the batch_size dim; Sum over the ensemble dim
        vfs_loss = (vf_weight * (vf_err**2)).mean(dim=0).sum()

        """
        Policy Loss
        """
        policy_logpp = dist.log_prob(actions)
        # [batch_size, 1]
        adv = (q_preds - vf_preds).mean(dim=-1)
        exp_adv = torch.exp(adv / self.beta)
        if self.clip_score is not None:
            exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        weights = exp_adv[:, 0].detach()
        policy_loss = (-policy_logpp * weights).mean()

        """
        Update networks
        """
        if self._n_train_steps_total % self.q_update_period == 0:
            self.qfs_optimizer.zero_grad()
            qfs_loss.backward()
            self.qfs_optimizer.step()

            self.vfs_optimizer.zero_grad()
            vfs_loss.backward()
            self.vfs_optimizer.step()

        if self._n_train_steps_total % self.policy_update_period == 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qfs, self.target_qfs, self.soft_target_tau)

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics["QF Loss"] = ptu.to_np(qfs_loss)
            self.eval_statistics["Policy Loss"] = np.mean(ptu.to_np(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Mean Q Predictions",
                    ptu.to_np(q_preds.mean(dim=-1)),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Mean Target Q Predictions",
                    ptu.to_np(target_vf_preds.mean(dim=-1)),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "rewards",
                    ptu.to_np(rewards[..., -1]),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "terminals",
                    ptu.to_np(terminals[..., -1]),
                )
            )
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            self.eval_statistics.update(policy_statistics)
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Advantage Weights",
                    ptu.to_np(weights),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Mean Advantage Score",
                    ptu.to_np(adv),
                )
            )

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Mean V Predictions",
                    ptu.to_np(vf_preds.mean(dim=-1)),
                )
            )
            self.eval_statistics["VF Loss"] = np.mean(ptu.to_np(vfs_loss))

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.policy,
            self.qfs,
            self.target_qfs,
            self.vfs,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.target_qfs,
            vfs=self.vfs,
        )


"""
Pipeline code
"""


def ensemble_iql_sanity(ctx):
    assert ctx.variant["d4rl"]
    assert ctx.variant["normalize_env"] is False
    assert ctx.variant["qf_class"] == ParallelMlp
    assert ctx.variant["policy_class"] == GaussianPolicy
    assert (
        ctx.variant["vf_kwargs"]["num_heads"] * 2
        == ctx.variant["qf_kwargs"]["num_heads"]
    )


def create_qv(ctx: PipelineCtx):
    obs_dim = ctx.eval_env.observation_space.low.size
    action_dim = ctx.eval_env.action_space.low.size

    qfs = ctx.variant["qf_class"](
        input_size=obs_dim + action_dim, output_size=1, **ctx.variant["qf_kwargs"]
    )
    vfs = ctx.variant["vf_class"](
        input_size=obs_dim, output_size=1, **ctx.variant["vf_kwargs"]
    )
    target_qfs = ctx.variant["qf_class"](
        input_size=obs_dim + action_dim, output_size=1, **ctx.variant["qf_kwargs"]
    )

    ctx.qfs = qfs
    ctx.vfs = vfs
    ctx.target_qfs = target_qfs


EnsembleIQLPipeline = Pipeline(
    "EnsembleIQLPipeline",
    [
        ensemble_iql_sanity,
        offline_init,
        create_eval_env,
        create_dataset_next_actions,
        optionally_normalize_dataset,
        create_qv,
        create_policy,
        create_trainer,
        create_eval_policy,
        create_eval_path_collector,
        create_replay_buffer,
        create_algorithm,
        load_demos,
        train,
    ],
)
