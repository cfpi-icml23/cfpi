""" 
Learn an ensemble of q with sarsa and pi_beta with NLL
"""
from typing import Tuple

from collections import OrderedDict, namedtuple

import eztils.torch as ptu
import numpy as np
import torch
import torch.optim as optim
from eztils import create_stats_ordered_dict
from torch import nn as nn
from torch.nn.functional import smooth_l1_loss

from cfpi.launchers.pipeline import Pipeline, PipelineCtx, Pipelines
from cfpi.launchers.pipeline_pieces import create_dataset_next_actions
from cfpi.pytorch.networks.mlp import QuantileMlp
from cfpi.pytorch.torch_rl_algorithm import TorchTrainer

SarsaLosses = namedtuple(
    "SarsaLosses",
    "qfs_loss",
)


def quantile_regression_loss(pred, target, tau, weight):
    pred = pred.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)

    weight = weight.detach().unsqueeze(-2)
    expanded_pred, expanded_target = torch.broadcast_tensors(pred, target)
    L = smooth_l1_loss(expanded_pred, expanded_target, reduction="none")  # (N, T, T)

    sign = torch.sign(expanded_pred - expanded_target) / 2.0 + 0.5
    rho = torch.abs(tau - sign) * L * weight
    return rho.sum(dim=-1).mean()


def get_tau(batch_size, num_quantiles=8):
    with torch.no_grad():
        presum_tau = ptu.rand(batch_size, num_quantiles) + 0.1
        presum_tau /= presum_tau.sum(dim=-1, keepdims=True)

        tau = torch.cumsum(
            presum_tau, dim=1
        )  # (N, T), note that they are tau1...tauN in the paper
        tau_hat = ptu.zeros_like(tau)
        tau_hat[:, 0:1] = tau[:, 0:1] / 2.0
        tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.0
    return tau_hat, presum_tau


def get_target_quantile(quantiles, tau_hat, target_percentile):
    x_idx = torch.arange(len(quantiles))
    y_idx = torch.min((tau_hat - target_percentile).abs(), dim=1)[1]
    target_percentiles = quantiles[x_idx, y_idx]
    return target_percentiles


class SarsaIQNTrainer(TorchTrainer):
    def __init__(
        self,
        eval_env,
        qfs,
        target_qfs,
        discount=0.99,
        reward_scale=1.0,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        num_quantiles=8,
        plotter=None,
        render_eval_paths=False,
        **kwargs,
    ):
        super().__init__()
        self.env = eval_env
        assert len(qfs) == 2
        self.qf1 = qfs[0]
        self.qf2 = qfs[1]
        self.target_qf1 = target_qfs[0]
        self.target_qf2 = target_qfs[1]
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = quantile_regression_loss

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

        self.num_quantiles = num_quantiles

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

        losses.qfs_loss.backward()

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
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SarsaLosses, OrderedDict]:
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        next_actions = batch["next_actions"]
        """
        QF Loss
        """

        assert isinstance(self.qf1, QuantileMlp)
        assert isinstance(self.qf2, QuantileMlp)

        batch_size = obs.shape[0]
        tau_hat_samples, presum_tau_samples = get_tau(
            batch_size * 2, self.num_quantiles
        )
        tau_hat, next_tau_hat = tau_hat_samples.reshape(2, batch_size, -1)
        presum_tau, next_presum_tau = presum_tau_samples.reshape(2, batch_size, -1)

        z1_pred = self.qf1(obs, actions, tau_hat)
        z2_pred = self.qf2(obs, actions, tau_hat)

        with torch.no_grad():
            target_z1_value = self.target_qf1(next_obs, next_actions, next_tau_hat)
            target_z2_value = self.target_qf2(next_obs, next_actions, next_tau_hat)

            z1_target = (
                self.reward_scale * rewards
                + (1.0 - terminals) * self.discount * target_z1_value
            )
            z2_target = (
                self.reward_scale * rewards
                + (1.0 - terminals) * self.discount * target_z2_value
            )

        qf1_loss = self.qf_criterion(z1_pred, z1_target, tau_hat, next_presum_tau)
        qf2_loss = self.qf_criterion(z2_pred, z2_target, tau_hat, next_presum_tau)
        qfs_loss = qf1_loss + qf2_loss

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            with torch.no_grad():
                eval_statistics["QF1 Loss"] = ptu.to_np(qf1_loss)
                eval_statistics["QF2 Loss"] = ptu.to_np(qf2_loss)

                q1_pred = (z1_pred * presum_tau).sum(-1, keepdim=True)
                q2_pred = (z2_pred * presum_tau).sum(-1, keepdim=True)
                q_preds = torch.cat([q1_pred, q2_pred], dim=-1)

                z_pred = (z1_pred + z2_pred) / 2
                quantile_40 = get_target_quantile(z_pred, tau_hat, 0.4)
                quantile_60 = get_target_quantile(z_pred, tau_hat, 0.6)
                quantile_80 = get_target_quantile(z_pred, tau_hat, 0.8)

            target_q1_value = (target_z1_value * presum_tau).sum(-1, keepdim=True)
            target_q2_value = (target_z2_value * presum_tau).sum(-1, keepdim=True)
            target_q_values = torch.cat([target_q1_value, target_q2_value], dim=-1)

            eval_statistics.update(
                create_stats_ordered_dict(
                    "Mean Q Predictions",
                    ptu.to_np(q_preds.mean(-1)),
                )
            )
            eval_statistics.update(
                create_stats_ordered_dict(
                    "Mean Target Q Predictions",
                    ptu.to_np(target_q_values.mean(dim=-1)),
                )
            )
            eval_statistics.update(
                create_stats_ordered_dict(
                    "Q std",
                    np.mean(
                        ptu.to_np(torch.std(q_preds, dim=-1)),
                    ),
                ),
            )
            eval_statistics.update(
                create_stats_ordered_dict(
                    "40 Quantile",
                    ptu.to_np(quantile_40),
                )
            )
            eval_statistics.update(
                create_stats_ordered_dict(
                    "60 Quantile",
                    ptu.to_np(quantile_60),
                )
            )
            eval_statistics.update(
                create_stats_ordered_dict(
                    "80 Quantile",
                    ptu.to_np(quantile_80),
                )
            )

        loss = SarsaLosses(
            qfs_loss=qfs_loss,
        )
        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    @property
    def optimizers(self):
        return [
            self.qf1_optimizer,
            self.qf2_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )


"""
Pipeline code
"""


def create_q_iqn(ctx: PipelineCtx):
    obs_dim = ctx.eval_env.observation_space.low.size
    action_dim = ctx.eval_env.action_space.low.size

    qf1 = ctx.variant["qf_class"](
        input_size=obs_dim + action_dim,
        **ctx.variant["qf_kwargs"],
    )
    qf2 = ctx.variant["qf_class"](
        input_size=obs_dim + action_dim,
        **ctx.variant["qf_kwargs"],
    )

    target_qf1 = ctx.variant["qf_class"](
        input_size=obs_dim + action_dim,
        **ctx.variant["qf_kwargs"],
    )
    target_qf2 = ctx.variant["qf_class"](
        input_size=obs_dim + action_dim,
        **ctx.variant["qf_kwargs"],
    )

    ctx.qfs = [qf1, qf2]
    ctx.target_qfs = [target_qf1, target_qf2]


SarsaIQNPipeline = Pipeline.from_(
    Pipelines.offline_zerostep_pac_pipeline, "SarsaIQNPipeline"
)
SarsaIQNPipeline.delete('pac_sanity_check')
SarsaIQNPipeline.delete('load_checkpoint_policy')
SarsaIQNPipeline.delete("create_eval_policy")
SarsaIQNPipeline.replace("load_checkpoint_iqn_q", create_q_iqn)
SarsaIQNPipeline.replace("create_dataset", create_dataset_next_actions)
