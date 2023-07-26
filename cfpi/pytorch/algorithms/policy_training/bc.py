"""  
Behavior Cloning Policy
"""
from typing import Tuple

from collections import OrderedDict, namedtuple

import eztils.torch as ptu
import numpy as np
import torch
import torch.optim as optim
from eztils import create_stats_ordered_dict

from cfpi.core.logging import add_prefix
from cfpi.data_management.hdf5_path_loader import load_hdf5_next_actions_and_val_data
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
from cfpi.pytorch.torch_rl_algorithm import TorchTrainer
from cfpi import conf 

BCLosses = namedtuple(
    "BCLosses",
    "policy_loss",
)


class BCTrainer(TorchTrainer):
    def __init__(self, policy, policy_lr=0.001, optimizer_class=optim.Adam, **kwargs):
        super().__init__()
        self.policy = policy

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.val_obs = None
        self.val_actions = None
        self.gt_val_log_prob = None

    def set_val_data(self, val_obs, val_actions, gt_val_log_prob):
        self.val_obs = val_obs
        self.val_actions = val_actions
        self.gt_val_log_prob = np.mean(ptu.to_np(gt_val_log_prob))

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[BCLosses, OrderedDict]:
        obs = batch["observations"]
        actions = batch["actions"]

        """
        Policy Loss
        """
        dist = self.policy(obs)
        _, log_pi = dist.sample_and_logprob()
        log_prob = self.policy.log_prob(obs, actions).unsqueeze(-1)
        policy_loss = -log_prob.mean()

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics["Policy Loss"] = np.mean(ptu.to_np(policy_loss))

            eval_statistics.update(
                create_stats_ordered_dict(
                    "Log Pis",
                    ptu.to_np(log_pi),
                )
            )
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)

            if self.gt_val_log_prob is not None:
                with torch.no_grad():
                    pred_val_log_prob = self.policy.log_prob(
                        self.val_obs, self.val_actions
                    )
                    eval_statistics["Pred Val Log Prob"] = np.mean(
                        ptu.to_np(pred_val_log_prob)
                    )
                    eval_statistics["GT Val Log Prob"] = self.gt_val_log_prob

                # kl_div = 0.5 * (
                #     (pred_val_Sigma / self.val_Sigma_gt - 1).sum(-1)
                #     + (self.val_Sigma_gt / pred_val_Sigma).prod(-1).log()
                #     + (self.val_Sigma_gt * (pred_val_mu - self.val_mu_gt) ** 2).sum(-1)
                # )

                # eval_statistics["Val KL Divergence"] = np.mean(ptu.to_np(policy_loss))

        loss = BCLosses(
            policy_loss=policy_loss,
        )

        return loss, eval_statistics

    def train_from_torch(self, batch):
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self._n_train_steps_total += 1

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
        ]

    @property
    def optimizers(self):
        return [
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
        )


def bc_sanity(ctx):
    assert ctx.variant["d4rl"]
    assert ctx.variant["normalize_env"] is False


def create_eval_policy(ctx: PipelineCtx):
    ctx.eval_policy = ctx.policy


BCPipeline = Pipeline(
    "offline_bc_pipeline",
    [
        bc_sanity,
        offline_init,
        create_eval_env,
        create_dataset_next_actions,
        optionally_normalize_dataset,
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


def load_demos_and_val_data(ctx: PipelineCtx):
    ctx.replay_buffer, val_obs, val_actions = load_hdf5_next_actions_and_val_data(
        ctx.dataset,
        ctx.replay_buffer,
        ctx.variant["train_ratio"],
        ctx.variant["fold_idx"],
    )

    action_space = ctx.eval_env._wrapped_env.action_space
    rg = ptu.from_numpy(action_space.high - action_space.low) / 2
    center = ptu.from_numpy(action_space.high + action_space.low) / 2

    obs_mean = ptu.from_numpy(ctx.eval_env._obs_mean)
    obs_std = ptu.from_numpy(ctx.eval_env._obs_std)

    val_obs_unnormalized = val_obs * obs_std[None] + obs_mean[None]
    val_actions_unnormalized = val_actions * rg[None] + center[None]

    params = torch.load(
        f'{conf.CHECKPOINT_PATH}/bc/{ctx.variant["env_id"]}/{ctx.variant["seed"]}/params.pt', 
        map_location="cpu",
    )

    with torch.no_grad():
        policy = params["trainer/policy"]
        policy.to(ptu.device)
        gt_val_log_prob = policy.log_prob(
            val_obs_unnormalized, val_actions_unnormalized
        )

    ctx.trainer.set_val_data(val_obs, val_actions, gt_val_log_prob)


BCWithValPipeline = Pipeline.from_(BCPipeline, "BCWithValPipeline")
BCWithValPipeline.replace("load_demos", load_demos_and_val_data)