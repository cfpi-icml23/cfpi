""" 
PAC with a Q Lower Bound, with pretrained q and pi beta from one step repository.
"""
import os.path as osp

import eztils.torch as ptu
import torch
from eztils.torch import Delta, TanhDelta

from cfpi import conf
from cfpi.launchers.pipeline import Pipeline, PipelineCtx, Pipelines
from cfpi.pytorch.algorithms.cfpi.single_gaussian import SG_CFPI_Trainer


class Deterministic_CFPI_Trainer(SG_CFPI_Trainer):
    def get_cfpi_action(self, obs) -> TanhDelta:
        behavior_action = self.behavior_policy(obs)
        if self.delta_range == [0.0, 0.0]:
            return Delta(behavior_action)

        self.sample_delta()
        assert self.iqn
        return self.compute_cfpi_action(obs, behavior_action)

    def compute_cfpi_action(self, obs, behavior_action: torch.Tensor):
        # * preliminaries

        pre_tanh_mu_beta = ptu.atanh(behavior_action)
        batch_size = obs.shape[0]
        pre_tanh_mu_beta.requires_grad_()
        mu_beta = torch.tanh(pre_tanh_mu_beta)

        # * calculate gradient of q lower bound w.r.t action
        # Get the lower bound of the Q estimate
        q_LB = self.calc_q_LB(obs, mu_beta)
        # Obtain the gradient of q_LB wrt to action
        # with action evaluated at mu_proposal
        grad = torch.autograd.grad(q_LB.sum(), pre_tanh_mu_beta)[
            0
        ]  #! note: this doesn't work on cpu.

        assert grad is not None
        assert pre_tanh_mu_beta.shape == grad.shape

        # * cacluate proposals
        # Set Sigma_beta as an identity matrix
        Sigma_beta = torch.ones_like(grad)

        denom = self.get_shift_denominator(grad, Sigma_beta)

        # [batch_size, num_deltas, action_dim]
        delta_mu = torch.sqrt(2 * self.delta) * (grad / denom).unsqueeze(1)

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


def deterministic_sanity_check(ctx: PipelineCtx):
    assert ctx.variant["checkpoint_params"] == "DET"


def load_checkpoint_policy(ctx: PipelineCtx):
    params = getattr(conf.CheckpointParams, ctx.variant["checkpoint_params"])
    base = osp.join(
        conf.CHECKPOINT_PATH,
        params.path,
        ctx.variant["env_id"],
        str(ctx.variant["seed"]),
    )

    policy_path = osp.join(base, params.file)
    ctx.policy = torch.load(policy_path, map_location="cpu")[params.key]


DETBasePipeline = Pipeline.from_(
    Pipelines.offline_zerostep_pac_pipeline, "DETBasePipeline"
)
DETBasePipeline.pipeline.insert(0, deterministic_sanity_check)
DETBasePipeline.replace("load_checkpoint_policy", load_checkpoint_policy)
