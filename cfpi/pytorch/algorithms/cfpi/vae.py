""" 
PAC with a Q Lower Bound, with pretrained q and pi beta from one step repository.
"""
import eztils.torch as ptu
import torch
import torch.optim as optim
from eztils.torch import Delta

from cfpi.launchers.pipeline import Pipeline, PipelineCtx, Pipelines
from cfpi.pytorch.algorithms.cfpi.deterministic import load_checkpoint_policy
from cfpi.pytorch.algorithms.cfpi.single_gaussian import SG_CFPI_Trainer


class Vae_CFPI_Trainer(SG_CFPI_Trainer):
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
        plotter=None,
        render_eval_paths=False,
        beta_LB=0.5,
        delta_range=None,
        num_delta=None,
        num_candidate_actions=10,
        target_quantile=0.7,
        IQN=True,
    ):
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
            plotter=plotter,
            render_eval_paths=render_eval_paths,
            beta_LB=beta_LB,
            delta_range=delta_range,
            num_delta=num_delta,
            target_quantile=target_quantile,
            IQN=IQN,
        )
        self.num_candidate_actions = num_candidate_actions

    def get_cfpi_action(self, obs) -> Delta:
        assert obs.shape[0] == 1
        obs_rep = obs.repeat(self.num_candidate_actions, 1)
        behavior_actions = self.behavior_policy.decode(obs_rep)

        qfs = self.calc_q_LB(obs_rep, behavior_actions)
        idx = torch.argmax(qfs, dim=0)
        selected_actions = behavior_actions[idx]
        if self.delta_range == [0.0, 0.0]:
            return Delta(selected_actions)

        self.sample_delta()
        assert self.iqn
        return self.compute_cfpi_action(obs, selected_actions.reshape(1, -1))

    def compute_cfpi_action(self, obs, behavior_actions: torch.Tensor):
        """
        Max proposals
        """
        num_actions = behavior_actions.shape[0]
        # * preliminaries
        pre_tanh_mu_beta = ptu.atanh(behavior_actions)
        # * calculate gradient of q lower bound w.r.t action
        pre_tanh_mu_beta.requires_grad_()
        # [num_candidate_actions, obs_dim]
        obs_exp = obs.repeat(num_actions, 1)
        # [num_candidate_actions, act_dim]
        mu_beta = torch.tanh(pre_tanh_mu_beta)

        # Get the lower bound of the Q estimate
        # [num_candidate_actions, ensemble_size]
        q_LB = self.calc_q_LB(obs_exp, mu_beta)

        # Obtain the gradient of q_LB wrt to a
        # with a evaluated at mu_proposal
        grad = torch.autograd.grad(
            q_LB.sum(), pre_tanh_mu_beta
        )  #! this returns a tuple!!
        # [num_candidate_actions, act_dim]
        grad = grad[0]

        assert grad is not None
        assert pre_tanh_mu_beta.shape == grad.shape

        # * calculate proposals
        Sigma_beta = torch.ones_like(grad)
        # [num_candidate_actions, ]
        denom = self.get_shift_denominator(grad, Sigma_beta)
        # [num_candidate_actions, 1, action_dim]
        direction = (torch.mul(Sigma_beta, grad) / denom).unsqueeze(1)

        # [num_candidate_actions, num_delta, action_dim]
        delta_mu = torch.sqrt(2 * self.delta) * direction
        # [num_candidate_actions * num_delta, action_dim]
        mu_proposal = (pre_tanh_mu_beta.unsqueeze(1) + delta_mu).reshape(
            self.num_delta * num_actions, -1
        )
        # [num_candidate_actions * num_delta, action_dim]
        tanh_mu_proposal = torch.tanh(mu_proposal)

        # * get the lower bounded q
        obs_exp = obs.repeat(self.num_delta * num_actions, 1)
        q_LB = self.calc_q_LB(obs_exp, tanh_mu_proposal)
        # * argmax the proposals
        _, idx = torch.max(q_LB, dim=0)
        select_mu_proposal = tanh_mu_proposal[idx]

        return Delta(select_mu_proposal)


def vae_sanity_check(ctx: PipelineCtx):
    assert ctx.variant["checkpoint_params"] == "VAE"


VaeBasePipeline = Pipeline.from_(
    Pipelines.offline_zerostep_pac_pipeline, "VaeBasePipeline"
)
VaeBasePipeline.pipeline.insert(0, vae_sanity_check)
VaeBasePipeline.replace("load_checkpoint_policy", load_checkpoint_policy)
