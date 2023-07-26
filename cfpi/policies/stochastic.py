import abc

from eztils.torch import Delta, to_torch

from cfpi.policies import Policy
from cfpi.policies.simple import elem_or_tuple_to_numpy
from cfpi.pytorch.networks.distribution_generator import DistributionGenerator


class TorchStochasticPolicy(DistributionGenerator, Policy, metaclass=abc.ABCMeta):
    def get_action(self, obs_np, time_step=None):
        actions = self.get_actions(obs_np[None], time_step)
        return actions.squeeze(), {}

    def get_actions(
        self,
        obs_np,
        time_step=None,
    ):
        if time_step is None:
            dist = self._get_dist_from_np(obs_np)
        else:
            dist = self._get_dist_from_np(obs_np, time_step)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(to_torch(x) for x in args)
        torch_kwargs = {k: to_torch(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist


class MakeDeterministic(TorchStochasticPolicy):
    def __init__(
        self,
        action_distribution_generator: DistributionGenerator,
    ):
        super().__init__()
        self._action_distribution_generator = action_distribution_generator

    def forward(self, *args, **kwargs):
        dist = self._action_distribution_generator.forward(*args, **kwargs)
        return Delta(dist.mle_estimate())


class MakeCfpiDeterministic(TorchStochasticPolicy):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def forward(self, *args, **kwargs):
        # with torch.enable_grad():
        dist: Delta = self.trainer.get_cfpi_action(*args, **kwargs)
        return dist
