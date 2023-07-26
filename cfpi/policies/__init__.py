import abc


class Policy(metaclass=abc.ABCMeta):
    """
    General policy interface.
    """

    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """

    def reset(self):
        pass


from cfpi.policies.gaussian_policy import (
    GaussianPolicy,
    TanhGaussianMixturePolicy,
    TanhGaussianPolicy,
)
from cfpi.policies.stochastic import MakeDeterministic, TorchStochasticPolicy

__all__ = [
    "Policy",
    "TorchStochasticPolicy",
    "MakeDeterministic",
    "TanhGaussianPolicy",
    "GaussianPolicy",
    "TanhGaussianMixturePolicy",
]
