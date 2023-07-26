from torch.nn import functional as F

from cfpi.variants.base import w
from cfpi.variants.base.policy import PolicyKwargs, Three256


class TunedSingleGaussianPolicyKwargs(PolicyKwargs):
    hidden_activation = w(F.leaky_relu)
    # layer_norm = True
    dropout = True
    dropout_kwargs = {"p": 0.1}
    hidden_sizes = [256, 256, 256, 256]


class ThreeLayerFourGaussianPolicyKwargs(Three256):
    num_gaussians = 4


class TunedPolicyKwargs(ThreeLayerFourGaussianPolicyKwargs):
    hidden_activation = w(F.leaky_relu)
    # layer_norm = True
    num_gaussians = 10
    dropout = True
    dropout_kwargs = {"p": 0.1}
    hidden_sizes = [256, 256, 256, 256]


class ThreeLayerTwelveGaussianPolicyKwargs(Three256):
    num_gaussians = 12


class ThreeLayerEightGaussianPolicyKwargs(Three256):
    num_gaussians = 8


class ThreeLayerTwoGaussianPolicyKwargs(Three256):
    num_gaussians = 2


class TwoLayerFourGaussianPolicyKwargs(PolicyKwargs):
    num_gaussians = 4
