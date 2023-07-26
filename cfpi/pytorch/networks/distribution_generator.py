from eztils.torch import (  # GaussianMixture as GaussianMixtureDistribution,
    Distribution,
    MultivariateDiagonalNormal,
)
from torch import nn


class DistributionGenerator(nn.Sequential, nn.Module):
    def forward(self, *input, **kwarg) -> Distribution:
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


class Gaussian(DistributionGenerator):
    def __init__(self, module, std=None, reinterpreted_batch_ndims=1):
        super().__init__(module)
        self.std = std
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, *input):
        if self.std:
            mean = super().forward(*input)
            std = self.std
        else:
            mean, log_std = super().forward(*input)
            std = log_std.exp()
        return MultivariateDiagonalNormal(
            mean, std, reinterpreted_batch_ndims=self.reinterpreted_batch_ndims
        )


# class GaussianMixture(ModuleToDistributionGenerator):
#     def forward(self, *input):
#         mixture_means, mixture_stds, weights = super().forward(*input)
#         return GaussianMixtureDistribution(mixture_means, mixture_stds, weights)
