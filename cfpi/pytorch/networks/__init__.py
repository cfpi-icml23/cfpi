"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch

from .mlp import ConcatMlp, ConcatMultiHeadedMlp, Mlp, ParallelMlp


class LinearTransform(torch.nn.Module):
    def __init__(self, m, b):
        super().__init__()
        self.m = m
        self.b = b

    def __call__(self, t):
        return self.m * t + self.b


__all__ = [
    "Mlp",
    "ConcatMlp",
    "ConcatMultiHeadedMlp",
    "ParallelMlp",
]
