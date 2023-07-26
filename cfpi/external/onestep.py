import math
from numbers import Number

import torch
import torch.distributions as D
from torch import nn
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

## bridge functions between their repo and ours


def soft_clamp(x, low, high):
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x


class GaussMLP(nn.Module):
    def __init__(self, state_dim, action_dim, width, depth, dist_type):
        super().__init__()
        self.net = MLP(
            input_shape=(state_dim), output_dim=2 * action_dim, width=width, depth=depth
        )
        self.log_std_bounds = (-5.0, 0.0)
        self.mu_bounds = (-1.0, 1.0)
        self.dist_type = dist_type

    def forward(self, s):
        s = torch.flatten(s, start_dim=1)
        mu, log_std = self.net(s).chunk(2, dim=-1)

        mu = soft_clamp(mu, *self.mu_bounds)
        log_std = soft_clamp(log_std, *self.log_std_bounds)

        std = log_std.exp()
        if self.dist_type == "normal":
            dist = D.Normal(mu, std)
        elif self.dist_type == "trunc":
            dist = TruncatedNormal(mu, std)
        elif self.dist_type == "squash":
            dist = SquashedNormal(mu, std)
        else:
            raise TypeError("Expected dist_type to be 'normal', 'trunc', or 'squash'")
        return dist


#! q_network.PY


class QMLP(nn.Module):
    def __init__(self, state_dim, action_dim, width, depth):
        super().__init__()
        self.net = MLP(
            input_shape=(state_dim + action_dim), output_dim=1, width=width, depth=depth
        )

    def forward(self, s, a):
        x = torch.cat(
            [torch.flatten(s, start_dim=1), torch.flatten(a, start_dim=1)], axis=-1
        )
        return self.net(x)


#! utils.PY


class MLP(nn.Module):
    def __init__(self, input_shape, output_dim, width, depth):
        super().__init__()
        input_dim = torch.tensor(input_shape).prod()
        layers = [
            nn.Linear(input_dim, width),
            # nn.LayerNorm(width), nn.Tanh()]
            nn.ReLU(),
        ]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x):
        return self.net(x)


CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://github.com/toshas/torch_truncnorm
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, a, b, eps=1e-8, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super().__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        self._dtype_min_gt_0 = torch.tensor(torch.finfo(self.a.dtype).eps).type_as(
            self.a
        )
        self._dtype_max_lt_1 = torch.tensor(1 - torch.finfo(self.a.dtype).eps).type_as(
            self.a
        )
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()

    @property
    def mean(self):
        return -(self._little_phi(self.b) - self._little_phi(self.a)) / self._Z

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        p = p.type_as(self.a)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://github.com/toshas/torch_truncnorm
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "a": constraints.real,
        "b": constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, a=-1, b=1, eps=1e-8, validate_args=None):
        self.loc, self.scale, self.a, self.b = broadcast_all(loc, scale, a, b)
        a_standard = (a - self.loc) / self.scale
        b_standard = (b - self.loc) / self.scale
        super().__init__(a_standard, b_standard, eps=eps, validate_args=validate_args)
        self._log_scale = self.scale.log()

    @property
    def mean(self):
        mean = super().mean
        return mean * self.scale + self.loc

    def _to_std_rv(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return value * self.scale + self.loc

    def cdf(self, value):
        return super().cdf(self._to_std_rv(value))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._from_std_rv(super().icdf(value))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return super().log_prob(self._to_std_rv(value)) - self._log_scale


from torch.distributions import Normal, TanhTransform, transformed_distribution


class SquashedNormal(transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def t_dtype_single(x):
    dtype = x.dtype
    if dtype == torch.int64:
        return torch.int32
    elif dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def torch_single_precision(x):
    if type(x) != torch.Tensor:
        x = torch.tensor(x)
    return x.type(t_dtype_single(x))


def mode(mixed_dist):
    component_probs = mixed_dist.mixture_distribution.probs
    mode_idx = torch.argmax(component_probs, dim=1)
    means = mixed_dist.component_distribution.mean
    batch_size = means.shape[0]
    mode = means[range(batch_size), mode_idx]
    return mode
