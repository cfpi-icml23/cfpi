import copy

import eztils.torch as ptu
import numpy as np
import torch
import torch.nn.functional as F
from eztils.torch import (
    GaussianMixture,
    MultivariateDiagonalNormal,
    TanhGaussianMixture,
    TanhNormal,
)
from torch import nn as nn

from cfpi.policies.stochastic import TorchStochasticPolicy
from cfpi.pytorch.networks import Mlp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5  # this used to be 20
MEAN_MIN = -9.0
MEAN_MAX = 9.0


# noinspection PyMethodOverriding
class TanhGaussianPolicy(Mlp, TorchStochasticPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
        self, hidden_sizes, obs_dim, action_dim, std=None, init_w=1e-3, **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = (
                torch.from_numpy(
                    np.array(
                        [
                            self.std,
                        ]
                    )
                )
                .float()
                .to(ptu.device)
            )

        return TanhNormal(mean, std)

    def log_prob(self, obs, actions):
        raw_actions = ptu.atanh(actions)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob


class GaussianPolicy(Mlp, TorchStochasticPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        std=None,
        init_w=1e-3,
        min_log_std=-6,
        max_log_std=0,
        std_architecture="shared",
        **kwargs,
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs,
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True)
                )
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = (
                torch.from_numpy(
                    np.array(
                        [
                            self.std,
                        ]
                    )
                )
                .float()
                .to(ptu.device)
            )

        return MultivariateDiagonalNormal(mean, std)


class TanhGaussianMixturePolicy(Mlp, TorchStochasticPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        weight_temperature=1.0,
        init_w=1e-3,
        num_gaussians=2,
        **kwargs,
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim * num_gaussians,
            init_w=init_w,
            **kwargs,
        )
        self.action_dim = action_dim
        self.num_gaussians = num_gaussians
        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]

        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim * num_gaussians)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)

        self.last_fc_weights = nn.Linear(last_hidden_size, num_gaussians)
        self.last_fc_weights.weight.data.uniform_(-init_w, init_w)
        self.last_fc_weights.bias.data.uniform_(-init_w, init_w)
        self.weight_temperature = weight_temperature

    def log_prob(self, obs, actions):
        raw_actions = ptu.atanh(actions)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)

        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        weights = F.softmax(
            self.last_fc_weights(h.detach()) / self.weight_temperature,
            dim=1,
        )

        mixture_weights = weights.reshape((-1, self.num_gaussians))
        mixture_means = mean.reshape(
            (
                -1,
                self.num_gaussians,
                self.action_dim,
            )
        )
        mixture_stds = std.reshape(
            (
                -1,
                self.num_gaussians,
                self.action_dim,
            )
        )

        tanh_gmm = TanhGaussianMixture(mixture_means, mixture_stds, mixture_weights)
        log_prob = tanh_gmm.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        weights = F.softmax(
            self.last_fc_weights(h.detach()) / self.weight_temperature,
            dim=1,
        )

        mixture_weights = weights.reshape((-1, self.num_gaussians))
        mixture_means = mean.reshape(
            (
                -1,
                self.num_gaussians,
                self.action_dim,
            )
        )
        mixture_stds = std.reshape(
            (
                -1,
                self.num_gaussians,
                self.action_dim,
            )
        )
        return TanhGaussianMixture(mixture_means, mixture_stds, mixture_weights)


class GaussianMixturePolicy(TanhGaussianMixturePolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        weight_temperature=1,
        init_w=0.001,
        num_gaussians=2,
        output_activation=torch.tanh,
        **kwargs,
    ):
        super().__init__(
            hidden_sizes,
            obs_dim,
            action_dim,
            weight_temperature,
            init_w,
            num_gaussians,
            **kwargs,
        )
        self.output_activation = output_activation

    def log_prob(self, obs, actions):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        mean = self.output_activation(mean)

        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        weights = F.softmax(
            self.last_fc_weights(h.detach()) / self.weight_temperature,
            dim=1,
        )

        mixture_weights = weights.reshape((-1, self.num_gaussians))
        mixture_means = mean.reshape(
            (
                -1,
                self.num_gaussians,
                self.action_dim,
            )
        )
        mixture_stds = std.reshape(
            (
                -1,
                self.num_gaussians,
                self.action_dim,
            )
        )
        gmm = GaussianMixture(mixture_means, mixture_stds, mixture_weights)
        log_prob = gmm.log_prob(value=actions)
        return log_prob

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = self.output_activation(mean)

        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        weights = F.softmax(
            self.last_fc_weights(h.detach()) / self.weight_temperature,
            dim=1,
        )

        mixture_weights = weights.reshape((-1, self.num_gaussians))
        mixture_means = mean.reshape(
            (
                -1,
                self.num_gaussians,
                self.action_dim,
            )
        )
        mixture_stds = std.reshape(
            (
                -1,
                self.num_gaussians,
                self.action_dim,
            )
        )
        return GaussianMixture(mixture_means, mixture_stds, mixture_weights)


class UnnormalizeTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(self, state_mean, state_std, policy: TanhGaussianPolicy):
        self.__dict__ = copy.deepcopy(policy.__dict__)
        self.state_mean = ptu.to_torch(state_mean)
        self.state_std = ptu.to_torch(state_std)

    def forward(self, obs):
        obs = self.unnormalize(obs)
        return super().forward(obs)

    def log_prob(self, obs):
        obs = self.unnormalize(obs)
        return super().log_prob(obs)

    def unnormalize(self, obs):
        return (obs * self.state_std) + self.state_mean


class UnnormalizeGaussianPolicy(GaussianPolicy):
    def __init__(self, state_mean, state_std, policy: GaussianPolicy):
        self.__dict__ = copy.deepcopy(policy.__dict__)
        self.state_mean = ptu.to_torch(state_mean)
        self.state_std = ptu.to_torch(state_std)

    def forward(self, obs):
        obs = self.unnormalize(obs)
        return super().forward(obs)

    def log_prob(self, obs):
        obs = self.unnormalize(obs)
        return super().log_prob(obs)

    def unnormalize(self, obs):
        return (obs * self.state_std) + self.state_mean
