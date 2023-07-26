import torch
import torch.nn.functional as F
from eztils.torch import randn, to_np, to_torch
from torch import nn

from cfpi.policies import Policy
from cfpi.pytorch.networks import Mlp
from cfpi.pytorch.normalizer import TorchFixedNormalizer


def elem_or_tuple_to_numpy(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(to_np(x) for x in elem_or_tuple)
    else:
        return to_np(elem_or_tuple)


def eval_np(module, *args, **kwargs):
    """
    Eval this module with a numpy interface
    Same as a call to __call__ except all Variable input/outputs are
    replaced with numpy equivalents.
    Assumes the output is either a single object or a tuple of objects.
    """
    torch_args = tuple(to_torch(x) for x in args)
    torch_kwargs = {k: to_torch(v) for k, v in kwargs.items()}
    outputs = module(*torch_args, **torch_kwargs)
    return elem_or_tuple_to_numpy(outputs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(self, *args, obs_normalizer: TorchFixedNormalizer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


# Vanilla Variational Auto-Encoder
# Adapted from https://github.com/sfujim/BCQ/blob/master/continuous_BCQ/BCQ.py
class VAEPolicy(nn.Module, Policy):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim, max_action):
        super().__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = randn((state.shape[0], self.latent_dim)).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self.decode, obs)
