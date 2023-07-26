import eztils.torch as ptu
import torch
from gym.spaces import Discrete

from cfpi.data_management.simple_replay_buffer import SimpleReplayBuffer
from cfpi.envs import get_dim


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(self, max_replay_buffer_size, env, env_info_sizes=None):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, "info_sizes"):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
        )

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        if isinstance(self._action_space, Discrete):
            new_action = ptu.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs,
        )


class EnvReplayBufferNextAction(EnvReplayBuffer):
    def __init__(self, max_replay_buffer_size, env, env_info_sizes=None):
        super().__init__(max_replay_buffer_size, env, env_info_sizes)
        self._next_actions = ptu.zeros(
            (max_replay_buffer_size, get_dim(self._action_space))
        )

    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        next_action,
        terminal,
        **kwargs,
    ):
        if isinstance(self._action_space, Discrete):
            new_next_action = ptu.zeros(self._action_dim)
            new_next_action[next_action] = 1
        else:
            new_next_action = next_action

        self._next_actions[self._top] = next_action
        return super().add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )

    def random_batch(self, batch_size):
        indices = ptu.randperm(self._size, dtype=torch.long)[:batch_size]
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            next_actions=self._next_actions[indices],  #! only diff
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch


class EnvReplayBufferNextActionNewAction(EnvReplayBufferNextAction):
    def __init__(self, max_replay_buffer_size, env, env_info_sizes=None):
        super().__init__(max_replay_buffer_size, env, env_info_sizes)
        self._new_next_actions = ptu.zeros(
            (max_replay_buffer_size, get_dim(self._action_space))
        )

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError

    def random_batch(self, batch_size):
        indices = ptu.randperm(self._size, dtype=torch.long)[:batch_size]
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            next_actions=self._next_actions[indices],
            new_next_actions=self._new_next_actions[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch
