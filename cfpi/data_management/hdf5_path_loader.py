import d4rl
import eztils.torch as ptu
import numpy as np
from torch import nn as nn


def d4rl_qlearning_dataset_with_next_actions(env):
    ds = d4rl.qlearning_dataset(env)
    ds["next_actions"] = np.roll(ds["actions"], -1, axis=0)
    for k in ds:  # discard the last action so all keys are the same length
        ds[k] = ds[k][:-1]

    return ds


def d4rl_qlearning_dataset_with_next_actions_new_actions(env):
    ds = d4rl_qlearning_dataset_with_next_actions(env)
    ds["new_next_actions"] = ds["next_actions"].copy()

    return ds


def load_hdf5(dataset, replay_buffer):
    _obs = dataset["observations"]
    N = _obs.shape[0]
    assert (
        replay_buffer._max_replay_buffer_size >= N
    ), "dataset does not fit in replay buffer"

    _actions = dataset["actions"]
    _next_obs = dataset["next_observations"]
    _rew = dataset["rewards"][:N]
    _done = dataset["terminals"][:N]

    replay_buffer._observations[:N] = ptu.to_torch(_obs[:N])
    replay_buffer._next_obs[:N] = ptu.to_torch(_next_obs[:N])
    replay_buffer._actions[:N] = ptu.to_torch(_actions[:N])
    replay_buffer._rewards[:N] = ptu.to_torch(np.expand_dims(_rew, 1)[:N])
    replay_buffer._terminals[:N] = ptu.to_torch(np.expand_dims(_done, 1)[:N])
    replay_buffer._size = N - 1
    replay_buffer._top = replay_buffer._size


def load_hdf5_next_actions(dataset, replay_buffer):
    load_hdf5(dataset, replay_buffer)
    _obs = dataset["observations"]
    N = _obs.shape[0]

    replay_buffer._next_actions[:N] = ptu.to_torch(dataset["next_actions"][:N])


def load_hdf5_next_actions_new_actions(dataset, replay_buffer):
    load_hdf5_next_actions(dataset, replay_buffer)
    _obs = dataset["observations"]
    N = _obs.shape[0]

    replay_buffer._new_next_actions[:N] = ptu.to_torch(dataset["new_next_actions"][:N])


def load_hdf5_next_actions_and_val_data(
    dataset, replay_buffer, train_raio=0.95, fold_idx=1
):
    _obs = dataset["observations"]
    _actions = dataset["actions"]
    _next_obs = dataset["next_observations"]
    _next_actions = dataset["next_actions"]
    _rew = dataset["rewards"]
    _done = dataset["terminals"]

    N = _obs.shape[0]
    assert (
        replay_buffer._max_replay_buffer_size >= N
    ), "dataset does not fit in replay buffer"

    assert np.array_equal(
        _next_actions[: N - 1],
        _actions[1:N],
    )

    # rng = np.random.default_rng()
    for _ in range(fold_idx):
        indices = np.random.permutation(N)
    tran_indices, val_indices = np.split(indices, [int(N * train_raio)])

    size = len(tran_indices)
    replay_buffer._observations[:size] = ptu.to_torch(_obs[tran_indices])
    replay_buffer._next_obs[:size] = ptu.to_torch(_next_obs[tran_indices])
    replay_buffer._actions[:size] = ptu.to_torch(_actions[tran_indices])
    replay_buffer._rewards[:size] = ptu.to_torch(np.expand_dims(_rew[tran_indices], 1))
    replay_buffer._terminals[:size] = ptu.to_torch(
        np.expand_dims(_done[tran_indices], 1)
    )
    replay_buffer._next_actions[:size] = ptu.to_torch(_next_actions[tran_indices])

    replay_buffer._size = size - 1
    replay_buffer._top = replay_buffer._size

    val_observations = ptu.to_torch(_obs[val_indices])
    val_actions = ptu.to_torch(_actions[val_indices])

    return replay_buffer, val_observations, val_actions
