from cfpi.pytorch.algorithms.q_training.sarsa_iqn import (
    SarsaIQNPipeline,
    SarsaIQNTrainer,
)
from cfpi.pytorch.networks.mlp import QuantileMlp
from cfpi.variants.base import w
from cfpi.variants.base.algorithm import OfflineAlgoKwargs
from cfpi.variants.base.q import QuantileMLPKwargs
from cfpi.variants.base.trainer import TrainerKwargs
from cfpi.variants.base.variant import OfflineVariant
from cfpi.data_management.env_replay_buffer import EnvReplayBufferNextAction

class SarsaIQNAlgoKwargs(OfflineAlgoKwargs):
    num_eval_steps_per_epoch = 0
    max_path_length = 0
    num_epochs = 0
    batch_size = 256
    start_epoch = -int(500)


class SarsaIQNTrainerKwargs(TrainerKwargs):
    num_quantiles = 8
    qf_lr = 3e-4
    target_update_period = 1


class VanillaVariant(OfflineVariant):
    d4rl = True
    algorithm = "sarsa-iqn"
    version = "normalize-env-neg-one-reward"
    env_id = "antmaze-umaze-v0"
    normalize_env = True
    seed = 2

    qf_class = w(QuantileMlp)
    qf_kwargs = QuantileMLPKwargs()
    trainer_cls = w(SarsaIQNTrainer)
    trainer_kwargs = SarsaIQNTrainerKwargs()
    algorithm_kwargs = SarsaIQNAlgoKwargs()
    
    replay_buffer_class = w(EnvReplayBufferNextAction)
    pipeline = w(SarsaIQNPipeline)

    snapshot_gap = 100
