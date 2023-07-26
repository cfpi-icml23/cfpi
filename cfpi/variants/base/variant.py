from cfpi.data_management.env_replay_buffer import (
    EnvReplayBuffer,
)
from cfpi.launchers.pipeline import Pipelines
from cfpi.policies.gaussian_policy import TanhGaussianPolicy
from cfpi.pytorch.networks.mlp import ConcatMlp
from cfpi.pytorch.torch_rl_algorithm import (
    OfflineTorchBatchRLAlgorithm,
    TorchTrainer,
    Trainer,
)
from cfpi.samplers.rollout_functions import rollout
from cfpi.variants.base import BaseModel, w
from cfpi.variants.base.algorithm import OfflineAlgoKwargs, PacAlgoKwargs
from cfpi.variants.base.policy import PolicyKwargs, Three256
from cfpi.variants.base.q import QFKwargs, QuantileMLPKwargs
from cfpi.variants.base.trainer import PacTrainerKwargs, TrainerKwargs


class PathLoaderKwargs(BaseModel):
    pass


class OfflineVariant(BaseModel):
    # require children to specify
    algorithm = ""
    version = ""
    env_id = ""
    seed = -1

    algorithm_kwargs = OfflineAlgoKwargs()
    policy_kwargs = PolicyKwargs()
    qf_kwargs = QFKwargs()
    trainer_kwargs = TrainerKwargs()
    path_loader_kwargs = PathLoaderKwargs()

    policy_class = w(TanhGaussianPolicy)
    qf_class = w(ConcatMlp)
    trainer_cls = w(TorchTrainer)
    alg_class = w(OfflineTorchBatchRLAlgorithm)
    replay_buffer_class = w(EnvReplayBuffer)
    rollout_fn = w(rollout)

    replay_buffer_size = int(2e6)

    snapshot_mode = "gap_and_last"
    snapshot_gap = 100


class PacVariant(OfflineVariant):
    trainer_cls = w(Trainer)  # require child to specify
    checkpoint_params = "SPECIFY"

    policy_kwargs = Three256()
    qf_kwargs = QuantileMLPKwargs()
    trainer_kwargs = PacTrainerKwargs()
    algorithm_kwargs = PacAlgoKwargs()
    IQN = True
    d4rl = True
    normalize_env = True

    pipeline = w(Pipelines.offline_zerostep_pac_pipeline)


