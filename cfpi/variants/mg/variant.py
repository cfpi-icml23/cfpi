from cfpi.data_management.env_replay_buffer import (
    EnvReplayBufferNextAction,
    EnvReplayBufferNextActionNewAction,
)
from cfpi.pytorch.algorithms.cfpi import mixture_gaussian as mg
from cfpi.pytorch.networks.mlp import QuantileMlp
from cfpi.variants.base import w
from cfpi.variants.base.q import QFKwargs, QuantileMLPKwargs
from cfpi.variants.base.variant import PacVariant
from cfpi.variants.mg.algorithm import IterativeAlgoKwargs, MultistepAlgoKwargs
from cfpi.variants.mg.trainer import (
    BaseMGTrainerKwargs,
    EasyBCQMGTrainerKwargs,
    IQLTrainerKwargs,
    MG8TrainerKwargs,
    MoreDeltasMGTrainerKwargs,
    MultistepTrainerKwargs,
)


class IQLQFKwargs(QFKwargs):
    hidden_sizes = [256, 256]


class BaseMGPacVariant(PacVariant):
    checkpoint_params = "MG4"

    trainer_kwargs = BaseMGTrainerKwargs()
    trainer_cls = w(mg.MG_CFPI_Trainer)
    pipeline = w(mg.MGBasePipeline)


class VanillaVariant(BaseMGPacVariant):
    checkpoint_params = "MG8"
    trainer_kwargs = MG8TrainerKwargs()

    algorithm = "PAC-MG8"
    version = "use-mean"
    env_id = "hopper-medium-replay-v2"
    seed = 1


class IQLVariant(BaseMGPacVariant):
    algorithm = "PAC-MG"
    version = "iql-eval-1000-policy"
    env_id = "hopper-medium-v2"
    seed = 2

    qf_kwargs = IQLQFKwargs()
    trainer_kwargs = IQLTrainerKwargs()
    normalize_env = True
    IQN = False
    pipeline = w(mg.MGIQLPipeline)


class VanillaMGAntMazePacVariant(BaseMGPacVariant):
    normalize_env = False
    IQN = False
    checkpoint_params = "MG12_WITHOUT_NORMALIZE"
    qf_kwargs = IQLQFKwargs()
    pipeline = w(mg.MGIQLAntMazePipeline)
    trainer_kwargs = IQLTrainerKwargs()

    algorithm = "PAC-MG12"
    version = "vanilla"
    env_id = "antmaze-umaze-diverse-v0"
    seed = 1


class MultistepMGAntMazePacVariant(VanillaMGAntMazePacVariant):
    algorithm = "MultiStep-PAC-MG12"
    version = "vanilla"
    env_id = "antmaze-umaze-v0"
    seed = 1

    replay_buffer_class = w(EnvReplayBufferNextActionNewAction)
    checkpoint_params = "MG12_WITHOUT_NORMALIZE"
    qf_class = w(QuantileMlp)
    qf_kwargs = QuantileMLPKwargs()
    trainer_kwargs = MultistepTrainerKwargs()
    algorithm_kwargs = MultistepAlgoKwargs()
    pipeline = w(mg.MultiStepMGAntMazePipeline)

    snapshot_gap = 1


class IterativeMGAntMazePacVariant(MultistepMGAntMazePacVariant):
    algorithm = "Iterative-PAC-MG8"
    version = "vanilla"
    env_id = "antmaze-umaze-v0"
    seed = 1

    replay_buffer_class = w(EnvReplayBufferNextAction)
    checkpoint_params = "MG8_WITHOUT_NORMALIZE"
    algorithm_kwargs = IterativeAlgoKwargs()
    pipeline = w(mg.IterativepMGAntMazePipeline)

    snapshot_gap = 5


class MoreDeltasMGPacVariant(BaseMGPacVariant):
    trainer_kwargs = MoreDeltasMGTrainerKwargs()

    algorithm = "PAC-MG4"
    version = "rebuttal-more-deltas"
    env_id = "hopper-medium-expert-v2"
    seed = 1


class EasyBCQMGPacVariant(BaseMGPacVariant):
    trainer_kwargs = EasyBCQMGTrainerKwargs()

    algorithm = "PAC-MG4"
    version = "easy-bcq"
    env_id = "hopper-medium-expert-v2"
    seed = 1


class BCVariant(BaseMGPacVariant):
    algorithm = "PAC-MG"
    version = "behavior-cloning-evaluation"
    env_id = "walker2d-medium-v2"
    seed = 2

    pipeline = w(mg.MGEvalBCPipeline)


class MGEnsembleAntMazePacVariant(VanillaMGAntMazePacVariant):
    pipeline = w(mg.MGEnsembleIQLAntMazePipeline)

    version = "iql-ensemblerebuttal"
    env_id = "antmaze-umaze-diverse-v0"
    seed = 2


class AntMazeIQLBCVariant(VanillaMGAntMazePacVariant):
    algorithm = "PAC-MG4"
    version = "behavior-cloning-evaluation-vanilla"
    env_id = "antmaze-medium-diverse-v0"
    seed = 2

    pipeline = w(mg.MGIQLAntMazeEvalBCPipeline)


class VanillaMGPacVariant(BaseMGPacVariant):
    algorithm = "PAC-MG4"
    version = "ablation-log-tau"
    env_id = "hopper-medium-expert-v2"
    seed = 1


class TwoGaussianBaseMGPacVariant(BaseMGPacVariant):
    checkpoint_params = "MG2"
