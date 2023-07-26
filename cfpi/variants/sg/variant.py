# * Setup ---------------------------------------------------


from cfpi.pytorch.algorithms.cfpi import single_gaussian
from cfpi.variants.base import w
from cfpi.variants.base.trainer import PacTrainerKwargs
from cfpi.variants.base.variant import PacVariant
from cfpi.variants.sg.trainer import (
    EasyBCQTrainerKwargs,
    EvalReverseKLTrainerKwargs,
    ImproveReverseKLTrainerKwargs,
    IQLTrainerKwargs,
)


class BaseSGPacVariant(PacVariant):
    checkpoint_params = "SG"

    trainer_cls = w(single_gaussian.SG_CFPI_Trainer)
    pipeline = w(single_gaussian.SGBasePipeline)


# * Runnable ------------------------------------------------
class VanillaVariant(BaseSGPacVariant):
    algorithm = "PAC-SG"
    version = "ablation-log-tau"
    env_id = "hopper-medium-v2"
    seed = 1


class GroundTruthVariant(BaseSGPacVariant):
    algorithm = "PAC"
    version = "ground_truth"
    env_id = "hopper-medium-v2"
    seed = 1

    pipeline = w(single_gaussian.GTExperiment)


class EpochBCVariant(BaseSGPacVariant):
    algorithm = "PAC"
    version = "variable_ensemble_size"
    env_id = "hopper-medium-replay-v2"
    seed = 1
    epoch_no = -250

    pipeline = w(single_gaussian.EpochBCExperiment)



class CQLVariant(BaseSGPacVariant):
    algorithm = "PAC-SG"
    version = "cql-baseline-3"
    pipeline = w(single_gaussian.CQLExperiment)
    trainer_kwargs = IQLTrainerKwargs()
    env_id = "hopper-medium-v2"
    normalize_env = False
    seed = 1


class BCVariant(BaseSGPacVariant):
    algorithm = "PAC-MG"
    version = "behavior-cloning-evaluation"
    env_id = "walker2d-medium-v2"
    seed = 2

    trainer_kwargs = PacTrainerKwargs()


class EasyBCQPacVariant(BaseSGPacVariant):
    trainer_kwargs = EasyBCQTrainerKwargs()

    algorithm = "PAC-SG"
    version = "easy-bcq"
    env_id = "hopper-medium-expert-v2"
    seed = 1
