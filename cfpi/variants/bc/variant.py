from cfpi.policies.gaussian_policy import (
    GaussianMixturePolicy,
    GaussianPolicy,
    TanhGaussianMixturePolicy,
)
from cfpi.pytorch.algorithms.policy_training.bc import (
    BCPipeline,
    BCTrainer,
    BCWithValPipeline,
)
from cfpi.variants.base import w
from cfpi.variants.base.policy import Three256
from cfpi.variants.base.variant import OfflineVariant
from cfpi.variants.bc.algorithm import BCAlgoKwargs
from cfpi.variants.bc.policy import (
    ThreeLayerEightGaussianPolicyKwargs,
    ThreeLayerFourGaussianPolicyKwargs,
    ThreeLayerTwelveGaussianPolicyKwargs,
    ThreeLayerTwoGaussianPolicyKwargs,
    TunedPolicyKwargs,
    TunedSingleGaussianPolicyKwargs,
    TwoLayerFourGaussianPolicyKwargs,
)
from cfpi.variants.bc.trainer import BCTrainerKwargs, TunedBCTrainerKwargs


class BaseBCVariant(OfflineVariant):
    trainer_cls = w(BCTrainer)

    policy_kwargs = Three256()
    trainer_kwargs = BCTrainerKwargs()
    algorithm_kwargs = BCAlgoKwargs()
    IQN = True
    d4rl = True
    normalize_env = True

    pipeline = w(BCPipeline)


# * Runnable ------------------------------------------------


class VanillaVariant(BaseBCVariant):
    algorithm = "behavior-cloning"
    version = "normalized-256-256-256"
    seed = 0
    normalize_env = False
    env_id = "hopper-medium-v2"


class TunedBCVariant(BaseBCVariant):
    seed = 0
    algorithm = "mg-behavior-cloning"
    version = "gaussian-tanh-before"
    trainer_kwargs = TunedBCTrainerKwargs()
    policy_kwargs = TunedSingleGaussianPolicyKwargs()
    policy_class = w(GaussianPolicy)


class MGBCVariant(VanillaVariant):
    seed = 0
    algorithm = "mg-behavior-cloning"
    version = "4-gaussian-fixed-atanh"

    policy_kwargs = TwoLayerFourGaussianPolicyKwargs()
    policy_class = w(TanhGaussianMixturePolicy)


class MGBCNormalizeWithValVariant(MGBCVariant):
    normalize_env = True
    version = "2-gaussian-3-layers-normalize-env-with-val"
    env_id = "hopper-medium-v2"
    seed = 0
    train_ratio = 0.95
    fold_idx = 2

    policy_kwargs = ThreeLayerTwoGaussianPolicyKwargs()
    algorithm_kwargs = BCAlgoKwargs()
    pipeline = w(BCWithValPipeline)

    snapshot_gap = 50


class TunedMGBCVariant(BaseBCVariant):
    seed = 0
    algorithm = "mg-behavior-cloning"
    version = "10-gaussian-correct-GM"
    env_id = "antmaze-umaze-diverse-v0"
    trainer_kwargs = TunedBCTrainerKwargs()
    normalize_env = False

    policy_kwargs = TunedPolicyKwargs()
    policy_class = w(GaussianMixturePolicy)


class TwelveGaussianMGBCAntMazeVariant(MGBCVariant):
    normalize_env = False
    version = "12-gaussian-3-layers-all-data-rebuttal"
    env_id = "antmaze-medium-diverse-v0"
    seed = 0

    policy_kwargs = ThreeLayerTwelveGaussianPolicyKwargs()


class EightGaussianMGBCAntMazeVariant(MGBCVariant):
    normalize_env = False
    version = "8-gaussian-3-layers-all-data-rebuttal"
    env_id = "antmaze-medium-diverse-v0"
    seed = 0

    policy_kwargs = ThreeLayerEightGaussianPolicyKwargs()
    pipeline = w(BCPipeline)


class FourGaussianMGBCAntMazeVariant(MGBCVariant):
    normalize_env = False
    version = "4-gaussian-3-layers-all-data-rebuttal"
    env_id = "antmaze-medium-diverse-v0"
    seed = 0

    policy_kwargs = ThreeLayerFourGaussianPolicyKwargs()
    pipeline = w(BCPipeline)


class FourGaussianMGBCNormalizeVariant(MGBCVariant):
    normalize_env = True
    version = "4-gaussian-3-layers-normalize-env-all-data"
    env_id = "hopper-medium-v2"
    seed = 0

    policy_kwargs = ThreeLayerFourGaussianPolicyKwargs()
    pipeline = w(BCPipeline)


class TwoGaussianMGBCNormalizeVariant(MGBCVariant):
    normalize_env = True
    version = "2g-normalize-env-all-data"
    env_id = "hopper-medium-v2"
    seed = 0

    policy_kwargs = ThreeLayerTwoGaussianPolicyKwargs()
    pipeline = w(BCPipeline)
