from cfpi.variants.base.trainer import PacTrainerKwargs
from cfpi.variants.mg.trainer import BaseMGTrainerKwargs


class IQLTrainerKwargs(BaseMGTrainerKwargs):
    IQN = False
    delta_range = [0.5, 0.5]
    num_delta = 1


class BCTrainerKwargs(PacTrainerKwargs):
    delta_range = [0.0, 0.0]


class EasyBCQTrainerKwargs(PacTrainerKwargs):
    easy_bcq = True
    num_candidate_actions = 10


class ImproveReverseKLTrainerKwargs(BCTrainerKwargs):
    alpha = 3.0
    num_delta = 1
    delta_range = [0.5, 0.5]


class EvalReverseKLTrainerKwargs(BCTrainerKwargs):
    alpha = 1.0
