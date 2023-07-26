from torch import optim

from cfpi.variants.base import w
from cfpi.variants.base.trainer import TrainerKwargs


class BCTrainerKwargs(TrainerKwargs):
    policy_lr = 1e-4


class TunedBCTrainerKwargs(TrainerKwargs):
    optimizer_class = w(optim.AdamW)
