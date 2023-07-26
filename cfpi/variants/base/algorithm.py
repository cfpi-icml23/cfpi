#! Algo Kwargs
from cfpi.variants.base import BaseModel


class AlgoKwargs(BaseModel):
    start_epoch = -1000  # offline epochs
    num_epochs = 0
    batch_size = 256
    max_path_length = 1000
    num_trains_per_train_loop = 1000


class OnlineAlgoKwargs(AlgoKwargs):
    num_expl_steps_per_train_loop = 0
    min_num_steps_before_training = 1000


class OfflineAlgoKwargs(AlgoKwargs):
    num_eval_steps_per_epoch = 5000


class PacAlgoKwargs(OfflineAlgoKwargs):
    zero_step = True
    num_eval_steps_per_epoch = 1000
    num_epochs = 100
