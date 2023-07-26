# * Setup --------------------------------------------------


from cfpi.variants.base.algorithm import OfflineAlgoKwargs


class BCAlgoKwargs(OfflineAlgoKwargs):
    start_epoch = -int(500)
    num_eval_steps_per_epoch = 1000
    num_epochs = 0
