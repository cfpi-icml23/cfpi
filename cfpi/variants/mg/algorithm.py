from cfpi.variants.base.algorithm import OfflineAlgoKwargs


class MultistepAlgoKwargs(OfflineAlgoKwargs):
    num_eval_steps_per_epoch = int(1e5)
    num_trains_per_train_loop = int(2e5)

    pre_calculate_new_next_actions = True
    num_epochs = 0
    start_epoch = -int(5)


class IterativeAlgoKwargs(OfflineAlgoKwargs):
    num_eval_steps_per_epoch = int(1e4)
    num_trains_per_train_loop = int(2e4)

    pre_calculate_new_next_actions = False
    num_epochs = 0
    start_epoch = -int(50)
