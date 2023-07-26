# * Setup --------------------------------------------------
from cfpi.variants.base.trainer import PacTrainerKwargs


class BaseMGTrainerKwargs(PacTrainerKwargs):
    use_max_lambda = True
    # action_selection_mode = "max"
    delta_range = [0.5, 0.5]  #! lower, upper
    # delta = [0.2, 0.5, 1.0, 2.0]
    # delta = [sqrt(log(x) * -2) for x in [0.99, 0.9, 0.6, 0.3, 0.15]]
    beta_LB = 1.0
    action_selection_mode = "max_from_both"
    num_delta = 1

    num_quantiles = 0
    target_quantile = 0.0
    policy_lr = 0.0
    qf_lr = 0.0
    soft_target_tau = 0.0
    target_update_period = 0


class MultistepTrainerKwargs(BaseMGTrainerKwargs):
    action_selection_mode = "max"
    delta_range = [0.5, 0.5]
    num_delta = 1
    beta_LB = 1.0
    IQN = True
    target_update_period = 1

    num_quantiles = 8
    target_quantile = 0.0
    policy_lr = 3e-4
    qf_lr = 3e-4
    soft_target_tau = 0.005


class MG8TrainerKwargs(BaseMGTrainerKwargs):
    action_selection_mode = "max_from_both"
    delta_range = [0.5, 0.5]
    num_delta = 1
    beta_LB = 1.0


class MoreDeltasMGTrainerKwargs(BaseMGTrainerKwargs):
    action_selection_mode = "max_from_both"
    delta_range = [0.5, 1.0]
    num_delta = 1000
    beta_LB = 1.0


class EasyBCQMGTrainerKwargs(BaseMGTrainerKwargs):
    action_selection_mode = "easy_bcq"
    num_candidate_actions = 10


class IQLTrainerKwargs(BaseMGTrainerKwargs):
    IQN = False
    delta_range = [0.0, 0.0]
    num_delta = 1
    trivial_threshold = 0.03
    action_selection_mode = "max_from_both"
