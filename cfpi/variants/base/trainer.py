from cfpi.variants.base import BaseModel


class TrainerKwargs(BaseModel):
    discount = 0.99
    policy_lr = 3e-4
    qf_lr = 1e-4
    reward_scale = 1
    soft_target_tau = 0.005
    target_update_period = 1


class SacTrainerKwargs(TrainerKwargs):
    use_automatic_entropy_tuning = False


class PacTrainerKwargs(TrainerKwargs):
    beta_LB = 1.0
    # delta: list = [0.0]
    delta_range = [0.2, 2.0]
    num_delta = 10
