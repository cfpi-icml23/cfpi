#! Policy Kwargs
from cfpi.variants.base import BaseModel


class PolicyKwargs(BaseModel):
    hidden_sizes = [1024, 1024]


class Two256(PolicyKwargs):
    hidden_sizes = [256, 256]


class Three256(PolicyKwargs):
    hidden_sizes = [256, 256, 256]


class Four256(PolicyKwargs):
    hidden_sizes = [256, 256, 256, 256]
