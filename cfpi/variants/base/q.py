from cfpi.variants.base import BaseModel


class QFKwargs(BaseModel):
    hidden_sizes = [1024, 1024]


class EnsembleQFKwargs(QFKwargs):
    num_heads = 10


class QuantileMLPKwargs(QFKwargs):
    hidden_sizes = [256, 256, 256]
    num_quantiles = 8
    embedding_size = 64
