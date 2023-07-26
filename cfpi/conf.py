from typing import Dict, List

import os
from os.path import abspath, join


import cfpi
from cfpi.variants.base import BaseModel
from cfpi import checkpoint_config
CheckpointParams = checkpoint_config

"""
Debug mode will
* skip confirmation when replacing directories
* change the data directory to ./tmp
* turn off wandb logging
"""
try:
    from cfpi.conf_private import DEBUG
except ImportError:
    DEBUG = False
    
DEBUG = True #!!
DISPLAY_WELCOME = True


class LogModel(BaseModel):
    repo_dir: str = abspath(join(os.path.dirname(cfpi.__file__), os.pardir))
    try:
        from cfpi.conf_private import rootdir

        rootdir: str = rootdir
    except ImportError:
        rootdir: str = repo_dir
    basedir: str = join(rootdir, "tmp" if DEBUG else "data")


Log = LogModel()

ENSEMBLE_MODEL_ROOT: str = join(Log.rootdir, "models")
CHECKPOINT_PATH: str = join(Log.rootdir, "checkpoints")

class GridSearch:
    """The closest thing to a namespace in python"""

    class Base(BaseModel):
        gridsearch_values: Dict[str, List]

    class Testing(Base):
        gridsearch_values: Dict[str, List] = {"delta": [1, 2], "beta_LB": [0]}

    class Full(Base):
        gridsearch_values: Dict[str, List] = {
            "delta": [
                -0.0,
                0.1417768376957354,
                0.4590436050264207,
                0.6680472308365775,
                1.5517556536555206,
            ],
            "beta_LB": [0.1, 0.5, 2],
        }

    class Mg(Base):
        gridsearch_values: Dict[str, List] = {"beta_LB": [0.1, 1.0]}

    class Pac(Base):
        gridsearch_values: Dict[str, List] = {"beta_LB": [1.0, 0.1]}

    class EnsembleSize(Base):
        gridsearch_values: Dict[str, List] = {"ensemble_size": [1, 2, 5, 10]}

    class QTrainedEpochs(Base):
        gridsearch_values: Dict[str, List] = {
            "q_trained_epochs": [1000, 1250, 1500, 1750, 2000]
        }

    class KFold(Base):
        gridsearch_values: Dict[str, List] = {"fold_idx": [1, 2, 3]}

    class TrqDelta(Base):
        gridsearch_values: Dict[str, List] = {"delta": [0.01, 0.001, 0.0001]}

    class ReverseKl(Base):
        gridsearch_values: Dict[str, List] = {"alpha": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}

    class Deltas(Base):
        gridsearch_values: Dict[str, List] = {
            "delta_range": [
                [0.0, 0.0],
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.3],
                [0.4, 0.4],
                [0.5, 0.5],
                [0.6, 0.6],
                [0.7, 0.7],
                [0.8, 0.8],
                [1.0, 1.0],
                [1.5, 1.5],
                [2.0, 2.0],
                [0.0, 1.0],
                [0.5, 1.0],
                [0.8, 1.2],
                [1.0, 1.5],
                [1, 2],
            ]
        }

    class FineGrainedDeltas(Base):
        gridsearch_values: Dict[str, List] = {
            "delta": [[0.5], [0.75], [1.0], [1.25], [1.5], [1.75], [2.0], [2.25]]
        }

    class DetDeltas(Base):
        gridsearch_values: Dict[str, List] = {
            "delta_range": [[0.5, 1.0], [0.25, 0.5], [0.0, 1.5], [1.0, 2.0]]
        }

    class CqlDeltas(Base):
        gridsearch_values: Dict[str, List] = {
            "delta_range": [[0.1, 0.1], [0.2, 0.5], [1.0, 1.0], [1.5, 1.5]]
        }

    class EasyBcq(Base):
        gridsearch_values: Dict[str, List] = {
            "num_candidate_actions": [2, 5, 10, 20, 50, 100]
        }


def lrange(i):
    return list(range(i))


class Parallel:
    class Base(BaseModel):
        seeds: List[int]
        envs: List[str]

    class Single(Base):
        seeds: List[int] = lrange(1)
        envs: List[str] = [
            "hopper-medium-v2",
            "walker2d-medium-v2",
            "halfcheetah-medium-v2",
        ]

    class Wide(Base):
        seeds: List[int] = lrange(10)
        envs: List[str] = [
            "hopper-medium-replay-v2",
            "walker2d-medium-replay-v2",
            "halfcheetah-medium-replay-v2",
            "hopper-medium-v2",
            "walker2d-medium-v2",
            "halfcheetah-medium-v2",
            "hopper-medium-expert-v2",
            "walker2d-medium-expert-v2",
            "halfcheetah-medium-expert-v2",
        ]

    class MediumReplay(Base):
        seeds: List[int] = lrange(10)
        envs: List[str] = [
            "hopper-medium-replay-v2",
            "walker2d-medium-replay-v2",
            "halfcheetah-medium-replay-v2",
        ]

    class MediumExpert(Base):
        seeds: List[int] = lrange(10)
        envs: List[str] = [
            "hopper-medium-expert-v2",
            "walker2d-medium-expert-v2",
            "halfcheetah-medium-expert-v2",
        ]

    class AntMaze(Base):
        seeds: List[int] = lrange(5)
        envs: List[str] = [
            "antmaze-umaze-v0",
            "antmaze-umaze-diverse-v0",
            "antmaze-medium-diverse-v0",
            "antmaze-medium-play-v0",
            "antmaze-large-play-v0",
            "antmaze-large-diverse-v0",
        ]


class WandbModel(BaseModel):
    is_on: bool = not DEBUG  # Whether or not to use wandb
    entity: str = "mdsac"
    project: str = "cfpi"


Wandb = WandbModel()

try:
    from cfpi.conf_private import *
except ImportError:
    pass
