from typing import Dict, List, Optional
import socket
from cfpi.variants.base import BaseModel



class CheckpointParam(BaseModel):
    envs: List[str]
    seeds: List[int]
    path: str
    key: Optional[str] = None
    file: Optional[str] = None
    itrs: Optional[Dict[str, List[int]]] = None
    validation_optimal_epochs: Optional[Dict[str, int]] = None

class Q_IQN(CheckpointParam):
    envs: List[str] = [
        "halfcheetah-medium-expert-v2",
        "halfcheetah-medium-replay-v2",
        "halfcheetah-medium-v2",
        "hopper-medium-expert-v2",
        "hopper-medium-replay-v2",
        "hopper-medium-v2",
        "walker2d-medium-expert-v2",
        "walker2d-medium-replay-v2",
        "walker2d-medium-v2",
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-diverse-v0",
        "antmaze-medium-play-v0",
    ]
    seeds: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    path: str = "q-iqn"
    key: Optional[str] = "trainer/qfs"
    file: Optional[str] = None
    itrs: Optional[Dict[str, List[int]]] = {
        "hopper-medium-expert-v2": [50, 100, 150, 200, 250, 300, 350, 400],
        "halfcheetah-medium-expert-v2": [50, 100, 150, 200, 250, 300, 350, 400],
        "hopper-medium-v2": [50, 100, 150, 200, 250, 300, 350, 400],
        "halfcheetah-medium-v2": [50, 100, 150, 200, 250, 300, 350, 400],
        "walker2d-medium-expert-v2": [50, 100, 150, 200, 250, 300, 350, 400],
        "hopper-medium-replay-v2": [50, 100, 150, 200, 250, 300, 350, 400],
        "walker2d-medium-replay-v2": [
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
        ],
        "halfcheetah-medium-replay-v2": [
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
        ],
        "walker2d-medium-v2": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "antmaze-umaze-v0": [100, 200, 300, 400, 500],
        "antmaze-umaze-diverse-v0": [100, 200, 300, 400, 500],
        "antmaze-medium-diverse-v0": [100, 200, 300, 400, 500],
        "antmaze-medium-play-v0": [100, 200, 300, 400, 500],
    }
    validation_optimal_epochs: Optional[Dict[str, int]] = {
        "halfcheetah-medium-expert-v2": 400,
        "halfcheetah-medium-replay-v2": 1500,
        "halfcheetah-medium-v2": 200,
        "hopper-medium-expert-v2": 400,
        "hopper-medium-replay-v2": 300,
        "hopper-medium-v2": 400,
        "walker2d-medium-expert-v2": 400,
        "walker2d-medium-replay-v2": 1100,
        "walker2d-medium-v2": 700,
        "antmaze-umaze-v0": 500,
        "antmaze-umaze-diverse-v0": 500,
        "antmaze-medium-diverse-v0": 500,
        "antmaze-medium-play-v0": 500,
    }


class Q_IQL(CheckpointParam):
    envs: List[str] = [
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-diverse-v0",
        "antmaze-medium-play-v0",
        "antmaze-large-play-v0",
        "antmaze-large-diverse-v0",
    ]
    seeds: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    path: str = "q-iql-models"
    key: Optional[str] = None
    file: Optional[str] = None
    itrs: Optional[Dict[str, List[int]]] = None
    validation_optimal_epochs: Optional[Dict[str, int]] = None


class SG(CheckpointParam):
    envs: List[str] = [
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-diverse-v0",
        "antmaze-medium-play-v0",
        "halfcheetah-medium-expert-v2",
        "halfcheetah-medium-replay-v2",
        "halfcheetah-medium-v2",
        "hopper-medium-expert-v2",
        "hopper-medium-replay-v2",
        "hopper-medium-v2",
        "walker2d-medium-expert-v2",
        "walker2d-medium-replay-v2",
        "walker2d-medium-v2",
    ]
    seeds: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    path: str = "sg"
    key: Optional[str] = "trainer/policy"
    file: Optional[str] = None
    itrs: Optional[Dict[str, List[int]]] = None
    validation_optimal_epochs: Optional[Dict[str, int]] = None

class MG4(CheckpointParam):
    envs: List[str] = [
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-diverse-v0",
        "antmaze-medium-play-v0",
        "walker2d-medium-expert-v2",
        "hopper-medium-expert-v2",
        "halfcheetah-medium-expert-v2",
        "hopper-medium-replay-v2",
        "halfcheetah-medium-replay-v2",
        "walker2d-medium-replay-v2",
        "hopper-medium-v2",
        "halfcheetah-medium-v2",
        "walker2d-medium-v2",
    ]
    seeds: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    path: str = "mg-4"
    key: Optional[str] = "trainer/policy"
    file: Optional[str] = "params.pt"
    itrs: Optional[Dict[str, List[int]]] = None
    validation_optimal_epochs: Optional[Dict[str, int]] = None



class MG8:
    envs: List[str] = [
        "walker2d-medium-expert-v2",
        "hopper-medium-expert-v2",
        "halfcheetah-medium-expert-v2",
        "hopper-medium-replay-v2",
        "halfcheetah-medium-replay-v2",
        "walker2d-medium-replay-v2",
        "hopper-medium-v2",
        "halfcheetah-medium-v2",
        "walker2d-medium-v2",
    ]
    seeds: List[int] = range(10)
    path: str = "mg-8-no-normalize"  # normalized
    key:Optional[str]  = "trainer/policy"
    file: Optional[str] = "params.pt"


class MG8_WITHOUT_NORMALIZE(CheckpointParam):
    envs: List[str] = [
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-diverse-v0",
        "antmaze-medium-play-v0",
        "antmaze-large-play-v0",
        "antmaze-large-diverse-v0",
    ]
    seeds: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    path: str = "mg-8"
    key: Optional[str] = "trainer/policy"


class MG12_WITHOUT_NORMALIZE:
    envs: List[str] = [
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-diverse-v0",
        "antmaze-medium-play-v0",
        "antmaze-large-play-v0",
        "antmaze-large-diverse-v0",
    ]
    seeds: List[int] = list(range(10))
    path: str = "mg-12-no-normalize"
    key: Optional[str] = "trainer/policy"
