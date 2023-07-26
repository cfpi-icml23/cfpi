import pickle

import numpy as np
import torch
from torch import nn as nn

PICKLE = "pickle"
NUMPY = "numpy"
JOBLIB = "joblib"
TORCH = "torch"


def load_file(local_path, file_type=None):
    if file_type is None:
        extension = local_path.split(".")[-1]
        if extension == "npy":
            file_type = NUMPY
        elif extension == "pkl":
            file_type = PICKLE
        elif extension == "joblib":
            file_type = JOBLIB
        elif extension == "pt":
            file_type = TORCH
        else:
            raise ValueError("Could not infer file type.")
    if file_type == NUMPY:
        object = np.load(open(local_path, "rb"), allow_pickle=True)
    elif file_type == TORCH:
        object = torch.load(local_path)
    else:
        # f = open(local_path, 'rb')
        # object = CPU_Unpickler(f).load()
        object = pickle.load(open(local_path, "rb"))
    print("loaded", local_path)
    return object
