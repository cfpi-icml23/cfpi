#!/usr/bin/env python3
import os

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
from typing_extensions import Annotated

import glob
import importlib
import importlib.util
import inspect
import json
import site
from enum import Enum

import typer
from eztils import bold, datestr, dict_to_safe_json, red
from pydantic import BaseSettings
from typer import Argument, Option

from cfpi.conf import GridSearch, Parallel
from cfpi.variants import SUPPORTED_ALGORITHMS

app: typer.Typer = typer.Typer(name="cfpi", no_args_is_help=True)


def find_attr(module, attr_substr):
    return [attr for attr in user_defined_attrs(module) if attr_substr in attr]


def user_defined_attrs(
    cls,
    excluded: list = None,
):
    excluded = excluded or ["Base", "frozen_enforced_dataclass"]
    return [
        attr for attr in dir(cls) if not attr.startswith("__") and attr not in excluded
    ]


def user_defined_attrs_dict(cls, excluded: list = None, string=False):
    excluded = excluded or ["Base", "frozen_enforced_dataclass"]
    return {
        k: str(v) if string else v
        for k, v in cls.__dict__.items()
        if not k.startswith("__") and k not in excluded
    }


def load_experiment(variant, alg: str):
    from cfpi.variants.base import FuncWrapper

    variant_module = importlib.import_module(f"cfpi.variants.{alg}.variant")
    variant: BaseSettings = getattr(variant_module, variant)

    variant_dict = variant().dict()
    for k in variant_dict:
        if isinstance(variant_dict[k], FuncWrapper):
            variant_dict[k] = variant_dict[k].f  # unwrap the functions and class
        if isinstance(variant_dict[k], dict):  # maybe do this recursively in the future
            for k2 in variant_dict[k]:
                if isinstance(variant_dict[k][k2], FuncWrapper):
                    variant_dict[k][k2] = variant_dict[k][k2].f

    return variant_dict


def print_section(name, content):
    from cfpi.core.logging import SEPARATOR

    bold(name.upper() + ":", "\n")
    print(content, SEPARATOR)


def list_to_dict(l):
    return {i: i for i in l}


@app.command()
def main(
    algorithm: Annotated[
        Enum("Algorithm", list_to_dict(user_defined_attrs(SUPPORTED_ALGORITHMS))),
        Argument(
            help="Specify algorithm to run. Find all supported algorithms in ./cfpi/variants/SUPPORTED_ALGORITHMS.py",
            autocompletion=lambda: user_defined_attrs(SUPPORTED_ALGORITHMS),
        ),
    ],
    variant: Annotated[
        str,
        Option(
            help="Specify which variant of the algorithm to run. Find all supported variant in ./cfpi/variants/<algorithm_name>.py",
        ),
    ] = 'VanillaVariant',
    parallel: Annotated[
        Enum("Parallel", list_to_dict(user_defined_attrs(Parallel))),
        Option(
            help="Run multiple versions of the algorithm on different environments and seeds.",
            autocompletion=lambda: user_defined_attrs(Parallel),
        ),
    ] = None,
    gridsearch: Annotated[
        Enum("GridSearch", list_to_dict(user_defined_attrs(GridSearch))),
        Option(
            help="Do a gridsearch. Only supported when parallel is also enabled",
            autocompletion=lambda: user_defined_attrs(GridSearch),
        ),
    ] = None,
    dry: Annotated[
        bool,
        Option(
            help="Just print the variant and pipeline.",
        ),
    ] = False,
):
    algorithm = algorithm.value
    parallel = parallel.value if parallel else None
    gridsearch = gridsearch.value if gridsearch else None

    import torch

    torch.multiprocessing.set_start_method("spawn")
    from cfpi import conf
    from cfpi.conf import GridSearch, Parallel
    from cfpi.launchers import (
        run_hyperparameters,
        run_parallel_pipeline_here,
        run_pipeline_here,
    )

    # remove mujoco locks
    for l in glob.glob(f"{site.getsitepackages()[0]}/mujoco_py/generated/*lock"):
        print(l)
        os.remove(l)

    variant = load_experiment(variant, algorithm)

    if dry:
        print_section("time", datestr())
        pipeline = variant["pipeline"]
        print_section("variant", json.dumps(dict_to_safe_json(variant), indent=2))
        print_section("pipeline", pipeline.composition)

    if gridsearch:
        if dry:
            print_section(
                "gridsearch args",
                inspect.getsource(getattr(GridSearch, gridsearch)),
            )
        else:
            run_hyperparameters(
                getattr(Parallel, parallel),
                variant,
                hyperparameters=(getattr(GridSearch, gridsearch))["gridsearch_values"],
            )
            return

    if parallel:
        if dry:
            print_section(
                "parallel args",
                inspect.getsource(getattr(Parallel, parallel)),
            )
        else:
            run_parallel_pipeline_here(getattr(Parallel, parallel), variant)
            return

    if dry:
        red("Debug mode: ", conf.DEBUG)
        red("Root dir", conf.Log.rootdir)
        return

    run_pipeline_here(
        variant=variant,
        snapshot_mode=variant.get("snapshot_mode", "gap_and_last"),
        snapshot_gap=variant.get("snapshot_gap", 100),
        gpu_id=variant.get("gpu_id", 0),
    )


if __name__ == "__main__":
    app()
