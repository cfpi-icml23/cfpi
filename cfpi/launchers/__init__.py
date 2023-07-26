"""
This file contains 'launchers', which are self-contained functions that take
in a configuration dictionary and runs a full experiment. The dictionary configures the
experiment. Examples include run_pipeline_here, run_parallel_pipeline_here, and run_hyperparameters

It is important that the functions are completely self-contained (i.e. they
import their own modules) so that they can be serialized.
"""
import copy
import datetime
import itertools
import os
import os.path as osp
import pickle
import random
import shutil
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import torch
from eztils import bold, green, query_yes_no, red
from eztils.git import generate_snapshot
from eztils.torch import seed_everything, set_gpu_mode
from tqdm import tqdm

import wandb
from cfpi import conf
from cfpi.core.logging import logger
from cfpi.core.multiprocessing import NestablePool

from .pipeline import Pipelines


def print_welcome():
    if not conf.DISPLAY_WELCOME:
        return
    logo = r"""
█░█ █▀▀ █▀ █▄▄   █▀█ █░░ █▄▀ █ ▀█▀
█▄█ █▄▄ ▄█ █▄█   █▀▄ █▄▄ █░█ █ ░█░
    """
    icon = r"""
                                 ,_-=(!7(7/zs_.
                             .='  ' .`/,/!(=)Zm.
               .._,,._..  ,-`- `,\ ` -` -`\\7//WW.
          ,v=~/.-,-\- -!|V-s.)iT-|s|\-.'   `///mK%.
        v!`i!-.e]-g`bT/i(/[=.Z/m)K(YNYi..   /-]i44M.
      v`/,`|v]-DvLcfZ/eV/iDLN\D/ZK@%8W[Z..   `/d!Z8m
     //,c\(2(X/NYNY8]ZZ/bZd\()/\7WY%WKKW)   -'|(][%4.
   ,\\i\c(e)WX@WKKZKDKWMZ8(b5/ZK8]Z7%ffVM,   -.Y!bNMi
   /-iit5N)KWG%%8%%%%W8%ZWM(8YZvD)XN(@.  [   \]!/GXW[
  / ))G8\NMN%W%%%%%%%%%%8KK@WZKYK*ZG5KMi,-   vi[NZGM[
 i\!(44Y8K%8%%%**~YZYZ@%%%%%4KWZ/PKN)ZDZ7   c=//WZK%!
,\v\YtMZW8W%%f`,`.t/bNZZK%%W%%ZXb*K(K5DZ   -c\\/KM48
-|c5PbM4DDW%f  v./c\[tMY8W%PMW%D@KW)Gbf   -/(=ZZKM8[
2(N8YXWK85@K   -'c|K4/KKK%@  V%@@WD8e~  .//ct)8ZK%8`
=)b%]Nd)@KM[  !'\cG!iWYK%%|   !M@KZf    -c\))ZDKW%`
YYKWZGNM4/Pb  '-VscP4]b@W%     'Mf`   -L\///KM(%W!
!KKW4ZK/W7)Z. '/cttbY)DKW%     -`  .',\v)K(5KW%%f
'W)KWKZZg)Z2/,!/L(-DYYb54%  ,,`, -\-/v(((KK5WW%f
 \M4NDDKZZ(e!/\7vNTtZd)8\Mi!\-,-/i-v((tKNGN%W%%
 'M8M88(Zd))///((|D\tDY\\KK-`/-i(=)KtNNN@W%%%@%[
  !8%@KW5KKN4///s(\Pd!ROBY8/=2(/4ZdzKD%K%%%M8@%%
   '%%%W%dGNtPK(c\/2\[Z(ttNYZ2NZW8W8K%%%%YKM%M%%.
     *%%W%GW5@/%!e]_tZdY()v)ZXMZW%W%%%*5Y]K%ZK%8[
      '*%%%%8%8WK\)[/ZmZ/Zi]!/M%%%%@f\ \Y/NNMK%%!
        'VM%%%%W%WN5Z/Gt5/b)((cV@f`  - |cZbMKW%%|
           'V*M%%%WZ/ZG\t5((+)L'-,,/  -)X(NWW%%
                `~`MZ/DZGNZG5(((\,    ,t\\Z)KW%@
                   'M8K%8GN8\5(5///]i!v\K)85W%%f
                     YWWKKKKWZ8G54X/GGMeK@WM8%@
                      !M8%8%48WG@KWYbW%WWW%%%@
                        VM%WKWK%8K%%8WWWW%%%@`
                          ~*%%%%%%W%%%%%%%@~
                             ~*MM%%%%%%@f`
                             """
    print(icon)
    print(logo)
    if conf.DEBUG:
        print("\n\nDebug mode on!\n\n")
    conf.DISPLAY_WELCOME = False


"""
Run experiment 
"""


def run_pipeline_here(
    variant,
    use_gpu=True,
    gpu_id=0,
    # Logger params:
    snapshot_mode="gap_and_last",
    snapshot_gap=100,
    git_infos=None,
    script_name=None,
    base_log_dir=None,
    force_randomize_seed=False,
    parallel=False,
    **setup_logger_kwargs,
):
    """
    Run an experiment locally without any serialization.
    This will add the 'log_dir' key to variant, and set variant['version'] to 'normal' if isn't already set.

    experiments. Note that one experiment may correspond to multiple seeds,.
    :param seed: Seed used for this experiment.
    :param use_gpu: Run with GPU. By default False.
    :param script_name: Name of the running script
    :return:
        trainer_cls=trainer_cls,
    """

    if not parallel:
        print_welcome()

    start = datetime.datetime.today()
    try:
        seed = variant.get("seed")
        algorithm = variant.get("algorithm")

        if force_randomize_seed or seed is None:
            seed = random.randint(0, 100000)
            variant["seed"] = seed
        logger.reset()

        actual_log_dir = setup_logger(
            algorithm=algorithm,
            variant=variant,
            seed=seed,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            base_log_dir=base_log_dir,
            git_infos=git_infos,
            script_name=script_name,
            parallel=parallel,
            env_id=variant["env_id"],
            **setup_logger_kwargs,
        )

        seed_everything(seed)
        set_gpu_mode(use_gpu, gpu_id)

        run_experiment_here_kwargs = dict(
            variant=variant,
            seed=seed,
            use_gpu=use_gpu,
            algorithm=algorithm,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            git_infos=git_infos,
            parallel=parallel,
            script_name=script_name,
            base_log_dir=base_log_dir,
            **setup_logger_kwargs,
        )
        with open(actual_log_dir + "/experiment.pkl", "wb") as handle:
            pickle.dump(
                dict(run_experiment_here_kwargs=run_experiment_here_kwargs),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        variant["log_dir"] = actual_log_dir

        return Pipelines.run_pipeline(variant)

    except Exception as e:
        exception_name, _, __ = sys.exc_info()
        if (
            exception_name is not None
            and not issubclass(exception_name, KeyboardInterrupt)
            and not issubclass(exception_name, FileExistsError)
        ):
            red(  # this doesn't get called in when running as a spawned process...why?
                f'{variant.get("algorithm")} seed: {variant.get("seed")} env_id: {variant.get("env_id")} started at {start.strftime("%I:%M %p %a %b %y")}, has crashed'
            )
            print(traceback.format_exc(), flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            if conf.Wandb.is_on:
                if wandb.run is not None:
                    wandb.alert(
                        title="Experiment Crash",
                        text=f'{variant.get("algorithm")} started at {start.strftime("%I:%M %p %a %b %y")}, has crashed',
                        level="ERROR",
                    )
            if conf.DEBUG:
                raise e

    green("Successfully finished")


"""
Multiprocessing
"""


def run_hyperparameters(parallel_cls, variant, hyperparameters: dict):
    print_welcome()

    if hyperparameters is None:
        raise Exception("No Hyperparameters given")

    all_experiment_combinations = []
    for kwarg_vals in list(itertools.product(*hyperparameters.values())):
        hp_string = ""
        trainer_kwargs = copy.deepcopy(variant["trainer_kwargs"])
        v = deepcopy(variant)
        for kw, val in zip(hyperparameters.keys(), kwarg_vals):
            hp_string += f"{kw[0]}={val}-"
            trainer_kwargs[kw] = val
            v[kw] = val
        v["trainer_kwargs"] = trainer_kwargs
        v["__gridsearch"] = hp_string[:-1]
        experiment_combinations = list(
            itertools.product(
                parallel_cls.seeds,
                parallel_cls.envs,
                (v,),
            )
        )

        all_experiment_combinations += experiment_combinations

    pool_run(list(enumerate(all_experiment_combinations)))


def run_parallel_pipeline_here(parallel_cls, variant):
    print_welcome()
    pool_run(
        list(
            enumerate(
                itertools.product(
                    parallel_cls.seeds,
                    parallel_cls.envs,
                    (variant,),
                )
            )
        )
    )


def pool_run(experiment_combinations):
    with torch.multiprocessing.Manager() as manager:
        d = manager.dict()
        with NestablePool(torch.cuda.device_count() * 2) as p:
            list(
                tqdm(
                    p.imap_unordered(
                        parallel_run_experiment_here_wrapper,
                        [(d, e) for e in experiment_combinations],
                    ),
                    total=len(experiment_combinations),
                )
            )


def parallel_run_experiment_here_wrapper(experiment_tuple):
    """A wrapper around run_experiment_here that uses just a single argument to work with multiprocessing pool map."""
    d, (i, (seed, env_id, variant)) = experiment_tuple

    cp = torch.multiprocessing.current_process().ident
    start = time.time()
    while cp is None:
        cp = torch.multiprocessing.current_process().ident
        time.sleep(1)
        if time.time() - start > 30:
            raise Exception("Couldn't get current process id!")
            # time out after thirty seconds
    bold(f"Running env_id: {env_id}, seed: {seed} with process {cp}")
    if torch.cuda.is_available():
        if d.get(cp) is None:
            gpu_id = int(i % torch.cuda.device_count())
            d[cp] = gpu_id
        else:
            gpu_id = d[cp]
    else:
        gpu_id = None
    variant = deepcopy(variant)
    variant["seed"] = seed
    variant["env_id"] = env_id
    run_pipeline_here(
        variant=variant,
        gpu_id=gpu_id,
        parallel=True,
        snapshot_mode=variant["snapshot_mode"],
        snapshot_gap=variant["snapshot_gap"],
    )


"""
Logging
"""


def create_log_dir(
    algorithm,
    env_id,
    variant,
    version="normal",
    seed=0,
    parallel=False,
    base_log_dir=None,
):
    """
    Creates and returns a unique log directory.

    :param algorithm: All experiments with this prefix will have log
    directories be under this directory.
    experiment.
    :param base_log_dir: The directory where all log should be saved.
    :return:
    """
    if variant.get("__gridsearch"):
        log_dir = (
            Path(base_log_dir or conf.Log.basedir)
            / algorithm
            / version
            / env_id
            / (variant.get("__gridsearch")).replace(" ", "_")
            / str(seed)
        )
    else:
        log_dir = (
            Path(base_log_dir or conf.Log.basedir)
            / algorithm
            / version
            / env_id
            / str(seed)
        )

    if osp.exists(log_dir):
        red(f"This experiment already exists: {log_dir}")
        if parallel:
            print("Exiting")
            raise FileExistsError
            # why do this hang?

        if conf.DEBUG or query_yes_no(
            "Would you like to replace the existing directory?"
        ):
            bold("Replacing this directory...")
            shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            bold("Replaced")
        else:
            print("Not replacing, exiting now")
            raise FileExistsError
    else:
        green(f"Running experiment in: {log_dir}")
        os.makedirs(log_dir, exist_ok=False)
    return str(log_dir)


def setup_logger(
    algorithm="default",
    env_id=None,
    variant=None,
    text_log_file="debug.log",
    variant_log_file="variant.json",
    tabular_log_file="progress.csv",
    snapshot_mode="last",
    snapshot_gap=1,
    log_tabular_only=False,
    git_infos=None,
    script_name=None,
    wandb_entity=conf.Wandb.entity,
    wandb_project=conf.Wandb.project,
    parallel=False,
    **create_log_dir_kwargs,
):
    """
    Set up logger to have some reasonable default settings.

    Will save log output to

        basedir/<algorithm>/<algorithm-version>/<env_id>/<seed>

    exp_name will be auto-generated to be unique.

    If log_dir is specified, then that directory is used as the output dir.

    :param algorithm: The sub-directory for this specific experiment.
    :param variant:
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param git_infos:
    :param script_name: If set, save the script name to this.
    :return:
    """
    if variant.get("version") is None:
        variant["version"] = "normal"

    log_dir = create_log_dir(
        algorithm,
        env_id,
        variant,
        version=variant["version"],
        parallel=parallel,
        **create_log_dir_kwargs,
    )
    if parallel:
        sys.stdout = open(osp.join(log_dir, "stdout.out"), "a")
        sys.stderr = open(osp.join(log_dir, "stderr.out"), "a")

    if conf.Wandb.is_on:
        wandb_group = f"{algorithm}-{variant['version']}-{env_id}"
        if variant.get("__gridsearch"):
            wandb_name = f"seed-{variant['seed']}-hp-{variant.get('__gridsearch')}"
        else:
            wandb_name = f"seed-{variant['seed']}"

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            name=wandb_name,
            config=variant,
            reinit=True,
        )
        wandb.run.log_code(os.path.join(conf.Log.repo_dir, "src"))

    if variant is not None:
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    logger.add_tabular_output(tabular_log_path)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)
    bold("Backing up folder...")
    generate_snapshot(conf.Log.repo_dir, log_dir, exclude=['checkpoints'])
    bold("Backed up!")
    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir
