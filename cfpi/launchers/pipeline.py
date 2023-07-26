from typing import Callable, List

import copy
import inspect

from cfpi import conf
from cfpi.pytorch.torch_rl_algorithm import Trainer


class PipelineCtx:
    def __init__(self, **kwargs) -> None:
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
        # These ones cannot be modified
        self.variant = None
        self.trainer_cls = None
        # feel free to add more
        self.eval_env = None
        self.dataset = None
        self.qfs = []
        self.target_qfs = []
        self.policy = None
        self.obs_mean = None
        self.obs_std = None
        self.trainer = None
        self.eval_policy = None
        self.eval_path_collector = None
        self.replay_buffer = None
        self.algorithm = None


class Pipeline:
    def __init__(self, name, pipeline) -> None:
        self.name: str = name

        self.pipeline: List[Callable] = pipeline

    @classmethod
    def from_(cls, previous_pipeline, name):
        return cls(name, copy.deepcopy(previous_pipeline.pipeline))

    def delete(self, func_name):
        found = None
        for i, f in enumerate(self.pipeline):
            if f.__name__ == func_name:
                found = i
                break
        if found is None:
            print(f"Failed to replace {func_name} in {self.name}")
        else:
            del self.pipeline[found]

    def replace(self, func_name, new_func):
        found = False
        for i, f in enumerate(self.pipeline):
            if f.__name__ == func_name:
                found = True
                self.pipeline[i] = new_func
                break

        if not found:
            print(f"Failed to replace {func_name} in {self.name}")

    @property
    def composition(self):
        return "\n\n".join([inspect.getsource(f) for f in self.pipeline])

    def __getitem__(self, index):
        return self.pipeline[index]

    def __str__(self) -> str:
        return f"<Pipeline {self.name}>:\n" + ",\n".join(
            [f.__name__ for f in self.pipeline]
        )

    @property
    def __name__(self):
        return str(self)


from .pipeline_pieces import (
    create_algorithm,
    create_dataset_next_actions,
    create_eval_env,
    create_eval_path_collector,
    create_pac_eval_policy,
    create_replay_buffer,
    create_trainer,
    load_checkpoint_iqn_q,
    load_checkpoint_policy,
    load_demos,
    offline_init,
    optionally_normalize_dataset,
    pac_sanity_check,
    train,
)


class Pipelines:
    @staticmethod
    def run_pipeline(variant, ctx: PipelineCtx = None, silent=True):
        try:
            pipeline: Pipeline = variant["pipeline"]
        except KeyError:
            raise Exception("Please add a pipeline to your variant!")

        if ctx is None:
            try:
                trainer_cls: Trainer = variant["trainer_cls"]
            except KeyError:
                raise Exception("Please add a <trainer_cls> to your variant!")

            ctx = PipelineCtx()
            ctx.variant = variant
            ctx.trainer_cls = trainer_cls

        # print variant and pipeline
        if not silent:
            print(pipeline)
            if conf.DEBUG:
                print(pipeline.composition)

        for f in pipeline:
            f(ctx)

    offline_zerostep_pac_pipeline = Pipeline(  # Don't need any training
        "ZeroStepPacExperiment",
        [
            pac_sanity_check,
            offline_init,
            create_eval_env,
            create_dataset_next_actions,
            optionally_normalize_dataset,
            load_checkpoint_iqn_q,
            load_checkpoint_policy,
            create_trainer,
            create_pac_eval_policy,
            create_eval_path_collector,
            create_replay_buffer,
            load_demos,
            create_algorithm,
            train,
        ],
    )
