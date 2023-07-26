from typing import Iterable

import abc
from collections import OrderedDict

from torch import nn as nn
from torch.cuda import empty_cache

import cfpi.conf as conf
import cfpi.core.gtimer as gt
import wandb
from cfpi.core.logging import logger
from cfpi.core.logging.eval_util import get_average_returns
from cfpi.core.rl_algorithm import BatchRLAlgorithm, Trainer
from cfpi.data_management.replay_buffer import ReplayBuffer
from cfpi.samplers.path_collector import MdpPathCollector


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: MdpPathCollector,
        evaluation_data_collector: MdpPathCollector,
        replay_buffer: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
        start_epoch=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch,
            min_num_steps_before_training,
            start_epoch,
        )

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class OfflineTorchBatchRLAlgorithm(TorchBatchRLAlgorithm):
    def __init__(
        self,
        trainer,
        evaluation_env,
        evaluation_data_collector: MdpPathCollector,
        replay_buffer: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        start_epoch=0,
        zero_step=False,
        pre_calculate_new_next_actions=False,
    ):
        super().__init__(
            trainer,
            None,  # set exploration_env to None
            evaluation_env,
            None,  # set expl data collector to None
            evaluation_data_collector,
            replay_buffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            None,  # set expl steps per train loop to None
            num_trains_per_train_loop,
            num_train_loops_per_epoch,
            None,  # set min_num_steps_before_training to None
            start_epoch,
        )
        self.normalized_scores = []
        self.zero_step = zero_step
        self.pre_calculate_new_next_actions = pre_calculate_new_next_actions
        assert self.expl_env is None

    def record_exploration(self):  # don't record exploration
        pass

    def log_additional(self, epoch):
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if eval_paths == []:
            return

        normalized_score = (
            self.eval_env.get_normalized_score(get_average_returns(eval_paths)) * 100
        )
        self.normalized_scores.append(normalized_score)
        logger.record_dict(
            OrderedDict(normalized_score=normalized_score),
            prefix="eval/",
        )

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot["trainer/" + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot["evaluation/" + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot["replay_buffer/" + k] = v
        return snapshot

    def _end_epoch(self, epoch, save_params=True):
        if self.pre_calculate_new_next_actions:
            for i in range(self.replay_buffer._size // self.batch_size + 1):
                indices = range(
                    i * self.batch_size,
                    min((i + 1) * self.batch_size, self.replay_buffer._size),
                )
                next_obs = self.replay_buffer._next_obs[indices]
                new_next_actions = self.trainer.get_cfpi_action(next_obs).mean
                self.replay_buffer._new_next_actions[
                    indices
                ] = new_next_actions.detach()
            empty_cache()
        gt.stamp("generate_new_next_actions")
        snapshot = self._get_snapshot()
        if save_params:
            logger.save_itr_params(epoch - self._start_epoch, snapshot)
        gt.stamp("saving")
        self._log_stats(epoch)

        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def train(self):
        if not self.zero_step:
            return super().train()
        else:
            self.offline_rl = True
            for i in range(self.num_epochs):
                self._begin_epoch(i)
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
                self._end_epoch(i, save_params=False)
            if conf.Wandb.is_on:
                table = wandb.Table(
                    data=list(enumerate(self.normalized_scores)),
                    columns=["step", "normalized score"],
                )
                histogram = wandb.plot.histogram(
                    table,
                    value="normalized score",
                    title="Normalized Score Distribution",
                )
                wandb.log({"Normalized Score Distribution": histogram})
                wandb.finish()

    def _train(self):
        self.training_mode(True)
        for _ in range(self.num_train_loops_per_epoch):
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                gt.stamp("sampling batch")
                self.trainer.train(train_data)
                gt.stamp("training")
        self.training_mode(False)
        # First train, then evaluate
        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp("evaluation sampling")


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, batch):
        self._num_train_steps += 1
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict(
            [
                ("num train calls", self._num_train_steps),
            ]
        )

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass
