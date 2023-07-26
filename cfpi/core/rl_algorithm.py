import abc
from collections import OrderedDict

import cfpi.core.gtimer as gt
from cfpi.core.logging import eval_util, logger
from cfpi.data_management.replay_buffer import ReplayBuffer
from cfpi.samplers.path_collector import MdpPathCollector


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times[f"time/{key} (s)"] = time
    times["time/epoch (s)"] = epoch_time
    times["time/total (s)"] = gt.get_times().total
    return times


class BaseRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: MdpPathCollector,
        evaluation_data_collector: MdpPathCollector,
        replay_buffer: ReplayBuffer,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError("_train must implemented by inherited class")

    def _begin_epoch(self, epoch):
        pass

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch - self._start_epoch, snapshot)
        gt.stamp("saving")
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot["trainer/" + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot["exploration/" + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot["evaluation/" + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot["replay_buffer/" + k] = v
        return snapshot

    def record_exploration(self):
        logger.record_dict(self.expl_data_collector.get_diagnostics(), prefix="expl/")
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, "get_diagnostics"):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix="expl/",
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="expl/",
        )

    def log_additonal(self, epoch):
        return

    def _log_stats(self, epoch):
        logger.log(f"Epoch {epoch} finished", with_timestamp=True)
        logger.record_dict({"epoch": epoch})

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(), prefix="replay_buffer/"
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix="trainer/")

        """
        Exploration
        """
        self.record_exploration()

        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix="eval/",
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, "get_diagnostics"):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix="eval/",
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="eval/",
        )
        self.log_additional(epoch)

        """
        Misc
        """
        gt.stamp("logging")
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
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
        start_epoch=0,  # negative epochs are offline, positive epochs are online
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self.offline_rl = self.epoch < 0
            self._begin_epoch(self.epoch)
            self._train()
            self._end_epoch(self.epoch)

    def _train(self):
        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            if not self.offline_rl:
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp("evaluation sampling")

        for _ in range(self.num_train_loops_per_epoch):
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            gt.stamp("exploration sampling")

            if not self.offline_rl:
                self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp("data storing")

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)
            gt.stamp("training")
            self.training_mode(False)


class Trainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, data):
        pass

    def end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}
