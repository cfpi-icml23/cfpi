"""
Based on rllab's logger.

https://github.com/rll/rllab
"""
import csv
import datetime
import errno
import json
import os
import os.path as osp
import sys
from collections import OrderedDict
from contextlib import contextmanager
from enum import Enum

import dateutil.tz
import portalocker
import torch

import wandb
from cfpi import conf
from cfpi.core.logging.tabulate import tabulate

SEPARATOR = "\n\n-----------------\n\n"


def add_prefix(log_dict: OrderedDict, prefix: str, divider=""):
    with_prefix = OrderedDict()
    for key, val in log_dict.items():
        with_prefix[prefix + divider + key] = val
    return with_prefix


class TerminalTablePrinter:
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os

        rows, _ = os.popen("stty size", "r").read().split()
        tabulars = self.tabulars[-(int(rows) - 3) :]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        from cfpi.launchers.pipeline import Pipeline

        if isinstance(o, type) or isinstance(o, Pipeline):
            return {"$class": o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {"$enum": o.__module__ + "." + o.__class__.__name__ + "." + o.name}
        elif callable(o):
            return {"$function": o.__module__ + "." + o.__name__}
        return json.JSONEncoder.default(self, o)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Logger:
    def __init__(self):
        self._prefixes = []
        self._prefix_str = ""

        self._tabular_prefixes = []
        self._tabular_prefix_str = ""

        self._tabular = []
        self._tabular_keys = {}

        self._text_outputs = []
        self._tabular_outputs = []

        self._text_fds = {}
        self._tabular_fds = {}
        self._tabular_header_written = set()

        self._snapshot_dir = None
        self._snapshot_mode = "all"
        self._snapshot_gap = 1

        self._log_tabular_only = False
        self.table_printer = TerminalTablePrinter()

    def reset(self):
        self.__init__()

    def _add_output(self, file_name, arr, fds, mode="a"):
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = "".join(self._prefixes)

    def add_text_output(self, file_name):
        self._add_output(file_name, self._text_outputs, self._text_fds, mode="a")

    def add_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        self._add_output(file_name, self._tabular_outputs, self._tabular_fds, mode="w")
        self._tabular_keys[file_name] = None

    def set_snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    def get_snapshot_dir(
        self,
    ):
        return self._snapshot_dir

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def set_log_tabular_only(self, log_tabular_only):
        self._log_tabular_only = log_tabular_only

    def log(self, s, with_prefix=True, with_timestamp=True):
        out = s
        if with_prefix:
            out = self._prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f %Z")
            out = f"{timestamp} | {out}"
        if not self._log_tabular_only:
            # Also log to stdout
            print(out)
            for fd in list(self._text_fds.values()):
                fd.write(out + "\n")
                fd.flush()
            sys.stdout.flush()

    def record_tabular(self, key, val):
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def record_dict(self, d, prefix=None):
        if prefix is not None:
            self.push_tabular_prefix(prefix)
        for k, v in d.items():
            self.record_tabular(k, v)
        if prefix is not None:
            self.pop_tabular_prefix()

    def push_tabular_prefix(self, key):
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = "".join(self._tabular_prefixes)

    def pop_tabular_prefix(
        self,
    ):
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = "".join(self._tabular_prefixes)

    def get_table_dict(
        self,
    ):
        return dict(self._tabular)

    @contextmanager
    def prefix(self, key):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    def log_variant(self, log_file, variant_data):
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)

    def dump_tabular(self, *args, **kwargs):
        wh = kwargs.pop("write_header", None)
        if len(self._tabular) > 0:
            if self._log_tabular_only:
                self.table_printer.print_tabular(self._tabular)
            else:
                for line in tabulate(self._tabular).split("\n"):
                    self.log(line, *args, **kwargs)
            tabular_dict = dict(self._tabular)
            # Also write to the csv files
            for filename, tabular_fd in list(self._tabular_fds.items()):
                # Only saves keys in first iteration to CSV!
                # (But every key is printed out in text)
                itr0_keys = self._tabular_keys.get(filename)
                if itr0_keys is None:
                    itr0_keys = list(sorted(tabular_dict.keys()))
                    self._tabular_keys[filename] = itr0_keys
                else:
                    prev_keys = set(itr0_keys)
                    curr_keys = set(tabular_dict.keys())
                    if curr_keys != prev_keys:
                        print("Warning: CSV key mismatch")
                        print("extra keys in 0th iter", prev_keys - curr_keys)
                        print("extra keys in cur iter", curr_keys - prev_keys)

                writer = csv.DictWriter(
                    tabular_fd,
                    fieldnames=itr0_keys,
                    extrasaction="ignore",
                )
                if wh or (
                    wh is None and tabular_fd not in self._tabular_header_written
                ):
                    writer.writeheader()
                    self._tabular_header_written.add(tabular_fd)
                writer.writerow(tabular_dict)
                tabular_fd.flush()
            del self._tabular[:]

    def pop_prefix(
        self,
    ):
        del self._prefixes[-1]
        self._prefix_str = "".join(self._prefixes)

    def save_itr_params(self, itr, params):
        if self._snapshot_dir:
            if self._snapshot_mode == "all":
                file_name = osp.join(self._snapshot_dir, "itr_%d.pt" % (itr + 1))
                torch.save(params, file_name)
            elif self._snapshot_mode == "last":
                # override previous params
                file_name = osp.join(self._snapshot_dir, "params.pt")
                torch.save(params, file_name)
            elif self._snapshot_mode == "gap":
                if (itr + 1) % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, "itr_%d.pt" % (itr + 1))
                    torch.save(params, file_name)
            elif self._snapshot_mode == "gap_and_last":
                if (itr + 1) % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, "itr_%d.pt" % (itr + 1))
                    torch.save(params, file_name)
                file_name = osp.join(self._snapshot_dir, "params.pt")
                torch.save(params, file_name)
            elif self._snapshot_mode == "none":
                pass
            else:
                raise NotImplementedError


def wlog(*args, **kwargs):
    if conf.Wandb.is_on:
        with portalocker.Lock(f"/tmp/wandb_log_lock_{os.getlogin()}"):
            wandb.log(*args, **kwargs)


class WandbLogger(Logger):
    def __init__(self, blacklist=None, highlight=None):
        super().__init__()
        if blacklist is None:
            blacklist = ["Epoch", "epoch", "eval/Average Returns"]
        if highlight is None:
            highlight = {
                "eval/Returns Mean": "Eval Returns Mean",
                "eval/Returns Std": "Eval Returns Std",
                "expl/Returns Mean": "Expl Returns Mean",
                "expl/Returns Std": "Expl Returns Std",
            }
        self.blacklist = blacklist
        self.highlight = highlight

    def dump_tabular(self, *args, **kwargs):
        logs = {k: float(v) for k, v in self.get_table_dict().items()}

        for b in self.blacklist:
            logs.pop(b, None)
        for old_key, new_key in self.highlight.items():
            try:
                logs[new_key] = logs.pop(old_key)
            except KeyError:
                continue

        wlog(logs, commit=True)
        super().dump_tabular(*args, **kwargs)

    def set_offline_rl(self):
        self.highlight = {
            "eval/normalized_score": "Eval Normalized Score",
            # "eval/path length Mean": "Eval Path length",
            "eval/Returns Mean": "Eval Returns Mean",
        }


logger = WandbLogger()
