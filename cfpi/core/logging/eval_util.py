"""
Common evaluation utilities.
"""

from collections import OrderedDict

import numpy as np
from eztils import create_stats_ordered_dict, list_of_dicts__to__dict_of_lists


def get_generic_path_information(paths, stat_prefix=""):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    if len(paths) == 0:
        return statistics
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(
        create_stats_ordered_dict("Rewards", rewards, stat_prefix=stat_prefix)
    )
    statistics.update(
        create_stats_ordered_dict("Returns", returns, stat_prefix=stat_prefix)
    )
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(
        create_stats_ordered_dict("Actions", actions, stat_prefix=stat_prefix)
    )
    statistics["Num Paths"] = len(paths)
    statistics[stat_prefix + "Average Returns"] = get_average_returns(paths)

    for info_key in ["env_infos", "agent_infos"]:
        if info_key in paths[0]:
            all_env_infos = [
                list_of_dicts__to__dict_of_lists(p[info_key]) for p in paths
            ]
            for k in all_env_infos[0].keys():
                final_ks = np.array([info[k][-1] for info in all_env_infos])
                first_ks = np.array([info[k][0] for info in all_env_infos])
                all_ks = np.concatenate([info[k] for info in all_env_infos])
                statistics.update(
                    create_stats_ordered_dict(
                        stat_prefix + k,
                        final_ks,
                        stat_prefix=f"{info_key}/final/",
                    )
                )
                statistics.update(
                    create_stats_ordered_dict(
                        stat_prefix + k,
                        first_ks,
                        stat_prefix=f"{info_key}/initial/",
                    )
                )
                statistics.update(
                    create_stats_ordered_dict(
                        stat_prefix + k,
                        all_ks,
                        stat_prefix=f"{info_key}/",
                    )
                )

    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)
