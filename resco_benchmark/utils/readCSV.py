"""Print distribution of queue size per episode - print average and rl bootstrapped interval"""

from collections import defaultdict
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import matplotlib
import string
from typing import Tuple, List
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

log_dir = Path.cwd() / "results"
print(log_dir)
names = [folder for folder in next(os.walk(log_dir.as_posix()))[1]]

metric = "queue"
output_file = "avg_{}.py".format(metric)
run_avg, run_sim = defaultdict(list), defaultdict(list)


ALGO_ID_TO_ALGO_LBL = {
    "IQL_NS": "IQL",
    "DVDN_NS": "DVDN",
    "VDN_NS": "VDN",
    "IA2C_NS": "INDA2C",
    "CENTRALV": "CENTRAL-V",
    "DNAA2C": "DNAA2C",
}


def sort_key(x: Path) -> int:
    return int(x.stem.split("_")[-1])


def savefig(suptitle: str, save_directory_path: Path = None) -> None:
    """Saves a figure, named after suptitle, if save_directory_path is provided

    Parameters:
    ----------
    suptitle: str
        The title.
    save_directory_path: Path = None
        Saves the reward plot on a pre-defined path.

    Returns:
    -------
    filename: str
        Space between words are filled with underscore (_).
    """
    if save_directory_path is not None:
        # iteractively check for path and builds where it doesnt exist.
        prev_path = None
        for sub_dir in save_directory_path.parts:
            if prev_path is None:
                prev_path = Path(sub_dir)
            else:
                prev_path = prev_path / sub_dir
            prev_path.mkdir(exist_ok=True)
        # uses directory
        file_path = save_directory_path / f"{_snakefy(suptitle)}"
        plt.savefig(f"{file_path.as_posix()}.png", format="png")
        plt.savefig(f"{file_path.as_posix()}.pdf", format="pdf")


def _snakefy(title_case: str) -> str:
    """Converts `Title Case` into `snake_case`

    Parameters
    ----------
    title_case: str
        Uppercase for new words and spaces to split then up.

    Returns
    -------
    filename: str
        Space between words are filled with underscore (_).
    """
    fmt = title_case.translate(str.maketrans("", "", string.punctuation))
    return "_".join(fmt.lower().split())


def get_filename(scenarios: List[str], labels: List[str], metric: str="queue") -> str:
    algos = "_".join([*map(lambda x: x.lower(), labels)])
    maps = "_".join(list(set(scenarios)))
    filename = f"{metric}-{algos}-{maps}"
    return filename


def get_marker_and_color(algoname: str) -> Tuple[str, str]:
    """Finds marker and color

    Parameters:
    -----------
    algoname: E.g, DVDN, VDN, IQL_NS

    Returns:
    --------
    marker code: The marker for the plot.
    color code:
    """
    if (
        algoname.startswith("DNAA2C")
        or algoname.startswith("DNAQL")
        or algoname.startswith("DVDN")
    ):
        marker, color = "^", "C0"
    elif algoname.endswith("_GT"):
        marker, color = "^", "C2"
    # elif algoname.startswith("DVA2C"):
    #     marker, color = "s", "C4"
    elif algoname.startswith("DVA2C") or algoname.startswith("PIC"):
        marker, color = "x", "C7"
    elif algoname.startswith("MADDPG"):
        marker, color = "|", "C3"
    elif algoname.startswith("IND") or algoname.startswith("IQL"):
        marker, color = "|", "C1"
    elif (
        algoname.startswith("CENTRAL")
        or algoname.startswith("MAA2C")
        or algoname.startswith("VDN")
    ):
        marker, color = "h", "C5"
    else:
        raise ValueError(f"{algoname} not recognizable.")
    return marker, color

if __name__ == "__main__":
    for name in names:
        split_name = name.split("-")
        print(split_name)
        map_name = split_name[2]
        average_per_episode = []
        average_per_timestep = []
        path = log_dir / name
        # for i in range(1, 10000):
        for trip_file_path in sorted(path.glob("metrics_*"), key=sort_key):
            # trip_file_name = log_dir + name + os.sep + "metrics_" + str(i) + ".csv"
            # if not os.path.exists(trip_file_name):
            #     print("No " + trip_file_name)
            #     break

            num_steps, total = 0, 0.0
            last_departure_time = 0
            last_depart_id = ""
            average_per_timestep.append([])
            # with open(trip_file_name) as fp:
            with trip_file_path.open("r") as fp:
                reward, wait, steps = 0, 0, 0
                for line in fp:  # each line is a timestep.
                    line = line.split("}")
                    queues = line[2]
                    signals = queues.split(":")
                    step_total = 0
                    for s, signal in enumerate(signals):
                        if s == 0:
                            continue
                        queue = signal.split(",")
                        queue = int(queue[0])
                        step_total += queue
                    step_avg = step_total / len(signals)
                    total += step_avg
                    num_steps += 1
                    average_per_timestep[-1].append(step_total / len(signals))

                assert num_steps == 360

            average = total / num_steps
            average_per_episode.append(average)

        run_name = " ".join(split_name)
        average_per_episode = np.asarray(average_per_episode)
        average_per_timestep = np.asarray(average_per_timestep).T

        run_avg[run_name].append(average_per_episode)
        run_sim[run_name].append(average_per_timestep)

    # Average delay.
    alg_res = []
    alg_name = []

    # new loop control variables.
    metrics = defaultdict(list)
    for run_name in run_avg:
        list_runs = run_avg[run_name]
        min_len = min([len(run) for run in list_runs])
        list_runs = [run[:min_len] for run in list_runs]
        avg_delays = np.sum(list_runs, 0) / len(list_runs)
        # err = np.std(list_runs, axis=0)
        res = bs.bootstrap(avg_delays, stat_func=bs_stats.mean, alpha=0.01)

        alg_name.append(run_name)
        alg_res.append(avg_delays)

        alg_name.append(run_name + "_yerr")
        alg_res.append(res.upper_bound - res.lower_bound)

        # plt.title(run_name)
        # plt.plot(avg_delays)
        # plt.show()

        label = ALGO_ID_TO_ALGO_LBL[run_name.split()[0].upper()]

        metrics["LABELS"].append(label)
        metrics["QUEUE"].append(f"{res.value:.2f}")
        metrics["UB"].append(f"{res.upper_bound:.2f}")
        metrics["LB"].append(f"{res.lower_bound:.2f}")
        metrics["ERROR"].append(
            f"({res.lower_bound - res.value:.2f}, {res.upper_bound - res.value:.2f})"
        )
        metrics["SCENARIOS"].append(run_name.split()[2])

    filename = get_filename(metrics["SCENARIOS"], metrics["LABELS"])
    df = pd.DataFrame(data=metrics).set_index("LABELS")
    print(df.to_string())
    df.to_csv(log_dir.parent / "stats" / f"{filename}.csv", sep=",")
    labels = []
    scenarios = []
    for run_name, sim_data in run_sim.items():
        # list_runs = run_avg[run_name]
        n_series = sim_data
        label = ALGO_ID_TO_ALGO_LBL[run_name.split()[0].upper()]
        scenarios.append(run_name.split()[2])

        print(label, scenarios[-1])
        labels.append(label)

        marker, color = get_marker_and_color(label)
        queues = []
        ubs = []
        lbs = []

        for i, sim_step in enumerate(np.concatenate(sim_data, axis=1)):
            res = bs.bootstrap(sim_step, stat_func=bs_stats.mean, alpha=0.01)
            queues.append(res.value)
            lbs.append(res.lower_bound)
            ubs.append(res.upper_bound)

        lbs, queues, ubs = np.asarray(lbs), np.asarray(queues), np.asarray(ubs)
        plt.plot(range(len(queues)), queues, label=label, linestyle="-", c=color)
        plt.fill_between(range(len(queues)), lbs, ubs, alpha=0.25, facecolor=color)
    plt.xlabel("In-episode Timesteps")
    plt.ylabel("Average Queue")
    plt.legend(loc="best")
    savefig(filename, log_dir.parent / "plots")
    # plt.show()

    np.set_printoptions(threshold=sys.maxsize)
    with open(output_file, "a") as out:
        for i, res in enumerate(alg_res):
            out.write("'{}': {},\n".format(alg_name[i], res.tolist()))
