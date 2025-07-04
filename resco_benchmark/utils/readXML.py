from collections import defaultdict
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Tuple, List
import numpy as np
import pandas as pd
import sys
from resco_benchmark.config.map_config import map_configs
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import matplotlib
import string

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# log_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'results' + os.sep)
log_dir = Path.cwd() / "results"
env_base = ".." + os.sep + "environments" + os.sep
names = [folder for folder in next(os.walk(log_dir))[1]]
metrics = ["timeLoss", "duration", "waitingTime"]
time_loss, duration, waiting_time = (
    defaultdict(list),
    defaultdict(list),
    defaultdict(list),
)

ALGO_ID_TO_ALGO_LBL = {
    "IQL_NS": "IQL",
    "DVDN_NS": "DVDN",
    "VDN_NS": "VDN",
    "IA2C_NS": "INDA2C",
    "CENTRALV": "CENTRAL-V",
    "DNAA2C": "DNAA2C",
}

METRIC_ID_TO_METRIC_LABEL = {
    "timeLoss": "DELAY",
    "duration": "TRIP_TIME",
    "waitingTime": "WAIT",
}


def sort_key(x: Path) -> int:
    return int(x.stem.split("_")[-1])


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


def get_filename(scenarios: List[str], labels: List[str], metric: str = "queue") -> str:
    algos = "_".join([*map(lambda x: x.lower(), labels)])
    maps = "_".join(list(set(scenarios)))
    filename = f"{metric}-{algos}-{maps}"
    return filename


for metric, data in zip(metrics, (time_loss, duration, waiting_time)):
    output_file = "avg_{}.py".format(metric.lower())
    run_avg = dict()

    for name in names:
        split_name = name.split("-")
        print(split_name)
        map_name = split_name[2]
        average_per_episode = []
        path = log_dir / name
        # for i in range(1, 10000):
        for trip_file_path in sorted(path.glob("tripinfo_*.xml"), key=sort_key):
            # trip_file_name = log_dir + name + os.sep + "tripinfo_" + str(i) + ".xml"
            # trip_file_name = log_dir / name / "tripinfo_" + str(i) + ".xml"
            # if not os.path.exists(trip_file_name):
            i = sort_key(trip_file_path)
            if not trip_file_path.exists():
                print("No " + trip_file_path.stem)
                break
            try:
                tree = ET.parse(trip_file_path.as_posix())
                root = tree.getroot()
                num_trips, total = 0, 0.0
                last_departure_time = 0
                last_depart_id = ""
                for child in root:
                    try:
                        num_trips += 1
                        total += float(child.attrib[metric])
                        if metric == "TIME_LOSS":
                            total += float(child.attrib["departDelay"])
                            depart_time = float(child.attrib["depart"])
                            if depart_time > last_departure_time:
                                last_departure_time = depart_time
                                last_depart_id = child.attrib["id"]
                    except Exception as e:
                        # raise e
                        break
                route_file_name = (
                    # env_base + map_name + os.sep + map_name + "_" + str(i) + ".rou.xml"
                    log_dir.parent
                    / "resco_benchmark"
                    / "environments"
                    / map_name
                    / f"{map_name}.rou.xml"
                )
                if metric == "TIME_LOSS":  # Calc. departure delays
                    # try:
                    tree = ET.parse(route_file_name.as_posix())
                    # except FileNotFoundError:
                    #     route_file_name = (
                    #         env_base + map_name + os.sep + map_name + ".rou.xml"
                    #     )
                    #     tree = ET.parse(route_file_name)
                    root = tree.getroot()
                    last_departure_time = None
                    for child in root:
                        if child.attrib["id"] == last_depart_id:
                            last_departure_time = float(
                                child.attrib["depart"]
                            )  # Get the time it was suppose to depart
                    never_departed = []
                    if last_departure_time is None:
                        raise Exception("Wrong trip file")
                    for child in root:
                        if child.tag != "vehicle":
                            continue
                        depart_time = float(child.attrib["depart"])
                        if depart_time > last_departure_time:
                            never_departed.append(depart_time)
                    never_departed = np.asarray(never_departed)
                    never_departed_delay = np.sum(
                        float(map_configs[map_name]["end_time"]) - never_departed
                    )
                    total += never_departed_delay
                    num_trips += len(never_departed)

                average = total / num_trips
                average_per_episode.append(average)
            except ET.ParseError as e:
                # raise e
                break

        run_name = (
            split_name[0]
            + " "
            + split_name[2]
            + " "
            + split_name[3]
            + " "
            + split_name[4]
            + " "
            + split_name[5]
        )
        average_per_episode = np.asarray(average_per_episode)

        if run_name in run_avg:
            run_avg[run_name].append(average_per_episode)
        else:
            run_avg[run_name] = [average_per_episode]

    alg_res = []
    alg_name = []
    min_value, max_value = 10_000, -1
    for run_name in run_avg:
        values = run_avg[run_name][0]
        min_value = min(min(values), min_value)
        max_value = max(max(values), max_value)
        res = bs.bootstrap(values, stat_func=bs_stats.mean, alpha=0.01)

        alg_name.append(run_name)
        alg_res.append(values)

        label = ALGO_ID_TO_ALGO_LBL[run_name.split()[0].upper()]
        marker, color = get_marker_and_color(label)

        data["LABELS"].append(label)
        data["VALUE"].append(f"{res.value:.2f}")
        data["UB"].append(f"{res.upper_bound:.2f}")
        data["LB"].append(f"{res.lower_bound:.2f}")
        data["ERROR"].append(
            f"({res.lower_bound - res.value:.2f}, {res.upper_bound - res.value:.2f})"
        )
        data["SCENARIOS"].append(run_name.split()[1])

        histtype = "bar" if label in ("DVDN", "DNAA2C") else "step"
        plt.hist(
            values,
            bins=100,
            histtype=histtype,
            color=color,
            label=label,
        )

        np.set_printoptions(threshold=sys.maxsize)
        target = log_dir.parent / output_file
        with target.open("a") as out:
            for i, res in enumerate(alg_res):
                out.write("'{}': {},\n".format(alg_name[i], res.tolist()))
    filename = get_filename(
        data["LABELS"],
        data["SCENARIOS"],
        metric=_snakefy(METRIC_ID_TO_METRIC_LABEL[metric].replace("_", "")),
    )

    df = pd.DataFrame(data=data).set_index("LABELS")
    print("#########", METRIC_ID_TO_METRIC_LABEL[metric], "#########")
    print(df.to_string())
    df.to_csv(log_dir.parent / "stats" / f"{filename}.csv", sep=",")
    plt.xlabel(f"Avg. {metric.replace('_', '').title()} (s)")
    plt.ylabel("Per Simulation")
    # plt.xlim(0.9 * min_value, 1.1 * max_value)
    # plt.ylim(350, 500)
    plt.legend(loc="best")
    savefig(filename, log_dir.parent / "plots")
    # plt.show()
