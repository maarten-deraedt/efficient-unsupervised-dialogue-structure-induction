import random
import time
import json
import itertools
import numpy as np

from models import Baseline, Regressor

state_numbers = {
    "simdial": {
        "restaurant": {"user": 7, "system": 8},
        "weather": {"user": 6, "system": 7},
        "bus": {"user": 7, "system": 8},
        "movie": {"user": 8, "system": 9},
    },
    "sgd": {
        "events_2": {"user": 100, "system": 46},
        "music_2": {"user": 28, "system": 10},
        "movies_1": {"user": 72, "system": 16},
        "homes_1": {"user": 71, "system": 24},
    },
    "camrest": {"user": 78, "system": 16},
    "dstc2": {"user": 95, "system": 49}
}


def combine_metrics(user_metrics, system_metrics):
    metrics = ["adjusted_mutual_info", "adjusted_rand_score", "fowlkes_mallows_score"]
    combined = dict()
    for metric in metrics:
        combined[metric] = (user_metrics[metric] + system_metrics[metric]) / 2
    combined["time_user"] = user_metrics["time_user"]
    combined["time_system"] = system_metrics["time_system"]
    combined["total_time"] = user_metrics["time_user"] + system_metrics["time_system"]
    return combined


def aggregate_metrics(metrics):
    aggregated_metrics = dict()
    for model in metrics.keys():
        keys = list(metrics[model][0].keys())
        aggregated_metrics[model] = dict()
        for key in keys:
            values = [d[key] for d in metrics[model]]
            aggregated_metrics[model][key + '_mean'] = np.mean(values)
            if "time" not in key:
                aggregated_metrics[model][key + '_std'] = np.std(values)
    return aggregated_metrics


def evaluate(clusterer, n_system, n_user):
   metrics_user = clusterer.cluster_eval(n_user, user=True, turn_eval=True)
   metrics_system = clusterer.cluster_eval(n_system, user=False, turn_eval=True)
   return metrics_user, metrics_system


def evaluate_turn(clusterer, n_system: int, n_user: int, task: str):
    user_first = True if task == "sgd" else False
    metrics_turn = clusterer.cluster_eval_turn(n_user=n_user, n_system=n_system, user_first=user_first)
    return metrics_turn


embedder_name = "all-MiniLM-L6-v2"
num_runs = 3
tasks = ["simdial", "simdial", "simdial", "simdial", "sgd", "sgd", "sgd", "sgd"] #"camrest", "dstc2"]
services = ["weather", "restaurant", "movie", "bus", "events_2", "music_2", "movies_1", "homes_1"] # None, None]
# tasks = ["sgd", "sgd", "sgd", "sgd", "camrest", "dstc2"]
# services = ["events_2", "music_2", "movies_1", "homes_1", None, None]
for task, service in zip(tasks, services):
    path_to_pickle = f"persisted_new/{task}-{service}-{embedder_name}.pkl" if service is not None else f"persisted_new/{task}-{embedder_name}.pkl"
    out_file = f"results/{task}/turn_{service.lower()}-{embedder_name}.json" if service is not None else f"results/{task}/{embedder_name}.json"
    n_user, n_system = (state_numbers[task][service]["user"], state_numbers[task][service]["system"]) \
        if service is not None else (state_numbers[task]["user"], state_numbers[task]["system"])

    print(f"{task} & {service} with #user states: {n_user} and #system states: {n_system}")

    all_metrics = {"c": [], "pc": [], "pcn": [], "int": [], "ext": [], "int_ext": []}
    for run in range(num_runs):
        start = time.time()
        seed = random.randrange(999, 99999999)
        c = Baseline(path_to_pickle=path_to_pickle, seed=seed, config="c")
        pc = Baseline(path_to_pickle=path_to_pickle, seed=seed, config="pc")
        pcn = Baseline(path_to_pickle=path_to_pickle, seed=seed, config="pcn")

        interpolator = Regressor(path_to_pickle=path_to_pickle, seed=seed, method="interpolate", features="neighbors")
        extrapolator = Regressor(path_to_pickle=path_to_pickle, seed=seed, method="extrapolate", features="neighbors")
        hybrid = Regressor(path_to_pickle=path_to_pickle, seed=seed, method="hybrid", features="neighbors")

        metrics_c = evaluate_turn(c, n_system=n_system, n_user=n_user, task=task)
        metrics_pc = evaluate_turn(pc, n_system=n_system, n_user=n_user, task=task)
        metrics_pcn = evaluate_turn(pcn, n_system=n_system, n_user=n_user, task=task)

        metrics_int = evaluate_turn(interpolator, n_system=n_system, n_user=n_user, task=task)
        metrics_ext = evaluate_turn(extrapolator, n_system=n_system, n_user=n_user, task=task)
        metrics_int_ext = evaluate_turn(hybrid, n_system=n_system, n_user=n_user, task=task)

        all_metrics["c"].append(metrics_c)
        all_metrics["pc"].append(metrics_pc)
        all_metrics["pcn"].append(metrics_pcn)

        all_metrics["int"].append(metrics_int)
        all_metrics["ext"].append(metrics_ext)
        all_metrics["int_ext"].append(metrics_int_ext)

        print(f"Took {time.time() - start} seconds for run {run + 1}.")

    aggregated = aggregate_metrics(all_metrics)

    with open(out_file, "w") as fp:
        json.dump(aggregated, fp, indent=2)