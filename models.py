from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    adjusted_mutual_info_score,
)
from dialogs import (
    Dataset,
    get_all_user_utterances,
    get_all_system_utterances,
)

import time
import json
import csv
import random
import collections
import numpy as np


def _get_embedding(utterance, direction, embedding_dim, steps, average: bool = False):
    embeddings = []
    for _ in range(steps):
        if utterance is not None:
            utterance = utterance.next_message if direction == 'next' else utterance.previous_message
            embeddings.append(np.zeros(shape=(embedding_dim,)) if utterance is None else utterance.embeddings)
    if len(embeddings) == 1 and steps == 2:
        embeddings.append(np.zeros(shape=(embedding_dim,)))
    if average:
        return np.mean(embeddings, axis=0)
    else:
        return np.hstack(embeddings)


def neighbor_embeddings(utterances):
    X_left, X_current, X_right, y = [], [], [], []
    embedding_dim = utterances[0].embeddings.shape[0]
    for utterance in utterances:
        X_left.append(_get_embedding(utterance, 'previous', embedding_dim, steps=1))
        X_current.append(utterance.embeddings)
        X_right.append(_get_embedding(utterance, 'next', embedding_dim, steps=1))
        y.append("".join(utterance.text_labels))

    return {
        "X_left": np.array(X_left),
        "X_current": np.array(X_current),
        "X_right": np.array(X_right),
        "y": np.array(y)
    }


def distant_neighbor_embeddings(utterances):
    # Perhaps we will have to add an option for averaging as well.
    X_left, X_current, X_right, y = [], [], [], []
    embedding_dim = utterances[0].embeddings.shape[0]
    for utterance in utterances:
        X_left.append(_get_embedding(utterance, 'previous', embedding_dim, steps=2))
        X_current.append(utterance.embeddings)
        X_right.append(_get_embedding(utterance, 'next', embedding_dim, steps=2))
        y.append("".join(utterance.text_labels))

    return {
        "X_left": np.array(X_left),
        "X_current": np.array(X_current),
        "X_right": np.array(X_right),
        "y": np.array(y)
    }


def _all_neighbors(utterance, direction, embedding_dim):
    embeddings = []
    while utterance is not None:
        utterance = utterance.next_message if direction == 'next' else utterance.previous_message
        if utterance is not None:
            embeddings.append(utterance.embeddings)
    if len(embeddings) < 1:
        return np.zeros(shape=(embedding_dim,))
    else:
        return np.mean(embeddings, axis=0)


def all_embeddings(utterances):
    embedding_dim = utterances[0].embeddings.shape[0]
    X_left, X_current, X_right, y = [], [], [], []
    for utterance in utterances:
        X_left.append(_all_neighbors(utterance, 'previous', embedding_dim))
        X_right.append(_all_neighbors(utterance, 'next', embedding_dim))
        X_current.append(utterance.embeddings)
        y.append("".join(utterance.text_labels))
    return {
        "X_left": np.array(X_left),
        "X_current": np.array(X_current),
        "X_right": np.array(X_right),
        "y": np.array(y)
    }


class Clusterer(object):
    def __init__(self, path_to_pickle: str, seed: int):
        self.dataset = Dataset.from_pickle(path_to_pickle)
        self.embedding_dim = self.dataset.dialogs[0].utterances[0].embeddings.shape[0]
        self.seed = seed
        self.user_utterances = get_all_user_utterances(self.dataset)
        self.system_utterances = get_all_system_utterances(self.dataset)
        self.X_user, self.X_system = None, None
        self.y_user, self.y_system = None, None
        self.cluster_time = None

    def cluster(self, n_clusters: int, user: bool = True):
        start = time.time()
        X, utterances = (self.X_user, self.user_utterances) if user else (self.X_system, self.system_utterances)
        y_pred = KMeans(n_clusters=n_clusters, max_iter=1000, random_state=self.seed).fit_predict(X)
        self.cluster_time = time.time() - start
        for i in range(y_pred.shape[0]):
            utterances[i].cluster = y_pred[i]

    def evaluate(self, out_file: str = None, user: bool = True, n_faulty_samples: int = None):
        if user:
            y = self.y_user
            utterances = self.user_utterances
        else:
            y = self.y_system
            utterances = self.system_utterances
        y_pred = [utterance.cluster for utterance in utterances]
        metrics = {
            "adjusted_mutual_info": adjusted_mutual_info_score(y, y_pred),
            "adjusted_rand_score": adjusted_rand_score(y, y_pred),
            "fowlkes_mallows_score": fowlkes_mallows_score(y, y_pred),
        }
        if user:
            metrics["time_user"] = self.cluster_time
        else:
            metrics["time_system"] = self.cluster_time

        clusters = dict()
        for cluster_id, utterance in zip(y_pred, utterances):
            cluster = clusters.setdefault(cluster_id, {"utterances": [], "labels": []})
            cluster["utterances"].append(utterance)
            cluster["labels"].append("".join(utterance.text_labels))

        rows = [
            [
                "Previous Utterance",
                "Utterance",
                "Next Utterance",
                "Label",
                "Cluster label",
                "Cluster ID",
                "Frequency",
            ]
        ]
        for cluster_id in clusters.keys():
            label_counter = collections.Counter(clusters[cluster_id]["labels"])
            total_counts = sum(label_counter.values())
            most_common_label = label_counter.most_common(1)[0][0]
            most_common_freq = label_counter.most_common(1)[0][1] / total_counts
            for utterance, label in zip(
                    clusters[cluster_id]["utterances"], clusters[cluster_id]["labels"]
            ):
                rows.append(
                    [
                        utterance.previous_message.utterance if utterance.previous_message is not None else None,
                        utterance.utterance,
                        utterance.next_message.utterance if utterance.next_message is not None else None,
                        label,
                        most_common_label,
                        cluster_id,
                        most_common_freq,
                    ]
                )

        # add error samples
        faulty_samples = []
        if n_faulty_samples is not None:
            lower_freq, upper_freq = 0.4, 0.9
            faulty_samples = list(filter(lambda row: row[3] != row[4] and 0.4 <= float(row[6]) <= 0.9, rows[1:]))
            faulty_samples = random.sample(faulty_samples, n_faulty_samples)

        if out_file is not None:
            with open(f"{out_file}.json", "w") as fp:
                json.dump(metrics, fp, indent=4)
            print(json.dumps(metrics, indent=4))

            with open(f"{out_file}.tsv", "w") as fp:
                writer = csv.writer(
                    fp, delimiter="\t", quotechar="|", quoting=csv.QUOTE_MINIMAL
                )
                for row in rows:
                    writer.writerow(row)
                for faulty_sample in faulty_samples:
                    writer.writerow(faulty_sample)

        return metrics

    import json
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score

    def evaluate_turn(self, out_file: str = None, user_first: bool = False):
        def process_utterances(utterances, is_user_first):
            turns, turn_labels, turn_predicted_labels = [], [], []
            for utterance in utterances:
                predicted_label = str(utterance.cluster)
                label = "".join(utterance.text_labels)
                other_utterance = utterance.next_message

                if is_user_first:
                    assert utterance.is_user
                    if other_utterance is not None:
                        assert other_utterance.is_system
                        label += "-" + "".join(other_utterance.text_labels)
                        predicted_label += "-" + str(other_utterance.cluster)
                else:
                    assert utterance.is_system
                    if other_utterance is not None:
                        assert other_utterance.is_user
                        label += "-" + "".join(other_utterance.text_labels)
                        predicted_label += "-" + str(other_utterance.cluster)

                turn_labels.append(label)
                turn_predicted_labels.append(predicted_label)
                turns.append((utterance.utterance, other_utterance.utterance if other_utterance else "N/A"))

            return turns, turn_labels, turn_predicted_labels

        if user_first:
            turns, turn_labels, turn_predicted_labels = process_utterances(self.user_utterances, True)
        else:
            turns, turn_labels, turn_predicted_labels = process_utterances(self.system_utterances, False)

        metrics = {
            "adjusted_rand_score": adjusted_rand_score(turn_labels, turn_predicted_labels),
            "adjusted_mutual_info": adjusted_mutual_info_score(turn_labels, turn_predicted_labels),
            "fowlkes_mallows_score": fowlkes_mallows_score(turn_labels, turn_predicted_labels),
        }

        print(f"There are {len(set(turn_labels))} turn labels for {out_file}")
        assert len(turn_labels) == len(turns)

        if out_file is not None:
            with open(f"{out_file}.json", "w") as fp:
                json.dump(metrics, fp, indent=4)
            print(json.dumps(metrics, indent=4))

        print(json.dumps(metrics))
        return metrics

    def cluster_eval(self, n_clusters: int, user: bool = True):
        self.cluster(n_clusters=n_clusters, user=user)
        return self.evaluate(out_file=None, user=user)

    def cluster_eval_turn(self, n_user: int, n_system: int, user_first: bool):
        self.cluster(n_clusters=n_user, user=True)
        self.cluster(n_clusters=n_system, user=False)
        return self.evaluate_turn(out_file=None, user_first=user_first)


class Baseline(Clusterer):
    def __init__(self, path_to_pickle: str, seed: int, config: str = "pcn"):
        super().__init__(path_to_pickle, seed)

        user_features = neighbor_embeddings(self.user_utterances)
        system_features = neighbor_embeddings(self.system_utterances)

        self.y_user, self.y_system = user_features["y"], system_features["y"]

        if config == "c":
            self.X_user = user_features["X_current"]
            self.X_system = system_features["X_current"]
        elif config == "pc":
            self.X_user = np.hstack((user_features["X_left"], user_features["X_current"]))
            self.X_system = np.hstack((system_features["X_left"], system_features["X_current"]))
        elif config == "pcn":
            self.X_user = np.hstack((user_features["X_left"], user_features["X_current"], user_features["X_right"]))
            self.X_system = np.hstack((system_features["X_left"], system_features["X_current"], system_features["X_right"]))


class Regressor(Clusterer):
    def __init__(self, path_to_pickle: str, seed: int, method: str, features: str = "neighbors"):
        super().__init__(path_to_pickle, seed)

        print(f"{method} using features: {features}.")

        feature_function = self.get_feature_function(features)
        user_features = feature_function(self.user_utterances)
        system_features = feature_function(self.system_utterances)

        X_user, Y_user = self.get_X_Y(user_features, features)
        X_system, Y_system = self.get_X_Y(system_features, features)

        self.y_user, self.y_system = user_features["y"], system_features["y"]

        self.X_user = self.fit_predict(X_user, Y_user, method)
        self.X_system = self.fit_predict(X_system, Y_system, method)

    def get_feature_function(self, features):
        if features == "neighbors":
            return neighbor_embeddings
        elif features == "distant_neighbors":
            return distant_neighbor_embeddings
        elif features == "all":
            return all_embeddings
        elif features in ["left", "right"]:
            return neighbor_embeddings

    def get_X_Y(self, features, feature_type):
        X, Y = None, None
        if feature_type in ["neighbors", "distant_neighbors", "all"]:
            X = np.hstack((features["X_left"], features["X_right"]))
        elif feature_type == "left":
            X = features["X_left"]
        elif feature_type == "right":
            X = features["X_right"]
        Y = features["X_current"]
        return X, Y

    def fit_predict(self, X, Y, method):
        if method == "interpolate":
            regressor = LinearRegression()
            regressor.fit(X, Y)
            return regressor.predict(X)
        elif method == "extrapolate":
            regressor = LinearRegression()
            regressor.fit(Y, X)
            return regressor.predict(Y)
        elif method == "hybrid":
            interpolator, extrapolator = LinearRegression(), LinearRegression()
            interpolator.fit(X, Y)
            extrapolator.fit(Y, X)
            return np.hstack((extrapolator.predict(Y), interpolator.predict(X)))