import json
from typing import Dict, List, Any, Optional, Set

import numpy as np
import pickle
import graphviz


class Utterance(object):
    def __init__(
            self,
            utterance: str,
            embeddings: np.ndarray = None,
            label: int = None,
            text_labels: List[str] = None,
            is_system: bool = False,
            author: str = None,
            turn: int = None,
            previous_message: "Utterance" = None,
            next_message: "Utterance" = None,
            properties: Dict[Any, Any] = None,
            dialog_id: int = None,
    ):
        self.utterance = utterance
        self.cluster = label
        self.is_system = is_system
        self.is_user = not is_system
        self.author = author
        self.turn = turn
        self.previous_message = previous_message
        self.next_message = next_message
        self.embeddings = embeddings
        self.dialog_id = dialog_id
        self.text_labels = text_labels

        if properties is None:
            self.properties = dict()
        else:
            self.properties = properties

    def get_utterance_length(self):
        return len(self.utterance.split(" "))


class Turn(object):
    """
    Class in which a is represented as the combination of the system utterance and user utterance.
    """

    def __init__(
            self,
            system_utterance: str,
            user_utterance: str,
            embedding: np.ndarray = None,
            turn_index: int = None,
            next_turn: "Turn" = None,
            properties: Dict[Any, Any] = None,
    ):
        self.system_utterance = system_utterance
        self.user_utterance = user_utterance
        self.next_turn = next_turn
        self.turn_index = turn_index
        self.embedding = embedding

        if properties is None:
            self.properties = dict()
        else:
            self.properties = properties


class Dialog(object):
    def __init__(self, utterances: List[Utterance], embedding: np.array = None, dialog_id: int = None):
        """
        :param utterances:
        :param embedding: vector representation of the dialog
            1) concatenation or mean of first k utterance embeddings
            2) TFIDF or LDA representation of the concatenation of all utterances
        """
        self.utterances = utterances
        self.n_turns = len(self.utterances)
        self.embedding = embedding
        self.dialog_id = dialog_id

        for i in range(self.n_turns):
            if i > 0:
                self.utterances[i].previous_message = self.utterances[i - 1]
            else:
                self.utterances[i].previous_message = None

            if i < self.n_turns - 1:
                self.utterances[i].next_message = self.utterances[i + 1]
            else:
                self.utterances[i].next_message = None

    def is_binary_dialog(self) -> bool:
        for utterance in self.utterances:
            if utterance.previous_message is not None and utterance.is_system == utterance.previous_message.is_system:
                return False
        return True

    def get_dialog_as_dict(self) -> Dict:
        dialog = {"turns": []}
        for utterance in self.utterances:
            is_system = utterance.is_system
            labels = utterance.text_labels
            dialog["turns"].append(
                {
                    "utterance": utterance.utterance,
                    "system": is_system,
                    "labels": labels,
                }
            )
        return dialog

    def get_user_utterances(self) -> List[Utterance]:
        return [utterance for utterance in self.utterances if utterance.is_user]

    def get_system_utterances(self) -> List[Utterance]:
        return [utterance for utterance in self.utterances if utterance.is_system]

    def get_first_k_user_utterances(self, k: int) -> List[Utterance]:
        user_utterances = self.get_user_utterances()

        if len(user_utterances) >= k:
            return user_utterances[:k]
        else:
            return user_utterances

    def get_first_user_utterance(self) -> List[Utterance]:
        return self.get_first_k_user_utterances(k=1)

    def get_dialog_as_single_document(self) -> str:
        """
        :param n_utterances: maximum number of utterances that should be included in the result;
            if n_turns < n_utterances then all utterances will be considered else if n_turns > n_utterances, only the
            first n_utterances are added to the result.
        :return: string representation of the concatenated utterances.
        """
        document = ""
        for utterance in self.utterances:
            document = document + " " + utterance.utterance
        return document.strip()

    def pretty(self):
        result = ""
        for utterance in self.utterances:
            if utterance.is_system:
                result += "SYSTEM: " + utterance.utterance
            else:
                result += "USER: " + utterance.utterance

            if len(utterance.text_labels) > 0:
                result += f"{utterance.text_labels}"
            result += "\n"
        return result

    def pretty_tex(self):
        result = ""
        for utterance in self.utterances:
            if utterance.is_system:
                result += f" \\texttt{{SYS}}: {utterance.utterance} & "
            else:
                result += f" \\texttt{{USR}}: {utterance.utterance} & "

            if len(utterance.text_labels) > 0:
                filtered_labels = map(lambda x: x.replace("'", "").replace("#", "-").replace("_", "").lower(),
                                      utterance.text_labels)
                result += f"{list(filtered_labels)}"
            result += "\\\\ \n"
        return result

    def get_mean_utterance_length(self):
        return np.mean([utterance.get_utterance_length() for utterance in self.utterances])


class TurnDialog(object):
    def __init__(self, turns: List[Turn]):
        self.turns = turns
        self.n_turns = len(self.turns)
        for i in range(self.n_turns):
            if i < self.n_turns - 1:
                self.turns[i].next_turn = self.turns[i + 1]
            else:
                self.turns[i].next_turn = None


class Dataset(object):
    def __init__(self, dialogs: List[Dialog]):
        self.dialogs = dialogs
        self.n = len(dialogs)

    @classmethod
    def from_pickle(cls, path_to_pickle_file: str) -> "Dataset":
        return pickle.load(open(path_to_pickle_file, "rb"))

    def persist(self, path_to_pickle_file: str) -> None:
        pickle.dump(self, open(path_to_pickle_file, "wb"))

    def get_mean_utterance_length(self):
        return np.mean([dialog.get_mean_utterance_length() for dialog in self.dialogs])

    def get_mean_turns(self):
        return np.mean([dialog.n_turns for dialog in self.dialogs])

    def export(self, path_to_file: str) -> None:
        dataset = {"dialogs": []}
        for dialog in self.dialogs:
            dataset["dialogs"].append(dialog.get_dialog_as_dict())

        with open(path_to_file, "w") as fp:
            json.dump(dataset, fp, indent=4)


def get_all_user_utterances(dataset: Dataset, labeled_only: bool = True) -> List[Utterance]:
    all_user_utterances = []
    for dialog in dataset.dialogs:
        all_user_utterances += dialog.get_user_utterances()

    if labeled_only:
        all_user_utterances = [utt for utt in all_user_utterances if len(utt.text_labels) > 0]
    return all_user_utterances


def get_all_system_utterances(dataset: Dataset, labeled_only: bool = True) -> List[Utterance]:
    all_system_utterances = []
    for dialog in dataset.dialogs:
        all_system_utterances += dialog.get_system_utterances()

    if labeled_only:
        all_system_utterances = [utt for utt in all_system_utterances if len(utt.text_labels) > 0]
    return all_system_utterances


def get_all_first_k_user_messages(dataset: Dataset, k: int) -> List[Utterance]:
    all_first_k_user_utterances = []
    for dialog in dataset.dialogs:
        first_k_utterances = dialog.get_first_k_user_utterances(k)
        if first_k_utterances is not None:
            all_first_k_user_utterances += first_k_utterances
    return all_first_k_user_utterances


class TurnDataset(object):
    def __init__(self, dialogs: List[TurnDialog]):
        self.dialogs = dialogs
        self.n = len(dialogs)

    @classmethod
    def from_pickle(cls, path_to_pickle_file: str) -> "Dataset":
        return pickle.load(open(path_to_pickle_file, "rb"))

    def persist(self, path_to_pickle_file: str) -> None:
        pickle.dump(self, open(path_to_pickle_file, "wb"))
