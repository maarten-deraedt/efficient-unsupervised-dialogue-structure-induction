from typing import List
from dialogs import Dataset, Utterance, Dialog

import os
import json


def load_dstc2(export=False) -> Dataset:
    train_file_list = "datasets_src/DSTC2/dstc2_traindev/scripts/config/dstc2_train.flist"
    with open(train_file_list, "r") as fp:
        dialog_dirs = [f"datasets_src/DSTC2/dstc2_traindev/data/" + line.strip() for line in fp.readlines()]

    dialogs = []
    for dialog_dir in dialog_dirs:
        with open(dialog_dir + "/label.json", "r") as fp:
            user_dict = json.load(fp)
        with open(dialog_dir + "/log.json", "r") as fp:
            system_dict = json.load(fp)
        user_messages = []
        for turn in user_dict["turns"]:
            user_utterance = turn["transcription"]
            user_labels = []
            acts = turn["semantics"]["json"]
            for act in acts:
                act_type = act["act"]
                if len(act["slots"]) == 0:
                    user_labels.append(act_type)
                else:
                    for slot in act["slots"]:
                        slot_type = slot[0] if slot[0] != "slot" else slot[1]
                        user_labels.append(f"{act_type}_{slot_type}")
            if len(user_labels) == 0:
                # user_labels = [turn["semantics"]["cam"]]
                pass # don't add noisy use utterances.
            else:
                user_messages.append(Utterance(utterance=user_utterance, text_labels=sorted(user_labels), is_system=False))
        system_messages = []
        for turn in system_dict["turns"]:
            system_utterance = turn["output"]["transcript"]
            acts = turn["output"]["dialog-acts"]
            system_labels = []
            for act in acts:
                act_type = act["act"]
                if len(act["slots"]) == 0:
                    system_labels.append(act_type)
                else:
                    for slot in act["slots"]:
                        slot_type = slot[0] if slot[0] != "slot" else slot[1]
                        system_labels.append(f"{act_type}_{slot_type}")
            system_messages.append(Utterance(utterance=system_utterance, text_labels=sorted(system_labels), is_system=True))
        messages = [utterance for pair in zip(system_messages, user_messages) for utterance in pair]
        dialogs.append(Dialog(messages))

    dataset = Dataset(dialogs=dialogs)
    if export:
        dataset.export("datasets_src/exported/DSTC2-parsed.json")
    return dataset


def load_simdial_json(json_file: str, export: bool = False):
    with open(json_file, "r") as fp:
        simdial = json.load(fp)
        print("loaded the simdial dataset into a dictionary!")
        dialogs = []
        for dialog in simdial["dialogs"]:
            utterances = []
            for turn in dialog:
                utterance = turn["utt"]
                is_system = True if turn["speaker"] == "SYS" else False
                text_labels = []
                for action in turn["actions"]:
                    act = action["act"]
                    if act == "query" or act == "kb_return":
                        text_labels.append(act)
                    elif act == "more_request" or act == "satisfy":
                        text_labels.append(act)
                    else:
                        parameters = action["parameters"]
                        label = act
                        if type(parameters) == list:
                            for param in parameters:
                                if type(param) == dict:
                                    label += list(param.keys())[0]
                                elif type(param) == list and len(param) > 0:
                                    param_name = param[0]
                                    label += param_name
                        text_labels.append(label)
                if "query" not in text_labels and "kb_return" not in text_labels:
                    utterances.append(Utterance(utterance=utterance, is_system=is_system, text_labels=sorted(text_labels)))
            dialogs.append(Dialog(utterances=utterances))

        dataset = Dataset(dialogs=dialogs)
        if export:
            dataset.export(json_file.replace(".json", "-parsed.json"))
        return dataset


def load_sgd_json(path_to_json: str):
    simple_acts = {"REQUEST_ALTS", "THANK_YOU", "GOODBYE", "NEGATE", "AFFIRM",
                   "REQ_MORE", "NOTIFY_FAILURE", "NOTIFY_SUCCESS"}
    with open(path_to_json, "r") as fp:
        sgd = json.load(fp)

    dialogs = []
    for dialog in sgd:
        utterances = []
        for turn in dialog["turns"]:
            is_system = True if turn["speaker"] != "USER" else False
            utterance = turn["utterance"]
            text_labels = []
            for frame in turn["frames"]:
                for action in frame["actions"]:
                    act = action["act"]
                    slot = action["slot"]
                    values = action["values"]
                    if act in simple_acts:
                        text_labels.append(act)
                    elif slot == "intent":
                        text_labels.append(act + "." + str(values))
                    else:
                        text_labels.append(act + "." + slot)
            utterances.append(Utterance(utterance=utterance, is_system=is_system, text_labels=sorted(text_labels)))
        dialogs.append(Dialog(utterances=utterances))

    return Dataset(dialogs=dialogs)


def load_sgd(target_service: str, split: str, export: bool = False):
    simple_acts = {"REQUEST_ALTS", "THANK_YOU", "GOODBYE", "NEGATE", "AFFIRM",
                   "REQ_MORE", "NOTIFY_FAILURE", "NOTIFY_SUCCESS"}

    target_dialogs = []
    services = []
    for file in os.listdir(f"datasets_src/SGD/{split}"):
        if file != "schema.json":
            with open(f"datasets_src/SGD/{split}/{file}", "r") as fp:
                dialogs = json.load(fp)
                for dialog in dialogs:
                    if len(dialog["services"]) == 1:
                        if target_service.lower() == dialog["services"][0].lower():
                            target_dialogs.append(dialog)
                    services.append(";".join(dialog["services"]))

    dialogs = []
    for dialog in target_dialogs:
        utterances = []
        for turn in dialog["turns"]:
            is_system = True if turn["speaker"] != "USER" else False
            utterance = turn["utterance"]
            text_labels = []
            for frame in turn["frames"]:
                for action in frame["actions"]:
                    act = action["act"]
                    slot = action["slot"]
                    values = action["values"]
                    if act in simple_acts:
                        text_labels.append(act)
                    elif slot == "intent":
                        text_labels.append(act + "." + str(values))
                    else:
                        text_labels.append(act + "." + slot)
            utterances.append(Utterance(utterance=utterance, is_system=is_system, text_labels=sorted(text_labels)))
        dialogs.append(Dialog(utterances=utterances))

    dataset = Dataset(dialogs=dialogs)
    if export:
        dataset.export(f"datasets_src/exported/SGD-{target_service}-parsed.json")
    return dataset


def load_camrest(export=False) -> Dataset:
    with open("datasets_src/CamRest/camrest676.json", "r") as fp:
        camrest = json.load(fp)
    dialogs: List[Dialog] = []
    for dialog in camrest:
        dial = dialog["dial"]
        utterances: List[Utterance] = []
        all_user_labels = []
        for i, turn in enumerate(dial):
            system_utterance = turn["sys"]["sent"].replace("<s>", " ").replace("</s>", "")
            user_utterance = turn["usr"]["transcript"].replace("<s>", " ").replace("</s>", "")
            system_labels = turn["sys"]["DA"]
            if len(system_labels) > 0:
                if type(system_labels[0]) == list:
                    system_labels = system_labels[0]
            new_user_labels = []
            for annotation in turn["usr"]["slu"]:
                act = annotation["act"]
                if act == "request":
                    slot = annotation["slots"][0][1]
                    value = ""
                else:
                    slot = annotation["slots"][0][0]
                    value = annotation["slots"][0][1]
                current_label = act + "." + slot
                annotation_str = json.dumps(annotation)
                if annotation_str not in all_user_labels:
                    new_user_labels.append(current_label)
                    all_user_labels.append(annotation_str)

            utterances.append(Utterance(utterance=user_utterance, is_system=False, text_labels=sorted(new_user_labels)))
            utterances.append(Utterance(utterance=system_utterance, is_system=True, text_labels=sorted(system_labels)))
        dialog = Dialog(utterances=utterances)
        dialogs.append(dialog)
    dataset = Dataset(dialogs)
    if export:
        dataset.export("datasets_src/exported/CamRest676-parsed.json")
    return dataset