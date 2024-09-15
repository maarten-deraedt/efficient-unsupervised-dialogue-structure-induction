from dialogs import (
    Dataset,
    TurnDataset,
    get_all_first_k_user_messages,
    Utterance,
)
from typing import List
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from dataloaders import load_simdial_json, load_sgd, load_camrest, load_dstc2

import os
import torch
import time
import numpy as np


# class EmbedderSentenceTransformer(object):
#     def __init__(self, model_name: str):
#         self.model_name = model_name
#         self.model = SentenceTransformer(model_name)
#         # self.model.max_seq_length = max_seq_length
#
#     def embed(self, sentence_batch: List[str]):
#         start = time.time()
#         encodings = self.model.encode(sentence_batch)
#         print(f"Took {time.time() - start} seconds to encode {len(sentence_batch)} sentences")
#         return encodings


def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    start = time.time()
    model = SentenceTransformer(model_name)
    print(
        "Took {} seconds to load encoding model: {}".format(
            time.time() - start, model_name
        )
    )
    return model


def encode_utterances_sentence_transformer(
    model: SentenceTransformer,
    dataset: Dataset,
    path_to_persist: str = None,
    first_k_utterances: int = None,
) -> None:
    start = time.time()

    if first_k_utterances is not None:
        utterances = get_all_first_k_user_messages(dataset, first_k_utterances)
        for utterance in utterances:
            utterance.embeddings = model.encode(
                [utterance.utterance.lower().replace("\"", "").replace(",", "")], convert_to_numpy=True
            ).flatten()
    else:
        n_utterances = 0
        n_dialogs = 0
        for dialog in dataset.dialogs:
            embeddings = model.encode([u.utterance for u in dialog.utterances], convert_to_numpy=True)
            n_dialogs += 1
            for i, utterance in enumerate(dialog.utterances):
                utterance.embeddings = embeddings[i, :]
                n_utterances += 1
        elapsed_time = time.time() - start
        print(
            "Took {} seconds to embed {} utterances; {} sentences/second".format(
                elapsed_time, n_utterances, n_utterances / elapsed_time
            )
        )
    if path_to_persist is not None:
        dataset.persist(path_to_persist)


if __name__=="__main__":
    dataset_names = ["camrest676", "dstc2", "simdial-weather", "simdial-restaurant", "simdial-movie", "simdial-bus",
                "sgd-music_2", "sgd-movies_1", "sgd-homes_1", "sgd-events_2"]
    embedder = get_sentence_transformer("all-MiniLM-L6-v2") # "average_word_embeddings_glove.6B.300d"
    path_to_persist = "datasets" if not torch.cuda.is_available() else "/myserverpath/dialoguestructureinduction/datasets"
    for dataset_name in dataset_names:
        if dataset_name == "camrest676":
            dataset = load_camrest(export=False)
        elif dataset_name == "dstc2":
            dataset = load_dstc2(export=False)
        elif dataset_name.split("-")[0] == "simdial":
            service = dataset_name.split("-")[1]
            dataset = load_simdial_json(f"datasets/SimDial/{service}-CleanSpec-2000.json")
        elif dataset_name.split("-")[0] == "sgd":
            service = dataset_name.split("-")[1]
            dataset = load_sgd(target_service=service, split="train")
        print(f"Starting to encode the utterances for {dataset_name}")
        start = time.time()
        encode_utterances_sentence_transformer(model=embedder, dataset=dataset, path_to_persist=f"{path_to_persist}/{dataset_name}.pkl")
        print(f"Took {time.time() - start} seconds to encode the utterances!")