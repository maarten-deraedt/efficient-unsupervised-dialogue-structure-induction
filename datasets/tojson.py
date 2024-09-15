from dialogs import Dataset

import os

for dataset_path in os.listdir():
    if ".pkl" in dataset_path:
        dataset = Dataset.from_pickle(dataset_path)
        dataset.export(f"exported_new/{dataset_path.replace('.pkl', '.json').replace('-all-MiniLM-L6-v2', '')}")