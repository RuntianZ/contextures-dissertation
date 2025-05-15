import numpy as np
from typing import Tuple
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sys

sys.path.append("..")
from anomaly.dataset_anomaly import AnomalyDetectionDataset
from framework.utils import fix_seed


def read_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"
    data = np.load(dataset_path)
    return data['X'], data['y']


def process_data(X, y, n_samples_threshold=1000, generate_duplicates=True):
    # if the dataset is too small, generating duplicate smaples up to n_samples_threshold
    if len(y) < n_samples_threshold and generate_duplicates:
        fix_seed(0)
        idx_duplicate = np.random.choice(np.arange(len(y)), n_samples_threshold, replace=True)
        X = X[idx_duplicate]
        y = y[idx_duplicate]

    # if the dataset is too large, subsampling for considering the computational cost
    if len(y) > 10000:
        fix_seed(0)
        idx_sample = np.random.choice(np.arange(len(y)), 10000, replace=False)
        X = X[idx_sample]
        y = y[idx_sample]
    return X, y


def tabzilla_split_dataset(X, y, num_splits=10, shuffle=True, seed=0):
    kf = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=seed)

    splits = kf.split(X, y)

    _split_indeces = []
    for train_indices, test_indices in splits:
        _split_indeces.append({"train": train_indices, "test": test_indices, "val": []})
    # Build validation set by using n+1 th test set.
    for split_idx in range(10):
        _split_indeces[split_idx]["val"] = _split_indeces[(split_idx + 1) % 10][
            "test"
        ].copy()
        _split_indeces[split_idx]["train"] = np.setdiff1d(
            _split_indeces[split_idx]["train"],
            _split_indeces[split_idx]["val"],
            assume_unique=True,
        )

    return _split_indeces


def generate_split(X, y, test_size=0.3, cv_n_folds=10):
    _split_indeces = []
    for split_idx in range(cv_n_folds):
        indices = np.arange(len(X))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, shuffle=True,
                                                       stratify=y, random_state=split_idx)
        _split_indeces.append({"train": train_indices, "test": test_indices, "val": []})
    # Build validation set by using n+1 th test set.
    for split_idx in range(cv_n_folds):
        _split_indeces[split_idx]["val"] = _split_indeces[(split_idx + 1) % cv_n_folds][
            "test"
        ].copy()
        _split_indeces[split_idx]["train"] = np.setdiff1d(
            _split_indeces[split_idx]["train"],
            _split_indeces[split_idx]["val"],
            assume_unique=True,
        )

    return _split_indeces


if __name__ == "__main__":
    from tqdm import tqdm

    dataset_folder = "Classical"
    # target_folder = f"{dataset_folder}_processed"
    target_folder = f"anomaly_data_truncated"

    for dset_path in tqdm(os.listdir(dataset_folder)):
        data_X, data_y = read_dataset(os.path.join(dataset_folder, dset_path))
        data_X, data_y = process_data(data_X, data_y)
        split_indeces = generate_split(data_X, data_y)
        dataset_kwargs = {
            "X": data_X,
            "y": np.array(data_y),
            "cat_idx": [],
            "target_type": "binary",
            "num_classes": 1,
            "split_indeces": split_indeces,
            "split_source": "random_init",
        }
        dataset_name = dset_path.split(".")[0].split("_")
        dataset_name = [dataset_name[0]] + ["ad"] + dataset_name[1:]
        dataset_name = "_".join(dataset_name)
        dataset = AnomalyDetectionDataset(dataset_name, **dataset_kwargs)
        dataset.target_encode()
        dataset.cat_idx = []
        dataset.cat_dims = []
        dataset.write(Path(target_folder) / dataset_name, overwrite=True)
