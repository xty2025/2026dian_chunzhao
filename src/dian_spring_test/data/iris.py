from __future__ import annotations

from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
LABEL_TO_ID = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}


def _load_iris_from_uci() -> tuple[np.ndarray, np.ndarray]:
    raw_text = urlopen(IRIS_URL, timeout=30).read().decode("utf-8")
    features = []
    labels = []
    for line in raw_text.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 5:
            continue
        features.append([float(value) for value in parts[:4]])
        labels.append(LABEL_TO_ID[parts[4]])
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def make_iris_loaders(
    batch_size: int = 16,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    x, y = _load_iris_from_uci()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    test_size = int(len(x) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_x = torch.from_numpy(x[train_indices])
    train_y = torch.from_numpy(y[train_indices])
    test_x = torch.from_numpy(x[test_indices])
    test_y = torch.from_numpy(y[test_indices])

    train_mean = train_x.mean(dim=0, keepdim=True)
    train_std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - train_mean) / train_std
    test_x = (test_x - train_mean) / train_std

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
