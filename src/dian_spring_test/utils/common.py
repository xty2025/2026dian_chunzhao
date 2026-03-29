from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(payload: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
