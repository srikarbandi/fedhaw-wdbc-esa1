"""Reproducibility helpers."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_torch(num_threads: int = 1) -> None:
    """Single-threaded CPU execution: deterministic and light on an 8 GB laptop."""
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
