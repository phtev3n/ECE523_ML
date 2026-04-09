from __future__ import annotations
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all relevant random seeds for reproducible training runs.

    Sets seeds for Python's random module, NumPy, and PyTorch (both CPU and
    all CUDA devices).  Note that full determinism also requires setting
    CUBLAS_WORKSPACE_CONFIG in the environment and using torch.use_deterministic_algorithms,
    which are not enforced here to avoid performance overhead.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """Return the best available torch device matching the requested name.

    Falls back to CPU silently if CUDA is requested but not available (e.g.
    when running on a login node or local machine without a GPU).

    Args:
        name: "cuda" or "cpu".
    """
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
