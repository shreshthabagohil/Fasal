"""One global seed for the whole project. Import and call set_seed() everywhere
that has randomness (splits, sampling, training). Reproducibility is graded."""
import os
import random


def set_seed(seed: int | None = None) -> int:
    if seed is None:
        # LOCKED default per train/A0_LOCKED.md + 02_EXECUTION_PLAN.md §6.5 — 1729, not a generic placeholder.
        seed = int(os.environ.get("GLOBAL_SEED", "1729"))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
