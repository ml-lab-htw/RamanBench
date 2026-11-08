"""Seeding for the Raman models: one seed per AutoGluon model, applied to numpy and torch.

AutoGluon gives every bagged child its own ``random_seed``. The models draw from numpy's and
torch's global generators (augmentation, weight initialisation, shuffling), so each fit seeds
both from that value before it draws anything (issue #171). Without this, two fits of the same
cell give different predictions.
"""

from __future__ import annotations

import numpy as np
import torch


def model_seed(model) -> int:
    """The AutoGluon seed of *model*, or 0 before AutoGluon has set one."""
    seed = getattr(model, "random_seed", None)
    if seed is None or seed == "NOTSET":
        return 0
    return int(seed)


def seed_global_rngs(seed: int) -> None:
    """Seed numpy's and torch's global generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
