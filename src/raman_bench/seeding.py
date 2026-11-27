"""Seeding for the Raman models: one seed per AutoGluon model, applied to numpy and torch.

AutoGluon gives every bagged child its own ``random_seed``. The models draw from numpy's and
torch's global generators (augmentation, weight initialisation, shuffling), so each fit seeds
both from that value before it draws anything (issue #171). Without this, two fits of the same
cell give different predictions.

:func:`get_seeds` is the seed list of a multi-repetition run.
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


def get_seeds(config):
    """Return list of random seeds based on config.

    If ``n_repetitions`` is present, returns ``[0, 1, ..., n_repetitions-1]``.
    If only ``random_state`` is present (legacy), returns ``[random_state]``.
    If neither is present, returns ``[42]``.

    Parameters
    ----------
    config : dict
        Benchmark configuration dictionary.

    Returns
    -------
    list[int]
        Ordered list of random seeds to use across repetitions.
    """
    n_repetitions = config.get("n_repetitions")
    if n_repetitions is not None:
        return list(range(n_repetitions))
    random_state = config.get("random_state", 42)
    return [random_state]
