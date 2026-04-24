"""Seed management for multi-repetition benchmark runs."""


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
