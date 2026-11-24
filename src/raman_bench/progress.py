"""Progress lines for the long result-reading loops."""

import time


def log_progress(logger, what: str, done: int, total: int, t0: float, every: int) -> None:
    """Log ``what: done/total, elapsed, ETA`` when *done* is a multiple of *every* or the last item."""
    if done % every and done != total:
        return
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done)
    logger.info("%s: %d/%d (%.0f%%), %.0fs elapsed, ~%.0fs left", what, done, total, 100 * done / total, elapsed, eta)
