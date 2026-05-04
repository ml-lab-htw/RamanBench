"""Logging utilities for the benchmark pipeline."""

import logging
from contextlib import contextmanager

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s:%(funcName)s:%(lineno)d: %(message)s"

_RUN_FORMAT = logging.Formatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


@contextmanager
def run_file_logger(log_path: str):
    """Attach a per-run FileHandler for the duration of a context.

    All log records emitted by any logger in the process are written to
    *log_path* while the context is active.  The handler is always
    removed and closed on exit, even if an exception occurs.

    The handler is attached both to the root logger and directly to the
    ``raman_bench`` logger tree so that AutoGluon / Ray reconfigurations
    of the root logger (which happen during HPO) cannot silence our logs.

    Parameters
    ----------
    log_path : str
        Path to the log file.  The file is created (or overwritten) when
        the context is entered.
    """
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_RUN_FORMAT)

    _loggers = [
        logging.getLogger(),           # root — catches everything by default
        logging.getLogger("raman_bench"),  # survives root reconfiguration by Ray/AG
    ]
    for lg in _loggers:
        lg.addHandler(handler)
    try:
        yield
    finally:
        handler.flush()
        for lg in _loggers:
            lg.removeHandler(handler)
        handler.close()
