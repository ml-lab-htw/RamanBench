"""Run a model on the RamanBench v1 protocol, so it can be compared with the leaderboard.

Two ways in:

- :func:`evaluate_estimator` runs any scikit-learn estimator (a ``classifier`` with
  ``predict_proba``, a ``regressor``, or both) through the same TabArena/AutoGluon bagged
  experiment every leaderboard model went through: same tasks, same outer folds, 3-fold
  bagging, same time budget and row caps (``data/precomputed/v1/protocol.json``).
- :func:`protocol_commands` prints the ``scripts/run_experiment.py`` calls for a model
  registered in RamanBench (``src/raman_bench/models/custom/<key>/``).

Either way the results land as cached ``results.pkl`` files under ``results_dir``, which
:func:`raman_bench.compare.compare` scores against the leaderboard::

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LogisticRegression
    from raman_bench.evaluate import evaluate_estimator
    from raman_bench.compare import compare

    evaluate_estimator(
        "MY_MODEL",
        classifier=LogisticRegression(max_iter=2000),
        regressor=PLSRegression(n_components=10),
        results_dir="results/my_model",
    )
    scores = compare("results/my_model")

The estimator sees the spectra as they come (no RamanBench preprocessing, which is
off by default for every model); put your preprocessing into a scikit-learn ``Pipeline``,
or switch on RamanBench's own steps with ``hyperparameters={"prep_snv_enabled": True, ...}``
(see :mod:`raman_bench.preprocessing.mixin`). Each fit gets a fresh ``sklearn.base.clone``
of your estimator (:class:`raman_bench.models.sklearn_estimator.Prep_SKLEARN`).
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
import shlex
import tempfile
import time

import pandas as pd

from raman_bench.compare import load_own_results, load_protocol, protocol_tasks

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")


def _check_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(f"name must be letters, digits, '.' and '-' (it names the result directories), got {name!r}")


def _task_settings(task: dict, protocol: dict) -> tuple[float, int | None]:
    """``(time_limit, max_train_samples)`` of *task* under *protocol*."""
    time_limit = protocol["time_limit_overrides"].get(task["dataset"], protocol["time_limit"])
    max_rows = protocol["max_train_samples_overrides"].get(task["dataset"])
    return float(time_limit), max_rows


def evaluate_estimator(
    name: str,
    *,
    classifier=None,
    regressor=None,
    tasks: list[str] | None = None,
    results_dir: str = "results/v1/user",
    hyperparameters: dict | None = None,
    use_mirror: bool = True,
    cache_dir: str = ".cache_v1",
    force_recompute: bool = False,
    verbosity: int = 1,
) -> pd.DataFrame:
    """Run scikit-learn estimators on the v1 protocol and return their per-fold results.

    Parameters
    ----------
    name
        Model key your results are stored and ranked under (e.g. ``"MY_PLS"``). It must not
        be a leaderboard model's key, unless you mean to replace it in
        :func:`raman_bench.compare.compare`.
    classifier, regressor
        Unfitted scikit-learn estimators; give at least one. The classifier needs
        ``predict_proba`` (binary tasks are scored by ROC AUC, multiclass by log loss).
        Tasks of a type you give no estimator for are skipped; the model is then ranked on
        the other task type only.
    tasks
        Task keys (``"<dataset>__<target_idx>"``) or dataset names to run; default all
        tasks of the protocol your estimators cover. Handy for a quick first look; for a
        leaderboard comparison run them all.
    results_dir
        Where the ``results.pkl`` files go (``<name>_c1_BAG_L1/<task>/<repeat>_<fold>/``).
        Finished folds are reused, so an interrupted run picks up where it stopped.
    hyperparameters
        Extra AutoGluon hyperparameters for the wrapper, e.g. RamanBench preprocessing
        switches (``{"prep_snv_enabled": True}``).
    verbosity
        0: errors only; 1 (default): one line per task and fold; 2: also AutoGluon's and
        TabArena's own logs, and per-fit preprocessing messages.

    Returns the tidy per-fold results (:func:`raman_bench.compare.load_own_results`) of
    *name* in *results_dir*. A fold that raises is logged and skipped; it is imputed when
    compared.
    """
    from tabarena.utils.config_utils import ConfigGenerator

    from raman_bench.models.sklearn_estimator import estimator_model_cls

    _check_name(name)
    if classifier is None and regressor is None:
        raise ValueError("Give a classifier, a regressor or both")
    if classifier is not None and not hasattr(classifier, "predict_proba"):
        raise ValueError(f"{type(classifier).__name__} has no predict_proba; RamanBench scores probabilities")
    task_types = [t for t, est in (("classification", classifier), ("regression", regressor)) if est is not None]
    selected = [t for t in protocol_tasks(tasks=tasks) if t["problem_type"] in task_types]
    if not selected:
        raise ValueError("No protocol task left for the estimators you gave")

    protocol = load_protocol()
    config = {"classifier": classifier, "regressor": regressor, **(hyperparameters or {})}
    generator = ConfigGenerator(
        model_cls=estimator_model_cls(name), name=name, manual_configs=[config], search_space={}
    )
    failed = []
    start = time.monotonic()
    with _quiet(verbosity):
        _run_tasks(
            selected, generator, protocol, failed,
            results_dir=results_dir, use_mirror=use_mirror, cache_dir=cache_dir,
            force_recompute=force_recompute, verbosity=verbosity,
        )
    logger.log(
        logging.INFO if verbosity else logging.DEBUG,
        "Finished %d task(s) in %.0f s, %d failed fold(s)", len(selected), time.monotonic() - start, len(failed),
    )
    if failed:
        logger.warning("Failed folds: %s", failed)
    results = load_own_results(results_dir)
    return results[results["model"] == name].reset_index(drop=True)


def _run_tasks(selected, generator, protocol, failed, *, results_dir, use_mirror, cache_dir, force_recompute, verbosity):
    """Run every (task, repeat, fold) of *selected*, appending failed folds to *failed*."""
    from raman_bench.experiment_utils import (
        bag_experiment_kwargs,
        build_task,
        load_dataframe,
    )

    n_splits = protocol["n_splits"]
    for i, task in enumerate(selected, 1):
        time_limit, max_rows = _task_settings(task, protocol)
        logger.info("[%d/%d] %s (%s)", i, len(selected), task["task"], task["problem_type"])
        dataset, df, sample_idx, problem_type = load_dataframe(
            task["dataset"], task["target_idx"], max_train_samples=max_rows, cache_dir=cache_dir, use_mirror=use_mirror
        )
        task_name, task_wrapper = build_task(
            dataset_name=task["dataset"],
            target_idx=task["target_idx"],
            df=df,
            raw_targets=dataset.targets,
            problem_type=problem_type,
            n_repeats=task["n_repeats"],
            n_splits=n_splits,
            sample_idx=sample_idx,
            min_samples_per_class=protocol["min_samples_per_class"],
        )
        if task_wrapper is None:
            logger.warning("Skipping %s: no usable labels after filtering", task["task"])
            continue
        for repeat in range(task["n_repeats"]):
            for fold in range(n_splits):
                # AutoGluon saves each fitted predictor; only the cached result is kept.
                with tempfile.TemporaryDirectory(prefix="raman_bench_ag_") as scratch:
                    (experiment,) = generator.generate_all_bag_experiments(
                        **bag_experiment_kwargs(
                            num_random_configs=0,
                            time_limit=time_limit,
                            num_bag_folds=protocol["num_bag_folds"],
                            scratch_dir=scratch,
                            verbosity=0 if verbosity < 2 else 2,
                        )
                    )
                    _run_fold(experiment, task_name, task_wrapper, repeat, fold, results_dir, force_recompute, failed)


def _run_fold(experiment, task_name, task_wrapper, repeat, fold, results_dir, force_recompute, failed):
    from raman_bench.experiment_utils import run_cached

    try:
        run_cached(
            experiment,
            task_name=task_name,
            task_wrapper=task_wrapper,
            repeat=repeat,
            fold=fold,
            results_dir=results_dir,
            force_recompute=force_recompute,
        )
    except Exception as e:  # one failing fold should not end a 135-task run
        logger.error("%s repeat %d fold %d failed: %s", task_name, repeat, fold, e)
        failed.append((task_name, repeat, fold, repr(e)))


@contextlib.contextmanager
def _quiet(verbosity: int):
    """Silence AutoGluon, TabArena and per-fit RamanBench logs (and stdout) below *verbosity* 2."""
    if verbosity >= 2:
        yield
        return
    from loguru import logger as loguru_logger

    names = ["autogluon", "tabarena", "openml", "raman_bench.preprocessing", "raman_bench.benchmark",
             "raman_bench.experiment_utils", "raman_data"]  # fmt: skip
    if verbosity == 0:
        names.append(__name__)
    loggers = [logging.getLogger(n) for n in names]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    loguru_logger.disable("tabarena")
    try:
        from huggingface_hub.utils import (
            are_progress_bars_disabled,
            disable_progress_bars,
            enable_progress_bars,
        )
    except ImportError:  # pragma: no cover
        are_progress_bars_disabled = None
    bars_were_off = are_progress_bars_disabled() if are_progress_bars_disabled else True
    if not bars_were_off:
        disable_progress_bars()
    try:
        # TabArena reports its result caching with print().
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)
        loguru_logger.enable("tabarena")
        if not bars_were_off:
            enable_progress_bars()


def protocol_commands(
    model_key: str,
    *,
    tasks: list[str] | None = None,
    task_type: str | None = None,
    results_dir: str = "results/v1/user",
    python: str = "python",
) -> list[str]:
    """``scripts/run_experiment.py`` calls that run a registered model on the v1 protocol.

    One command per (task, repeat, fold), with the protocol's folds, bagging, time budget
    and row caps filled in. Run them in any order or in parallel; then
    ``compare(results_dir)``.
    """
    protocol = load_protocol()
    commands = []
    for task in protocol_tasks(task_type=task_type, tasks=tasks):
        time_limit, max_rows = _task_settings(task, protocol)
        for repeat in range(task["n_repeats"]):
            for fold in range(protocol["n_splits"]):
                args = [
                    python, "scripts/run_experiment.py",
                    "--dataset", task["dataset"], "--target-idx", str(task["target_idx"]),
                    "--model", model_key, "--repeat", str(repeat), "--fold", str(fold),
                    "--config-index", "0", "--num-random-configs", "0",
                    "--n-repeats", str(task["n_repeats"]), "--n-splits", str(protocol["n_splits"]),
                    "--num-bag-folds", str(protocol["num_bag_folds"]), "--time-limit", f"{time_limit:g}",
                    "--min-samples-per-class", str(protocol["min_samples_per_class"]),
                    "--results-dir", results_dir,
                ]  # fmt: skip
                if max_rows is not None:
                    args += ["--max-train-samples", str(max_rows)]
                commands.append(shlex.join(args))
    return commands
