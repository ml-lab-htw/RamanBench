"""Which results a protocol change makes stale (raman_bench.stale_results)."""

import json

from raman_bench import stale_results as sr

OLD = {
    "time_limit": 600,
    "time_limit_overrides": {"mlrod": 10800},
    "model_time_limit_overrides": {"MITRA": {"mlrod": 10800}, "LR": 800000},
    "max_train_samples_overrides": {"mlrod": 10000},
    "model_max_train_samples_overrides": {"CAUSILO": 5000, "LIMIX": {"mlrod": 2000}},
}
NEW = {"time_limit": 10800, "max_train_samples": 30000, "tasks": []}


def task(dataset, n):
    return {"task": f"{dataset}__0", "dataset": dataset, "num_instances": n}


def test_caps_and_limits_the_old_scope_resolves():
    assert sr.old_row_cap(OLD, "RF", "mlrod") == 10000
    assert sr.old_row_cap(OLD, "CAUSILO", "mlrod") == 5000  # the model cap is smaller
    assert sr.old_row_cap(OLD, "LIMIX", "mlrod") == 2000 and sr.old_row_cap(OLD, "LIMIX", "other") is None
    assert sr.old_time_limit(OLD, "RF", "other") == 600 and sr.old_time_limit(OLD, "RF", "mlrod") == 10800
    assert sr.old_time_limit(OLD, "LR", "other") == 800000


def test_a_result_is_stale_when_the_row_cut_changed():
    big = task("mlrod", 130061)
    assert sr.stale_reason("RF", big, 5.0, OLD, NEW) == "rows"  # 10000 rows before, 30000 now
    assert sr.stale_reason("CAUSILO", task("small", 4000), 5.0, OLD, NEW) == ""  # 4000 rows either way
    assert sr.stale_reason("CAUSILO", task("mid", 8000), 5.0, OLD, NEW) == "rows"  # model cap 5000 before, all 8000 now
    assert sr.stale_reason("RF", task("mid", 8000), 5.0, OLD, NEW) == ""  # 8000 rows either way


def test_a_result_is_stale_when_it_was_bound_by_the_old_time_limit():
    small = task("small", 100)
    assert sr.stale_reason("MITRA", small, 590.0, OLD, NEW) == "time"
    assert sr.stale_reason("MITRA", small, 100.0, OLD, NEW) == ""  # stopped well before the limit
    assert sr.stale_reason("LR", small, 700000.0, OLD, NEW) == ""  # the old limit was higher than the new one
    assert sr.stale_reason(None, small, 599.0, OLD, NEW) == ""  # no model key: a baseline with its own budget


def test_classify_and_quarantine_move_the_fold_directories(tmp_path):
    NEW["tasks"] = [task("mlrod", 130061)]
    results = [
        {"framework": "RF_c1_BAG_L1", "task_metadata": {"name": "mlrod__0", "repeat": 0, "fold": 1},
         "method_metadata": {"model_type": "RF"}, "time_train_s": 3.0},
        {"framework": "RF_c1_BAG_L1", "task_metadata": {"name": "gone__0", "repeat": 0, "fold": 0},
         "method_metadata": {"model_type": "RF"}, "time_train_s": 3.0},
    ]
    table = sr.classify(results, OLD, NEW)
    assert table["reason"].tolist() == ["rows"]  # the task outside the protocol is ignored
    fold = tmp_path / "data/RF_c1_BAG_L1/mlrod__0/0_1"
    fold.mkdir(parents=True)
    (fold / "results.pkl").write_text("x")
    assert sr.quarantine(tmp_path / "data", table, tmp_path / "stale") == 1
    assert not fold.exists() and (tmp_path / "stale/RF_c1_BAG_L1/mlrod__0/0_1/results.pkl").exists()
    json.dumps(table.to_dict())
