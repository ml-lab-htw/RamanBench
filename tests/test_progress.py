import logging
import time

from raman_bench.progress import log_progress


def test_log_progress_every_n_and_last(caplog):
    log = logging.getLogger("progress_test")
    with caplog.at_level(logging.INFO, logger="progress_test"):
        for done in range(1, 8):
            log_progress(log, "Step", done, 7, time.time() - 10, every=3)
    assert [r.getMessage().split(",")[0] for r in caplog.records] == [
        "Step: 3/7 (43%)", "Step: 6/7 (86%)", "Step: 7/7 (100%)",
    ]
    assert "left" in caplog.records[0].getMessage()
