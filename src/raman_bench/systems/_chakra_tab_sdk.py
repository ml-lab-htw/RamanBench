"""Chakra-Tab through YHat Labs' Python library (``pip install yhatlabs``).

The library sends a table past the API's direct size (medium and full: 25,000 rows x 100 columns) as an
async job and waits for it, which the plain ``requests`` client in ``_chakra_tab_vendored`` cannot. Spectra
have hundreds of columns, so every fold needs this path. Selected with ``CHAKRA_TAB_SDK=1``; the key is
still read from ``CHAKRA_TAB_KEY``. The endpoint is the library's default (``YHAT_BASE_URL`` overrides it).
"""

from __future__ import annotations

import hashlib
import os
import time

import pandas as pd

from raman_bench.systems._chakra_tab_vendored import _KEY_ENV
from raman_bench.systems._chakra_tab_vendored import ChakraTabSystemModel as _VendoredModel

_REQUEST_TIMEOUT_S = 600.0  # per HTTP request; a job is polled for up to the library's own wait (two hours)


class ChakraTabSdkSystemModel(_VendoredModel):
    """The vendored system model with its API call replaced by ``yhatlabs.Client.predict``."""

    def _call(self, X_test: pd.DataFrame) -> dict:
        import yhatlabs

        key = hashlib.sha256(pd.util.hash_pandas_object(X_test, index=False).to_numpy().tobytes()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        client = yhatlabs.Client(api_key=os.environ[_KEY_ENV], timeout=_REQUEST_TIMEOUT_S)
        start = time.monotonic()
        res = client.predict(
            self._train,
            self._target,
            X_test.reset_index(drop=True),
            task=self._problem_type,
            preset=self.preset,
            eval_metric=self._metric,
            time_limit=self._time_limit,
            seed=self._seed,
            # Tables over 8 MB as parquet would go through POST /v1/uploads; /v1/table/predict then answers 422
            # ("train must be a list of row objects or parquet_b64"). Empty refs send the table in the body.
            _refs={},
        )
        self._api_calls.append(
            {"rows": len(X_test), "attempts": 1, "wall_s": time.monotonic() - start, "failure": None},
        )
        out = {"predictions": res.predictions.to_numpy(), "fit": res.fit}
        if self._problem_type != "regression":
            out["probabilities"] = res.probabilities.to_numpy()
            out["classes"] = list(res.classes)
        self.fit_info = res.fit
        self._cache = {key: out}
        return out
