"""Tests for shape-changing preprocessing steps' interaction with the mixin's
DataFrame reconstruction.

``crop`` (and later ``gcu``/``lvse``) change ``n_features``, so the mixin
cannot simply reuse the pre-preprocessing column names when rebuilding a
DataFrame from the transformed array (see ``_output_feature_cols`` in
``mixin.py``, and the docstring caveat in ``crop_spectra``).
"""

from raman_bench.preprocessing.mixin import _output_feature_cols


def test_output_feature_cols_unchanged_width():
    cols = ["a", "b", "c"]
    assert _output_feature_cols(cols, 3) == cols


def test_output_feature_cols_changed_width_falls_back_to_positional():
    cols = ["a", "b", "c", "d", "e"]
    out = _output_feature_cols(cols, 3)
    assert out == ["feature_0", "feature_1", "feature_2"]
    assert len(out) == 3
