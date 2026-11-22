"""The version is written in three places; a release sets all three (RELEASING.md, step 1)."""

import re
import tomllib
from pathlib import Path

import raman_bench

ROOT = Path(__file__).resolve().parents[1]


def test_versions_agree():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    citation = re.search(r'^version: "([^"]+)"', (ROOT / "CITATION.cff").read_text(), re.M).group(1)
    assert raman_bench.__version__ == pyproject == citation
