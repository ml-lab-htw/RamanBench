"""The config folders are one per benchmark version, and what the code and the v0.1 config point at exists."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


def test_only_version_folders_and_the_readme_are_in_configs():
    names = {p.name for p in CONFIGS.iterdir()}
    assert names <= {"README.md", "v0.1", "v1", "v2"}, names


def test_v0_1_config_points_at_files_that_exist():
    config = json.loads((CONFIGS / "v0.1" / "benchmark.json").read_text())
    for key in ("datasets_regression", "datasets_classification", "models"):
        assert (CONFIGS / "v0.1" / config[key]).is_file(), key


def test_v1_and_v2_have_the_shared_files():
    for version in ("v1", "v2"):
        for name in ("scope_default.json", "target_list.json", "exclusions.json"):
            assert (CONFIGS / version / name).is_file(), (version, name)
    assert not (CONFIGS / "v1" / "quality_exclusions.json").exists()
