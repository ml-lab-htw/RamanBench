"""Integration test: download datasets and validate their structure.

This test downloads a few datasets, validates their format and integrity,
then cleans up the cache.

Can be run as: python tests/test_data_download_integration.py
"""

import json
import os
import shutil
import tempfile

import numpy as np

from raman_bench.benchmark import RamanBenchmark, configure_benchmark


def test_benchmark_init_and_dataset_listing(temp_cache):
    """Test that we can initialize a benchmark and list available datasets."""
    bm = RamanBenchmark(
        dataset_names_classification=None,
        dataset_names_regression=None,
        cache_dir=temp_cache,
    )

    assert len(bm.dataset_names_classification) > 0
    assert len(bm.dataset_names_regression) > 0
    print(f"✓ Found {len(bm.dataset_names_classification)} classification datasets")
    print(f"✓ Found {len(bm.dataset_names_regression)} regression datasets")


def test_load_single_classification_dataset(temp_cache):
    """Download and validate a single classification dataset."""
    bm = RamanBenchmark(
        dataset_names_classification=["wheat_lines"],
        dataset_names_regression=[],
        cache_dir=temp_cache,
        use_mirror=False,  # use original sources, not mirror
    )

    # Initialize datasets (this will download and cache)
    bm.init_datasets()

    # Check that the dataset was loaded and indexed
    assert "wheat_lines" in bm._index
    assert bm._index["wheat_lines"] >= 1

    # Load a train/test split
    train_df, test_df, key, task_type = bm[0]

    # Validate structure
    assert train_df is not None
    assert test_df is not None
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert "target" in train_df.columns
    assert "target" in test_df.columns

    # Validate shapes
    assert train_df.shape[1] > 1  # at least features + target
    assert test_df.shape[1] == train_df.shape[1]

    # Validate no NaNs in target
    assert not train_df["target"].isna().any()
    assert not test_df["target"].isna().any()

    print(f"✓ Loaded {key}: train shape={train_df.shape}, test shape={test_df.shape}")


def test_load_single_regression_dataset(temp_cache):
    """Download and validate a single regression dataset."""
    bm = RamanBenchmark(
        dataset_names_classification=[],
        dataset_names_regression=["amino_acids_glycine"],
        cache_dir=temp_cache,
        use_mirror=False,  # use original sources, not mirror
    )

    # Initialize datasets
    bm.init_datasets()

    # Check that the dataset was loaded
    assert "amino_acids_glycine" in bm._index
    assert bm._index["amino_acids_glycine"] >= 1

    # Load a train/test split
    train_df, test_df, key, task_type = bm[0]

    # Validate structure
    assert train_df is not None
    assert test_df is not None
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert "target" in train_df.columns

    # Validate that target values are numeric (regression)
    train_targets = train_df["target"].values
    assert np.issubdtype(train_targets.dtype, np.number)

    print(f"✓ Loaded {key}: train shape={train_df.shape}, test shape={test_df.shape}")


def test_configure_benchmark_with_defaults(temp_cache):
    """Test configure_benchmark function with default parameters."""
    config = {
        "dataset_names_classification": ["wheat_lines"],
        "dataset_names_regression": ["amino_acids_glycine"],
        "test_size": 0.2,
        "random_state": 42,
        "cache_dir": temp_cache,
    }

    bm = configure_benchmark(config, init_benchmark=True)

    # Verify benchmark is initialized
    assert bm.is_initialized
    assert len(bm) >= 2  # at least 2 datasets (1 clf + 1 reg)

    print(f"✓ Configured benchmark with {len(bm)} keys")


def test_configure_benchmark_with_mirror_flag(temp_cache):
    """Test that mirror flag is read from config."""
    config = {
        "dataset_names_classification": ["wheat_lines"],
        "dataset_names_regression": [],
        "test_size": 0.2,
        "random_state": 42,
        "cache_dir": temp_cache,
        "use_mirror": False,  # explicitly use original sources
        "mirror_repo": "HTW-KI-Werkstatt/RamanBench",
    }

    bm = configure_benchmark(config, init_benchmark=False)

    # Check that flags were set correctly
    assert bm.use_mirror is False
    assert bm.mirror_repo == "HTW-KI-Werkstatt/RamanBench"

    print(f"✓ Mirror flag correctly set: use_mirror={bm.use_mirror}")


def test_multiple_datasets_no_crash(temp_cache):
    """Test loading multiple datasets in sequence without crashes."""
    datasets_to_test = [
        ("wheat_lines", "classification"),
        ("amino_acids_glycine", "regression"),
    ]

    for dataset_name, task_type in datasets_to_test:
        if task_type == "classification":
            bm = RamanBenchmark(
                dataset_names_classification=[dataset_name],
                dataset_names_regression=[],
                cache_dir=temp_cache,
                use_mirror=False,
            )
        else:
            bm = RamanBenchmark(
                dataset_names_classification=[],
                dataset_names_regression=[dataset_name],
                cache_dir=temp_cache,
                use_mirror=False,
            )

        bm.init_datasets()

        # Try to get the first split
        train_df, test_df, key, _ = bm[0]
        assert train_df is not None
        assert test_df is not None
        print(f"✓ {task_type:15s} {dataset_name:30s}")


def test_cache_directory_structure(temp_cache):
    """Verify that cache directory structure is created correctly."""
    bm = RamanBenchmark(
        dataset_names_classification=["wheat_lines"],
        dataset_names_regression=[],
        cache_dir=temp_cache,
    )

    bm.init_datasets()

    # Check that cache directories exist
    assert os.path.isdir(bm.cache_dir_raw), "Raw cache dir not created"
    assert os.path.isdir(bm.cache_dir_processed), "Processed cache dir not created"

    # Check that index file was created
    assert os.path.isfile(bm._index_file), "Index file not created"

    # Verify index file is valid JSON
    with open(bm._index_file) as f:
        index = json.load(f)
    assert isinstance(index, dict)
    assert "wheat_lines" in index

    print(f"✓ Cache structure verified")
    print(f"  Raw cache: {bm.cache_dir_raw}")
    print(f"  Processed cache: {bm.cache_dir_processed}")
    print(f"  Index file: {bm._index_file}")


def run_all_tests():
    """Run all tests in sequence."""
    temp_dir = tempfile.mkdtemp(prefix="raman_test_")
    print(f"📦 Using temporary cache: {temp_dir}\n")

    tests = [
        ("Benchmark initialization and dataset listing", test_benchmark_init_and_dataset_listing),
        ("Load single classification dataset", test_load_single_classification_dataset),
        ("Load single regression dataset", test_load_single_regression_dataset),
        ("Configure benchmark with defaults", test_configure_benchmark_with_defaults),
        ("Configure benchmark with mirror flag", test_configure_benchmark_with_mirror_flag),
        ("Load multiple datasets", test_multiple_datasets_no_crash),
        ("Cache directory structure", test_cache_directory_structure),
    ]

    failed = []
    for i, (test_name, test_func) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {test_name}")
        print("─" * 70)
        try:
            test_func(temp_dir)
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed.append((test_name, e))
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback

            traceback.print_exc()
            failed.append((test_name, e))

    print("\n" + "=" * 70)
    if failed:
        print(f"❌ {len(failed)}/{len(tests)} tests FAILED\n")
        for test_name, error in failed:
            print(f"  ✗ {test_name}: {error}")
        return False
    else:
        print(f"✅ ALL {len(tests)} TESTS PASSED!")

    print("\n🧹 Cleaning up cache...")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"✓ Removed: {temp_dir}")

    return True


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
