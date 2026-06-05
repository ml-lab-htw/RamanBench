"""Verify that all datasets from the RamanBench paper can be loaded."""

import tempfile
import shutil
import os

from raman_bench.benchmark import RamanBenchmark


def verify_all_datasets():
    """Load all available datasets and report their status."""
    temp_cache = tempfile.mkdtemp(prefix="raman_verify_")

    print("🔬 RamanBench Full Dataset Verification")
    print("=" * 80)

    try:
        # Initialize benchmark with ALL datasets
        print("\n📦 Initializing benchmark with all datasets...")
        bm = RamanBenchmark(
            dataset_names_classification=None,  # Load all
            dataset_names_regression=None,      # Load all
            cache_dir=temp_cache,
            use_mirror=True,  # Use HuggingFace mirror (faster, more reliable in CI)
        )

        print(f"✓ Found {len(bm.dataset_names_classification)} classification datasets")
        print(f"✓ Found {len(bm.dataset_names_regression)} regression datasets")

        total_datasets = len(bm.dataset_names_classification) + len(bm.dataset_names_regression)
        print(f"\n📊 Total: {total_datasets} datasets\n")

        # Initialize (load indices)
        print("⏳ Initializing datasets (this may take a moment)...")
        bm.init_datasets()

        print(f"✓ Benchmark initialized with {len(bm)} dataset-target keys\n")

        # Report classification datasets
        print("\n📋 CLASSIFICATION DATASETS")
        print("─" * 80)
        clf_datasets = sorted(bm.dataset_names_classification)
        for i, name in enumerate(clf_datasets, 1):
            n_targets = bm._index.get(name, 0)
            print(f"{i:2d}. {name:45s} (targets: {n_targets})")

        # Report regression datasets
        print("\n📋 REGRESSION DATASETS")
        print("─" * 80)
        reg_datasets = sorted(bm.dataset_names_regression)
        for i, name in enumerate(reg_datasets, 1):
            n_targets = bm._index.get(name, 0)
            print(f"{i:2d}. {name:45s} (targets: {n_targets})")

        # Summary statistics
        print("\n" + "=" * 80)
        print("📊 SUMMARY STATISTICS")
        print("─" * 80)

        total_targets = sum(bm._index.values())
        print(f"Total datasets:           {len(bm._index)}")
        print(f"Total dataset-target keys: {len(bm)}")
        print(f"Classification datasets:  {len(bm.dataset_names_classification)}")
        print(f"Regression datasets:      {len(bm.dataset_names_regression)}")

        # Paper mentions 74 datasets / 163 targets, let's check
        print("\n📄 Paper Reference:")
        print(f"Paper claims: 74 datasets, 163 targets")
        print(f"Loaded:       {len(bm._index)} datasets, {total_targets} targets")

        if len(bm._index) == 74 or len(bm._index) >= 74:
            print("✓ Dataset count matches or exceeds paper (good!)")
        else:
            print(f"⚠ Only {len(bm._index)} datasets loaded (paper has 74)")

        # Test loading a few datasets
        print("\n" + "=" * 80)
        print("🧪 SAMPLE DATA VALIDATION")
        print("─" * 80)

        test_samples = [
            bm.dataset_names_classification[0] if bm.dataset_names_classification else None,
            bm.dataset_names_regression[0] if bm.dataset_names_regression else None,
        ]

        for dataset_name in test_samples:
            if dataset_name is None:
                continue

            try:
                # Get the first key for this dataset
                keys = [k for k in bm._key_list if k.startswith(dataset_name + "_")]
                if keys:
                    idx = bm._key_list.index(keys[0])
                    train_df, test_df, key, task_type = bm[idx]

                    if train_df is not None:
                        print(f"\n✓ {dataset_name}")
                        print(f"  Type: {task_type}")
                        print(f"  Train: {train_df.shape} | Test: {test_df.shape}")
                        print(f"  Features: {train_df.shape[1] - 1}")  # -1 for target column
            except Exception as e:
                print(f"✗ {dataset_name}: {e}")

        print("\n" + "=" * 80)
        print("✅ VERIFICATION COMPLETE!")
        print("=" * 80)

    finally:
        print(f"\n🧹 Cleaning up cache: {temp_cache}")
        if os.path.exists(temp_cache):
            shutil.rmtree(temp_cache, ignore_errors=True)
            print("✓ Cache cleaned")


if __name__ == "__main__":
    verify_all_datasets()
