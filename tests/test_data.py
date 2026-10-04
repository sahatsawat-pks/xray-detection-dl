"""
test_data.py — Data contract tests (Data tier).

Verifies dataset integrity, split properties, class distribution,
and transform correctness. No cloud SDK imports.

These tests require the actual dataset to be present at dataset/dataset.csv
and dataset/images/. Skip gracefully if data is not available.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check if dataset exists for conditional skipping
DATA_CSV = PROJECT_ROOT / "dataset" / "dataset.csv"
IMAGE_ROOT = PROJECT_ROOT / "dataset" / "images"
HAS_DATA = DATA_CSV.exists() and IMAGE_ROOT.exists()

skip_no_data = pytest.mark.skipif(
    not HAS_DATA, reason="Dataset not available (run `make data` first)"
)


@skip_no_data
class TestDataLoading:
    """Verify dataset loads correctly."""

    def test_metadata_loads(self):
        """CSV metadata should parse without errors."""
        from src.dataset import load_metadata
        records = load_metadata(str(DATA_CSV), str(IMAGE_ROOT))
        assert len(records) > 0, "No records loaded from dataset.csv"

    def test_expected_record_count(self):
        """Should load ~4,083 records (some may be missing images)."""
        from src.dataset import load_metadata
        records = load_metadata(str(DATA_CSV), str(IMAGE_ROOT))
        assert len(records) >= 3000, f"Too few records: {len(records)}"

    def test_record_structure(self):
        """Each record should have 'path' and 'label' keys."""
        from src.dataset import load_metadata
        records = load_metadata(str(DATA_CSV), str(IMAGE_ROOT))
        for rec in records[:10]:
            assert "path" in rec, "Record missing 'path'"
            assert "label" in rec, "Record missing 'label'"
            assert rec["label"] in (0, 1), f"Invalid label: {rec['label']}"

    def test_images_exist(self):
        """All referenced image paths should exist on disk."""
        from src.dataset import load_metadata
        records = load_metadata(str(DATA_CSV), str(IMAGE_ROOT))
        for rec in records[:50]:  # Spot-check first 50
            assert os.path.exists(rec["path"]), f"Missing image: {rec['path']}"


@skip_no_data
class TestDataSplits:
    """Verify train/val/test split properties."""

    def test_no_data_leakage(self):
        """No image should appear in more than one split."""
        from src.dataset import load_metadata, split_data
        records = load_metadata(str(DATA_CSV), str(IMAGE_ROOT))
        train, val, test = split_data(records)

        train_paths = {r["path"] for r in train}
        val_paths = {r["path"] for r in val}
        test_paths = {r["path"] for r in test}

        assert len(train_paths & val_paths) == 0, "Train/val overlap detected!"
        assert len(train_paths & test_paths) == 0, "Train/test overlap detected!"
        assert len(val_paths & test_paths) == 0, "Val/test overlap detected!"

    def test_split_ratios(self):
        """Splits should approximately match 70/15/15 ratio."""
        from src.dataset import load_metadata, split_data
        records = load_metadata(str(DATA_CSV), str(IMAGE_ROOT))
        train, val, test = split_data(records)

        total = len(train) + len(val) + len(test)
        train_ratio = len(train) / total
        val_ratio = len(val) / total
        test_ratio = len(test) / total

        assert 0.60 < train_ratio < 0.80, f"Train ratio out of range: {train_ratio:.2f}"
        assert 0.10 < val_ratio < 0.25, f"Val ratio out of range: {val_ratio:.2f}"
        assert 0.10 < test_ratio < 0.25, f"Test ratio out of range: {test_ratio:.2f}"

    def test_all_records_accounted(self):
        """Total split count should equal total records."""
        from src.dataset import load_metadata, split_data
        records = load_metadata(str(DATA_CSV), str(IMAGE_ROOT))
        train, val, test = split_data(records)
        assert len(train) + len(val) + len(test) == len(records)

    def test_deterministic_splits(self):
        """Same seed should produce identical splits."""
        from src.dataset import load_metadata, split_data
        records = load_metadata(str(DATA_CSV), str(IMAGE_ROOT))
        train1, val1, test1 = split_data(records, seed=42)
        train2, val2, test2 = split_data(records, seed=42)

        assert len(train1) == len(train2)
        paths1 = sorted(r["path"] for r in train1)
        paths2 = sorted(r["path"] for r in train2)
        assert paths1 == paths2, "Splits are not deterministic with same seed"


@skip_no_data
class TestClassDistribution:
    """Verify class balance properties."""

    def test_both_classes_present(self):
        """Both classes should be present in every split."""
        from src.dataset import load_metadata, split_data
        records = load_metadata(str(DATA_CSV), str(IMAGE_ROOT))
        train, val, test = split_data(records)

        for name, split in [("train", train), ("val", val), ("test", test)]:
            labels = {r["label"] for r in split}
            assert 0 in labels, f"Class 0 missing from {name} split"
            assert 1 in labels, f"Class 1 missing from {name} split"

    def test_class_imbalance_ratio(self):
        """Dataset should show the expected ~4.7:1 imbalance ratio."""
        from src.dataset import load_metadata
        records = load_metadata(str(DATA_CSV), str(IMAGE_ROOT))
        n_neg = sum(1 for r in records if r["label"] == 0)
        n_pos = sum(1 for r in records if r["label"] == 1)
        ratio = n_neg / n_pos
        assert 3.0 < ratio < 7.0, f"Imbalance ratio {ratio:.1f} outside expected range"


@skip_no_data
class TestTransforms:
    """Verify image transforms produce correct output."""

    def test_train_transform_shape(self):
        """Training transforms should produce (3, 224, 224) tensors."""
        import numpy as np
        from PIL import Image

        from src.dataset import get_transforms

        transform = get_transforms("train", img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224), f"Wrong shape: {tensor.shape}"

    def test_val_transform_shape(self):
        """Validation transforms should produce (3, 224, 224) tensors."""
        import numpy as np
        from PIL import Image

        from src.dataset import get_transforms

        transform = get_transforms("val", img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_deterministic(self):
        """Validation/test transforms should be deterministic."""
        import numpy as np
        import torch
        from PIL import Image

        from src.dataset import get_transforms

        transform = get_transforms("val", img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        t1 = transform(img)
        t2 = transform(img)
        assert torch.allclose(t1, t2), "Validation transforms should be deterministic"
