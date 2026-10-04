"""
dataset.py — FractureDataset for bone fracture detection.
Supports train/val/test splits from the dataset.csv metadata.
"""

import csv
import os
import random

# Allow PIL to load truncated/corrupted images without crashing
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── Label mapping ────────────────────────────────────────────────────────────
CLASSES = ["Non-Fractured", "Fractured"]  # 0, 1


def load_metadata(csv_path: str, img_root: str) -> list[dict]:
    """Return list of {path, label} dicts, skipping images that don't exist."""
    records = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = int(row["fractured"])
            subdir = "Fractured" if label == 1 else "Non_fractured"
            img_path = os.path.join(img_root, subdir, row["image_id"])
            if os.path.exists(img_path):
                records.append({"path": img_path, "label": label})
    return records


def split_data(records: list[dict], val_ratio=0.15, test_ratio=0.15, seed=42):
    """Stratified train / val / test split."""
    random.seed(seed)
    frac   = [r for r in records if r["label"] == 1]
    nonfrac = [r for r in records if r["label"] == 0]

    def _split(items):
        items = items.copy()
        random.shuffle(items)
        n = len(items)
        n_test = int(n * test_ratio)
        n_val  = int(n * val_ratio)
        return items[n_test + n_val:], items[n_test:n_test + n_val], items[:n_test]

    tr_f, va_f, te_f = _split(frac)
    tr_n, va_n, te_n = _split(nonfrac)

    train = tr_f + tr_n
    val   = va_f + va_n
    test  = te_f + te_n

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


# ── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(split: str, img_size: int = 224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if split == "train":
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            normalize,
        ])


# ── Dataset ──────────────────────────────────────────────────────────────────
class FractureDataset(Dataset):
    def __init__(self, records: list[dict], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, rec["label"]


# ── DataLoaders factory ───────────────────────────────────────────────────────
def make_loaders(
    csv_path: str,
    img_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 2,
):
    records = load_metadata(csv_path, img_root)
    train_recs, val_recs, test_recs = split_data(records)

    print(f"Dataset — Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)}")

    train_ds = FractureDataset(train_recs, get_transforms("train", img_size))
    val_ds   = FractureDataset(val_recs,   get_transforms("val",   img_size))
    test_ds  = FractureDataset(test_recs,  get_transforms("test",  img_size))

    # MPS (Apple Silicon) does not support multi-process DataLoader or pin_memory
    is_cuda = torch.cuda.is_available()
    safe_workers  = num_workers if is_cuda else 0
    safe_pin      = bool(is_cuda)
    common_kwargs = dict(num_workers=safe_workers, pin_memory=safe_pin)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **common_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **common_kwargs)

    # Class weights for imbalanced loss
    n_neg = sum(1 for r in train_recs if r["label"] == 0)
    n_pos = sum(1 for r in train_recs if r["label"] == 1)
    total = n_neg + n_pos
    class_weights = torch.tensor([total / (2 * n_neg), total / (2 * n_pos)], dtype=torch.float)
    print(f"Class weights → Non-Fractured: {class_weights[0]:.3f} | Fractured: {class_weights[1]:.3f}")

    return train_loader, val_loader, test_loader, class_weights
