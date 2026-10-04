"""
evaluate.py — Evaluation utilities: metrics, confusion matrix, learning curves,
              Grad-CAM visualisation, and per-model comparison.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

CLASSES = ["Non-Fractured", "Fractured"]
COLORS  = {"Non-Fractured": "#4C9BE8", "Fractured": "#E85C5C"}


# ── Core evaluation ───────────────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict:
    model.eval()
    # Ensure criterion weights (if any) are on the correct device
    criterion = criterion.to(device)
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0
    n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
        total_loss += loss.item() * imgs.size(0)
        n          += imgs.size(0)

    y, p, pr = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    return {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(y, p),
        "precision": precision_score(y, p, zero_division=0),
        "recall":    recall_score(y, p, zero_division=0),
        "f1":        f1_score(y, p, zero_division=0),
        "auc":       roc_auc_score(y, pr),
        "labels":    y,
        "preds":     p,
        "probs":     pr,
    }


# ── Learning curves ────────────────────────────────────────────────────────────
def plot_learning_curves(history: dict, model_name: str, save_dir: str = "../results"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0F0F1A")

    epochs = range(1, len(history["train_loss"]) + 1)

    panel_cfg = [
        ("Loss",     ["train_loss", "val_loss"],    ["#4C9BE8", "#E85C5C"]),
        ("Accuracy", ["train_acc",  "val_acc"],     ["#4C9BE8", "#E85C5C"]),
        ("Val F1 / AUC", ["val_f1", "val_auc"],    ["#A78BFA", "#34D399"]),
    ]

    for ax, (title, keys, cols) in zip(axes, panel_cfg, strict=True):
        ax.set_facecolor("#1A1A2E")
        for key, col in zip(keys, cols, strict=True):
            label = key.replace("_", " ").title()
            ax.plot(epochs, history[key], color=col, linewidth=2.5, label=label)
        ax.set_title(title, color="white", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", color="#AAAAAA")
        ax.tick_params(colors="#AAAAAA")
        ax.spines[:].set_color("#333355")
        ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=11)
        ax.grid(alpha=0.15, color="#FFFFFF")

    plt.suptitle(f"Learning Curves — {model_name}", color="white", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
    plt.close()
    print(f"[plot] Saved: {path}")
    return path


# ── Confusion matrix ───────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str = "../results"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#1A1A2E")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = range(len(CLASSES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASSES, color="white", fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASSES, color="white", fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "black", fontsize=16)

    ax.set_ylabel("True Label", color="white", fontsize=13)
    ax.set_xlabel("Predicted Label", color="white", fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name}", color="white", fontsize=14)
    ax.tick_params(colors="white")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
    plt.close()
    print(f"[plot] Saved: {path}")
    return path


# ── ROC curve ─────────────────────────────────────────────────────────────────
def plot_roc_curve(results_list: list[dict], save_dir: str = "../results"):
    """
    results_list: [{"name": str, "labels": array, "probs": array}, ...]
    """
    os.makedirs(save_dir, exist_ok=True)
    palette = ["#4C9BE8", "#E85C5C", "#A78BFA", "#34D399"]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#1A1A2E")

    for i, r in enumerate(results_list):
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        auc = roc_auc_score(r["labels"], r["probs"])
        ax.plot(fpr, tpr, color=palette[i % len(palette)],
                linewidth=2.5, label=f"{r['name']} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title("ROC Curves — All Models", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=11)
    ax.grid(alpha=0.15, color="#FFFFFF")

    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
    plt.close()
    print(f"[plot] Saved: {path}")
    return path


# ── Model comparison table image ───────────────────────────────────────────────
def plot_comparison_table(rows: list[dict], save_dir: str = "../results"):
    """
    rows: [{"Model": str, "Accuracy": float, "Precision": float,
             "Recall": float, "F1": float, "AUC": float}, ...]
    """
    os.makedirs(save_dir, exist_ok=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    x = np.arange(len(metrics))
    width = 0.25
    palette = ["#4C9BE8", "#E85C5C", "#A78BFA"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#1A1A2E")

    for i, row in enumerate(rows):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=row["Model"],
                      color=palette[i % len(palette)], alpha=0.9, edgecolor="#FFFFFF33")
        for bar, v in zip(bars, vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", color="white", fontsize=9)

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x + width * (len(rows) - 1) / 2)
    ax.set_xticklabels(metrics, color="white", fontsize=13)
    ax.set_ylabel("Score", color="white")
    ax.set_title("Model Performance Comparison", color="white", fontsize=15, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333355")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=12)
    ax.grid(axis="y", alpha=0.15, color="#FFFFFF")

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
    plt.close()
    print(f"[plot] Saved: {path}")
    return path


# ── Grad-CAM (basic) ──────────────────────────────────────────────────────────
class GradCAM:
    """Simple Grad-CAM for any model with a named conv layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model  = model
        self.grads  = None
        self.feats  = None
        target_layer.register_forward_hook(self._save_feats)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_feats(self, _, __, output):
        self.feats = output.detach()

    def _save_grads(self, _, __, grad_output):
        self.grads = grad_output[0].detach()

    def generate(self, img_tensor: torch.Tensor, class_idx: int = 1):
        """Returns CAM heatmap as numpy array [H, W]."""
        self.model.eval()
        logits = self.model(img_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.feats).sum(dim=1, keepdim=True) # (1, 1, H, W)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualise_gradcam(model, img_tensor, target_layer, img_orig,
                      pred_label: str, conf: float, save_path: str):
    """Overlay Grad-CAM heatmap on original image and save."""
    gcam = GradCAM(model, target_layer)
    cam  = gcam.generate(img_tensor.unsqueeze(0), class_idx=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("#0F0F1A")

    axes[0].imshow(img_orig, cmap="gray")
    axes[0].set_title("Original X-Ray", color="white", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(img_orig, cmap="gray")
    axes[1].imshow(cam, cmap="jet", alpha=0.5,
                   extent=[0, img_orig.width, img_orig.height, 0])
    col = COLORS.get(pred_label, "white")
    axes[1].set_title(f"Grad-CAM | {pred_label} ({conf:.1%})", color=col, fontsize=13)
    axes[1].axis("off")

    for ax in axes:
        ax.set_facecolor("#1A1A2E")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
    plt.close()
