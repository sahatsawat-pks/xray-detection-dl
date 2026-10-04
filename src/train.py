"""
train.py — MLflow-tracked training pipeline.

Trains one or all models with full experiment tracking, lineage tagging,
and artifact logging. Replaces the original trainer.py with MLOps-grade
tracking via MLflow.

Usage:
    python -m src.train                    # Train all models
    python -m src.train --model ModelFinal # Train one model
"""

import argparse
import contextlib
import copy
import json
import os
import sys
from datetime import datetime

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.config import (
    BATCH_SIZE,
    DATA_CSV,
    IMAGE_ROOT,
    IMG_SIZE,
    LEARNING_RATES,
    MODEL_DIR,
    NUM_EPOCHS,
    PATIENCE,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
    get_device,
    set_seed,
)
from src.dataset import make_loaders
from src.evaluate import (
    plot_comparison_table,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_roc_curve,
)
from src.models import ModelBaseline, ModelFinal, ModelImproved, count_params

# dl_utils — kept at root for backward compatibility
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from dl_utils import compute_performance, test, train_one_epoch

# ── MLflow integration (optional — degrades gracefully) ───────────────────────
try:
    import mlflow

    from src.tracking import (
        log_epoch_metrics,
        log_lineage_tags,
        log_model_artifact,
        log_test_metrics,
        log_training_params,
        setup_mlflow,
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[warn] MLflow not installed — training will run without experiment tracking")


# ── Model registry ───────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "ModelBaseline": {
        "class": ModelBaseline,
        "kwargs": {},
        "use_weights": False,
    },
    "ModelImproved": {
        "class": ModelImproved,
        "kwargs": {"freeze_layers": 6},
        "use_weights": True,
    },
    "ModelFinal": {
        "class": ModelFinal,
        "kwargs": {"dropout": 0.4},
        "use_weights": True,
    },
}


def train_model(model_name: str) -> dict:
    """
    Train one model with full MLflow tracking.

    Returns a dict with test metrics and the MLflow run ID.
    """
    config = MODEL_CONFIGS[model_name]
    device = get_device()

    print("=" * 62)
    print(f"  Training: {model_name}")
    print("=" * 62)

    # ── Seed ──────────────────────────────────────────────────────────────────
    set_seed(SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dl, valid_dl, test_dl, class_weights = make_loaders(
        DATA_CSV, IMAGE_ROOT, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = config["class"](**config["kwargs"]).to(device)
    params = count_params(model)
    print(f"  Total params    : {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")
    print(f"  Frozen params   : {params['frozen']:,}\n")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────────────
    lr = LEARNING_RATES[model_name]
    loss_fn = (
        nn.CrossEntropyLoss(weight=class_weights.to(device))
        if config["use_weights"]
        else nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=lr * 0.01,
    )

    # ── TensorBoard ───────────────────────────────────────────────────────────
    run_tag = f"train_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_tag}")

    # ── MLflow run ────────────────────────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        try:
            setup_mlflow()
            mlflow.start_run(run_name=f"{model_name}-seed{SEED}")
            log_lineage_tags(model_name, SEED)
            log_training_params(model_name, {
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": lr,
                "weight_decay": WEIGHT_DECAY,
                "patience": PATIENCE,
                "img_size": IMG_SIZE,
                "seed": SEED,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "use_class_weights": config["use_weights"],
            })
        except Exception as e:
            print(f"[warn] MLflow setup failed: {e}")
            MLFLOW_AVAILABLE_LOCAL = False
        else:
            MLFLOW_AVAILABLE_LOCAL = True
    else:
        MLFLOW_AVAILABLE_LOCAL = False

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_acc", "val_acc",
        "val_f1", "val_precision", "val_recall", "val_auc",
    ]}

    best_val_f1 = 0.0
    no_improve = 0
    best_weights = None
    epochs_run = 0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1} / {NUM_EPOCHS}")
        epochs_run = epoch + 1

        # Train one epoch
        train_one_epoch(train_dl, model, loss_fn, optimizer,
                        epoch, device, writer, log_step_interval=50)

        # Evaluate
        train_loss, tr_preds, tr_trues, tr_probs = test(train_dl, model, loss_fn, device)
        val_loss, va_preds, va_trues, va_probs = test(valid_dl, model, loss_fn, device)

        train_perf = compute_performance(tr_preds, tr_trues, tr_probs)
        val_perf = compute_performance(va_preds, va_trues, va_probs)

        # TensorBoard scalars
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_perf["accuracy"], epoch)
        writer.add_scalar("Accuracy/valid", val_perf["accuracy"], epoch)
        writer.add_scalar("F1-Score/train", train_perf["f1"], epoch)
        writer.add_scalar("F1-Score/valid", val_perf["f1"], epoch)
        writer.add_scalar("Recall/valid", val_perf["recall"], epoch)
        writer.add_scalar("Precision/valid", val_perf["precision"], epoch)
        writer.add_scalar("AUC/valid", val_perf["auc"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # MLflow epoch metrics
        if MLFLOW_AVAILABLE_LOCAL:
            with contextlib.suppress(Exception):
                log_epoch_metrics(epoch, train_loss, val_loss, train_perf, val_perf)

        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_perf["accuracy"])
        history["val_acc"].append(val_perf["accuracy"])
        history["val_f1"].append(val_perf["f1"])
        history["val_precision"].append(val_perf["precision"])
        history["val_recall"].append(val_perf["recall"])
        history["val_auc"].append(val_perf["auc"])

        print(
            f"  TrainLoss={train_loss:.4f}  TrainAcc={train_perf['accuracy']:.4f} | "
            f"ValLoss={val_loss:.4f}  ValAcc={val_perf['accuracy']:.4f}  "
            f"F1={val_perf['f1']:.4f}  AUC={val_perf['auc']:.4f}"
        )

        # Early stopping
        if val_perf["f1"] > best_val_f1:
            best_val_f1 = val_perf["f1"]
            best_weights = copy.deepcopy(model.state_dict())
            os.makedirs(MODEL_DIR, exist_ok=True)
            ckpt_path = os.path.join(MODEL_DIR, f"{model_name}_best_vloss.pth")
            torch.save(best_weights, ckpt_path)
            print(f"  ✅ Best checkpoint saved (Val F1={best_val_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  ⏹  Early stopping at epoch {epoch + 1}")
                break

        scheduler.step()

    writer.close()
    print(f"[{model_name}] Training completed after {epochs_run} epochs.")

    # ── Restore best weights ──────────────────────────────────────────────────
    model.load_state_dict(best_weights)

    # ── Test set evaluation ───────────────────────────────────────────────────
    test_loss, te_preds, te_trues, te_probs = test(test_dl, model, loss_fn, device)
    test_perf = compute_performance(te_preds, te_trues, te_probs)

    print(f"\n{'=' * 55}")
    print(f"  {model_name} — Test Results")
    print(f"{'=' * 55}")
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"  {metric.capitalize():12s}: {test_perf[metric]:.4f}")
    print(f"{'=' * 55}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_learning_curves(history, model_name, RESULTS_DIR)
    plot_confusion_matrix(
        te_trues.cpu().numpy(), te_preds.cpu().numpy(), model_name, RESULTS_DIR,
    )

    # ── Save result JSON ──────────────────────────────────────────────────────
    row = {
        "Model": model_name,
        "Accuracy": round(test_perf["accuracy"], 4),
        "Precision": round(test_perf["precision"], 4),
        "Recall": round(test_perf["recall"], 4),
        "F1": round(test_perf["f1"], 4),
        "AUC": round(test_perf["auc"], 4),
        "labels": te_trues.cpu().numpy().tolist(),
        "probs": te_probs.cpu().numpy().tolist(),
    }
    with open(os.path.join(RESULTS_DIR, f"{model_name}_results.json"), "w") as f:
        json.dump(row, f, indent=2)

    # ── MLflow: log test metrics + artifacts ──────────────────────────────────
    run_id = None
    if MLFLOW_AVAILABLE_LOCAL:
        try:
            log_test_metrics(test_perf)
            ckpt_path = os.path.join(MODEL_DIR, f"{model_name}_best_vloss.pth")
            log_model_artifact(ckpt_path, RESULTS_DIR)
            run_id = mlflow.active_run().info.run_id
            mlflow.end_run()
        except Exception as e:
            print(f"[warn] MLflow logging failed: {e}")
            with contextlib.suppress(Exception):
                mlflow.end_run()

    row["run_id"] = run_id or "local"
    return row


def main():
    parser = argparse.ArgumentParser(description="Train fracture detection models")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Train a specific model (default: train all)",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using {device} device\n")

    # Determine which models to train
    models_to_train = [args.model] if args.model else list(MODEL_CONFIGS.keys())

    table_rows = []
    roc_inputs = []

    for model_name in models_to_train:
        row = train_model(model_name)
        table_rows.append(row)
        roc_inputs.append({
            "name": model_name,
            "labels": row["labels"],
            "probs": row["probs"],
        })

    # ── Final comparison ──────────────────────────────────────────────────────
    if len(table_rows) > 1:
        print("\n" + "=" * 62)
        print("  📊 FINAL MODEL COMPARISON")
        print("=" * 62)
        print(f"\n{'Model':<22} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
        print("-" * 62)
        for r in table_rows:
            print(f"{r['Model']:<22} {r['Accuracy']:>8.4f} {r['Precision']:>8.4f} "
                  f"{r['Recall']:>8.4f} {r['F1']:>8.4f} {r['AUC']:>8.4f}")
        print()

        plot_roc_curve(roc_inputs, RESULTS_DIR)
        plot_comparison_table(table_rows, RESULTS_DIR)

        best = max(table_rows, key=lambda r: r["F1"])
        print(f"\n🏆 Best model by F1-score: {best['Model']}  (F1={best['F1']:.4f})")

    print("\nDone! Checkpoints → models/  |  Charts → results/  |  Tracking → MLflow UI")


if __name__ == "__main__":
    main()
