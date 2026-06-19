"""
LUNA-SITE | Layer 8: Physics-Regularized CNN Training Loop
===========================================================
Trains the LunarIceCNN with physics regularization.
Produces saved weights for finale inference-only deployment.

Key design decisions for the 30-hour finale:
  - Saves best checkpoint by validation F1 (not just loss)
  - Exports inference-ready TorchScript model
  - Generates training curves for dashboard display
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import logging

sys.path.insert(0, os.path.dirname(__file__))
from cnn_ice_detector import LunarIceCNN, PhysicsRegularizedLoss, mc_dropout_predict
from lunar_dataset import LunarDFSARDataset, generate_and_load_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, n_classes: int = 3):
    """Compute per-class precision, recall, F1."""
    metrics = {}
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec  = tp / (tp + fp + 1e-8)
        rec   = tp / (tp + fn + 1e-8)
        f1    = 2 * prec * rec / (prec + rec + 1e-8)
        metrics[f"class_{c}"] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
    macro_f1 = np.mean([v["f1"] for v in metrics.values()])
    metrics["macro_f1"] = float(macro_f1)
    return metrics


def train(
    n_epochs:    int   = 30,
    batch_size:  int   = 64,
    lr:          float = 1e-3,
    alpha_phys:  float = 0.10,
    save_dir:    str   = "checkpoints",
    data_dir:    str   = "data/synthetic",
    device_str:  str   = "auto",
):
    os.makedirs(save_dir, exist_ok=True)

    device = (
        torch.device("cuda") if device_str == "auto" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Training on: {device}")

    # Dataset
    dataset = generate_and_load_dataset(data_dir, n_samples=2000)
    n_val   = int(0.15 * len(dataset))
    n_test  = int(0.10 * len(dataset))
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # Model
    model     = LunarIceCNN(dropout_p=0.3).to(device)
    criterion = PhysicsRegularizedLoss(alpha=alpha_phys)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history  = {"train_loss": [], "val_loss": [], "val_f1": [], "lr": []}
    best_f1  = 0.0
    best_ckpt = os.path.join(save_dir, "best_model.pt")

    for epoch in range(1, n_epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            stokes  = batch["stokes"].to(device)
            labels  = batch["label"].to(device)
            cpr     = batch["cpr"].to(device)
            dop     = batch["dop"].to(device)

            optimizer.zero_grad()
            logits = model(stokes)
            loss, info = criterion(logits, labels, cpr, dop)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += info["total_loss"]

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss_total, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                stokes  = batch["stokes"].to(device)
                labels  = batch["label"].to(device)
                cpr     = batch["cpr"].to(device)
                dop     = batch["dop"].to(device)

                logits  = model(stokes)
                loss, _ = criterion(logits, labels, cpr, dop)
                val_loss_total += loss.item()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_val_loss = val_loss_total / len(val_loader)
        metrics      = compute_metrics(np.array(all_preds), np.array(all_labels))
        val_f1       = metrics["macro_f1"]

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      val_f1,
                "val_loss":    avg_val_loss,
                "metrics":     metrics,
            }, best_ckpt)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"LR: {current_lr:.2e}"
            )

    # ── Final test ──
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            stokes = batch["stokes"].to(device)
            logits = model(stokes)
            preds  = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds.tolist())
            test_labels.extend(batch["label"].numpy().tolist())

    test_metrics = compute_metrics(np.array(test_preds), np.array(test_labels))
    logger.info(f"\n=== FINAL TEST RESULTS ===")
    logger.info(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    for c in range(3):
        m = test_metrics[f"class_{c}"]
        names = {0: "Regolith", 1: "Ice", 2: "Rock"}
        logger.info(f"  {names[c]}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    # Export TorchScript for finale inference
    ts_path = os.path.join(save_dir, "luna_ice_model.ts")
    scripted = torch.jit.trace(model, torch.randn(1, 4, 64, 64))
    scripted.save(ts_path)
    logger.info(f"TorchScript model saved → {ts_path}")

    # Save history
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump({"history": history, "test_metrics": test_metrics,
                   "best_val_f1": best_f1}, f, indent=2)

    logger.info(f"Best Val F1: {best_f1:.4f} | Weights → {best_ckpt}")
    return model, history, test_metrics


if __name__ == "__main__":
    train(n_epochs=30, batch_size=64, lr=1e-3)
