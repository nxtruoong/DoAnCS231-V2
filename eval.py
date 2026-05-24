"""Evaluation + figures for a trained ResNet-18+CBAM checkpoint.

Produces, into --out-dir:
- metrics.json          full sklearn classification_report (precision,
                        recall, f1-score, support per class + accuracy,
                        macro avg, weighted avg)
- confusion_matrix.png
- per_driver_accuracy.csv + per_driver_accuracy.png
- training_curves.png   (loss + accuracy) — needs --history-json
- attention_grid.png    one sample per class with SAM heatmap overlay
- failures.png          top misclassified val images with overlay

Run on Kaggle:
    python eval.py --ckpt /kaggle/working/run1/best.pt \\
        --data-root /kaggle/input/competitions/state-farm-distracted-driver-detection \\
        --splits-dir /kaggle/working/splits \\
        --out-dir /kaggle/working/run1/eval \\
        --history-json /kaggle/working/run1/history.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from augment import CLASSES, StateFarmDataset, build_eval_transform, load_stats
from model import build_model
from model_resnet50 import build_resnet50_cbam


CLASS_NAMES = {
    "c0": "safe driving",
    "c1": "text right",
    "c2": "phone right",
    "c3": "text left",
    "c4": "phone left",
    "c5": "radio",
    "c6": "drinking",
    "c7": "reach behind",
    "c8": "hair/makeup",
    "c9": "talk passenger",
}


def load_model(ckpt_path: Path, use_ema: bool = True, use_cbam: bool = True) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    use_cbam = not saved_args.get("no_cbam", not use_cbam)
    backbone = saved_args.get("backbone", "resnet18")
    pretrained = bool(saved_args.get("pretrained", False))
    if backbone == "resnet50":
        model = build_resnet50_cbam(num_classes=10, use_cbam=use_cbam,
                                    pretrained=pretrained)
    else:
        model = build_model(num_classes=10, use_cbam=use_cbam)
    state = ckpt["ema" if use_ema else "model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_preds, all_labels, all_probs = [], [], []
    model.to(device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return (np.concatenate(all_labels), np.concatenate(all_preds),
            np.concatenate(all_probs))


def write_classification_report(y_true, y_pred, out_path: Path) -> dict:
    target_names = [f"{c} ({CLASS_NAMES[c]})" for c in CLASSES]
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, output_dict=True, zero_division=0,
    )
    text = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, zero_division=0,
    )
    out_path.write_text(text)
    (out_path.parent / "metrics.json").write_text(json.dumps(report_dict, indent=2))
    print(text)
    return report_dict


def plot_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=True,
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (counts; color = row-normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_driver_breakdown(y_true, y_pred, val_df: pd.DataFrame,
                         out_csv: Path, out_png: Path) -> pd.DataFrame:
    df = val_df.copy()
    df["pred"] = y_pred
    df["true"] = y_true
    df["correct"] = (df["pred"] == df["true"]).astype(int)
    table = df.groupby("subject")["correct"].agg(["count", "mean"]).rename(
        columns={"count": "n_images", "mean": "accuracy"},
    )
    table.to_csv(out_csv)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table.index, table["accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Held-out subject")
    ax.set_title("Per-driver accuracy on held-out val set")
    for i, (acc, n) in enumerate(zip(table["accuracy"], table["n_images"])):
        ax.text(i, acc + 0.02, f"{acc:.2f}\n(n={n})", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return table


def plot_training_curves(history_path: Path, out_path: Path) -> None:
    history = json.loads(history_path.read_text())
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].plot(epochs, [h["ema_val_loss"] for h in history], label="val (EMA)", linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].plot(epochs, [h["ema_val_acc"] for h in history], label="val (EMA)", linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].set_title("Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def overlay_sam(rgb_uint8: np.ndarray, sam_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """rgb_uint8: HxWx3 uint8 in [0,255]. sam_map: HxW in [0,1]."""
    heat = (sam_map - sam_map.min()) / (sam_map.max() - sam_map.min() + 1e-8)
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_uint8, 1 - alpha, heat_color, alpha, 0.0)


def _denorm_to_uint8(tensor: torch.Tensor, mean, std) -> np.ndarray:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = tensor.cpu() * std_t + mean_t
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def attention_grid(model, loader, mean, std, device, out_path: Path,
                   n_per_class: int = 1) -> None:
    if not getattr(model, "use_cbam", True):
        print("Model has no CBAM — skipping attention grid.")
        return
    picked: dict[int, list[tuple[np.ndarray, np.ndarray, int]]] = {i: [] for i in range(10)}
    model.to(device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        sam = model.last_spatial_attention()  # (B,1,h,w)
        sam_up = F.interpolate(sam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        for i in range(images.size(0)):
            cls = int(labels[i])
            if len(picked[cls]) >= n_per_class:
                continue
            rgb = _denorm_to_uint8(images[i], mean, std)
            picked[cls].append((rgb, sam_up[i, 0].cpu().numpy(), int(preds[i])))
        if all(len(v) >= n_per_class for v in picked.values()):
            break

    cols = 10
    rows = n_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    if rows == 1:
        axes = np.array([axes])
    for c in range(10):
        for r in range(n_per_class):
            ax = axes[r, c]
            if r < len(picked[c]):
                rgb, sam_map, pred = picked[c][r]
                ax.imshow(overlay_sam(rgb, sam_map))
                color = "green" if pred == c else "red"
                ax.set_title(f"{CLASSES[c]} -> {CLASSES[pred]}", fontsize=9, color=color)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def failure_cases(model, loader, mean, std, device, out_path: Path,
                  n: int = 12) -> None:
    if not getattr(model, "use_cbam", True):
        print("Model has no CBAM — failure overlays will skip heatmap.")
    model.to(device)
    failures: list[tuple[np.ndarray, np.ndarray | None, int, int, float]] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        sam = model.last_spatial_attention()
        sam_up = (F.interpolate(sam, size=images.shape[-2:], mode="bilinear", align_corners=False)
                  if sam is not None else None)
        wrong = (preds.cpu() != labels).nonzero(as_tuple=True)[0]
        for idx in wrong:
            true_c = int(labels[idx])
            pred_c = int(preds[idx])
            conf = float(probs[idx, pred_c])
            rgb = _denorm_to_uint8(images[idx], mean, std)
            heat = sam_up[idx, 0].cpu().numpy() if sam_up is not None else None
            failures.append((rgb, heat, true_c, pred_c, conf))
        if len(failures) >= n * 4:  # gather extras to rank by confidence
            break

    failures.sort(key=lambda t: -t[4])  # highest-confidence wrong predictions first
    failures = failures[:n]
    if not failures:
        print("No failures found — model is perfect on val (suspect bug).")
        return

    cols = 4
    rows = (len(failures) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for i, (rgb, heat, t, p, conf) in enumerate(failures):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        img = overlay_sam(rgb, heat) if heat is not None else rgb
        ax.imshow(img)
        ax.set_title(f"true {CLASSES[t]} | pred {CLASSES[p]} ({conf:.2f})", fontsize=8, color="red")
        ax.axis("off")
    for i in range(len(failures), rows * cols):
        r, c = divmod(i, cols); axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--history-json", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false")
    ap.add_argument("--img-size", type=int, default=224,
                    help="Eval input resolution. Should match the value used at training.")
    ap.add_argument("--imagenet-stats", action="store_true",
                    help="Force ImageNet mean/std normalization (auto-enabled "
                         "if the ckpt was trained with --imagenet-stats).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_args = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("args", {})
    use_imagenet_stats = bool(ckpt_args.get("imagenet_stats", False)) or args.imagenet_stats
    if use_imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
        mean, std = load_stats(args.splits_dir / "stats.json")
    eval_tx = build_eval_transform(mean, std, size=args.img_size)
    val_df = pd.read_csv(args.splits_dir / "val.csv")
    val_ds = StateFarmDataset(
        args.splits_dir / "val.csv", args.data_root / "imgs" / "train", eval_tx,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = load_model(args.ckpt, use_ema=args.use_ema)
    y_true, y_pred, _ = collect_predictions(model, val_loader, device)

    write_classification_report(y_true, y_pred, args.out_dir / "classification_report.txt")
    plot_confusion_matrix(y_true, y_pred, args.out_dir / "confusion_matrix.png")
    per_driver_breakdown(y_true, y_pred, val_df,
                         args.out_dir / "per_driver_accuracy.csv",
                         args.out_dir / "per_driver_accuracy.png")

    if args.history_json and args.history_json.exists():
        plot_training_curves(args.history_json, args.out_dir / "training_curves.png")

    attention_grid(model, val_loader, mean, std, device,
                   args.out_dir / "attention_grid.png", n_per_class=1)
    failure_cases(model, val_loader, mean, std, device,
                  args.out_dir / "failures.png", n=12)

    print(f"\nAll artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
