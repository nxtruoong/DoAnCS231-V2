"""Training loop for two-stream ResNet-18+CBAM (Run 7).

Mirrors train.py with three differences:
- Dataset yields (full, face, label) per item.
- Model forward takes two tensors.
- CutMix applies to both streams with shared permutation + box.

Run on Kaggle:
    python train_twostream.py \\
        --data-root /kaggle/input/competitions/state-farm-distracted-driver-detection \\
        --splits-dir /kaggle/working/splits \\
        --out-dir /kaggle/working/run7
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from augment import load_stats, mixup_loss
from augment_twostream import (
    POSE_DIM, PoseFusionDataset, TwoStreamDataset,
    build_face_eval_transform, build_face_train_transform,
    build_full_eval_transform, build_full_train_transform,
    cutmix_twostream, load_pose_lookup,
)
from model_twostream import build_posefusion, build_twostream


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        m_params = dict(model.named_parameters())
        s_params = dict(self.shadow.named_parameters())
        for k, sp in s_params.items():
            mp = m_params[k].detach()
            if sp.dtype.is_floating_point:
                sp.mul_(self.decay).add_(mp, alpha=1.0 - self.decay)
            else:
                sp.copy_(mp)
        m_buf = dict(model.named_buffers())
        for k, sb in self.shadow.named_buffers():
            sb.copy_(m_buf[k])


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, ema,
                    *, use_cutmix: bool, cutmix_alpha: float, cutmix_p: float):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for full, face, labels in pbar:
        full = full.to(device, non_blocking=True)
        face = face.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_cutmix:
            full, face, lbl_a, lbl_b, lam = cutmix_twostream(
                full, face, labels, alpha=cutmix_alpha, p=cutmix_p,
            )
        else:
            lbl_a, lbl_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)
        logits = model(full, face)
        loss = mixup_loss(criterion, logits, lbl_a, lbl_b, lam)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(_unwrap(model))

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.4f}")

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for full, face, labels in tqdm(loader, desc="val", leave=False):
        full = full.to(device, non_blocking=True)
        face = face.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(full, face)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
    return total_loss / total, total_correct / total


def train_one_epoch_pose(model, loader, optimizer, scheduler, criterion, device, ema):
    """Run 8 pose-fusion train loop. No CutMix on this path (pose vec
    is per-image; mixing yaw / wrist positions is meaningless)."""
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for full, pose, labels in pbar:
        full = full.to(device, non_blocking=True)
        pose = pose.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(full, pose)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(_unwrap(model))

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.4f}")

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate_pose(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for full, pose, labels in tqdm(loader, desc="val", leave=False):
        full = full.to(device, non_blocking=True)
        pose = pose.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(full, pose)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
    return total_loss / total, total_correct / total


def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, best_val_acc, args_dict):
    pose_fusion = bool(args_dict.get("pose_fusion", False))
    torch.save({
        "epoch": epoch,
        "model": _unwrap(model).state_dict(),
        "ema": ema.shadow.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "args": args_dict,
        "two_stream": not pose_fusion,
        "pose_fusion": pose_fusion,
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--cutmix-alpha", type=float, default=0.5)
    ap.add_argument("--cutmix-p", type=float, default=0.20)
    ap.add_argument("--no-cutmix", action="store_true")
    ap.add_argument("--no-cbam", action="store_true")
    ap.add_argument("--full-size", type=int, default=384)
    ap.add_argument("--face-size", type=int, default=224)
    ap.add_argument("--top-frac", type=float, default=0.50)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-parallel", action="store_true")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Path to checkpoint (.pt) to resume from")
    # Run 8 pose-fusion flags
    ap.add_argument("--pose-fusion", action="store_true",
                    help="Use PoseFusionCBAM: single CNN+CBAM + 36-d MediaPipe pose")
    ap.add_argument("--pose-parquet", type=Path, default=None,
                    help="Path to splits/pose.parquet (required if --pose-fusion)")
    # Run 9 backbone selector
    ap.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18",
                    help="CNN backbone for the full stream (default resnet18). "
                         "resnet50 is the Run 9 ImageNet-pretrained path.")
    ap.add_argument("--pretrained", action="store_true",
                    help="Load ImageNet pretrained weights for the backbone "
                         "(currently only honoured for --backbone resnet50).")
    ap.add_argument("--imagenet-stats", action="store_true",
                    help="Normalize with ImageNet mean/std instead of "
                         "splits/stats.json. Recommended with --pretrained.")
    args = ap.parse_args()

    if args.pose_fusion and args.pose_parquet is None:
        ap.error("--pose-fusion requires --pose-parquet PATH")

    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
        mean, std = load_stats(args.splits_dir / "stats.json")
    tx_full_train = build_full_train_transform(mean, std, size=args.full_size)
    tx_full_eval  = build_full_eval_transform(mean, std, size=args.full_size)

    img_root = args.data_root / "imgs" / "train"

    if args.pose_fusion:
        pose_lookup = load_pose_lookup(args.pose_parquet)
        train_ds = PoseFusionDataset(args.splits_dir / "train.csv", img_root,
                                     tx_full_train, pose_lookup)
        val_ds   = PoseFusionDataset(args.splits_dir / "val.csv", img_root,
                                     tx_full_eval, pose_lookup)
    else:
        tx_face_train = build_face_train_transform(mean, std, size=args.face_size,
                                                   top_frac=args.top_frac)
        tx_face_eval  = build_face_eval_transform(mean, std, size=args.face_size,
                                                  top_frac=args.top_frac)
        train_ds = TwoStreamDataset(args.splits_dir / "train.csv", img_root,
                                    tx_full_train, tx_face_train)
        val_ds   = TwoStreamDataset(args.splits_dir / "val.csv", img_root,
                                    tx_full_eval, tx_face_eval)

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.pose_fusion:
        model = build_posefusion(num_classes=10, use_cbam=not args.no_cbam,
                                 pose_dim=POSE_DIM, backbone=args.backbone,
                                 pretrained=args.pretrained).to(device)
    else:
        if args.backbone != "resnet18":
            raise SystemExit("--backbone resnet50 currently only supported with --pose-fusion")
        model = build_twostream(num_classes=10, use_cbam=not args.no_cbam).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)
    steps_per_epoch = len(train_loader)
    total_steps   = steps_per_epoch * args.epochs
    warmup_steps  = max(1, steps_per_epoch * args.warmup_epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema = EMA(_unwrap(model), decay=args.ema_decay)

    history: list[dict] = []
    best_val_acc = 0.0
    es_best = 0.0
    epochs_no_improve = 0
    start_epoch = 1
    use_cutmix = not args.no_cutmix
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    log_path = args.out_dir / "history.json"

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        _unwrap(model).load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_acc = float(ckpt.get("best_val_acc", 0.0))
        es_best = best_val_acc
        if log_path.exists():
            history = json.loads(log_path.read_text())
            history = [h for h in history if h["epoch"] < start_epoch]
        print(f"Resumed from {args.resume} at epoch {start_epoch} "
              f"(best_val_acc={best_val_acc:.4f})")
    if args.pose_fusion:
        print(f"Pose-fusion training {args.epochs} epochs | backbone={args.backbone}"
              f"{' (ImageNet)' if args.pretrained else ''} | full={args.full_size} "
              f"pose_dim={POSE_DIM} | batch={args.batch_size} | cutmix=disabled | "
              f"device={device} | gpus={torch.cuda.device_count()}")
    else:
        print(f"Two-stream training {args.epochs} epochs | full={args.full_size} "
              f"face={args.face_size} top_frac={args.top_frac} | batch={args.batch_size} | "
              f"cutmix=({args.cutmix_p},{args.cutmix_alpha}) | device={device} | "
              f"gpus={torch.cuda.device_count()}")

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if args.pose_fusion:
            train_loss, train_acc = train_one_epoch_pose(
                model, train_loader, optimizer, scheduler, criterion, device, ema,
            )
            val_loss, val_acc = evaluate_pose(model, val_loader, criterion, device)
            ema_val_loss, ema_val_acc = evaluate_pose(
                ema.shadow.to(device), val_loader, criterion, device,
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, device, ema,
                use_cutmix=use_cutmix, cutmix_alpha=args.cutmix_alpha, cutmix_p=args.cutmix_p,
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            ema_val_loss, ema_val_acc = evaluate(
                ema.shadow.to(device), val_loader, criterion, device,
            )

        elapsed = (time.time() - t_start) / 60.0
        print(f"[ep {epoch:02d}/{args.epochs}] "
              f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"ema val acc={ema_val_acc:.4f} | elapsed={elapsed:.1f} min")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "ema_val_loss": ema_val_loss, "ema_val_acc": ema_val_acc,
            "lr": scheduler.get_last_lr()[0],
        })
        log_path.write_text(json.dumps(history, indent=2))

        ema_acc = max(val_acc, ema_val_acc)
        if ema_acc > best_val_acc:
            best_val_acc = ema_acc
            save_checkpoint(args.out_dir / "best.pt", model, ema, optimizer, scheduler,
                            epoch, best_val_acc, args_dict)
        if epoch % args.ckpt_every == 0:
            save_checkpoint(args.out_dir / f"ckpt_e{epoch:02d}.pt",
                            model, ema, optimizer, scheduler, epoch, best_val_acc, args_dict)

        if args.early_stop_patience > 0:
            if ema_acc >= es_best + args.early_stop_min_delta:
                es_best = ema_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"\nEarly stop at ep {epoch}/{args.epochs}, best={es_best:.4f}")
                break

    save_checkpoint(args.out_dir / "final.pt", model, ema, optimizer, scheduler,
                    epoch, best_val_acc, args_dict)
    print(f"\nDone. Best val acc: {best_val_acc:.4f}. "
          f"Total: {(time.time() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
