"""Training loop for ResNet-18 + CBAM on State Farm (subject-wise split).

Defaults match the project plan:
- SGD momentum=0.9, weight_decay=5e-4
- Cosine LR 0.1 -> 0 over 40 epochs
- CrossEntropy with label smoothing 0.1
- CutMix p=0.5, alpha=1.0
- EMA decay 0.999
- batch 128 (single T4 + DataParallel on a second T4 if available)
- Checkpoint every 5 epochs to --out-dir

Tier-1 fallback trigger: at epoch 20, if val acc < 0.50, the script
aborts so the operator can restart with --no-cutmix --no-grayscale. The
augmentation toggles flip the relevant transforms / batch ops.

Run on Kaggle:
    python train.py \\
        --data-root /kaggle/input/competitions/state-farm-distracted-driver-detection \\
        --splits-dir /kaggle/working/splits \\
        --out-dir /kaggle/working/run1
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from augment import (
    StateFarmDataset, build_eval_transform, build_train_transform, build_trivialaugment_transform,
    cutmix_batch, load_stats, mixup_loss,
)
from model import build_model
from model_resnet50 import build_resnet50_cbam


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    ema: EMA,
    *,
    use_cutmix: bool,
    cutmix_alpha: float,
    cutmix_p: float,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_cutmix:
            images, lbl_a, lbl_b, lam = cutmix_batch(
                images, labels, alpha=cutmix_alpha, p=cutmix_p,
            )
        else:
            lbl_a, lbl_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
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
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
    return total_loss / total, total_correct / total


def save_checkpoint(path: Path, model: nn.Module, ema: EMA, optimizer, scheduler,
                    epoch: int, best_val_acc: float, args_dict: dict) -> None:
    torch.save({
        "epoch": epoch,
        "model": _unwrap(model).state_dict(),
        "ema": ema.shadow.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "args": args_dict,
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--cutmix-alpha", type=float, default=0.5)
    ap.add_argument("--cutmix-p", type=float, default=0.3)
    ap.add_argument("--no-cutmix", action="store_true")
    ap.add_argument("--no-grayscale", action="store_true",
                    help="Tier-1 fallback: drop RandomGrayscale from train aug.")
    ap.add_argument("--no-cbam", action="store_true",
                    help="Ablation baseline: backbone without CBAM blocks.")
    ap.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18",
                    help="resnet18 = Run 6 from-scratch backbone. "
                         "resnet50 = Run 10 ImageNet-pretrained backbone.")
    ap.add_argument("--pretrained", action="store_true",
                    help="Use ImageNet pretrained weights (resnet50 only).")
    ap.add_argument("--imagenet-stats", action="store_true",
                    help="Normalize with ImageNet mean/std instead of "
                         "dataset stats. Recommended with --pretrained.")
    ap.add_argument("--minimal-aug", action="store_true",
                    help="Smoke test: strip ColorJitter/Grayscale/Blur/Erasing; "
                    "keep only RandomResizedCrop + Normalize.")
    ap.add_argument("--trivialaugment", action="store_true",
                    help="Use TrivialAugmentWide instead of hand-tuned ColorJitter+Blur+Grayscale stack.")
    ap.add_argument("--img-size", type=int, default=224,
                    help="Train/eval input resolution. RandomResizedCrop and CenterCrop both use this.")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--tier1-epoch", type=int, default=20)
    ap.add_argument("--tier1-threshold", type=float, default=0.50)
    ap.add_argument("--early-stop-patience", type=int, default=8,
                    help="Stop if max(val_acc, ema_val_acc) does not improve by "
                         ">= --early-stop-min-delta for N consecutive epochs. 0 disables.")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.005,
                    help="Minimum gain over best-so-far counted as improvement.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-parallel", action="store_true",
                    help="Wrap model with DataParallel (use on Kaggle T4x2).")
    args = ap.parse_args()

    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
        mean, std = load_stats(args.splits_dir / "stats.json")
    from torchvision import transforms
    if args.minimal_aug:
        train_tx = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif args.trivialaugment:
        train_tx = build_trivialaugment_transform(mean, std, size=args.img_size)
    else:
        train_tx = build_train_transform(mean, std, size=args.img_size)
        if args.no_grayscale:
            train_tx.transforms = [t for t in train_tx.transforms
                                   if not isinstance(t, transforms.RandomGrayscale)]
    eval_tx = build_eval_transform(mean, std, size=args.img_size)

    img_root = args.data_root / "imgs" / "train"
    train_ds = StateFarmDataset(args.splits_dir / "train.csv", img_root, train_tx)
    val_ds = StateFarmDataset(args.splits_dir / "val.csv", img_root, eval_tx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone == "resnet50":
        model = build_resnet50_cbam(num_classes=10, use_cbam=not args.no_cbam,
                                    pretrained=args.pretrained).to(device)
    else:
        if args.pretrained:
            raise SystemExit("--pretrained only supported with --backbone resnet50")
        model = build_model(num_classes=10, use_cbam=not args.no_cbam).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, steps_per_epoch * args.warmup_epochs)
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
    use_cutmix = not args.no_cutmix
    args_dict = vars(args).copy()
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in args_dict.items()}

    log_path = args.out_dir / "history.json"
    print(f"Training {args.epochs} epochs | backbone={args.backbone} | "
          f"pretrained={args.pretrained} | imagenet_stats={args.imagenet_stats} | "
          f"use_cbam={not args.no_cbam} | "
          f"use_cutmix={use_cutmix} | no_grayscale={args.no_grayscale} | "
          f"trivialaugment={args.trivialaugment} | img_size={args.img_size} | "
          f"ema_decay={args.ema_decay} | "
          f"device={device} | gpus={torch.cuda.device_count()}")

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, ema,
            use_cutmix=use_cutmix, cutmix_alpha=args.cutmix_alpha, cutmix_p=args.cutmix_p,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        ema_val_loss, ema_val_acc = evaluate(ema.shadow.to(device), val_loader, criterion, device)

        elapsed = (time.time() - t_start) / 60.0
        print(f"[ep {epoch:02d}/{args.epochs}] "
              f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"ema val acc={ema_val_acc:.4f} | "
              f"elapsed={elapsed:.1f} min")

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

        if epoch == args.tier1_epoch and ema_acc < args.tier1_threshold:
            print(f"\nTier-1 trigger hit: val_acc {ema_acc:.4f} < {args.tier1_threshold} "
                  f"at epoch {epoch}. Aborting. Restart with "
                  f"--no-cutmix --no-grayscale.")
            return

        if args.early_stop_patience > 0:
            if ema_acc >= es_best + args.early_stop_min_delta:
                es_best = ema_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"\nEarly stop: no improvement >= {args.early_stop_min_delta} "
                      f"for {epochs_no_improve} epochs (best={es_best:.4f}). "
                      f"Stopping at epoch {epoch}/{args.epochs}.")
                break

    save_checkpoint(args.out_dir / "final.pt", model, ema, optimizer, scheduler,
                    epoch, best_val_acc, args_dict)
    print(f"\nDone. Best val acc: {best_val_acc:.4f}. "
          f"Total time: {(time.time() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
