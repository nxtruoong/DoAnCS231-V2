# RUN10_HOWTO — ResNet-50 + ImageNet + CBAM (image-only)

Single-stream Kaggle T4×2 recipe. No MediaPipe, no pose, no second
stream. Goal: isolate the contribution of a deeper, ImageNet-pretrained
backbone with CBAM, against Run 6 (ResNet-18 + CBAM from scratch) and
Run 8/9 (pose / dual-stream variants).

For design context see `log.md` (Run 10 entry). This file is the
runbook.

---

## 0. Prerequisites

- Subject-wise splits already produced (`splits/train.csv`,
  `splits/val.csv`, `splits/stats.json`). Re-use whatever Run 6/7/8
  produced; `stats.json` is read only when `--imagenet-stats` is off.
- Code dataset attached at `/kaggle/input/driver-distraction-cbam` or
  cloned to `/kaggle/working/code`. Must contain:
  - `train.py` (patched: `--backbone`, `--pretrained`, `--imagenet-stats`)
  - `eval.py`  (patched: same auto-detect)
  - `model.py`
  - `model_resnet50.py`
  - `augment.py`

GitHub mirror:
```python
!rm -rf /kaggle/working/code
!git clone https://github.com/nxtruoong/DoAnCS231 /kaggle/working/code
CODE_DIR = "/kaggle/working/code"
```

---

## 1. Paths (Cell 1)

```python
import os, sys
COMP_DIR = "/kaggle/input/competitions/state-farm-distracted-driver-detection"
CODE_DIR = "/kaggle/input/driver-distraction-cbam"   # or /kaggle/working/code
WORK     = "/kaggle/working"
RUN      = f"{WORK}/run10"

assert os.path.exists(COMP_DIR + "/driver_imgs_list.csv"), "Competition dataset not attached"
assert os.path.exists(CODE_DIR + "/model_resnet50.py"),    "Run 10 code missing"
assert os.path.exists(WORK + "/splits/stats.json"),        "Run data_prep.py first"

sys.path.insert(0, CODE_DIR)
print("OK. GPU count:", __import__("torch").cuda.device_count())
```

---

## 2. Smoke test (Cell 2, ~5 min)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train.py",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    f"{WORK}/run10_smoke",
    "--backbone", "resnet50",
    "--pretrained",
    "--imagenet-stats",
    "--epochs", "2",
    "--batch-size", "64",
    "--num-workers", "4",
    "--lr", "0.01",
    "--warmup-epochs", "1",
    "--img-size", "224",
    "--data-parallel",
], check=True)
```

Expect: 2 epochs complete, no OOM. Val acc already 0.40–0.70 by
epoch 2 because of ImageNet init (much higher than from-scratch
Run 6 at the same point). If OOM → drop `--batch-size` to 48 or 32.

---

## 3. Full Run 10 training (Cell 3, ~2-3 hr)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train.py",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    RUN,
    "--backbone", "resnet50",
    "--pretrained",
    "--imagenet-stats",
    "--epochs", "30",
    "--batch-size", "64",
    "--num-workers", "4",
    "--lr", "0.01",
    "--warmup-epochs", "2",
    "--weight-decay", "1e-4",
    "--ema-decay", "0.999",
    "--cutmix-alpha", "0.5",
    "--cutmix-p", "0.20",
    "--label-smoothing", "0.1",
    "--img-size", "224",
    "--early-stop-patience", "8",
    "--early-stop-min-delta", "0.005",
    "--ckpt-every", "5",
    "--data-parallel",
], check=True)
```

**Why these hyperparameters differ from Run 6:**
- `--lr 0.01` (vs Run 6 `0.1`): pretrained weights need gentler LR or
  they de-rail in epoch 1.
- `--weight-decay 1e-4` (vs `5e-4`): standard for fine-tuning.
- `--epochs 30` (vs `50`): pretrained converges faster.
- `--imagenet-stats`: pretrained backbone expects ImageNet
  normalization, not StateFarm dataset stats.

**Checkpoints land in** `/kaggle/working/run10/`: `best.pt`,
`ckpt_e05.pt`, ..., `final.pt`.

**Watch milestones** (Run 10 should beat Run 6 at every checkpoint):

| ep | target ema val acc | Run 6 actual |
|---:|---:|---:|
| 05 | ≥ 0.70 | ~0.50 |
| 10 | ≥ 0.82 | ~0.78 |
| 20 | ≥ 0.87 | 0.82  |
| 30 | ≥ 0.88 | —     |

If val acc stalls below 0.80 by epoch 10 → suspect `--lr` too high.
Halve to `0.005` and re-run.

---

## 4. Evaluation (Cell 4)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/eval.py",
    "--ckpt",        f"{RUN}/best.pt",
    "--data-root",   COMP_DIR,
    "--splits-dir",  f"{WORK}/splits",
    "--out-dir",     f"{RUN}/eval",
    "--history-json", f"{RUN}/history.json",
    "--batch-size", "128",
    "--num-workers", "4",
    "--img-size", "224",
], check=True)
```

`--imagenet-stats` is auto-detected from the checkpoint's saved args,
so no need to pass it again. Same goes for `--backbone resnet50`.

Artifacts written to `/kaggle/working/run10/eval/`:
- `metrics.json`, `classification_report.txt`
- `confusion_matrix.png`
- `per_driver_accuracy.{csv,png}`
- `training_curves.png` (if `history.json` present)
- `attention_grid.png` (SAM overlays from `cbam4`)
- `failures.png`

---

## 5. Ablation knobs

| Want to test | Flag combo |
|---|---|
| ResNet-50 from scratch (no ImageNet) | drop `--pretrained` and `--imagenet-stats`; raise `--lr` to `0.1` |
| ResNet-50 + ImageNet, no CBAM       | add `--no-cbam` |
| Same recipe on ResNet-18            | `--backbone resnet18` (default), drop `--pretrained` |

---

## 6. Troubleshooting

- **OOM at batch 64.** ResNet-50 is ~4× params of ResNet-18. Drop
  `--batch-size` to 48 or 32. If still OOM, drop `--img-size` to 192.
- **First-epoch val acc < 0.20.** `--lr` too high for pretrained
  weights; halve.
- **Eval crashes loading ckpt.** Backbone mismatch — eval.py
  auto-detects from `saved_args.backbone`. If the ckpt predates the
  patch, pass `--backbone resnet50` manually (would require a small
  eval.py edit) or re-train.
