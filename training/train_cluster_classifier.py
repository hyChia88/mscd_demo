#!/usr/bin/env python3
"""Phase 6.1.3 — train a ResNet-18 size_band classifier on AP floorplan crops.

Reads `manifest.jsonl` produced by `10_build_cluster_crops.py`, trains a 6-way
band classifier (door_M / door_L / window_S / window_M / window_L / window_XL),
and writes the best checkpoint + confusion-matrix metrics to disk.

Why band-level (not full 15-way cluster):
  G9 VLM precision was 31.6% on the 15-way cluster vocabulary because the
  model has no scale signal at floorplan-thumbnail resolution. 62% of its
  errors were size-class swaps (S↔M↔L↔XL), the discriminator a CNN-on-fixed-
  pixel-scale-crop *should* see. Collapsing prediction granularity to bands
  matches perception capability to retrieval granularity.

Usage:
  python mscd_demo/training/train_cluster_classifier.py \\
      --manifest data_curation/datasets/synth_v0.5_ap/cluster_crops_ap/manifest.jsonl \\
      --out-dir mscd_demo/models/cluster_classifier_ap \\
      --epochs 30 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


# Class order — fixed for label-index stability across runs.
DEFAULT_CLASSES = [
    "door_L", "door_M",
    "window_S", "window_M", "window_L", "window_XL",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CropDataset(Dataset):
    """Load 192×192 RGB crops + size_band label from a manifest split."""

    def __init__(
        self,
        manifest_path: Path,
        crops_root: Path,
        split: str,
        classes: List[str],
        train_aug: bool,
    ):
        self.crops_root = crops_root
        self.classes = classes
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.records: List[Dict] = []
        with manifest_path.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("split") != split:
                    continue
                if rec.get("size_band") not in self.cls_to_idx:
                    continue
                self.records.append(rec)

        if train_aug:
            self.tx = transforms.Compose([
                transforms.RandomChoice([
                    transforms.Lambda(lambda im: im),                          # identity
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270)),
                ]),
                transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),     # ±~8px on 192
                transforms.ColorJitter(brightness=0.15, contrast=0.10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]      # ImageNet
                ),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        img = Image.open(self.crops_root / rec["crop_path"]).convert("RGB")
        x = self.tx(img)
        y = self.cls_to_idx[rec["size_band"]]
        return x, y, rec["guid"]


def _build_model(num_classes: int) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1
    m = resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _class_weights(records: List[Dict], classes: List[str]) -> torch.Tensor:
    """Inverse-frequency class weights for weighted CE."""
    counts = Counter(r["size_band"] for r in records)
    n = len(records)
    weights = []
    for c in classes:
        cnt = counts.get(c, 0) or 1
        weights.append(n / (len(classes) * cnt))
    return torch.tensor(weights, dtype=torch.float32)


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    train = optim is not None
    model.train(train)
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y, _guid in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optim.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = loss_fn(logits, y)
        if train:
            loss.backward()
            optim.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def _evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    classes: List[str],
    device: torch.device,
) -> Dict:
    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()
    ys: List[int] = []
    ps: List[int] = []
    confs: List[float] = []
    guids: List[str] = []
    with torch.no_grad():
        for x, y, guid in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            top_p, top_i = prob.max(dim=1)
            ys.extend(y.tolist())
            ps.extend(top_i.cpu().tolist())
            confs.extend(top_p.cpu().tolist())
            guids.extend(guid)
    if not ys:
        return {"n": 0}
    cm = confusion_matrix(ys, ps, labels=list(range(len(classes))))
    report = classification_report(
        ys, ps, labels=list(range(len(classes))),
        target_names=classes, output_dict=True, zero_division=0,
    )
    return {
        "n": len(ys),
        "accuracy": float(np.mean(np.array(ys) == np.array(ps))),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "per_class": {
            classes[i]: {
                "precision": report[classes[i]]["precision"],
                "recall": report[classes[i]]["recall"],
                "f1": report[classes[i]]["f1-score"],
                "support": report[classes[i]]["support"],
            }
            for i in range(len(classes))
        },
        "predictions": [
            {"guid": g, "y_true": classes[y], "y_pred": classes[p], "confidence": float(c)}
            for g, y, p, c in zip(guids, ys, ps, confs)
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Path to manifest.jsonl from 10_build_cluster_crops.py")
    ap.add_argument("--crops-root", type=Path, default=None,
                    help="Root directory where crop_path is resolved. Defaults to manifest's parent.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output dir for checkpoint + metrics.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="Fixed class order for label index stability.")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Disable ImageNet pretrained weights (cold-start). Off by default.")
    args = ap.parse_args()

    _set_seed(args.seed)
    crops_root = args.crops_root or args.manifest.parent
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CropDataset(args.manifest, crops_root, "train", args.classes, train_aug=True)
    val_ds   = CropDataset(args.manifest, crops_root, "val",   args.classes, train_aug=False)
    test_ds  = CropDataset(args.manifest, crops_root, "test",  args.classes, train_aug=False)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={args.classes}", flush=True)
    if len(train_ds) == 0:
        sys.exit("[fatal] no training records — check manifest")

    cls_w = _class_weights(train_ds.records, args.classes)
    print(f"[loss] class weights (inverse frequency): "
          f"{[f'{c}={w:.2f}' for c, w in zip(args.classes, cls_w.tolist())]}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(len(args.classes)).to(device)
    if args.no_pretrained:
        # Re-init with default torch init by loading a randomly-initialised resnet18.
        from torchvision.models import resnet18 as _r18
        model.load_state_dict(_r18(weights=None).state_dict(), strict=False)
        model.fc = nn.Linear(512, len(args.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=cls_w.to(device))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Checkpoint selection: prefer minimum val_loss over max val_acc.
    # Reason — val n is small (~17) so val_acc has high variance and ties early
    # with under-converged epochs; val_loss is smoother and tracks generalisation
    # better at this dataset size. Tie-break by later epoch to avoid locking in
    # an early under-converged checkpoint.
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_path = args.out_dir / "best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = _epoch(model, train_loader, loss_fn, optim, device)
        va_loss, va_acc = _epoch(model, val_loader,   loss_fn, None,  device) if len(val_ds) else (0.0, 0.0)
        sched.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        improved = len(val_ds) > 0 and va_loss <= best_val_loss
        if improved:
            best_val_loss = va_loss
            best_val_acc = va_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": args.classes,
                "epoch": epoch,
                "val_loss": va_loss,
                "val_acc": va_acc,
            }, best_path)
        print(f"[ep {epoch:>3}/{args.epochs}] tr_loss={tr_loss:.3f} tr_acc={tr_acc:.3f} "
              f"va_loss={va_loss:.3f} va_acc={va_acc:.3f}{'  *best*' if improved else ''}", flush=True)

    # Fall back to last-epoch checkpoint if validation was empty (rare).
    if not best_path.exists():
        torch.save({"model_state_dict": model.state_dict(), "classes": args.classes,
                    "epoch": args.epochs, "val_acc": 0.0}, best_path)
        print("[warn] no validation set — saved last-epoch as best", flush=True)

    # Evaluate best checkpoint on test
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = _evaluate_test(model, test_loader, args.classes, device)
    print(f"\n[test] n={test_metrics.get('n', 0)}  "
          f"accuracy={test_metrics.get('accuracy', 0.0):.3f}", flush=True)
    if test_metrics.get("per_class"):
        print("[test] per-class precision / recall / f1 (support):")
        for c, m in test_metrics["per_class"].items():
            print(f"  {c:>10}  P={m['precision']:.2f}  R={m['recall']:.2f}  "
                  f"F1={m['f1']:.2f}  n={int(m['support'])}", flush=True)

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (args.out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"\n[done] saved best.pt + test_metrics.json + history.json to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
