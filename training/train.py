"""Обучение U-Net. Лучшая эпоха выбирается по IoU воды на val.

    python training/train.py --images data/stern/images --masks data/stern/masks --out checkpoints
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dataset import SternDataset, find_pairs, split_pairs
from model import UNet
from torch.utils.data import DataLoader

WATER = 2


def dice_loss(probs, target, eps=1e-6):
    C = probs.shape[1]
    onehot = nn.functional.one_hot(target, C).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).flatten(2).sum(-1)
    denom = probs.flatten(2).sum(-1) + onehot.flatten(2).sum(-1)
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, w_ce=1.0, w_dice=2.0):
        super().__init__()
        self.ce, self.w_ce, self.w_dice = nn.CrossEntropyLoss(), w_ce, w_dice

    def forward(self, logits, target):
        return self.w_ce * self.ce(logits, target) + self.w_dice * dice_loss(logits.softmax(1), target)


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    loss_sum, crit = 0.0, CombinedLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        p = logits.argmax(1)
        conf += torch.bincount((y * n_classes + p).flatten().cpu(), minlength=n_classes ** 2).view(n_classes, -1)
    tp = conf.diag().float()
    iou = tp / (conf.sum(0) + conf.sum(1) - tp).clamp(min=1)
    dice = 2 * tp / (conf.sum(0) + conf.sum(1)).clamp(min=1)
    return dict(loss=loss_sum / max(len(loader), 1), iou=iou.tolist(), dice=dice.tolist(),
                acc=float(tp.sum() / conf.sum().clamp(min=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default="checkpoints_v3")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-c", type=int, default=32)
    ap.add_argument("--size", type=int, nargs=2, default=[288, 512], metavar=("H", "W"))
    ap.add_argument("--resize", choices=["linear", "area"], default="linear",
                    help="linear: как текущая модель; area: с антиалиасингом")
    ap.add_argument("--split", choices=["time", "day"], default="time")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(a.images, a.masks)
    train_pairs, val_pairs = split_pairs(pairs, a.val_ratio, a.split)
    print("Пар: %d | train %d | val %d | устройство %s" % (len(pairs), len(train_pairs), len(val_pairs), device))
    tl = DataLoader(SternDataset(train_pairs, a.size, True, a.resize), a.batch, shuffle=True,
                    num_workers=a.workers, drop_last=True)
    vl = DataLoader(SternDataset(val_pairs, a.size, False, a.resize), a.batch, num_workers=a.workers)

    n_classes = 4
    model = UNet(n_classes=n_classes, in_channels=3, base_c=a.base_c).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    crit = CombinedLoss()
    hist, best = [], -1.0
    train_cfg = dict(vars(a), n_classes=n_classes)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        m = evaluate(model, vl, device, n_classes)
        rec = dict(epoch=ep, train_loss=tot / max(len(tl), 1), **m)
        hist.append(rec)
        print("ep %3d  train %.4f  val %.4f  IoU воды %.4f  Dice воды %.4f  (%.0f с)" % (
            ep, rec["train_loss"], m["loss"], m["iou"][WATER], m["dice"][WATER], time.time() - t0))
        if m["iou"][WATER] > best:
            best = m["iou"][WATER]
            torch.save(dict(epoch=ep, model_state=model.state_dict(), metrics=m, config=train_cfg), out / "best.pth")
        (out / "history.json").write_text(json.dumps(hist, indent=1))
    torch.save(model.state_dict(), out / "final_model.pth")
    print("Лучший IoU воды %.4f -> %s" % (best, out / "best.pth"))


if __name__ == "__main__":
    main()
