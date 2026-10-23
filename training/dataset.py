"""Датасет: пары кадр–маска, ресайз как на судне (icechannel.segmenter), разбиение train/val."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icechannel.segmenter import DEFAULT_META, resize_area, resize_linear  # noqa: E402

PALETTE = {tuple(c): i for i, c in enumerate(DEFAULT_META["colors"])}
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif")


def find_pairs(images_dir: str, masks_dir: str) -> List[Tuple[str, str]]:
    """Маска ищется по имени кадра: точное совпадение или имя кадра + суффикс (_mask)."""
    imgs = {p.stem: p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMG_EXT}
    pairs = []
    for m in sorted(Path(masks_dir).iterdir()):
        if m.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        img = imgs.get(m.stem)
        if img is None:
            cands = [s for s in imgs if m.stem.startswith(s)]
            img = imgs[max(cands, key=len)] if cands else None
        if img is not None:
            pairs.append((str(img), str(m)))
    return sorted(pairs, key=lambda p: Path(p[0]).name)


def rgb_mask_to_classes(mask_rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for color, cls in PALETTE.items():
        out[np.all(mask_rgb == color, axis=-1)] = cls
    return out


def resize_pair(img: np.ndarray, mask: np.ndarray, hw, mode: str = "linear"):
    """Кадр — как на судне; маска — ближайший сосед по центрам ячеек."""
    x = resize_area(img, hw) if mode == "area" else resize_linear(img, hw)
    h, w = mask.shape
    ys = np.clip(((np.arange(hw[0]) + 0.5) * h / hw[0]).astype(int), 0, h - 1)
    xs = np.clip(((np.arange(hw[1]) + 0.5) * w / hw[1]).astype(int), 0, w - 1)
    return np.clip(x, 0, 255).astype(np.uint8), mask[ys][:, xs]


def split_pairs(pairs, val_ratio: float = 0.2, mode: str = "time"):
    """time: последние по времени кадры в val; day: последние дни целиком."""
    if mode == "day":
        days = sorted({Path(p[0]).name[:8] for p in pairs})
        n_val_days = max(1, int(round(len(days) * val_ratio)))
        val_days = set(days[-n_val_days:])
        return ([p for p in pairs if Path(p[0]).name[:8] not in val_days],
                [p for p in pairs if Path(p[0]).name[:8] in val_days])
    n_val = max(1, int(len(pairs) * val_ratio))
    return pairs[:-n_val], pairs[-n_val:]


class SternDataset:
    """torch Dataset (без жёсткой зависимости при импорте модуля)."""

    def __init__(self, pairs, hw=(288, 512), augment=False, resize="linear"):
        import albumentations as A
        import cv2
        from albumentations.pytorch import ToTensorV2

        self.pairs, self.hw, self.resize = pairs, tuple(hw), resize
        norm = [A.Normalize(mean=DEFAULT_META["mean"], std=DEFAULT_META["std"]), ToTensorV2()]
        if augment:
            self.tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=(-0.03, 0.03), scale=(0.95, 1.05), rotate=(-3, 3),
                         border_mode=cv2.BORDER_CONSTANT, p=0.35),
                A.OneOf([A.RandomBrightnessContrast(0.2, 0.2),
                         A.HueSaturationValue(8, 15, 10)], p=0.6),
                A.RandomGamma(gamma_limit=(85, 115), p=0.2),
                *norm,
            ])
        else:
            self.tf = A.Compose(norm)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = np.asarray(Image.open(ip).convert("RGB"))
        mask = rgb_mask_to_classes(np.asarray(Image.open(mp).convert("RGB")))
        img, mask = resize_pair(img, mask, self.hw, self.resize)
        t = self.tf(image=img, mask=mask)
        return t["image"], t["mask"].long()
