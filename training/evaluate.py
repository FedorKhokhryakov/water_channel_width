"""Сравнение с ручной разметкой: IoU воды и разница ширины (модель против разметки).

    python training/evaluate.py      # data/stern + models/model.onnx -> out/eval
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icechannel.archives import decode_image, parse_timestamp  # noqa: E402
from icechannel.camera import CameraModel  # noqa: E402
from icechannel.config import load_config  # noqa: E402
from icechannel.pipeline import FrameProcessor  # noqa: E402
from icechannel.segmenter import Segmenter  # noqa: E402
from icechannel.storage import ResultStore  # noqa: E402


def gt_labels(path: Path, colors, grid_hw):
    from PIL import Image

    im = np.asarray(Image.open(str(path)).convert("RGB"))
    lab = np.zeros(im.shape[:2], np.uint8)
    for i, c in enumerate(colors):
        lab[np.all(im == c, axis=-1)] = i
    h, w = lab.shape
    ys = np.clip(((np.arange(grid_hw[0]) + 0.5) * h / grid_hw[0]).astype(int), 0, h - 1)
    xs = np.clip(((np.arange(grid_hw[1]) + 0.5) * w / grid_hw[1]).astype(int), 0, w - 1)
    return lab[ys][:, xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config/config.example.yaml")
    ap.add_argument("--frames", default="data/stern/images")
    ap.add_argument("--gt", default="data/stern/masks", help="ручные RGB-маски")
    ap.add_argument("--out", default="out/eval")
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["output"]["min_free_mb"] = 0
    seg = Segmenter(cfg.path("model"), cfg["model"]["threads"], cfg["model"]["tta_flip"])
    cam = CameraModel.from_file(cfg.path("calibration"), tuple(cfg["camera"]["calib_image_size"]))
    grid = tuple(seg.meta["input_size"])
    water = seg.class_index("water")
    store = ResultStore(cfg, Path(a.out))
    gts = {p.stem: p for p in Path(a.gt).iterdir() if p.suffix.lower() in (".png", ".jpg")}
    rows = []
    for f in sorted(Path(a.frames).iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        g = gts.get(f.stem) or next((p for s, p in gts.items() if s.startswith(f.stem)), None)
        img = decode_image(f.read_bytes())
        size = (img.shape[1], img.shape[0])
        ts = parse_timestamp(f.name)
        pred = FrameProcessor(cfg, cam, seg).process_image(img, ts, f.name)
        pred["result"]["model"] = seg.version
        store.save(pred, img, seg.meta["colors"])
        r = dict(file=f.name, status=pred["result"]["status"], flags=pred["result"]["flags"],
                 width_pred=pred["result"].get("width_m", ""), pitch=pred["result"].get("pitch_deg"))
        if g is not None:
            gl = gt_labels(g, seg.meta["colors"], grid)
            pl = pred["details"]["labels"]
            inter = np.sum((gl == water) & (pl == water))
            union = np.sum((gl == water) | (pl == water))
            gres = FrameProcessor(cfg, cam, None, seg.classes).process_prediction(gl, None, size, ts, f.name)["result"]
            r.update(iou_water=round(float(inter) / max(int(union), 1), 4), width_gt=gres.get("width_m", ""),
                     gt_status=gres["status"])
            if r["width_gt"] != "" and r["width_pred"] != "":
                r["diff_m"] = round(float(r["width_pred"]) - float(r["width_gt"]), 2)
        rows.append(r)
        print(r)

    out = Path(a.out) / "eval.csv"
    keys = ["file", "status", "flags", "pitch", "iou_water", "width_gt", "width_pred", "diff_m", "gt_status"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    d = np.array([r["diff_m"] for r in rows if "diff_m" in r], float)
    iou = np.array([r["iou_water"] for r in rows if "iou_water" in r], float)
    if len(iou):
        print("\nIoU воды: среднее %.3f, мин %.3f (n=%d)" % (iou.mean(), iou.min(), len(iou)))
    if len(d):
        print("Ширина (модель − разметка): медиана %+.2f м, медиана |Δ| %.2f м, макс |Δ| %.2f м (n=%d)" % (
            np.median(d), np.median(np.abs(d)), np.abs(d).max(), len(d)))
    print("Таблица:", out, "| overlay:", Path(a.out) / "overlays")


if __name__ == "__main__":
    main()
