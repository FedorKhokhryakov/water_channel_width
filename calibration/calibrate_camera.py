"""Калибровка камеры по шахматной доске.

    python calibration/calibrate_camera.py --images "calibration/chessboard/*.png" [--rational]
"""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icechannel.camera import CameraModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob снимков доски")
    ap.add_argument("--out", default="calibration/stern_camera.yaml")
    ap.add_argument("--board", type=int, nargs=2, default=[9, 6], help="внутренние углы")
    ap.add_argument("--square-mm", type=float, default=80.0)
    ap.add_argument("--rational", action="store_true")
    a = ap.parse_args()

    objp = np.zeros((a.board[0] * a.board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.board[0], 0:a.board[1]].T.reshape(-1, 2) * a.square_mm
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj, imgp, sizes = [], [], set()
    paths = sorted(glob(a.images))
    for p in paths:
        g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        sizes.add(g.shape[::-1])
        ok, c = cv2.findChessboardCorners(g, tuple(a.board), None)
        if ok:
            obj.append(objp)
            imgp.append(cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit))
    assert len(sizes) == 1, "снимки разного размера: %s" % sizes
    assert obj, "доска не найдена ни на одном снимке"
    size = sizes.pop()
    flags = cv2.CALIB_RATIONAL_MODEL if a.rational else 0
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, imgp, size, None, None, flags=flags)
    print("Доска найдена на %d из %d снимков, кадр %s, RMS %.3f px" % (len(obj), len(paths), size, rms))

    cam = CameraModel(K, dist.ravel(), size)
    pts = np.concatenate([c.reshape(-1, 2) for c in imgp])
    cover = (np.ptp(pts[:, 0]) / size[0], np.ptp(pts[:, 1]) / size[1])
    print("Покрытие кадра углами доски: %.0f%% x %.0f%%" % (cover[0] * 100, cover[1] * 100))
    print("Дисторсия обратима на %.0f%% кадра" % (cam.valid_fraction() * 100))
    if cam.valid_fraction() < 0.98:
        print("ВНИМАНИЕ: в углах кадра дисторсия необратима, нужны снимки у краёв или --rational")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(image_size=list(size), camera_matrix=K.tolist(), dist_coeffs=dist.ravel().tolist(),
                            rms_px=float(rms), n_images=len(obj)), f, allow_unicode=True, sort_keys=False)
    print("Сохранено:", a.out)


if __name__ == "__main__":
    main()
