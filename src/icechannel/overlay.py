"""Контрольное изображение: вода, кромки, горизонт и сечения с шириной."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

STATUS_COLORS = {"ok": (60, 200, 90), "suspect": (255, 170, 0), "fail": (230, 60, 60)}


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def colorize(labels: np.ndarray, colors) -> np.ndarray:
    pal = np.asarray(colors, dtype=np.uint8)
    return pal[np.clip(labels, 0, len(pal) - 1)]


def render(img_rgb: Optional[np.ndarray], out: Dict[str, Any], colors, out_width: int = 1280,
           water_color=(43, 169, 236)):
    from PIL import Image, ImageDraw

    det, res = out["details"], out["result"]
    labels = det["labels"]
    W0, H0 = det["image_size"]
    if img_rgb is None:
        img_rgb = colorize(labels, colors)
    scale = out_width / float(W0)
    out_h = int(round(H0 * scale))
    base = Image.fromarray(img_rgb).convert("RGB").resize((out_width, out_h), Image.BILINEAR)

    comp = det.get("component")
    if comp is not None:
        m = Image.fromarray((comp * 110).astype(np.uint8)).resize((out_width, out_h), Image.NEAREST)
        tint = Image.new("RGB", base.size, water_color)
        base = Image.composite(tint, base, m)
    draw = ImageDraw.Draw(base)
    gw, gh = labels.shape[1], labels.shape[0]
    sx, sy = out_width / gw, out_h / gh

    e = det.get("edges")
    if e is not None:
        for y, xl, xr, k in zip(e["y"], e["xl"], e["xr"], e["keep"]):
            col = (255, 230, 0) if k else (150, 150, 150)
            r = 1.5
            for x in (xl, xr):
                cx, cy = (x + 0.5) * sx, (y + 0.5) * sy
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)

    cam, plane = det["camera"], det.get("plane")
    k_img = out_width / float(W0)
    if plane is not None:
        xs = np.linspace(-1.2, 1.2, 120)
        ys = plane.horizon_y(xs)
        pts = cam.project_normalized(np.column_stack([xs, ys])) * k_img
        ok = (np.isfinite(pts).all(axis=1) & (pts[:, 0] > -50) & (pts[:, 0] < out_width + 50)
              & (xs ** 2 + ys ** 2 < 0.95 * cam.max_valid_r2))
        if ok.sum() > 1:
            draw.line([tuple(p) for p in pts[ok]], fill=(0, 255, 255), width=1)

    font = _font(16)
    m = det.get("measure")
    if m is not None and plane is not None:
        coef = np.asarray(m["centerline"])
        slope_coef = np.polyder(coef) if len(coef) > 1 else np.array([0.0])
        for zk, wk in m["sections"].items():
            if not math.isfinite(wk):
                continue
            xc, s = float(np.polyval(coef, zk)), float(np.polyval(slope_coef, zk))
            p = np.array([1.0, -s]) / math.hypot(1.0, s)
            X = np.array([xc - p[0] * wk / 2, xc + p[0] * wk / 2])
            Z = np.array([zk - p[1] * wk / 2, zk + p[1] * wk / 2])
            uv = cam.project_points(plane.ground_to_camera(X, Z)) * k_img
            if not np.isfinite(uv).all():
                continue
            draw.line([tuple(uv[0]), tuple(uv[1])], fill=(255, 80, 0), width=3)
            draw.text((uv[1][0] + 6, uv[1][1] - 9), "%.1f m @%dm" % (wk, zk), fill=(255, 255, 255),
                      font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    status = res.get("status", "fail")
    title = "%s  %s  W=%s m (smooth %s)  pitch=%s roll=%s  [%s]%s" % (
        res.get("time", ""), status.upper(), res.get("width_m", "-"), res.get("width_smooth_m", "-"),
        res.get("pitch_deg", "-"), res.get("roll_deg", "-"), res.get("horizon_source", ""),
        ("  " + res["flags"]) if res.get("flags") else ("  " + res["reason"] if res.get("reason") else ""))
    draw.rectangle([0, 0, out_width, 24], fill=(0, 0, 0))
    draw.rectangle([0, 0, 8, 24], fill=STATUS_COLORS.get(status, (200, 200, 200)))
    draw.text((14, 3), title, fill=(255, 255, 255), font=font)
    return base
