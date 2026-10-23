"""Обработка одного кадра: сегментация -> горизонт -> кромки -> ширина в метрах -> флаги."""
from __future__ import annotations

import math
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from .camera import CameraModel, model_to_image_coords
from .channel import class_margin, connected_components, horizon_points, main_water_component, row_edges
from .geometry import GroundPlane, HorizonTracker, dip_deg, fit_line_robust
from .segmenter import softmax
from .width import measure_ground, quality_flags

SUSPECT_FLAGS = {"trend", "noisy", "uncertain", "jump", "no_near", "horizon_default"}


class FrameProcessor:
    """Обработка кадров одной камеры (хранит историю горизонта и ширины)."""

    def __init__(self, cfg, camera: CameraModel, segmenter=None, classes=None):
        self.cfg = cfg
        self.camera = camera
        self.segmenter = segmenter
        names = list(classes or (segmenter.classes if segmenter else
                                 ["background", "ice", "water", "ship"]))
        self.cls = {k: names.index(k) for k in ("background", "ice", "water", "ship")}
        self.tracker = HorizonTracker(cfg["horizon"]["history_minutes"])
        self.history: Deque[Tuple[float, float]] = deque(maxlen=500)

    def process_image(self, img_rgb: np.ndarray, timestamp: Optional[datetime] = None,
                      name: str = "") -> Dict[str, Any]:
        if self.segmenter is None:
            raise RuntimeError("Модель не загружена")
        t0 = time.perf_counter()
        logits = self.segmenter.predict(img_rgb)
        t_model = time.perf_counter() - t0
        res = self.process_prediction(np.argmax(logits, axis=0).astype(np.uint8), logits,
                                      (img_rgb.shape[1], img_rgb.shape[0]), timestamp, name)
        res["result"]["model_ms"] = round(t_model * 1000, 1)
        res["result"]["proc_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res

    def process_prediction(self, labels: np.ndarray, logits: Optional[np.ndarray],
                           image_size: Tuple[int, int], timestamp: Optional[datetime] = None,
                           name: str = "") -> Dict[str, Any]:
        """labels (h, w) и logits (C, h, w) или None -> результат кадра."""
        t_epoch = timestamp.timestamp() if timestamp else time.time()
        mcfg, qcfg = self.cfg["measure"], self.cfg["quality"]
        h, w = labels.shape
        cam = self.camera.scaled(image_size)
        grid = (w, h)

        def to_norm(pts_xy: np.ndarray) -> np.ndarray:
            return cam.undistort_points(model_to_image_coords(pts_xy, grid, image_size))

        scale = w / 512.0  # параметры в px заданы для сетки 512 по ширине
        result: Dict[str, Any] = dict(
            time=timestamp.isoformat(sep=" ") if timestamp else "", file=name, status="fail",
            reason="", flags="")
        details: Dict[str, Any] = dict(labels=labels, image_size=image_size, camera=cam)

        # 1. горизонт -> ориентация камеры
        margin_sky = class_margin(logits, self.cls["background"]) if logits is not None else None
        hp = horizon_points(labels, margin_sky, self.cls["background"], self.cls["ship"],
                            tuple(self.cfg["horizon"]["x_range"]), step=max(1, int(round(2 * scale))))
        plane, source, h_rms_px, h_slope = self._estimate_plane(hp, to_norm, cam, t_epoch)
        details.update(plane=plane, horizon_pts=hp)
        result.update(pitch_deg=round(plane.pitch_deg, 3), roll_deg=round(plane.roll_deg, 3),
                      horizon_source=source, horizon_rms_px=_r(h_rms_px, 2))

        # 2. канал
        water = labels == self.cls["water"]
        comp = main_water_component(water, close_px=max(1, int(round(mcfg["close_px"] * scale))) | 1,
                                    merge_gap_frac=mcfg["merge_gap_frac"])
        details["component"] = comp
        if comp is None:
            result["reason"] = "no_water"
            return dict(result=result, details=details)
        margin_w = class_margin(logits, self.cls["water"]) if logits is not None else None
        e = row_edges(comp, labels, margin_w, self.cls["ice"], self.cls["ship"],
                      occlusion_px=max(1, int(round(mcfg["occlusion_px"] * scale))))
        width_px = e["xr"] - e["xl"]
        center = 0.5 * (e["xr"] + e["xl"])
        edge_m = mcfg["edge_margin_frac"] * w
        keep = (e["ok_left"] & e["ok_right"]
                & (width_px >= mcfg["min_width_frac"] * w)
                & (np.abs(center - w / 2.0) <= mcfg["center_tol_frac"] * w)
                & (e["xl"] >= edge_m) & (e["xr"] <= w - 1 - edge_m))
        details["edges"] = dict(e, keep=keep)
        result["rows_total"] = int(len(keep))
        result["rows_rejected_occluded"] = int(np.sum(~(e["ok_left"] & e["ok_right"])))
        if keep.sum() < mcfg["min_rows"]:
            result["reason"] = "few_rows"
            return dict(result=result, details=details)

        ys = e["y"][keep]
        nl = to_norm(np.column_stack([e["xl"][keep], ys]))
        nr = to_norm(np.column_stack([e["xr"][keep], ys]))
        XL, ZL, _ = plane.rays_to_ground(nl)
        XR, ZR, _ = plane.rays_to_ground(nr)
        # судно в канале за кормой: перед его носом вода шире, эту зону и всё за ним не берём
        vessel_z = self._vessel_distance(labels, comp, to_norm, plane, scale)
        result["vessel_z_m"] = _r(vessel_z, 1)
        if math.isfinite(vessel_z):
            far = 0.5 * (ZL + ZR) > vessel_z - float(mcfg["vessel_margin_m"])
            XL, ZL, XR, ZR = (np.where(far, np.nan, a) for a in (XL, ZL, XR, ZR))
        m = measure_ground(XL, ZL, XR, ZR, mcfg)
        if m is None:
            result["reason"] = "few_rows"
            return dict(result=result, details=details)
        details.update(measure=m, ground=dict(XL=XL, ZL=ZL, XR=XR, ZR=ZR, y=ys))

        # 3. диагностика
        entropy = self._edge_entropy(logits, e, keep) if logits is not None else float("nan")
        slant = self._slant_bottom(comp, labels, to_norm, plane)
        pitch_vp = self._pitch_vanishing(nl, nr, m["rows"]["used"], m["rows"]["Z"], h_slope, plane)

        flags = list(quality_flags(m, qcfg, entropy, source))
        if math.isfinite(vessel_z):
            flags.append("vessel")
        w_main = m["width_m"]
        recent = [wv for (tv, wv) in self.history if 0 <= t_epoch - tv <= qcfg["smooth_minutes"] * 60 * 3]
        if len(recent) >= 5 and math.isfinite(w_main):
            med = float(np.median(recent))
            mad = max(1.0, 1.4826 * float(np.median(np.abs(np.asarray(recent) - med))))
            if abs(w_main - med) > qcfg["jump_k"] * mad:
                flags.append("jump")
        status = "suspect" if SUSPECT_FLAGS.intersection(flags) else "ok"
        if status == "ok" and math.isfinite(w_main):
            self.history.append((t_epoch, w_main))
        window = [wv for (tv, wv) in self.history if 0 <= t_epoch - tv <= qcfg["smooth_minutes"] * 60]
        smooth = float(np.median(window[-int(qcfg["smooth_window"]):])) if window else float("nan")

        result.update(
            status=status, flags=";".join(flags),
            width_m=_r(w_main), width_smooth_m=_r(smooth), width_near_m=_r(m["width_near_m"]),
            width_far_m=_r(m["width_far_m"]), width_median_m=_r(m["width_median_m"]),
            width_row_median_m=_r(m["width_row_median_m"]), width_mad_m=_r(m["width_mad_m"]),
            width_sigma_m=_r(m["width_sigma_m"]),
            **{"w_%dm" % k: _r(v) for k, v in m["sections"].items()},
            n_rows=m["n_rows"], z_min_m=_r(m["z_min_m"], 1), z_max_m=_r(m["z_max_m"], 1),
            trend_m_per_100m=_r(m["trend_m_per_100m"]), heading_deg=_r(m["heading_deg"]),
            centerline_offset_m=_r(m["centerline_offset_m"]),
            edge_entropy=_r(entropy, 3), slant_bottom_m=_r(slant), pitch_vp_deg=_r(pitch_vp, 3),
        )
        return dict(result=result, details=details)

    def _estimate_plane(self, hp: np.ndarray, to_norm, cam: CameraModel, t: float):
        ccfg, hcfg = self.cfg["camera"], self.cfg["horizon"]
        H = float(ccfg["height_m"])
        dip = dip_deg(H) if ccfg.get("dip_correction", True) else 0.0
        offset = float(ccfg.get("pitch_offset_deg", 0.0))
        rms_px, slope = float("nan"), None
        if len(hp) >= int(hcfg["min_points"]):
            n = to_norm(hp)
            fit = fit_line_robust(n[:, 0], n[:, 1])
            if fit is not None:
                a, b, rms, inl = fit
                rms_px = rms * cam.focal_px
                plane = GroundPlane.from_horizon_line(a, b, H).tilted(dip + offset)
                ok = (inl.sum() >= int(hcfg["min_points"]) and rms_px <= float(hcfg["max_rms_px"])
                      and abs(plane.pitch_deg - float(ccfg["pitch_ref_deg"]) - offset) <= float(ccfg["pitch_tol_deg"])
                      and abs(plane.roll_deg) <= float(ccfg["roll_max_deg"]))
                slope = a
                if ok:
                    self.tracker.add(t, plane.pitch_deg, plane.roll_deg)
                    return plane, "mask", rms_px, slope
        est = self.tracker.estimate(t)
        if est is not None:
            return GroundPlane.from_angles(est[0], est[1], H), "history", rms_px, None
        return (GroundPlane.from_angles(float(ccfg["pitch_ref_deg"]) + offset, 0.0, H),
                "default", rms_px, None)

    @staticmethod
    def _edge_entropy(logits: np.ndarray, e: Dict[str, np.ndarray], keep: np.ndarray) -> float:
        """Средняя нормированная энтропия softmax у кромок."""
        C, h, w = logits.shape
        ys = e["y"][keep].astype(int)
        xs = np.concatenate([np.round(e["xl"][keep]), np.round(e["xr"][keep])]).astype(int)
        yy = np.concatenate([ys, ys])
        pts_x = np.clip(np.concatenate([xs - 1, xs, xs + 1]), 0, w - 1)
        pts_y = np.concatenate([yy, yy, yy])
        p = softmax(logits[:, pts_y, pts_x].astype(np.float64), axis=0)
        ent = -(p * np.log(p + 1e-12)).sum(axis=0) / math.log(C)
        return float(ent.mean())

    def _slant_bottom(self, comp, labels, to_norm, plane: GroundPlane) -> float:
        """Наклонная дальность до ближайшей воды на оси (по чертежу ≈ 93.4 м)."""
        h, w = comp.shape
        c0, c1 = int(w * 0.46), int(w * 0.54)
        rows = np.flatnonzero(comp[:, c0:c1].any(axis=1))
        if not rows.size:
            return float("nan")
        yb = int(rows.max())
        below = labels[min(h - 1, yb + 2), c0:c1]
        # палуба размечена как «судно» или «фон»
        if not np.any((below == self.cls["ship"]) | (below == self.cls["background"])):
            return float("nan")
        xs = np.flatnonzero(comp[yb, c0:c1]) + c0
        _, _, s = plane.rays_to_ground(to_norm(np.array([[xs.mean(), yb + 0.5]])))
        return float(s[0])

    def _vessel_distance(self, labels, comp, to_norm, plane: GroundPlane, scale: float) -> float:
        """Дистанция до ближайшего чужого судна в створе канала (класс «судно» выше кормы), м."""
        h, w = labels.shape
        rows = np.flatnonzero(comp.any(axis=1))
        cols = np.flatnonzero(comp.any(axis=0))
        bottom = int(rows.max()) - max(2, int(round(3 * scale)))
        x0, x1 = max(0, int(cols.min() - 0.05 * w)), min(w, int(cols.max() + 0.05 * w) + 1)
        ship = np.zeros_like(comp)
        ship[:, x0:x1] = labels[:, x0:x1] == self.cls["ship"]
        if not ship.any():
            return float("nan")
        lab, stats = connected_components(ship)
        best = float("nan")
        for idx, st in enumerate(stats, start=1):
            if st["area"] < 15 * scale * scale or st["ymax"] >= h - 2 or st["ymax"] >= bottom:
                continue  # мелочь, собственная корма (касается низа кадра) или ниже канала
            yb = st["ymax"]
            xs = np.flatnonzero(lab[yb] == idx)
            _, Z, _ = plane.rays_to_ground(to_norm(np.array([[xs.mean(), yb + 0.5]])))
            if np.isfinite(Z[0]) and Z[0] > 0 and (math.isnan(best) or Z[0] < best):
                best = float(Z[0])
        return best

    @staticmethod
    def _pitch_vanishing(nl, nr, used, Z, h_slope, plane: GroundPlane) -> float:
        """Наклон камеры по точке схода кромок (диагностика)."""
        sel = used & (Z <= 250.0)
        if sel.sum() < 6:
            return float("nan")
        fl = fit_line_robust(nl[sel, 1], nl[sel, 0])  # x = c·y + d
        fr = fit_line_robust(nr[sel, 1], nr[sel, 0])
        if fl is None or fr is None or abs(fl[0] - fr[0]) < 1e-3:
            return float("nan")
        yv = (fr[1] - fl[1]) / (fl[0] - fr[0])
        xv = fl[0] * yv + fl[1]
        a = h_slope if h_slope is not None else -plane.n[0] / plane.n[1]
        vp_plane = GroundPlane.from_horizon_line(a, yv - a * xv, plane.height)
        return vp_plane.pitch_deg


def _r(v, nd: int = 2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    return round(v, nd) if math.isfinite(v) else ""
