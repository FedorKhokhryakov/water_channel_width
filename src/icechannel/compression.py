"""Сжатие льда по скорости закрытия канала.

Участок канала на дистанции Z за кормой прорезан Z/V секунд назад (V — скорость ледокола),
поэтому наклон ширины по дистанции dW/dZ даёт скорость закрытия одной кромки: v = −(dW/dZ)·V/2.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np


def frame_slope(Z: np.ndarray, W: np.ndarray, used: np.ndarray, ccfg: Dict[str, Any]) -> Tuple[float, str]:
    """Наклон ширины по дистанции (м на м) в зоне [z_min, z_max] -> (наклон, причина отказа)."""
    sel = used & np.isfinite(Z) & np.isfinite(W) & (Z >= ccfg["z_min_m"]) & (Z <= ccfg["z_max_m"])
    if sel.sum() < int(ccfg["min_rows"]) or np.ptp(Z[sel]) < float(ccfg["min_span_m"]):
        return float("nan"), "few_rows"
    z, w = Z[sel], W[sel]
    coef = np.polyfit(z, w, 1)
    res = w - np.polyval(coef, z)
    mad = 1.4826 * np.median(np.abs(res)) + 1e-6
    ok = np.abs(res) <= 3 * mad + 0.1
    if ok.sum() >= int(ccfg["min_rows"]):
        coef = np.polyfit(z[ok], w[ok], 1)
    return float(coef[0]), ""


def score(v: float, ccfg: Dict[str, Any]) -> str:
    """Скорость закрытия кромки, м/с -> балл сжатия по таблице."""
    for thr, s in zip(ccfg["thresholds_mps"], ccfg["scores"]):
        if v < thr:
            return s
    return ccfg["scores"][-1]


class CompressionTracker:
    """Скользящая медиана скорости закрытия по годным кадрам."""

    def __init__(self, window_minutes: float):
        self.window_s = float(window_minutes) * 60.0
        self.items: Deque[Tuple[float, float]] = deque(maxlen=1000)

    def add(self, t: float, v: float) -> None:
        self.items.append((t, v))

    def estimate(self, t: float) -> Tuple[float, int]:
        vals = [v for (ti, v) in self.items if 0 <= t - ti <= self.window_s]
        return (float(np.median(vals)) if vals else float("nan")), len(vals)


def evaluate(m: Optional[Dict[str, Any]], result: Dict[str, Any], speed: Optional[float],
             speed_source: str, t: float, tracker: CompressionTracker, ccfg: Dict[str, Any]) -> Dict[str, Any]:
    """Колонки сжатия для результата кадра."""
    out: Dict[str, Any] = dict(speed_mps=speed if speed is not None else "", speed_source=speed_source,
                               closing_slope_m_per_100m="", closing_speed_ms="", closing_note="",
                               compression_speed_ms="", compression_n=0, compression_score="")
    note = ""
    slope = float("nan")
    if m is None or result.get("status") != "ok":
        note = "no_measure"
    else:
        slope, note = frame_slope(m["rows"]["Z"], m["rows"]["W"], m["rows"]["used"], ccfg)
        vz = result.get("vessel_z_m")
        heading = result.get("heading_deg")
        if not note and vz not in ("", None) and float(vz) < float(ccfg["vessel_clear_m"]):
            note = "vessel"
        elif not note and heading not in ("", None) and abs(float(heading)) > float(ccfg["max_heading_deg"]):
            note = "turn"
        elif not note and speed is None:
            note = "no_speed"
        elif not note and speed < float(ccfg["min_speed_mps"]):
            note = "slow"
    if math.isfinite(slope):
        out["closing_slope_m_per_100m"] = round(slope * 100, 2)
        if speed is not None:
            out["closing_speed_ms"] = round(-slope * speed / 2.0, 4)
    if not note:
        tracker.add(t, -slope * speed / 2.0)
    out["closing_note"] = note
    v, n = tracker.estimate(t)
    out["compression_n"] = n
    if n >= int(ccfg["min_frames"]):
        out["compression_speed_ms"] = round(v, 4)
        out["compression_score"] = score(v, ccfg)
    return out
