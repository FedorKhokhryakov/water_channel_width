"""Ширина канала на воде: перпендикулярно осевой линии, по строкам и в сечениях."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _polyfit_robust(z: np.ndarray, x: np.ndarray, deg: int, iterations: int = 3) -> np.ndarray:
    ok = np.isfinite(z) & np.isfinite(x)
    deg = int(min(deg, max(0, ok.sum() - 2)))
    coef = np.polyfit(z[ok], x[ok], deg)
    for _ in range(iterations):
        res = x - np.polyval(coef, z)
        mad = 1.4826 * np.median(np.abs(res[ok])) + 1e-6
        new_ok = ok & (np.abs(res) <= 3 * mad + 0.05)
        if new_ok.sum() < deg + 2 or np.array_equal(new_ok, ok):
            break
        ok = new_ok
        coef = np.polyfit(z[ok], x[ok], deg)
    return coef


def measure_ground(XL: np.ndarray, ZL: np.ndarray, XR: np.ndarray, ZR: np.ndarray,
                   cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Ширины по строкам и сводка кадра по точкам кромок на воде; None, если строк мало."""
    XL, ZL, XR, ZR = (np.asarray(a, dtype=np.float64) for a in (XL, ZL, XR, ZR))
    Zm = 0.5 * (ZL + ZR)
    Xm = 0.5 * (XL + XR)
    used = np.isfinite(Zm) & np.isfinite(Xm) & (Zm > 0) & (Zm <= float(cfg["z_max_m"]))
    if used.sum() < int(cfg["min_rows"]):
        return None

    coef = _polyfit_robust(Zm[used], Xm[used], int(cfg.get("centerline_degree", 1)))
    slope = np.polyval(np.polyder(coef), Zm) if len(coef) > 1 else np.zeros_like(Zm)
    dX, dZ = XR - XL, ZR - ZL
    W = np.abs(dX - dZ * slope) / np.sqrt(1.0 + slope ** 2)
    W_row = np.abs(dX)  # вдоль строки кадра, без учёта направления канала

    z_near = float(cfg["z_near_m"])
    near = used & (Zm <= z_near)
    far = used & (Zm > z_near)
    Wu, Zu = W[used], Zm[used]
    med = float(np.median(Wu))
    mad = float(1.4826 * np.median(np.abs(Wu - med)))

    def _median(mask: np.ndarray) -> float:
        return float(np.median(W[mask])) if mask.sum() >= 3 else float("nan")

    sections = {}
    hw0 = float(cfg.get("section_halfwidth_m", 20.0))
    for zk in cfg.get("sections_m", []):
        hw = max(hw0, 0.1 * float(zk))
        sel = used & (np.abs(Zm - float(zk)) <= hw)
        sections[int(zk)] = float(np.median(W[sel])) if sel.sum() >= 2 else float("nan")

    trend = float(np.polyfit(Zu, Wu, 1)[0] * 100.0) if np.ptp(Zu) > 20 else float("nan")
    w_near, w_all = _median(near), med
    primary = w_near if (cfg.get("primary", "near") == "near" and math.isfinite(w_near)) else w_all
    heading = math.degrees(math.atan(float(np.median(slope[used]))))
    return dict(
        width_m=primary,
        width_near_m=w_near,
        width_far_m=_median(far),
        width_median_m=w_all,
        width_row_median_m=float(np.median(W_row[used])),
        width_mad_m=mad,
        width_sigma_m=mad / math.sqrt(max(int(used.sum()), 1)),
        sections=sections,
        n_rows=int(used.sum()),
        z_min_m=float(Zu.min()),
        z_max_m=float(Zu.max()),
        trend_m_per_100m=trend,
        heading_deg=heading,
        centerline_offset_m=float(np.polyval(coef, min(z_near, float(Zu.max())))),
        centerline=coef.tolist(),
        rows=dict(Z=Zm, W=W, used=used),
    )


def quality_flags(m: Dict[str, Any], qcfg: Dict[str, Any], edge_entropy: float,
                  horizon_source: str) -> Sequence[str]:
    flags = []
    if math.isfinite(m["trend_m_per_100m"]) and abs(m["trend_m_per_100m"]) > qcfg["trend_max_m_per_100m"]:
        flags.append("trend")
    if m["width_median_m"] > 0 and m["width_mad_m"] / m["width_median_m"] > qcfg["rel_mad_max"]:
        flags.append("noisy")
    if math.isfinite(edge_entropy) and edge_entropy > qcfg["entropy_max"]:
        flags.append("uncertain")
    if horizon_source != "mask":
        flags.append("horizon_" + horizon_source)
    if not math.isfinite(m["width_near_m"]):
        flags.append("no_near")
    return flags
