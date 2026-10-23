"""Канал в маске: компонента воды, кромки по строкам, точки горизонта (numpy, без OpenCV)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _shift_reduce(mask: np.ndarray, k: int, axis: int, op) -> np.ndarray:
    if k <= 1:
        return mask
    r = k // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, k - 1 - r)
    fill = op is np.minimum  # эрозия не съедает края кадра
    p = np.pad(mask, pad, mode="constant", constant_values=fill)
    n = mask.shape[axis]
    out = np.take(p, range(0, n), axis=axis)
    for s in range(1, k):
        out = op(out, np.take(p, range(s, s + n), axis=axis))
    return out


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    return _shift_reduce(_shift_reduce(mask, k, 0, np.maximum), k, 1, np.maximum)


def erode(mask: np.ndarray, k: int) -> np.ndarray:
    return _shift_reduce(_shift_reduce(mask, k, 0, np.minimum), k, 1, np.minimum)


def binary_close(mask: np.ndarray, k: int) -> np.ndarray:
    return erode(dilate(mask, k), k) if k > 1 else mask




def connected_components(mask: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """8-связные компоненты (run-length + union-find) -> (labels, статистики)."""
    h, w = mask.shape
    m = np.zeros((h, w + 2), dtype=np.int8)
    m[:, 1:-1] = mask.astype(bool)
    d = np.diff(m, axis=1)
    rs, cs = np.nonzero(d == 1)
    _, ce = np.nonzero(d == -1)  # концы серий (исключительно)
    n = len(rs)
    labels = np.zeros((h, w), dtype=np.int32)
    if n == 0:
        return labels, []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    row_start = np.searchsorted(rs, np.arange(h + 1))
    for y in range(1, h):
        i, a1 = int(row_start[y - 1]), int(row_start[y])
        j, b1 = int(row_start[y]), int(row_start[y + 1])
        while i < a1 and j < b1:
            if cs[i] <= ce[j] and cs[j] <= ce[i]:  # перекрытие с учётом диагоналей
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            if ce[i] < ce[j]:
                i += 1
            else:
                j += 1
    roots = np.array([find(i) for i in range(n)])
    uniq, comp = np.unique(roots, return_inverse=True)
    comp = comp + 1
    stats: List[Dict[str, int]] = [dict(area=0, ymin=h, ymax=-1, xmin=w, xmax=-1) for _ in uniq]
    for k in range(n):
        y, s, e, c = int(rs[k]), int(cs[k]), int(ce[k]), int(comp[k])
        labels[y, s:e] = c
        st = stats[c - 1]
        st["area"] += e - s
        st["ymin"] = min(st["ymin"], y)
        st["ymax"] = max(st["ymax"], y)
        st["xmin"] = min(st["xmin"], s)
        st["xmax"] = max(st["xmax"], e - 1)
    return labels, stats


def _row_extent(comp: np.ndarray, y: int) -> Optional[Tuple[int, int]]:
    xs = np.flatnonzero(comp[y])
    return (int(xs[0]), int(xs[-1])) if xs.size else None


def main_water_component(water: np.ndarray, close_px: int = 3, min_area_frac: float = 0.001,
                         merge_gap_frac: float = 0.15) -> Optional[np.ndarray]:
    """Основная компонента воды + куски канала, отрезанные перемычкой льда."""
    h, w = water.shape
    closed = binary_close(water.astype(bool), close_px)
    labels, stats = connected_components(closed)
    cands = []
    for idx, st in enumerate(stats, start=1):
        if st["area"] < min_area_frac * h * w:
            continue
        hh = st["ymax"] - st["ymin"] + 1
        center_dist = abs((st["xmin"] + st["xmax"]) / 2.0 - w / 2.0) / w
        cands.append((st["area"] * (1 + hh / h) * (1 - center_dist), idx))
    if not cands:
        return None
    cands.sort(reverse=True)
    main_idx = cands[0][1]
    comp = labels == main_idx
    members = {main_idx}
    gap_max = merge_gap_frac * h
    changed = True
    while changed:
        changed = False
        ys = np.flatnonzero(comp.any(axis=1))
        top, bot = int(ys[0]), int(ys[-1])
        for _, idx in cands[1:]:
            if idx in members:
                continue
            st = stats[idx - 1]
            cx = (st["xmin"] + st["xmax"]) / 2.0
            if 0 < top - st["ymax"] <= gap_max:      # кусок выше (дальше по каналу)
                ext = _row_extent(comp, top)
            elif 0 < st["ymin"] - bot <= gap_max:    # кусок ниже (ближе к корме)
                ext = _row_extent(comp, bot)
            else:
                continue
            if ext is not None and ext[0] <= cx <= ext[1]:
                comp |= labels == idx
                members.add(idx)
                changed = True
    return comp




def _subpixel_crossing(m0: float, m1: float, x0: float) -> float:
    """Переход «запаса» через ноль между центрами пикселей x0 (m0) и x0+1 (m1)."""
    if (m0 > 0) == (m1 > 0) or m0 == m1:
        return x0 + 0.5
    return x0 + float(np.clip(m0 / (m0 - m1), 0.0, 1.0))


def row_edges(comp: np.ndarray, labels: np.ndarray, margin: Optional[np.ndarray],
              ice_class: int, ship_class: int, occlusion_px: int = 2) -> Dict[str, np.ndarray]:
    """Левая и правая кромка канала по строкам; кромка годится, только если за ней лёд."""
    h, w = comp.shape
    ys, xl, xr, okl, okr = [], [], [], [], []
    k = max(1, int(occlusion_px))
    for y in np.flatnonzero(comp.any(axis=1)):
        xs = np.flatnonzero(comp[y])
        left, right = int(xs[0]), int(xs[-1])
        row_lab = labels[y]
        if margin is not None and left > 0:
            el = _subpixel_crossing(margin[y, left - 1], margin[y, left], left - 1)
        else:
            el = left - 0.5
        if margin is not None and right < w - 1:
            er = _subpixel_crossing(margin[y, right], margin[y, right + 1], right)
        else:
            er = right + 0.5
        lo = row_lab[max(0, left - k):left]
        ro = row_lab[right + 1:right + 1 + k]
        good_l = left - k >= 0 and lo.size > 0 and lo[-1] == ice_class and not np.any(lo == ship_class)
        good_r = right + k <= w - 1 and ro.size > 0 and ro[0] == ice_class and not np.any(ro == ship_class)
        ys.append(y)
        xl.append(el)
        xr.append(er)
        okl.append(good_l)
        okr.append(good_r)
    return dict(y=np.asarray(ys, dtype=np.float64), xl=np.asarray(xl), xr=np.asarray(xr),
                ok_left=np.asarray(okl, dtype=bool), ok_right=np.asarray(okr, dtype=bool))




def horizon_points(labels: np.ndarray, margin_sky: Optional[np.ndarray], sky_class: int,
                   ship_class: int, x_range=(0.15, 0.85), step: int = 2) -> np.ndarray:
    """Точки видимого горизонта: первая сверху строка «не неба» в каждом столбце."""
    h, w = labels.shape
    pts = []
    for c in range(int(w * x_range[0]), int(w * x_range[1]), max(1, step)):
        col = labels[:, c]
        not_sky = np.flatnonzero(col != sky_class)
        if not_sky.size == 0 or not_sky[0] == 0:
            continue
        v0 = int(not_sky[0])
        if col[v0] == ship_class:
            continue
        if margin_sky is not None:
            y = _subpixel_crossing(margin_sky[v0 - 1, c], margin_sky[v0, c], v0 - 1)
        else:
            y = v0 - 0.5
        pts.append((float(c), y))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def class_margin(logits: np.ndarray, cls: int) -> np.ndarray:
    """Логит класса минус максимум остальных: >0 там, где класс побеждает."""
    others = np.delete(logits, cls, axis=0).max(axis=0)
    return logits[cls] - others
