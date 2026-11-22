"""Скорость судна из таблицы «время — скорость» (analytics/nmea_speed.py) по времени кадра."""
from __future__ import annotations

import bisect
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


class SpeedTable:
    def __init__(self, path, max_gap_min: float = 10.0):
        self.t, self.v = [], []
        with open(Path(path), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    self.t.append(datetime.fromisoformat(r["time_local"]).timestamp())
                    self.v.append(float(r["speed_mps"]))
                except (KeyError, ValueError):
                    continue
        order = sorted(range(len(self.t)), key=self.t.__getitem__)
        self.t = [self.t[i] for i in order]
        self.v = [self.v[i] for i in order]
        self.max_gap = max_gap_min * 60.0

    def __len__(self) -> int:
        return len(self.t)

    def at(self, ts: Optional[datetime]) -> Optional[float]:
        """Скорость, м/с: интерполяция между соседними записями; None, если рядом записей нет."""
        if ts is None or not self.t:
            return None
        x = ts.timestamp()
        i = bisect.bisect_left(self.t, x)
        lo, hi = i - 1, i
        near_lo = lo >= 0 and x - self.t[lo] <= self.max_gap
        near_hi = hi < len(self.t) and self.t[hi] - x <= self.max_gap
        if near_lo and near_hi and self.t[hi] > self.t[lo]:
            k = (x - self.t[lo]) / (self.t[hi] - self.t[lo])
            return self.v[lo] + k * (self.v[hi] - self.v[lo])
        if near_hi:
            return self.v[hi]
        if near_lo:
            return self.v[lo]
        return None
