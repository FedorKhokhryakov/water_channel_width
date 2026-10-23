"""Камера над плоской водой: ориентация по горизонту и проекция точек на воду.

Горизонт в нормализованных координатах — прямая n·(x, y, 1) = 0, где n — нормаль к воде.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class GroundPlane:
    """Плоскость воды относительно камеры: нормаль ``n`` и высота камеры ``height``."""

    def __init__(self, n_up: np.ndarray, height: float):
        n = _unit(np.asarray(n_up, dtype=np.float64))
        if n[1] > 0:  # «вверх» в кадре — это −y
            n = -n
        self.n = n
        self.height = float(height)
        ez = np.array([0.0, 0.0, 1.0])
        self.f = _unit(ez - ez.dot(n) * n)   # горизонтальное направление «вдоль оси камеры»
        self.lat = np.cross(self.f, n)        # горизонтальное направление «вправо по кадру»

    @classmethod
    def from_angles(cls, pitch_deg: float, roll_deg: float, height: float) -> "GroundPlane":
        th, ph = math.radians(pitch_deg), math.radians(roll_deg)
        n = np.array([math.cos(th) * math.sin(ph), -math.cos(th) * math.cos(ph), -math.sin(th)])
        return cls(n, height)

    @classmethod
    def from_horizon_line(cls, a: float, b: float, height: float) -> "GroundPlane":
        """Горизонт y = a·x + b в нормализованных координатах (без дисторсии)."""
        return cls(np.array([a, -1.0, b]), height)

    @property
    def pitch_deg(self) -> float:
        return math.degrees(math.asin(float(np.clip(-self.n[2], -1, 1))))

    @property
    def roll_deg(self) -> float:
        return math.degrees(math.atan2(self.n[0], -self.n[1]))

    def tilted(self, delta_deg: float) -> "GroundPlane":
        """Наклонить ось камеры вниз на delta градусов (поправка на dip и pitch_offset_deg)."""
        d = math.radians(delta_deg)
        return GroundPlane(math.cos(d) * self.n - math.sin(d) * self.f, self.height)

    def rays_to_ground(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Нормализованные точки -> (X поперёк, Z вдоль оси, наклонная дальность), м; выше горизонта NaN."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        r = np.column_stack([xy, np.ones(len(xy))])
        d = r.dot(self.n)
        d = np.where(d < -1e-9, d, np.nan)
        t = -self.height / d
        P = r * t[:, None]
        return P.dot(self.lat), P.dot(self.f), np.linalg.norm(P, axis=1)

    def ground_to_camera(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).ravel()
        Z = np.asarray(Z, dtype=np.float64).ravel()
        return X[:, None] * self.lat + Z[:, None] * self.f - self.height * self.n

    def horizon_y(self, x: np.ndarray) -> np.ndarray:
        """y линии горизонта (нормализованные координаты) для заданных x."""
        return -(self.n[0] * np.asarray(x, dtype=np.float64) + self.n[2]) / self.n[1]


def dip_deg(height_m: float) -> float:
    """Понижение видимого горизонта с учётом стандартной рефракции: 1.76′·√h."""
    return 1.76 / 60.0 * math.sqrt(max(height_m, 0.0))


def fit_line_robust(x: np.ndarray, y: np.ndarray, iterations: int = 4,
                    k: float = 3.0) -> Optional[Tuple[float, float, float, np.ndarray]]:
    """y = a·x + b, МНК с отбрасыванием выбросов по MAD. -> (a, b, rms, inliers)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    for _ in range(iterations):
        A = np.column_stack([x[ok], np.ones(ok.sum())])
        (a, b), *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        res = y - (a * x + b)
        mad = 1.4826 * np.median(np.abs(res[ok])) + 1e-12
        new_ok = np.isfinite(res) & (np.abs(res) <= k * mad + 1e-9)
        if new_ok.sum() < 3 or np.array_equal(new_ok, ok):
            break
        ok = new_ok
    A = np.column_stack([x[ok], np.ones(ok.sum())])
    (a, b), *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    rms = float(np.sqrt(np.mean((y[ok] - (a * x[ok] + b)) ** 2)))
    return float(a), float(b), rms, ok


class HorizonTracker:
    """Запасной горизонт: медиана удачных оценок за последние минуты."""

    def __init__(self, history_minutes: float = 30.0, maxlen: int = 200):
        self.window_s = float(history_minutes) * 60.0
        self.items: Deque[Tuple[float, float, float]] = deque(maxlen=maxlen)

    def add(self, t: float, pitch: float, roll: float) -> None:
        self.items.append((t, pitch, roll))

    def estimate(self, t: float) -> Optional[Tuple[float, float]]:
        recent = [(p, r) for (ti, p, r) in self.items if abs(t - ti) <= self.window_s]
        if len(recent) < 3:
            return None
        arr = np.array(recent)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
