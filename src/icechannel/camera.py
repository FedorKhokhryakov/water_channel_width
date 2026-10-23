"""Модель камеры (K + дисторсия OpenCV) для пересчёта точек без cv2."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import load_yaml


class CameraModel:
    """Pinhole-камера с дисторсией OpenCV: 5 коэффициентов или 8 (рациональная модель)."""

    def __init__(self, K: np.ndarray, dist: Sequence[float], image_size: Tuple[int, int]):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        d = np.zeros(8, dtype=np.float64)
        dist = np.asarray(dist, dtype=np.float64).ravel()
        d[: min(8, dist.size)] = dist[:8]
        self.dist = d
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self._max_r2: Optional[float] = None

    @classmethod
    def from_file(cls, path, default_size: Tuple[int, int] = (2560, 1440)) -> "CameraModel":
        """Читает YAML/JSON: camera_matrix + dist_coeffs (или distorsion_coeffs), image_size."""
        data = load_yaml(Path(path))
        K = np.array(data["camera_matrix"], dtype=np.float64)
        dist = None
        for key in ("dist_coeffs", "distortion_coeffs", "distorsion_coeffs"):
            if key in data:
                dist = np.array(data[key], dtype=np.float64).ravel()
                break
        if dist is None:
            dist = np.zeros(5)
        if "image_size" in data:
            size = tuple(int(v) for v in data["image_size"])
        elif "image_width" in data and "image_height" in data:
            size = (int(data["image_width"]), int(data["image_height"]))
        else:
            size = tuple(default_size)
        return cls(K, dist, size)

    def scaled(self, size: Tuple[int, int]) -> "CameraModel":
        """Та же камера для кадра другого разрешения (пропорциональное масштабирование)."""
        size = (int(size[0]), int(size[1]))
        if size == self.image_size:
            return self
        sx = size[0] / self.image_size[0]
        sy = size[1] / self.image_size[1]
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 1] *= sy
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        return CameraModel(K, self.dist, size)

    def _radial(self, r2):
        k1, k2, p1, p2, k3, k4, k5, k6 = self.dist
        return (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)))

    def _distort_normalized(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p1, p2 = self.dist[2], self.dist[3]
        r2 = x * x + y * y
        radial = self._radial(r2)
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return xd, yd

    def undistort_points(self, uv: np.ndarray, iterations: int = 30) -> np.ndarray:
        """Пиксели искажённого кадра -> нормализованные координаты без дисторсии (NaN, если необратимо)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        skew = self.K[0, 1]
        yd = (uv[:, 1] - cy) / fy
        xd = (uv[:, 0] - cx - skew * yd) / fx
        if not np.any(self.dist):
            return np.stack([xd, yd], axis=1)
        p1, p2 = self.dist[2], self.dist[3]
        x, y = xd.copy(), yd.copy()
        for _ in range(iterations):
            r2 = x * x + y * y
            radial = self._radial(r2)
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        # два шага Ньютона для точности у краёв кадра
        h = 1e-7
        for _ in range(2):
            fx0, fy0 = self._distort_normalized(x, y)
            ex, ey = fx0 - xd, fy0 - yd
            ax, ay = self._distort_normalized(x + h, y)
            bx, by = self._distort_normalized(x, y + h)
            j11, j21 = (ax - fx0) / h, (ay - fy0) / h
            j12, j22 = (bx - fx0) / h, (by - fy0) / h
            det = j11 * j22 - j12 * j21
            ok = np.abs(det) > 1e-12
            det = np.where(ok, det, 1.0)
            x = np.where(ok, x - (j22 * ex - j12 * ey) / det, x)
            y = np.where(ok, y - (-j21 * ex + j11 * ey) / det, y)
        # за радиусом обратимости (углы кадра) решения нет -> NaN
        r2 = x * x + y * y
        fx0, fy0 = self._distort_normalized(x, y)
        err_px = np.hypot(fx0 - xd, fy0 - yd) * self.focal_px
        bad = ~np.isfinite(err_px) | (err_px > 0.05) | (r2 > self.max_valid_r2)
        x = np.where(bad, np.nan, x)
        y = np.where(bad, np.nan, y)
        return np.stack([x, y], axis=1)

    @property
    def max_valid_r2(self) -> float:
        """r² (без дисторсии), до которого радиальная функция монотонна (обратима)."""
        if self._max_r2 is None:
            r = np.linspace(0, 3, 30001)
            with np.errstate(all="ignore"):
                rd = r * self._radial(r * r)
                deriv = np.gradient(rd, r)
            bad = np.flatnonzero(~np.isfinite(deriv) | (deriv <= 0.05))
            self._max_r2 = float(r[bad[0]] ** 2) if bad.size else float("inf")
        return self._max_r2

    def valid_fraction(self, step: int = 40) -> float:
        """Доля кадра, где дисторсия обратима (для проверки калибровки)."""
        w, h = self.image_size
        u, v = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
        xy = self.undistort_points(np.column_stack([u.ravel(), v.ravel()]))
        return float(np.isfinite(xy[:, 0]).mean())

    def project_normalized(self, xy: np.ndarray) -> np.ndarray:
        """Нормализованные координаты без дисторсии -> пиксели искажённого кадра."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        xd, yd = self._distort_normalized(xy[:, 0], xy[:, 1])
        u = self.K[0, 0] * xd + self.K[0, 1] * yd + self.K[0, 2]
        v = self.K[1, 1] * yd + self.K[1, 2]
        return np.stack([u, v], axis=1)

    def project_points(self, P: np.ndarray) -> np.ndarray:
        """Точки в системе камеры (x вправо, y вниз, z вперёд) -> пиксели; z<=0 -> NaN."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        z = np.where(P[:, 2] > 1e-9, P[:, 2], np.nan)
        return self.project_normalized(np.stack([P[:, 0] / z, P[:, 1] / z], axis=1))

    @property
    def focal_px(self) -> float:
        return float(np.sqrt(self.K[0, 0] * self.K[1, 1]))

    def __repr__(self) -> str:  # pragma: no cover
        return "CameraModel(size=%s, fx=%.1f, fy=%.1f, dist=%s)" % (
            self.image_size, self.K[0, 0], self.K[1, 1], np.round(self.dist[:5] if not np.any(self.dist[5:]) else self.dist, 4).tolist())


def model_to_image_coords(uv_model: np.ndarray, model_size: Tuple[int, int],
                          image_size: Tuple[int, int]) -> np.ndarray:
    """Координаты в сетке выхода сети (w_m, h_m) -> пиксели исходного кадра (центры пикселей)."""
    uv = np.asarray(uv_model, dtype=np.float64).reshape(-1, 2)
    sx = image_size[0] / model_size[0]
    sy = image_size[1] / model_size[1]
    return np.stack([(uv[:, 0] + 0.5) * sx - 0.5, (uv[:, 1] + 0.5) * sy - 0.5], axis=1)


def optional_camera(path: Optional[str], default_size=(2560, 1440)) -> Optional[CameraModel]:
    if not path:
        return None
    return CameraModel.from_file(path, default_size)
