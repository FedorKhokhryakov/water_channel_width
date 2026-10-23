"""Сегментация кадра ONNX-моделью; препроцессинг общий с training/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_META: Dict[str, Any] = {
    "input_size": [288, 512],                 # (h, w) входа сети
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "classes": ["background", "ice", "water", "ship"],
    "colors": [[0, 0, 0], [236, 236, 242], [43, 169, 236], [241, 47, 72]],
    # linear = cv2.INTER_LINEAR (текущая модель), area = усреднение по площади
    "resize": "linear",
    "input_name": "image",
    "output_name": "logits",
}

_RESIZE_CACHE: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, ...]] = {}


def _linear_taps(src: int, dst: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Индексы и веса как у cv2.resize(INTER_LINEAR): полупиксельные центры, без антиалиасинга."""
    scale = src / dst
    pos = (np.arange(dst, dtype=np.float64) + 0.5) * scale - 0.5
    pos = np.clip(pos, 0, src - 1)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, src - 1)
    w1 = (pos - i0).astype(np.float32)
    return i0, i1, w1


def resize_linear(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Билинейное уменьшение/увеличение, совместимое с cv2.INTER_LINEAR. img: HxWxC uint8 -> float32."""
    h, w = img.shape[:2]
    oh, ow = int(out_hw[0]), int(out_hw[1])
    key = (h, w, oh, ow)
    if key not in _RESIZE_CACHE:
        _RESIZE_CACHE[key] = _linear_taps(h, oh) + _linear_taps(w, ow)
    y0, y1, wy, x0, x1, wx = _RESIZE_CACHE[key]
    # сначала нужные строки, потом float: без копии всего кадра
    rows = img[y0].astype(np.float32) * (1.0 - wy)[:, None, None] + img[y1].astype(np.float32) * wy[:, None, None]
    return rows[:, x0] * (1.0 - wx)[None, :, None] + rows[:, x1] * wx[None, :, None]


def resize_area(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Усреднение по площади (антиалиасинг). Для кратного масштаба — точное блочное среднее."""
    h, w = img.shape[:2]
    oh, ow = int(out_hw[0]), int(out_hw[1])
    if h % oh == 0 and w % ow == 0:
        fy, fx = h // oh, w // ow
        return img.reshape(oh, fy, ow, fx, -1).mean(axis=(1, 3), dtype=np.float32)
    from PIL import Image  # BOX-фильтр Pillow = усреднение по площади

    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(img[..., c])).resize(
        (ow, oh), Image.BOX), dtype=np.float32) for c in range(img.shape[2])]
    return np.stack(chans, axis=-1)


def preprocess(img_rgb: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """RGB uint8 HxWx3 -> тензор 1x3xHxW float32 (ImageNet-нормализация, как в обучении)."""
    out_hw = tuple(meta["input_size"])
    if img_rgb.shape[:2] == out_hw:
        x = img_rgb.astype(np.float32)
    elif meta.get("resize", "linear") == "area":
        x = resize_area(img_rgb, out_hw)
    else:
        x = resize_linear(img_rgb, out_hw)
    x = x / 255.0
    x = (x - np.asarray(meta["mean"], np.float32)) / np.asarray(meta["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=axis, keepdims=True)
    return z


def load_meta(model_path: Path) -> Dict[str, Any]:
    meta = dict(DEFAULT_META)
    side = Path(model_path).with_suffix(".json")
    if side.exists():
        meta.update(json.loads(side.read_text(encoding="utf-8")))
    return meta


class Segmenter:
    """Обёртка над onnxruntime. predict() -> логиты (C, h, w) в сетке входа сети."""

    def __init__(self, model_path, threads: int = 2, tta_flip: bool = False,
                 meta: Optional[Dict[str, Any]] = None):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError("Нет файла модели: %s (см. training/export_onnx.py)" % self.model_path)
        self.meta = meta or load_meta(self.model_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_cpu_mem_arena = False  # меньше пиковой памяти на общем судовом компьютере
        self.session = ort.InferenceSession(str(self.model_path), sess_options=so,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.tta_flip = bool(tta_flip)
        self.version = self.meta.get("version") or self.model_path.stem

    @property
    def classes(self) -> List[str]:
        return list(self.meta["classes"])

    def class_index(self, name: str) -> int:
        return self.classes.index(name)

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        x = preprocess(img_rgb, self.meta)
        if self.tta_flip:
            batch = np.concatenate([x, x[..., ::-1]], axis=0)
            out = self._run(batch)
            return 0.5 * (out[0] + out[1][..., ::-1])
        return self._run(x)[0]

    def _run(self, x: np.ndarray) -> np.ndarray:
        try:
            return self.session.run(None, {self.input_name: np.ascontiguousarray(x)})[0]
        except Exception:
            if x.shape[0] == 1:
                raise
            return np.concatenate([self.session.run(None, {self.input_name: x[i:i + 1]})[0]
                                   for i in range(x.shape[0])], axis=0)
