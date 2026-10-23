"""Конфигурация: значения по умолчанию + YAML. Относительные пути — от папки конфига."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "paths": {
        "inbox": "/data/inbox",              # куда судовая система складывает zip-архивы
        "output": "/data/ice_channel",       # куда пишем результаты
        "model": "models/model.onnx",        # рядом должен лежать model.json (метаданные)
        "calibration": "calibration/stern_camera.yaml",
    },
    "scan": {
        "interval_sec": 10,
        "pattern": "*stern*.zip",            # какие архивы брать (glob по имени)
        "recursive": False,
        "settle_sec": 5,                     # файл не менялся столько секунд -> копирование завершено
        "process_existing": True,            # обработать то, что уже лежит в папке при старте
        "max_age_hours": 0,                  # 0 = без ограничения; иначе более старые архивы пропускаются
        "bad_zip_retry_min": 10,             # битый zip ждём столько минут (вдруг ещё копируется)
        "image_patterns": ["*stern_cameraupscale*", "*stern_camera*", "*.jpg", "*.jpeg", "*.png"],
        "meta_keys": [],                     # пути "a.b.c" в JSON из архива, которые надо сохранить в CSV
    },
    "model": {
        "threads": 2,                        # потоки onnxruntime (не занимаем весь судовой компьютер)
        "tta_flip": False,                   # усреднение с отражённым кадром (x2 время, чуть точнее)
    },
    "camera": {
        "height_m": 21.0,                    # высота камеры над ватерлинией (по чертежу)
        "pitch_ref_deg": 12.0,               # типичный наклон камеры вниз
        "pitch_tol_deg": 3.0,                # горизонт дальше этого от pitch_ref считается ошибкой
        "roll_max_deg": 4.0,                 # максимально правдоподобный крен
        "pitch_offset_deg": 0.0,             # постоянная поправка наклона (после калибровки по данным)
        "dip_correction": True,              # видимый горизонт ниже истинного (≈0.13° при 21 м)
        "calib_image_size": [2560, 1440],    # размер кадра, для которого задана калибровка
    },
    "horizon": {
        "x_range": [0.15, 0.85],             # столбцы кадра, по которым ищется горизонт
        "max_rms_px": 15.0,                  # СКО точек горизонта от прямой (px исходного кадра)
        "min_points": 12,
        "history_minutes": 30,               # запасной горизонт = медиана удачных за это время
    },
    "measure": {
        "z_max_m": 400.0,                    # дальше этой дистанции воду не используем
        "z_near_m": 250.0,                   # «ближняя» (надёжная) зона
        "sections_m": [100, 150, 200, 250, 300],
        "section_halfwidth_m": 20.0,         # полуширина окна усреднения сечения (растёт с дистанцией)
        "centerline_degree": 1,              # 1 — прямой канал, 2 — допускается изгиб
        "primary": "near",                   # какая ширина главная: near (до z_near) | all (до z_max)
        "min_rows": 5,
        "min_width_frac": 0.015,             # строки уже этой доли ширины кадра (вершина у горизонта) не берём
        "edge_margin_frac": 0.01,            # кромка у края кадра -> канал обрезан кадром, строка не берётся
        "center_tol_frac": 0.30,             # центр канала дальше этой доли от центра кадра -> не канал
        "occlusion_px": 2,                   # за кромкой должен быть лёд, а не судно/рамка (px сетки сети)
        "close_px": 3,                       # морфологическое закрытие маски воды (px сетки сети)
        "vessel_margin_m": 150.0,            # судно за кормой в канале: строки ближе этого к нему не берём
        "merge_gap_frac": 0.15,              # куски канала, разорванные льдом, склеиваются через такой разрыв
    },
    "quality": {
        "trend_max_m_per_100m": 12.0,        # |тренд ширины по дистанции| больше -> не канал / плохой горизонт
        "rel_mad_max": 0.20,
        "entropy_max": 0.45,                 # средняя нормированная энтропия softmax на кромках
        "jump_k": 5.0,                       # скачок относительно скользящей медианы, в MAD
        "smooth_window": 5,                  # сглаженная ширина: медиана последних N надёжных кадров
        "smooth_minutes": 20,
    },
    "output": {
        "csv": True,                         # results/YYYY-MM-DD.csv
        "profiles": True,                    # profiles/YYYY-MM-DD.jsonl — построчные ширины
        "masks": True,                       # masks/…png — маска в сетке сети (~5 КБ)
        "overlays": True,                    # overlays/…jpg — кадр с разметкой для контроля
        "overlay_every": 1,                  # сохранять каждый N-й overlay (подозрительные — всегда)
        "overlay_width": 1280,
        "overlay_quality": 75,
        "overlays_max_mb": 300,              # старые overlay удаляются сверх лимита
        "masks_max_mb": 200,
        "min_free_mb": 500,                  # меньше свободного места -> пишем только CSV
        "on_done": "keep",                   # что делать с входным архивом: keep | move | delete
        "done_dir": "",
    },
    "logging": {"level": "INFO", "max_mb": 5, "backups": 3},
    "web": {"host": "127.0.0.1", "port": 8585},
}


def load_yaml(path: Path) -> Dict[str, Any]:
    """YAML (через PyYAML) или JSON — по расширению."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml  # noqa: WPS433 — PyYAML нужен только для чтения конфигов

    return yaml.safe_load(text) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """dict с удобным разрешением путей."""

    base_dir: Path = Path(".")

    def path(self, key: str) -> Path:
        p = Path(str(self["paths"][key])).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    data: Dict[str, Any] = {}
    base = Path.cwd()
    if path:
        p = Path(path).expanduser().resolve()
        data = load_yaml(p)
        base = p.parent
    unknown = set(data) - set(DEFAULTS)
    if unknown:
        raise ValueError("Неизвестные секции в конфиге: %s" % ", ".join(sorted(unknown)))
    cfg = Config(deep_merge(DEFAULTS, data))
    if overrides:
        cfg = Config(deep_merge(cfg, overrides))
    cfg.base_dir = base
    return cfg
