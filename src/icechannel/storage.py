"""Запись результатов (CSV, профили, маски, overlay) с лимитами места на диске."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

log = logging.getLogger("icechannel")

BASE_COLUMNS = [
    "time", "file", "status", "reason", "flags",
    "width_m", "width_smooth_m", "width_near_m", "width_far_m", "width_median_m",
    "width_row_median_m", "width_mad_m", "width_sigma_m",
]
TAIL_COLUMNS = [
    "n_rows", "rows_total", "rows_rejected_occluded", "z_min_m", "z_max_m", "trend_m_per_100m",
    "heading_deg", "centerline_offset_m", "vessel_z_m", "pitch_deg", "roll_deg", "horizon_source",
    "horizon_rms_px", "pitch_vp_deg", "edge_entropy", "slant_bottom_m", "model_ms", "proc_ms",
    "model", "meta",
]


def csv_columns(sections: Iterable[int]) -> List[str]:
    return BASE_COLUMNS + ["w_%dm" % int(s) for s in sections] + TAIL_COLUMNS


def _day(result: Dict[str, Any]) -> str:
    t = str(result.get("time") or "")
    return t[:10] if len(t) >= 10 else time.strftime("%Y-%m-%d")


def _stem(result: Dict[str, Any]) -> str:
    return Path(str(result.get("file") or "frame")).stem


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return round(v, 3) if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


class ProcessedState:
    """Список обработанных архивов: append-only TSV, в памяти — множество имён."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.names = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    name = line.split("\t", 1)[0].strip()
                    if name:
                        self.names.add(name)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def add(self, name: str, status: str) -> None:
        if name in self.names:
            return
        self.names.add(name)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("%s\t%s\t%s\n" % (name, status, time.strftime("%Y-%m-%d %H:%M:%S")))


class ResultStore:
    def __init__(self, cfg, output: Optional[Path] = None):
        self.cfg = cfg
        self.ocfg = cfg["output"]
        self.root = Path(output) if output else cfg.path("output")
        self.columns = csv_columns(cfg["measure"]["sections_m"])
        for sub in ("results", "profiles", "masks", "overlays", "state", "logs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._n = 0
        self._low_space_warned = False

    def free_mb(self) -> float:
        return shutil.disk_usage(str(self.root)).free / 1e6

    def low_space(self) -> bool:
        low = self.free_mb() < float(self.ocfg["min_free_mb"])
        if low and not self._low_space_warned:
            log.warning("Мало места (%.0f МБ свободно): маски и overlay не сохраняются", self.free_mb())
        self._low_space_warned = low
        return low

    @staticmethod
    def enforce_limit(folder: Path, max_mb: float) -> int:
        """Удаляет самые старые файлы, пока папка не уложится в лимит. -> число удалённых."""
        files = []
        total = 0
        for dirpath, _, names in os.walk(str(folder)):
            for n in names:
                p = os.path.join(dirpath, n)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, p))
                total += st.st_size
        limit = max_mb * 1e6
        removed = 0
        files.sort()
        for _, size, p in files:
            if total <= limit:
                break
            try:
                os.remove(p)
                total -= size
                removed += 1
            except OSError:
                pass
        for dirpath, _, _ in os.walk(str(folder), topdown=False):
            if dirpath != str(folder) and not os.listdir(dirpath):
                os.rmdir(dirpath)
        return removed

    def save(self, out: Dict[str, Any], img_rgb: Optional[np.ndarray], colors) -> None:
        res = out["result"]
        day = _day(res)
        if self.ocfg["csv"]:
            self._append_csv(self.root / "results" / ("%s.csv" % day), res)
        (self.root / "latest.json").write_text(
            json.dumps({k: _jsonable(v) for k, v in res.items()}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        if self.ocfg["profiles"] and "measure" in out["details"]:
            self._append_profile(self.root / "profiles" / ("%s.jsonl" % day), res, out["details"])
        if self.low_space():
            return
        self._n += 1
        if self.ocfg["masks"]:
            self._save_mask(self.root / "masks" / day / ("%s.png" % _stem(res)),
                            out["details"]["labels"], colors)
        every = max(1, int(self.ocfg["overlay_every"]))
        if self.ocfg["overlays"] and (res.get("status") != "ok" or self._n % every == 0):
            from .overlay import render

            try:
                im = render(img_rgb, out, colors, int(self.ocfg["overlay_width"]))
                p = self.root / "overlays" / day / ("%s.jpg" % _stem(res))
                p.parent.mkdir(parents=True, exist_ok=True)
                im.save(str(p), quality=int(self.ocfg["overlay_quality"]), optimize=True)
            except Exception:  # контрольная картинка не должна ронять обработку
                log.exception("Не удалось сохранить overlay")
        if self._n % 50 == 1:
            self.enforce_limit(self.root / "overlays", float(self.ocfg["overlays_max_mb"]))
            self.enforce_limit(self.root / "masks", float(self.ocfg["masks_max_mb"]))

    def _append_csv(self, path: Path, res: Dict[str, Any]) -> None:
        new = not path.exists()
        row = {k: res.get(k, "") for k in self.columns}
        if isinstance(row.get("meta"), dict):
            row["meta"] = json.dumps(row["meta"], ensure_ascii=False) if row["meta"] else ""
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    @staticmethod
    def _append_profile(path: Path, res: Dict[str, Any], det: Dict[str, Any]) -> None:
        rows = det["measure"]["rows"]
        g = det["ground"]

        def lst(a, nd):
            return [round(float(v), nd) if math.isfinite(float(v)) else None for v in a]

        rec = dict(time=res.get("time"), file=res.get("file"), Z=lst(rows["Z"], 1), W=lst(rows["W"], 2),
                   used=[bool(u) for u in rows["used"]], XL=lst(g["XL"], 2), XR=lst(g["XR"], 2),
                   centerline=[float(c) for c in det["measure"]["centerline"]])
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    @staticmethod
    def _save_mask(path: Path, labels: np.ndarray, colors) -> None:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(labels.astype(np.uint8), mode="P")
        pal = [c for rgb in colors for c in rgb]
        im.putpalette(pal + [0] * (768 - len(pal)))
        im.save(str(path), optimize=True)

    def recent_results(self, n: int = 100) -> List[Dict[str, str]]:
        files = sorted((self.root / "results").glob("*.csv"))[-2:]
        rows: List[Dict[str, str]] = []
        for f in files:
            with open(f, encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
        return rows[-n:]

    def write_status(self, status: Dict[str, Any]) -> None:
        p = self.root / "state" / "status.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(str(tmp), str(p))
