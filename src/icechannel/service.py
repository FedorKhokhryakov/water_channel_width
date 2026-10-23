"""Сервис на судне: следит за папкой с архивами, обрабатывает новые, пишет результаты."""
from __future__ import annotations

import logging
import logging.handlers
import os
import shutil
import signal
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from . import __version__
from .archives import InboxScanner, decode_image, extract_meta, parse_timestamp, read_archive
from .camera import CameraModel
from .pipeline import FrameProcessor
from .segmenter import Segmenter
from .storage import ProcessedState, ResultStore

log = logging.getLogger("icechannel")


def setup_logging(cfg, log_dir: Optional[Path], foreground: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(cfg["logging"]["level"]).upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            str(log_dir / "service.log"), maxBytes=int(float(cfg["logging"]["max_mb"]) * 1e6),
            backupCount=int(cfg["logging"]["backups"]), encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if foreground:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)


class Engine:
    """Модель + геометрия + хранилище. Общий код для сервиса и разовой обработки."""

    def __init__(self, cfg, output: Optional[Path] = None, segmenter=None):
        self.cfg = cfg
        self.camera = CameraModel.from_file(cfg.path("calibration"),
                                            tuple(cfg["camera"]["calib_image_size"]))
        self.segmenter = segmenter or Segmenter(cfg.path("model"), cfg["model"]["threads"],
                                                cfg["model"]["tta_flip"])
        self.processor = FrameProcessor(cfg, self.camera, self.segmenter)
        self.store = ResultStore(cfg, output)
        self.colors = self.segmenter.meta["colors"]
        self._warm_start()

    def _warm_start(self) -> None:
        """После перезапуска восстанавливаем историю ширины и горизонта из последних CSV."""
        n = 0
        for row in self.store.recent_results(200):
            try:
                t = datetime.fromisoformat(row["time"]).timestamp()
                if row.get("status") == "ok" and row.get("width_m"):
                    self.processor.history.append((t, float(row["width_m"])))
                    n += 1
                if row.get("horizon_source") == "mask" and row.get("pitch_deg"):
                    self.processor.tracker.add(t, float(row["pitch_deg"]), float(row["roll_deg"]))
            except (KeyError, ValueError):
                continue
        if n:
            log.info("История восстановлена: %d кадров", n)

    def process_image(self, img, ts: Optional[datetime], name: str,
                      meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = self.processor.process_image(img, ts, name)
        out["result"]["model"] = self.segmenter.version
        out["result"]["meta"] = meta or ""
        self.store.save(out, img, self.colors)
        return out


class Service:
    def __init__(self, cfg, engine: Optional[Engine] = None):
        self.cfg = cfg
        self.scfg = cfg["scan"]
        self.engine = engine or Engine(cfg)
        self.store = self.engine.store
        self.state = ProcessedState(self.store.root / "state" / "processed.tsv")
        self.scanner = InboxScanner(cfg.path("inbox"), self.scfg["pattern"], self.scfg["recursive"],
                                    self.scfg["settle_sec"])
        self.stop_requested = False
        self.stats: Dict[str, Any] = dict(started=_now(), processed=0, ok=0, suspect=0, fail=0,
                                          errors=0, last_file="", last_width_m="", last_time="")
        self._bad_since: Dict[str, float] = {}

    def handle(self, path: Path) -> str:
        name = path.name
        ts = parse_timestamp(name) or datetime.fromtimestamp(path.stat().st_mtime)
        max_age = float(self.scfg["max_age_hours"])
        if max_age > 0 and (datetime.now() - ts).total_seconds() > max_age * 3600:
            return "skipped_old"
        try:
            if path.suffix.lower() == ".zip":
                _, data, meta_all = read_archive(path, self.scfg["image_patterns"],
                                                      bool(self.scfg["meta_keys"]))
            else:
                data, meta_all = path.read_bytes(), {}
        except zipfile.BadZipFile:
            first = self._bad_since.setdefault(name, time.time())
            if time.time() - first < float(self.scfg["bad_zip_retry_min"]) * 60:
                return "retry"
            log.error("Битый архив: %s", name)
            return "bad_zip"
        self._bad_since.pop(name, None)
        if data is None:
            log.warning("В архиве нет кадра по шаблонам %s: %s", self.scfg["image_patterns"], name)
            return "no_image"
        img = decode_image(data)
        meta = extract_meta(meta_all, self.scfg["meta_keys"]) if meta_all else {}
        out = self.engine.process_image(img, ts, name, meta)
        r = out["result"]
        log.info("%s %s W=%s м (сглаж. %s) pitch=%s roll=%s %s%s [%s мс]", name, r["status"],
                 r.get("width_m", "-"), r.get("width_smooth_m", "-"), r.get("pitch_deg"),
                 r.get("roll_deg"), r.get("horizon_source"),
                 (" " + r["flags"]) if r.get("flags") else (" " + r["reason"] if r.get("reason") else ""),
                 r.get("proc_ms"))
        self.stats.update(last_file=name, last_width_m=r.get("width_m", ""), last_time=r.get("time"))
        return r["status"]

    def _finish(self, path: Path, status: str) -> None:
        self.state.add(path.name, status)
        action = self.cfg["output"]["on_done"]
        try:
            if action == "delete":
                path.unlink()
            elif action == "move":
                dst = Path(self.cfg["output"]["done_dir"] or (self.store.root / "done"))
                dst.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dst / path.name))
        except OSError as exc:
            log.warning("Не удалось %s %s: %s", action, path.name, exc)

    def run(self, once: bool = False) -> None:
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        inbox = self.cfg.path("inbox")
        log.info("=== icechannel %s: старт, inbox=%s, output=%s ===", __version__, inbox, self.store.root)
        if not self.scfg["process_existing"]:
            n = 0
            for p in self.scanner.list_all():
                if p.name not in self.state:
                    self.state.add(p.name, "skipped_existing")
                    n += 1
            log.info("Пропущено существующих архивов: %d", n)
        waited = False
        while not self.stop_requested:
            if not inbox.exists():
                log.warning("Нет папки inbox: %s", inbox)
            else:
                for path in self.scanner.ready(skip=self.state.names):
                    if self.stop_requested:
                        break
                    try:
                        status = self.handle(path)
                    except Exception:  # один плохой кадр не должен останавливать сервис
                        log.exception("Ошибка обработки %s", path.name)
                        status = "error"
                        self.stats["errors"] += 1
                    if status == "retry":
                        continue
                    self._finish(path, status)
                    self.stats["processed"] += 1
                    if status in ("ok", "suspect", "fail"):
                        self.stats[status] += 1
                    self._write_status()
            self._write_status()
            if once:
                if self.scanner.pending and not waited:
                    log.info("Ещё копируются: %d архивов, жду %.0f с", self.scanner.pending,
                             2 * float(self.scfg["settle_sec"]) + 1)
                    self._sleep(2 * float(self.scfg["settle_sec"]) + 1)
                    waited = True
                    continue
                break
            self._sleep(float(self.scfg["interval_sec"]))
        log.info("=== остановка: обработано %d ===", self.stats["processed"])

    def _write_status(self) -> None:
        st = dict(self.stats, heartbeat=_now(), pid=os.getpid(), version=__version__,
                  free_mb=round(self.store.free_mb()))
        try:
            self.store.write_status(st)
        except OSError:
            pass

    def _sleep(self, sec: float) -> None:
        end = time.time() + sec
        while not self.stop_requested and time.time() < end:
            time.sleep(min(0.5, end - time.time()))

    def _on_signal(self, signum, frame) -> None:  # noqa: ARG002
        log.info("Получен сигнал %s, завершаю после текущего кадра", signum)
        self.stop_requested = True


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
