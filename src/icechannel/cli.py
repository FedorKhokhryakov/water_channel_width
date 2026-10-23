"""ice-channel run | process | serve | check"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np

from . import __version__
from .config import load_config

log = logging.getLogger("icechannel")


def _cmd_run(args) -> int:
    from .service import Service, setup_logging

    cfg = load_config(args.config)
    out = cfg.path("output")
    setup_logging(cfg, out / "logs", foreground=not args.quiet)
    Service(cfg).run(once=args.once)
    return 0


def _load_mask(path: Path, grid_hw) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Готовая label-маска полного кадра -> сетка сети (выборка центров ячеек)."""
    from PIL import Image

    lab = np.asarray(Image.open(str(path)))
    if lab.ndim == 3:
        raise ValueError("%s: ожидается label-маска (индексы классов), а не RGB" % path.name)
    h, w = lab.shape
    gh, gw = grid_hw
    ys = np.clip(((np.arange(gh) + 0.5) * h / gh).astype(int), 0, h - 1)
    xs = np.clip(((np.arange(gw) + 0.5) * w / gw).astype(int), 0, w - 1)
    return lab[ys][:, xs].astype(np.uint8), (w, h)


def _cmd_process(args) -> int:
    from .archives import decode_image, iter_local_inputs, parse_timestamp, read_archive
    from .camera import CameraModel
    from .pipeline import FrameProcessor
    from .segmenter import DEFAULT_META, load_meta
    from .service import Engine, setup_logging
    from .storage import ResultStore

    cfg = load_config(args.config)
    if args.no_overlays:
        cfg["output"]["overlays"] = False
    out_dir = Path(args.out) if args.out else cfg.path("output")
    setup_logging(cfg, None, foreground=True)
    files = list(iter_local_inputs(args.inputs))
    files.sort(key=lambda p: (parse_timestamp(p.name) or datetime.min, p.name))
    if args.limit:
        files = files[: args.limit]
    log.info("Файлов: %d -> %s", len(files), out_dir)

    if args.masks:
        # без модели: измерение по готовым маскам
        model_path = cfg.path("model")
        meta = load_meta(model_path) if model_path.exists() else dict(DEFAULT_META)
        camera = CameraModel.from_file(cfg.path("calibration"), tuple(cfg["camera"]["calib_image_size"]))
        proc = FrameProcessor(cfg, camera, None, meta["classes"])
        store = ResultStore(cfg, out_dir)
        grid = tuple(meta["input_size"])
        for i, f in enumerate(files, 1):
            labels, size = _load_mask(f, grid)
            name = f.name.replace("_mask_label", "")
            out = proc.process_prediction(labels, None, size, parse_timestamp(f.name), name)
            out["result"]["model"] = "mask"
            img = None
            if args.images:
                stem = name.rsplit(".", 1)[0]
                for ext in (".jpg", ".jpeg", ".png"):
                    p = Path(args.images) / (stem + ext)
                    if p.exists():
                        img = decode_image(p.read_bytes())
                        break
            store.save(out, img, meta["colors"])
            if i % 200 == 0:
                log.info("%d/%d", i, len(files))
        log.info("Готово: %s", out_dir / "results")
        return 0

    engine = Engine(cfg, out_dir)
    t0 = time.time()
    for i, f in enumerate(files, 1):
        try:
            if f.suffix.lower() == ".zip":
                member, data, _ = read_archive(f, cfg["scan"]["image_patterns"])
                if data is None:
                    log.warning("%s: нет кадра", f.name)
                    continue
            else:
                data = f.read_bytes()
            r = engine.process_image(decode_image(data), parse_timestamp(f.name), f.name)["result"]
            log.info("[%d/%d] %s %s W=%s м %s", i, len(files), f.name, r["status"], r.get("width_m", "-"),
                     r.get("flags") or r.get("reason") or "")
        except Exception:
            log.exception("Ошибка: %s", f)
    log.info("Готово за %.1f с: %s", time.time() - t0, out_dir / "results")
    return 0


def _cmd_serve(args) -> int:
    from .web import serve

    cfg = load_config(args.config)
    serve(cfg, host=args.host or cfg["web"]["host"], port=args.port or int(cfg["web"]["port"]))
    return 0


def _cmd_check(args) -> int:
    from .camera import CameraModel
    from .segmenter import Segmenter

    cfg = load_config(args.config)
    ok = True

    def line(flag: bool, text: str) -> None:
        nonlocal ok
        ok &= flag
        print(("  OK   " if flag else "  FAIL ") + text)

    print("icechannel %s, Python %s" % (__version__, sys.version.split()[0]))
    try:
        cam = CameraModel.from_file(cfg.path("calibration"), tuple(cfg["camera"]["calib_image_size"]))
        line(True, "калибровка %s: %r" % (cfg.path("calibration"), cam))
    except Exception as exc:
        line(False, "калибровка: %s" % exc)
    inbox, out = cfg.path("inbox"), cfg.path("output")
    line(inbox.exists(), "inbox: %s" % inbox)
    try:
        out.mkdir(parents=True, exist_ok=True)
        line(True, "output: %s" % out)
    except OSError as exc:
        line(False, "output %s: %s" % (out, exc))
    try:
        seg = Segmenter(cfg.path("model"), cfg["model"]["threads"], cfg["model"]["tta_flip"])
        h, w = seg.meta["input_size"]
        img = (np.random.RandomState(0).rand(1440, 2560, 3) * 255).astype(np.uint8)
        seg.predict(img)
        t0 = time.perf_counter()
        for _ in range(3):
            seg.predict(img)
        dt = (time.perf_counter() - t0) / 3
        line(True, "модель %s (%.1f МБ), вход %dx%d, классы %s, %.0f мс/кадр" % (
            seg.model_path.name, seg.model_path.stat().st_size / 1e6, w, h, seg.classes, dt * 1000))
    except Exception as exc:
        line(False, "модель: %s" % exc)
    print("Итог:", "всё в порядке" if ok else "есть проблемы")
    return 0 if ok else 1


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ice-channel", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="сервис: следить за inbox")
    r.add_argument("-c", "--config", required=True)
    r.add_argument("--once", action="store_true", help="один проход по inbox и выход")
    r.add_argument("-q", "--quiet", action="store_true", help="не дублировать лог в консоль")
    r.set_defaults(func=_cmd_run)

    pr = sub.add_parser("process", help="разовая обработка файлов/папок")
    pr.add_argument("inputs", nargs="+", help="кадры, zip-архивы или папки")
    pr.add_argument("-c", "--config")
    pr.add_argument("-o", "--out", help="папка результатов (по умолчанию paths.output)")
    pr.add_argument("--masks", action="store_true",
                    help="входы — готовые label-маски полного кадра, модель не нужна")
    pr.add_argument("--images", help="папка с исходными кадрами для overlay в режиме --masks")
    pr.add_argument("--no-overlays", action="store_true")
    pr.add_argument("--limit", type=int, default=0)
    pr.set_defaults(func=_cmd_process)

    s = sub.add_parser("serve", help="веб-страница результатов")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(func=_cmd_serve)

    c = sub.add_parser("check", help="проверка установки")
    c.add_argument("-c", "--config", required=True)
    c.set_defaults(func=_cmd_check)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        sys.exit(2)
    sys.exit(args.func(args))

