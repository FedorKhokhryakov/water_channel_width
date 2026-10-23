"""Сканирование папки с zip-архивами и чтение кадра из архива в память."""
from __future__ import annotations

import fnmatch
import io
import json
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_TS_RE = re.compile(r"(\d{8})[_-]?(\d{6})")


def parse_timestamp(name: str) -> Optional[datetime]:
    """'20241018_092316_stern_lidar.zip' -> datetime(2024, 10, 18, 9, 23, 16)."""
    m = _TS_RE.search(Path(name).name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def decode_image(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def pick_member(names: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
    """Первый файл-изображение, подходящий под шаблоны (в порядке приоритета шаблонов)."""
    images = [n for n in names if n.lower().endswith(IMAGE_EXT) and not n.endswith("/")]
    for pat in patterns:
        pat = pat.lower()
        for n in sorted(images):
            if fnmatch.fnmatch(os.path.basename(n).lower(), pat):
                return n
    return None


def read_archive(path: Path, image_patterns: Sequence[str],
                 want_meta: bool = False) -> Tuple[Optional[str], Optional[bytes], Dict[str, Any]]:
    """-> (имя кадра, байты кадра, JSON из архива). BadZipFile пробрасывается."""
    meta: Dict[str, Any] = {}
    with zipfile.ZipFile(str(path)) as zf:
        names = zf.namelist()
        member = pick_member(names, image_patterns)
        data = zf.read(member) if member else None
        if want_meta:
            for n in names:
                if n.lower().endswith(".json"):
                    try:
                        meta[Path(n).stem] = json.loads(zf.read(n).decode("utf-8", "replace"))
                    except (ValueError, KeyError):
                        pass
    return member, data, meta


def extract_meta(meta: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """Достаёт значения по путям 'файл.ключ.подключ' (или 'ключ.подключ' — ищется во всех JSON)."""
    out: Dict[str, Any] = {}
    for key in keys:
        parts = key.split(".")
        roots = [meta.get(parts[0])] if parts[0] in meta else list(meta.values())
        tail = parts[1:] if parts[0] in meta else parts
        for root in roots:
            v: Any = root
            for p in tail:
                v = v.get(p) if isinstance(v, dict) else None
            if v is not None and not isinstance(v, (dict, list)):
                out[key] = v
                break
    return out


class InboxScanner:
    """Находит новые архивы и отдаёт только «устоявшиеся» (копирование завершено)."""

    def __init__(self, inbox: Path, pattern: str = "*.zip", recursive: bool = False,
                 settle_sec: float = 5.0):
        self.inbox = Path(inbox)
        self.pattern = pattern
        self.recursive = recursive
        self.settle_sec = float(settle_sec)
        self._last: Dict[str, Tuple[int, int]] = {}
        self.pending = 0  # сколько архивов ещё копируется (не устоялись)

    def _iter_files(self) -> Iterator[os.DirEntry]:
        stack = [str(self.inbox)]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_dir(follow_symlinks=False):
                            if self.recursive:
                                stack.append(e.path)
                        elif fnmatch.fnmatch(e.name.lower(), self.pattern.lower()):
                            yield e
            except FileNotFoundError:
                return

    def list_all(self) -> List[Path]:
        return [Path(e.path) for e in self._iter_files()]

    def ready(self, skip: Optional[set] = None) -> List[Path]:
        now = time.time()
        out: List[Tuple[float, str, Path]] = []
        seen: Dict[str, Tuple[int, int]] = {}
        self.pending = 0
        for e in self._iter_files():
            key = e.path
            if skip is not None and e.name in skip:
                continue
            try:
                st = e.stat()
            except FileNotFoundError:
                continue
            sig = (st.st_size, st.st_mtime_ns)
            seen[key] = sig
            stable_before = self._last.get(key) == sig
            old_enough = now - st.st_mtime >= self.settle_sec
            if st.st_size > 0 and old_enough and (stable_before or now - st.st_mtime >= 2 * self.settle_sec):
                ts = parse_timestamp(e.name)
                out.append((ts.timestamp() if ts else st.st_mtime, e.name, Path(e.path)))
            else:
                self.pending += 1
        self._last = seen
        out.sort()
        return [p for _, _, p in out]


def iter_local_inputs(paths: Sequence[str], pattern: str = "*") -> Iterator[Path]:
    """Файлы для разовой обработки: изображения, zip или папки с ними."""
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in IMAGE_EXT + (".zip",) and fnmatch.fnmatch(f.name, pattern):
                    yield f
        elif p.exists():
            yield p
