"""Скорость судна из логов icescanner внутри архивов (строки «NMEA dictionary: {...}»).

Скорость общая для судна, поэтому годятся архивы любых лидаров (fore, port, star, stern).
Из zip читается только icescanner.log, облака точек не распаковываются.

    python analytics/nmea_speed.py /mnt/f/sibir_datasets_2025 -o out/speed.csv
    python -m icechannel process ... --speed-csv out/speed.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

NMEA_RE = re.compile(r"NMEA dictionary: (\{.*\})")
TS_RE = re.compile(r"'timestamp': datetime\.datetime\(([\d, ]+)\)")


def _field(text: str, key: str):
    m = re.search(r"'%s': (?:Decimal\()?'?([^',)}]+)" % key, text)
    if not m or m.group(1) == "None":
        return None
    return m.group(1)


def parse_log(text: str):
    """-> записи {time_utc, speed_kn, course_deg, lat, lon} из текста лога."""
    for m in NMEA_RE.finditer(text):
        d = m.group(1)
        t = TS_RE.search(d)
        sp = _field(d, "ground_speed")
        if not t or sp is None:
            continue
        parts = [int(x) for x in t.group(1).split(",")]
        yield dict(time_utc=datetime(*parts), speed_kn=float(sp), course_deg=_field(d, "true_course"),
                   lat=_field(d, "latitude"), lon=_field(d, "longitude"))


STATS = {"zips": 0, "example": None}


def _is_log(name: str) -> bool:
    n = Path(name).name.lower()
    return n.endswith(".log") or ".log." in n  # icescanner.log, icescanner.log.1


def iter_logs(paths):
    """-> (имя архива, текст лога) из zip-архивов, папок с ними и отдельных .log."""
    for p in map(Path, paths):
        if not p.exists():
            print("нет такого пути:", p)
            continue
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_file() and _is_log(f.name):
                yield f.name, f.read_text("utf-8", "replace")
            elif f.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(f) as zf:
                        STATS["zips"] += 1
                        if STATS["example"] is None:
                            STATS["example"] = (f.name, zf.namelist()[:30])
                        for n in zf.namelist():
                            if _is_log(n):
                                yield f.name, zf.read(n).decode("utf-8", "replace")
                except zipfile.BadZipFile:
                    print("битый архив:", f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="папки с архивами, zip или .log")
    ap.add_argument("-o", "--out", default="out/speed.csv")
    ap.add_argument("--tz-hours", type=float, default=3.0,
                    help="время в базе NMEA — UTC, в именах архивов — UTC+3")
    a = ap.parse_args()

    recs, n_arch, n_logs = {}, set(), 0
    for name, text in iter_logs(a.inputs):
        n_logs += 1
        for r in parse_log(text):
            recs[r["time_utc"]] = r
            n_arch.add(name)
    if not recs:
        ex = STATS["example"]
        raise SystemExit("Записей NMEA не найдено: архивов %d, логов %d%s" % (
            STATS["zips"], n_logs, "\nСостав архива %s: %s" % ex if ex else ""))
    tz = timedelta(hours=a.tz_hours)
    rows = sorted(recs.values(), key=lambda r: r["time_utc"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_local", "time_utc", "speed_kn", "speed_mps", "course_deg", "lat", "lon"])
        for r in rows:
            w.writerow([r["time_utc"] + tz, r["time_utc"], r["speed_kn"], round(r["speed_kn"] * 0.514444, 3),
                        r["course_deg"] or "", r["lat"] or "", r["lon"] or ""])
    gaps = [(b["time_utc"] - a_["time_utc"]).total_seconds() / 60 for a_, b in zip(rows, rows[1:])]
    big = sum(g > 30 for g in gaps)
    print("Логов: %d, архивов с NMEA: %d, записей: %d" % (n_logs, len(n_arch), len(rows)))
    print("Период (местное): %s — %s" % (rows[0]["time_utc"] + tz, rows[-1]["time_utc"] + tz))
    print("Скорость, уз: мин %.1f, медиана %.1f, макс %.1f" % (
        min(r["speed_kn"] for r in rows), sorted(r["speed_kn"] for r in rows)[len(rows) // 2],
        max(r["speed_kn"] for r in rows)))
    print("Разрывов больше 30 мин: %d" % big)
    print("Сохранено:", a.out)


if __name__ == "__main__":
    main()
