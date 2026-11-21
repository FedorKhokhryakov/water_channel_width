"""Сводка сжатия по результатам сервиса: баллы по дням и часам, график.

    python -m icechannel process --masks analytics/baseline/masks -c config/config.example.yaml \
        -o out/masks_all --no-overlays --speed 5
    python analytics/compression_report.py --plot out/compression.png
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

ORDER = ["0-1", "1", "1-2", "2", "2-3", "3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="out/masks_all/results", help="папка с CSV icechannel")
    ap.add_argument("--plot", help="сохранить график")
    a = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(a.results, "*.csv")))
    if not csvs:
        raise SystemExit("В %s нет CSV" % a.results)
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df["day"] = df["time"].dt.strftime("%Y-%m-%d")
    df["closing_note"] = df["closing_note"].fillna("")
    valid = df[df["closing_note"] == ""]
    src = df["speed_source"].dropna().unique().tolist()
    print("Скорость судна: %s" % (src or ["нет"]))
    print("Кадров: %d, годных для сжатия: %d" % (len(df), len(valid)))
    print("Почему не годны:", df.loc[df["closing_note"] != "", "closing_note"].value_counts().to_dict())

    rows = []
    for day, g in df.groupby("day"):
        v = g[g["closing_note"] == ""]
        sc = g["compression_score"].dropna()
        rows.append(dict(day=day, frames=len(g), valid=len(v),
                         closing_ms=round(v["closing_speed_ms"].median(), 3) if len(v) else None,
                         score=sc.mode().iloc[0] if len(sc) else "—"))
    print(pd.DataFrame(rows).to_string(index=False))

    if a.plot:
        import matplotlib.pyplot as plt

        thr = [0.03, 0.06, 0.11, 0.21, 0.29]
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.scatter(valid["time"], valid["closing_speed_ms"], s=4, alpha=0.4, label="кадр")
        sm = df.dropna(subset=["compression_speed_ms"])
        ax.plot(sm["time"], sm["compression_speed_ms"], "r.", ms=3, label="медиана за 30 мин")
        edges = [-0.15] + thr + [0.4]
        for i, s in enumerate(ORDER):
            if i:
                ax.axhline(edges[i], color="k", lw=0.5, ls=":")
            ax.text(1.005, (max(edges[i], -0.05) + edges[i + 1]) / 2, "%s б." % s, va="center",
                    fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_ylim(edges[0], edges[-1])
        ax.set_ylabel("скорость закрытия кромки, м/с")
        ax.set_title("Сжатие по закрытию канала (скорость судна: %s)" % ", ".join(map(str, src or ["нет"])))
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(a.plot, dpi=120)
        print("График:", a.plot)


if __name__ == "__main__":
    main()
