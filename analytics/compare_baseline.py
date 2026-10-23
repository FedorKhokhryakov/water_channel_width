"""Сравнение с прежним расчётом ширины (analytics/baseline/channel_width.csv).

    python -m icechannel process --masks analytics/baseline/masks -c config/config.example.yaml \
        -o out/masks_all --no-overlays
    python analytics/compare_baseline.py --plot out/cmp.png
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

SECTIONS = ("w_100m", "w_150m", "w_200m", "w_250m", "w_300m")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="analytics/baseline/channel_width.csv")
    ap.add_argument("--new", default="out/masks_all/results", help="папка с CSV icechannel")
    ap.add_argument("--plot", help="сохранить график")
    a = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(a.new, "*.csv")))
    if not csvs:
        raise SystemExit("В %s нет CSV, сначала запустите process --masks (см. начало файла)" % a.new)
    old = pd.read_csv(a.old)
    old["key"] = old["file"].str[:15]
    new = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    new["key"] = new["file"].astype(str).str[:15]
    df = old.merge(new, on="key", suffixes=("_old", ""))
    print("Кадров: %d общих (прежних %d, новых %d)" % (len(df), len(old), len(new)))
    print("Статусы:", new["status"].value_counts().to_dict())

    ok = df[df["status"] == "ok"].copy()
    for col_new, col_old in [("width_median_m", "W_median_m"), ("width_near_m", "W_near_m"),
                             ("width_far_m", "W_far_m"), ("pitch_deg", "theta_deg")]:
        d = num(ok[col_new]) - num(ok[col_old])
        print("%-15s - %-11s: медиана %+.2f, MAD %.2f" % (
            col_new, col_old, d.median(), 1.4826 * (d - d.median()).abs().median()))

    has = num(ok["vessel_z_m"]).notna()
    print("Сечения, медиана (м):      " + "  ".join("%7s" % c for c in SECTIONS))
    for name, sel in (("судно сзади", has), ("без судна", ~has)):
        vals = [num(ok.loc[sel, c]).median() for c in SECTIONS]
        print("  %-12s (n=%4d): " % (name, sel.sum()) + "  ".join("%7.2f" % v for v in vals))

    if a.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].scatter(ok["W_median_m"], ok["width_median_m"], s=3)
        ax[0].plot([0, 80], [0, 80], "k--", lw=1)
        ax[0].set_xlabel("прежний расчёт, м"), ax[0].set_ylabel("icechannel, м")
        t = pd.to_datetime(ok["key"], format="%Y%m%d_%H%M%S")
        ax[1].scatter(t, ok["width_m"], s=2, label="width_m")
        ax[1].plot(t, num(ok["width_smooth_m"]), "r-", lw=0.8, label="сглаженная")
        ax[1].legend(), ax[1].set_ylabel("ширина, м")
        plt.tight_layout()
        plt.savefig(a.plot, dpi=120)
        print("График:", a.plot)


if __name__ == "__main__":
    main()
