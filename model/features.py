"""Feature building for the demand model.

Everything here uses ONLY columns a real store could observe (see
data/README.md). Targets are log1p(sales), z-scored per item with train-period
statistics; sellout days are flagged as censored (sales is a lower bound of
demand there).
"""
import json
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "store_synth.csv")

CONTEXT_DAYS = 28
TAUS = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99])

HOLIDAY_NAMES = [
    "new_years_day", "super_bowl", "valentines", "easter", "mothers_day",
    "memorial_day", "july4", "labor_day", "halloween", "thanksgiving_eve",
    "thanksgiving", "christmas_eve", "christmas", "new_years_eve",
]
WEATHER_KINDS = ["sunny", "cloudy", "rain", "snow"]

TRAIN_END = "2024-12-31"   # train+val period; test = 2025
VAL_START = "2024-11-04"   # last ~8 weeks of the train period


def load():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item", "date"]).reset_index(drop=True)
    return df


def train_stats(df):
    """Per-item mean/std of log1p(sales) on open train days, + global temp stats."""
    tr = df[(df.date <= TRAIN_END) & (df.is_closed == 0)]
    stats = {}
    for item, grp in tr.groupby("item"):
        x = np.log1p(grp.sold.values)
        stats[item] = dict(mean=float(x.mean()), std=float(max(x.std(), 1e-3)))
    return dict(items=stats,
                tmax_mean=float(tr.tmax_f.mean()), tmax_std=float(tr.tmax_f.std()))


def covariate_vector(row, start_date, days_to_holiday):
    dow = np.zeros(7); dow[int(row.dow)] = 1.0
    doy = row.date.timetuple().tm_yday
    fourier = [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
               np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25)]
    hol = np.zeros(len(HOLIDAY_NAMES))
    if row.holiday:
        hol[HOLIDAY_NAMES.index(row.holiday)] = 1.0
    wx = np.zeros(4); wx[WEATHER_KINDS.index(row.weather)] = 1.0
    trend = (row.date - start_date).days / (3 * 365.0)
    return np.concatenate([
        dow, fourier, hol,
        [1.0 if row.holiday else 0.0],
        [min(days_to_holiday, 21) / 21.0],
        [row.tmax_z],
        wx,
        [float(row.snow_tomorrow)], [float(row.payday)], [trend],
    ]).astype(np.float32)


def build(df=None):
    """Returns dict with tensless numpy arrays for train/val/test splits."""
    if df is None:
        df = load()
    stats = train_stats(df)
    start_date = df.date.min()

    # days-to-next-holiday lookup over the full calendar
    cal = df[["date", "holiday"]].drop_duplicates("date").sort_values("date")
    hol_dates = cal[cal.holiday != ""].date.values
    def days_to_next(d):
        future = hol_dates[hol_dates >= np.datetime64(d)]
        return int((future[0] - np.datetime64(d)) / np.timedelta64(1, "D")) if len(future) else 21

    df = df.copy()
    df["holiday"] = df["holiday"].fillna("")
    df["tmax_z"] = (df.tmax_f - stats["tmax_mean"]) / stats["tmax_std"]

    items = sorted(df.item.unique())
    item_idx = {k: i for i, k in enumerate(items)}

    rows = dict(item=[], date=[], iidx=[], ctx=[], cov=[], y=[], cens=[])
    for item, grp in df.groupby("item"):
        grp = grp.reset_index(drop=True)
        st = stats["items"][item]
        sales_z = (np.log1p(grp.sold.values) - st["mean"]) / st["std"]
        ctx_mat = np.stack([
            sales_z,
            grp.stockout.values.astype(float),
            grp.is_closed.values.astype(float),
            grp.tmax_z.values,
            (grp.weather == "rain").values.astype(float),
            (grp.weather == "snow").values.astype(float),
        ], axis=1).astype(np.float32)

        for t in range(CONTEXT_DAYS, len(grp)):
            row = grp.iloc[t]
            if row.is_closed:
                continue
            rows["item"].append(item)
            rows["date"].append(row.date)
            rows["iidx"].append(item_idx[item])
            rows["ctx"].append(ctx_mat[t - CONTEXT_DAYS:t])
            rows["cov"].append(covariate_vector(row, start_date, days_to_next(row.date)))
            rows["y"].append(sales_z[t])
            rows["cens"].append(float(row.stockout))

    out = dict(
        items=items, stats=stats, taus=TAUS,
        date=np.array(rows["date"], dtype="datetime64[ns]"),
        iidx=np.array(rows["iidx"], dtype=np.int64),
        ctx=np.stack(rows["ctx"]),
        cov=np.stack(rows["cov"]),
        y=np.array(rows["y"], dtype=np.float32),
        cens=np.array(rows["cens"], dtype=np.float32),
        item=np.array(rows["item"]),
    )
    d = out["date"]
    out["split"] = np.where(
        d > np.datetime64(TRAIN_END), "test",
        np.where(d >= np.datetime64(VAL_START), "val", "train"),
    )
    return out


def inverse_transform(z, item, stats):
    """z-scored log1p sales -> units."""
    st = stats["items"][item]
    return np.expm1(z * st["std"] + st["mean"])


if __name__ == "__main__":
    b = build()
    for s in ("train", "val", "test"):
        m = b["split"] == s
        print(s, m.sum(), "rows | censored:", round(b["cens"][m].mean(), 3))
    print("ctx shape", b["ctx"].shape, "| cov shape", b["cov"].shape)
