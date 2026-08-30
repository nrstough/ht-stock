"""Feature building for the demand model.

Everything here uses ONLY columns a real store could observe (see
data/README.md). Targets are log1p(sales), z-scored per item with train-period
statistics; sellout days are flagged as censored (sales is a lower bound of
demand there).

A store's export is not the simulator's CSV: it starts on a different date,
runs for a different number of days, is missing days, and carries items that
opened last month. So everything that used to be a module constant -- the data
path, the split boundaries, the holiday and weather vocabularies, the trend
denominator -- now lives in one plain, JSON-serializable FEATURE SPEC dict.
legacy_spec() holds the exact values model/artifacts/demandnet.pt was trained
with, three known defects included, and is the default whenever the frame is
the simulator's; spec_for_panel() derives honest values from a panel's own date
range. Both keep 6 context channels and, on the standard US calendar, 35
covariates in the same order, because the frozen checkpoint is shape-locked to
them.
"""
import argparse
import functools
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd

from ht import schema

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "store_synth.csv")

CONTEXT_DAYS = 28
TAUS = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99])

HOLIDAY_NAMES = [
    "new_years_day", "super_bowl", "valentines", "easter", "mothers_day",
    "memorial_day", "july4", "labor_day", "halloween", "thanksgiving_eve",
    "thanksgiving", "christmas_eve", "christmas", "new_years_eve",
]
WEATHER_KINDS = ["sunny", "cloudy", "rain", "snow"]

TRAIN_END = "2024-12-31"   # train+val period; test = 2025  (legacy, frozen)
VAL_START = "2024-11-04"   # last ~8 weeks of the train period

# Split floors. Each is a statement about what a metric needs to be stable:
# 28 test days is four observations of every weekday, 14 val days is two, and
# 84 train days is 28 warm-up plus 56 target days (eight of every weekday).
# They cohere exactly -- at 126 days the rule below yields 28/14/84 with none
# to spare, which is why MIN_PANEL_DAYS is 126 and not a round number.
MIN_PANEL_DAYS = 126
MIN_PANEL_DAYS_NO_TEST = 98
MIN_PANEL_DAYS_SHORT = 70
MIN_ITEM_TRAIN_DAYS = 84
MIN_TRAIN_DAYS = 84
MIN_TRAIN_DAYS_SHORT = 56
MAX_WINDOW_GAP_DAYS = 5
FEATURE_VERSION = 2

CTX_CHANNELS = ["sales_z", "stockout", "is_closed", "tmax_z", "rain", "snow"]

REQUIRED_COLUMNS = ("date", "item", "dow", "holiday", "payday", "is_closed",
                    "tmax_f", "weather", "snow_tomorrow", "sold", "stockout")


class InsufficientHistory(ValueError):
    """The panel (or every item in it) is too short to split and train."""


class EmptySplit(ValueError):
    """The resolved boundaries left a split with no usable rows."""


class SpecMismatch(ValueError):
    """A checkpoint's recorded feature spec does not match the one just built."""


def legacy_spec():
    """The exact feature configuration model/artifacts/demandnet.pt was trained with.

    Three of these fields name defects rather than choices, and they are pinned
    deliberately so the published numbers reproduce from a named value instead
    of from an accident:

    holiday_countdown="off" -- build() used to select holiday dates BEFORE
      filling the NaNs in `holiday`, and NaN != "" is True, so every date read
      as a holiday and covariate 26 came out identically 0.0. The checkpoint was
      trained with that covariate dead. Anyone normalizing the frame first would
      silently have woken it up and changed the published dollar figures.
    trend_days=1095.0 -- the hardcoded 3*365 denominator, which happens to equal
      the synthetic panel's own (max-min).days. trend_start pins the origin the
      denominator counts from, which used to be whatever frame build() was handed.
    stats_scope="train_val" -- normalization statistics run to 2024-12-31 while
      the validation window starts 2024-11-04, so the z-scoring sees val. Only
      first and second moments leak, never targets; spec_for_panel fixes it.
    """
    return {
        "version": FEATURE_VERSION,
        "context_days": CONTEXT_DAYS,
        "taus": [float(t) for t in TAUS],
        "holiday_names": list(HOLIDAY_NAMES),
        "weather_kinds": list(WEATHER_KINDS),
        "holiday_countdown": "off",
        "holiday_horizon": 21,
        "fourier_harmonics": 2,
        "fourier_period": 365.25,
        "include_trend": True,
        "trend_days": 1095.0,
        "trend_start": "2023-01-01",
        "stats_scope": "train_val",
        "stats_end": TRAIN_END,
        "train_end": TRAIN_END,
        "val_start": VAL_START,
        "test_start": "2025-01-01",
        "require_contiguous_context": False,
        "min_item_train_days": 0,
        "unknown_vocab": "raise",
    }


def spec_for_panel(df, *, val_days=None, test_days=None, no_test=False,
                   allow_short=False, **overrides):
    """Derive a feature spec from a panel's own date range and vocabulary."""
    dates = _target_dates(df)
    sp = resolve_splits(dates, context_days=CONTEXT_DAYS, val_days=val_days,
                        test_days=test_days, no_test=no_test, allow_short=allow_short)
    span = sp["span_days"]

    # An annual cosine cannot be identified from under a year of data, and a
    # second harmonic fitted on exactly one year is fitting that one year.
    harmonics = 2 if span >= 540 else (1 if span >= 365 else 0)

    train = df[pd.to_datetime(df["date"]) <= pd.Timestamp(sp["train_end"])]
    seen = {str(h) for h in train.get("holiday", pd.Series(dtype=str)).fillna("") if str(h)}
    extra = sorted(seen - set(HOLIDAY_NAMES))
    kinds = list(WEATHER_KINDS)
    if "unknown" in set(df.get("weather", pd.Series(dtype=str)).astype(str)):
        kinds.append("unknown")

    spec = {
        "version": FEATURE_VERSION,
        "context_days": CONTEXT_DAYS,
        "taus": [float(t) for t in TAUS],
        # append, never reorder: the first 14 one-hot slots must stay where the
        # checkpoint expects them, which also keeps cov_dim at 35 for any store
        # on the standard US calendar.
        "holiday_names": list(HOLIDAY_NAMES) + extra,
        "weather_kinds": kinds,
        "holiday_countdown": "days",
        "holiday_horizon": 21,
        "fourier_harmonics": harmonics,
        "fourier_period": 365.25,
        "include_trend": span >= 120,
        "trend_days": float(max(span - 1, 1)),
        # The trend covariate is (date - origin) / trend_days, so the ORIGIN is as much a
        # part of the layout as the denominator. Left to the scored frame, re-exporting the
        # same store with a shorter history shifts it silently and spec_hash cannot see it.
        "trend_start": str(pd.Timestamp(dates.min()).date()),
        "stats_scope": "train",
        "stats_end": sp["train_end"],
        "train_end": sp["train_end"],
        "val_start": sp["val_start"],
        "test_start": sp["test_start"],
        "require_contiguous_context": True,
        # the per-item floor has to move with the panel floor: --allow-short relaxes the
        # store-level requirement to 70 days, which leaves a 56-day train window, so an 84-day
        # per-item floor would exclude every item and the escape hatch could never succeed
        "min_item_train_days": MIN_TRAIN_DAYS_SHORT if allow_short else MIN_ITEM_TRAIN_DAYS,
        "unknown_vocab": "zero",
    }
    spec.update(overrides)
    return spec


def spec_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def load(path=DATA):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item", "date"]).reset_index(drop=True)
    return df


def _target_dates(df):
    """Dates that could carry a target: open, with a finite sold value."""
    d = pd.to_datetime(df["date"])
    keep = np.ones(len(df), dtype=bool)
    if "is_closed" in df:
        keep &= df["is_closed"].fillna(0).values.astype(float) == 0
    if "sold" in df:
        keep &= np.isfinite(pd.to_numeric(df["sold"], errors="coerce").values)
    if not keep.any():
        raise InsufficientHistory("panel has no open rows with a finite `sold` value")
    return d[keep].values


def resolve_splits(dates, *, context_days=CONTEXT_DAYS, val_days=None, test_days=None,
                   no_test=False, allow_short=False):
    """Chronological train/val/test boundaries derived from the panel's own range.

    test = 20% of the span capped into [28, 365], val = 10% capped into [14, 84],
    and whatever is left must clear the train floor. Boundaries are dates, so the
    three windows are contiguous with no gap and no overlap.
    """
    d = pd.to_datetime(pd.Series(np.asarray(dates))).dropna()
    if d.empty:
        raise InsufficientHistory("no dates to split")
    first, last = d.min().normalize(), d.max().normalize()
    span = int((last - first).days) + 1

    if allow_short:
        test_n, val_n, floor = 0, 14, MIN_TRAIN_DAYS_SHORT
    else:
        test_n = 0 if no_test else int(np.clip(round(0.20 * span), 28, 365))
        val_n = int(np.clip(round(0.10 * span), 14, 84))
        floor = MIN_TRAIN_DAYS
    if test_days is not None:
        test_n = int(test_days)
    if val_days is not None:
        val_n = int(val_days)

    train_n = span - test_n - val_n
    if train_n < floor:
        need = floor + val_n + test_n
        raise InsufficientHistory(
            f"panel covers {span} days ({first.date()}..{last.date()}); this split needs "
            f"{need} days ({context_days} context + {floor - context_days} train targets + "
            f"{val_n} val + {test_n} test). Ask the store for 104 weeks of item movement, "
            f"or pass --no-test (needs {MIN_PANEL_DAYS_NO_TEST}) or --allow-short "
            f"(needs {MIN_PANEL_DAYS_SHORT})."
        )

    day = pd.Timedelta(days=1)
    test_start = (last - (test_n - 1) * day) if test_n else None
    val_start = (test_start if test_start is not None else last + day) - val_n * day
    return dict(
        span_days=span,
        train_end=str((val_start - day).date()),
        val_start=str(val_start.date()),
        test_start=None if test_start is None else str(test_start.date()),
        thin=bool(allow_short),
    )


def train_stats(df, stats_end, scope="train", val_start=None, min_item_days=0):
    """Per-item mean/std of log1p(sales) on open days up to stats_end, + temp stats.

    scope names what that boundary covers: "train" (the derived path, stats_end ==
    train_end) or "train_val" (the legacy path, whose stats_end sits past val_start
    and so z-scores with the validation window included).
    """
    if scope not in ("train", "train_val"):
        raise ValueError(f"unknown stats scope {scope!r}")
    d = pd.to_datetime(df["date"])
    tr = df[(d <= pd.Timestamp(stats_end)) & (df.is_closed == 0) & np.isfinite(df.sold)]
    stats, skipped = {}, {}
    for item, grp in tr.groupby("item"):
        x = np.log1p(grp.sold.values)
        if len(x) < max(min_item_days, 1):
            skipped[item] = int(len(x))
            continue
        stats[item] = dict(mean=float(x.mean()), std=float(max(x.std(), 1e-3)))
    tmax_mean, tmax_std = float(tr.tmax_f.mean()), float(tr.tmax_f.std())
    if not np.isfinite(tmax_mean):          # a store with no weather feed at all
        tmax_mean = 0.0
    if not np.isfinite(tmax_std) or tmax_std <= 0:
        tmax_std = 1.0
    return dict(items=stats, tmax_mean=tmax_mean, tmax_std=tmax_std, skipped=skipped)


@functools.lru_cache(maxsize=32)
def _index_map(names):
    return {name: i for i, name in enumerate(names)}


def _one_hot(value, names, unknown_vocab, what):
    v = np.zeros(len(names))
    i = _index_map(tuple(names)).get(value, -1)
    if i < 0:
        if unknown_vocab == "raise":
            raise ValueError(f"{what} {value!r} is not in the spec vocabulary {list(names)}")
        return v                      # all zeros: unseen, not silently miscoded
    v[i] = 1.0
    return v


def context_matrix(grp, item_stats, spec=None):
    """(n, 6) float32 encoder context for one item's rows, in panel order.

    Channels: z-scored log1p sales, stockout, is_closed, tmax_z, rain, snow.
    grp must carry a tmax_z column, or item_stats must carry tmax_mean/tmax_std.
    """
    sold = pd.to_numeric(grp["sold"], errors="coerce").values.astype(float)
    z = (np.log1p(sold) - item_stats["mean"]) / item_stats["std"]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    if "tmax_z" in grp:
        tmax_z = grp["tmax_z"].values.astype(float)
    else:
        tmax_z = (grp["tmax_f"].values.astype(float) - item_stats["tmax_mean"]) \
            / item_stats["tmax_std"]
    tmax_z = np.nan_to_num(tmax_z, nan=0.0, posinf=0.0, neginf=0.0)
    weather = grp["weather"].astype(str).values
    return np.stack([
        z,
        grp["stockout"].values.astype(float),
        grp["is_closed"].values.astype(float),
        tmax_z,
        (weather == "rain").astype(float),
        (weather == "snow").astype(float),
    ], axis=1).astype(np.float32)


def covariate_vector(row, spec, start_date, days_to_holiday):
    """The known-at-forecast-time covariates for one item-day, in frozen order."""
    unknown = spec["unknown_vocab"]
    dow = np.zeros(7)
    dow[int(row.dow)] = 1.0
    doy = row.date.timetuple().tm_yday
    period = spec["fourier_period"]
    fourier = []
    for k in range(1, spec["fourier_harmonics"] + 1):
        fourier += [np.sin(2 * k * np.pi * doy / period), np.cos(2 * k * np.pi * doy / period)]
    holiday = row.holiday if isinstance(row.holiday, str) else ""
    hol = _one_hot(holiday, spec["holiday_names"], unknown, "holiday") if holiday \
        else np.zeros(len(spec["holiday_names"]))
    wx = _one_hot(str(row.weather), spec["weather_kinds"], unknown, "weather kind")
    horizon = spec["holiday_horizon"]
    countdown = 0.0 if spec["holiday_countdown"] == "off" \
        else min(days_to_holiday, horizon) / float(horizon)
    trend = [(row.date - start_date).days / spec["trend_days"]] if spec["include_trend"] else []
    return np.concatenate([
        dow, fourier, hol,
        [1.0 if holiday else 0.0],
        [countdown],
        [row.tmax_z],
        wx,
        [float(row.snow_tomorrow)], [float(row.payday)], trend,
    ]).astype(np.float32)


def cov_layout(spec):
    """{block: [start, stop]} for the covariate vector this spec produces."""
    widths = [
        ("dow", 7),
        ("fourier", 2 * spec["fourier_harmonics"]),
        ("holiday", len(spec["holiday_names"])),
        ("is_holiday", 1),
        ("holiday_countdown", 1),
        ("tmax_z", 1),
        ("weather", len(spec["weather_kinds"])),
        ("snow_tomorrow", 1),
        ("payday", 1),
        ("trend", 1 if spec["include_trend"] else 0),
    ]
    out, at = {}, 0
    for name, w in widths:
        out[name] = [at, at + w]
        at += w
    return out


def attach_lags(b, df):
    """Ridge's lag features, aligned onto b's rows. Stored as b["lags"]."""
    from . import baselines          # imported here: baselines imports features

    frame = df[df["item"].isin(b["items"])].sort_values(["item", "date"])
    frame = frame.reset_index(drop=True)          # _lag_features scatters by label
    lag_full = baselines._lag_features(frame, b["stats"])
    pos = {(it, d): i for i, (it, d) in enumerate(zip(frame.item.values, frame.date.values))}
    b["lags"] = lag_full[[pos[(it, d)] for it, d in zip(b["item"], b["date"])]]
    return b["lags"]


def _resolve_spec(df, spec, from_default):
    if spec is not None:
        return spec
    if from_default or any(c in df.columns for c in schema.SIM_ONLY):
        return legacy_spec()          # the simulator's panel: keep the frozen behaviour
    return spec_for_panel(df)


def _days_to_holiday_map(df, spec):
    """days-to-next-holiday per date, from the CALENDAR rather than the panel's own rows.

    A holiday falling just past the panel's last date is invisible in the panel, so every
    target in the final `holiday_horizon` days would read "no holiday within three weeks" --
    and those are exactly the days the test window ends on and the first served morning
    follows. The panel's own holiday column is unioned in so a store's local closures file
    still counts down.
    """
    if spec["holiday_countdown"] == "off":
        return {}
    from ht import calendar as ht_calendar

    cal = df[["date", "holiday"]].drop_duplicates("date").sort_values("date")
    horizon = int(spec["holiday_horizon"])
    ahead = ht_calendar.holiday_map(pd.Timestamp(cal.date.min()).date(),
                                    (pd.Timestamp(cal.date.max())
                                     + pd.Timedelta(days=horizon)).date())
    hol = sorted(set(cal.loc[cal.holiday.astype(str) != "", "date"])
                 | {np.datetime64(pd.Timestamp(d)) for d in ahead})
    days = ht_calendar.days_to_next_holiday(cal.date.values, np.array(hol, dtype="datetime64[ns]"),
                                            horizon=horizon)
    return dict(zip(cal.date.values, [int(v) for v in days]))


def _item_guard(df, spec, stats):
    """Split the roster into items with enough train history and items without."""
    d = pd.to_datetime(df["date"])
    train = df[(d <= pd.Timestamp(spec["train_end"])) & (df.is_closed == 0)
               & np.isfinite(df.sold)]
    open_days = train.groupby("item").size().to_dict()
    required = max(int(spec["min_item_train_days"]), 1)
    kept, excluded = [], []
    for item in sorted(df.item.unique()):
        n = int(open_days.get(item, 0))
        if n >= required and item in stats["items"]:
            kept.append(item)
        else:
            reason = "no training history" if item not in stats["items"] \
                else f"short history, {n} of {required} days"
            excluded.append(dict(item=item, open_train_days=n, required=required,
                                 reason=reason))
    return kept, excluded


def build(df=None, spec=None, path=None, stats=None):
    """Returns dict with tensorless numpy arrays for train/val/test splits.

    `stats` pins the per-item normalizers instead of refitting them from this frame. Every
    path that SCORES an existing checkpoint passes meta["stats"], because a z-scoring fitted
    on a different slice of the same store is a different model input for the same day.
    """
    from_default = df is None
    if df is None:
        df = load(path or DATA)
    spec = _resolve_spec(df, spec, from_default)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"panel is missing required column(s): {', '.join(missing)}")

    df = df.copy()
    df["holiday"] = df["holiday"].fillna("")
    if stats is None:
        stats = train_stats(df, spec["stats_end"], spec["stats_scope"],
                            spec["val_start"], spec["min_item_train_days"])
    df["tmax_z"] = ((df.tmax_f - stats["tmax_mean"]) / stats["tmax_std"]).fillna(0.0)

    items, excluded = _item_guard(df, spec, stats)
    if not items:
        rows = "\n".join(f"  {e['item']}: {e['open_train_days']} of {e['required']} days"
                         for e in excluded)
        raise InsufficientHistory(
            "no item has enough training history to forecast:\n" + rows)
    item_idx = {k: i for i, k in enumerate(items)}
    df = df[df["item"].isin(items)]

    start_date = pd.Timestamp(spec["trend_start"]) if spec.get("trend_start") else df.date.min()
    d2h = _days_to_holiday_map(df, spec)
    context_days = int(spec["context_days"])
    contiguous = bool(spec["require_contiguous_context"])
    known_col = "stockout_known" in df.columns

    rows = dict(item=[], date=[], iidx=[], ctx=[], cov=[], y=[], cens=[])
    dropped = {}
    for item, grp in df.groupby("item"):
        grp = grp.reset_index(drop=True)
        st = stats["items"][item]
        sales_z = (np.log1p(grp.sold.values) - st["mean"]) / st["std"]
        ctx_mat = context_matrix(grp, st, spec)
        dates = grp.date.values
        closed = grp.is_closed.values.astype(float)
        known = grp.stockout_known.values.astype(float) if known_col else None

        for t in range(context_days, len(grp)):
            row = grp.iloc[t]
            if row.is_closed or not np.isfinite(row.sold):
                continue
            if contiguous and not _window_ok(dates, closed, t, context_days):
                dropped[item] = dropped.get(item, 0) + 1
                continue
            rows["item"].append(item)
            rows["date"].append(row.date)
            rows["iidx"].append(item_idx[item])
            rows["ctx"].append(ctx_mat[t - context_days:t])
            rows["cov"].append(covariate_vector(row, spec, start_date, d2h.get(dates[t], 0)))
            rows["y"].append(sales_z[t])
            # censored only where the sellout flag could actually be evaluated
            rows["cens"].append(float(row.stockout) * (1.0 if known is None else known[t]))

    if not rows["y"]:
        raise EmptySplit("no target rows survived the context, closure and contiguity guards")

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
    out["split"] = _label_splits(out["date"], spec)
    _assert_occupancy(out, spec)

    sellout_source = "unknown"
    if "sellout_source" in df.columns and len(df):
        sellout_source = str(df.sellout_source.mode().iloc[0])
    known_share = 1.0 if not known_col else float(df.stockout_known.astype(float).mean())
    out.update(
        spec=spec, spec_hash=spec_hash(spec), cov_layout=cov_layout(spec),
        ctx_channels=list(CTX_CHANNELS),
        span_days=int((df.date.max() - df.date.min()).days) + 1,
        train_end=spec["train_end"], val_start=spec["val_start"],
        test_start=spec["test_start"], excluded_items=excluded, dropped_rows=dropped,
        censoring_known=bool(known_share > 0 and sellout_source != "none"),
        sellout_source=sellout_source,
    )
    attach_lags(out, df)
    return out


def _window_ok(dates, closed, t, context_days):
    """The 28 rows behind t really are the 28 calendar days behind t."""
    span = (dates[t] - dates[t - context_days]) / np.timedelta64(1, "D")
    if int(span) != context_days:
        return False
    return closed[t - context_days:t].sum() <= MAX_WINDOW_GAP_DAYS


def _label_splits(dates, spec):
    val_start = np.datetime64(spec["val_start"])
    if spec["test_start"] is None:
        return np.where(dates >= val_start, "val", "train")
    return np.where(dates >= np.datetime64(spec["test_start"]), "test",
                    np.where(dates >= val_start, "val", "train"))


def _assert_occupancy(out, spec):
    counts = {s: int((out["split"] == s).sum()) for s in ("train", "val", "test")}
    bad = [s for s in ("train", "val") if counts[s] == 0]
    if spec["test_start"] is not None and counts["test"] == 0:
        bad.append("test")
    val_dates = len(np.unique(out["date"][out["split"] == "val"]))
    if bad or val_dates < 7:
        raise EmptySplit(
            f"panel {np.datetime_as_string(out['date'].min(), unit='D')}.."
            f"{np.datetime_as_string(out['date'].max(), unit='D')} split at "
            f"train_end {spec['train_end']} / val_start {spec['val_start']} / test_start "
            f"{spec['test_start']} gives rows {counts} over {val_dates} val dates"
        )


def inverse_transform(z, item, stats):
    """z-scored log1p sales -> units."""
    st = stats["items"][item]
    return np.expm1(z * st["std"] + st["mean"])


def assert_compatible(meta, b):
    """Refuse to score a checkpoint against features it was not trained on."""
    if "spec" not in meta:
        # the frozen artifact records only dimensions; that is all we can check
        warnings.warn("meta.json has no feature spec: falling back to a dimension-only "
                      "compatibility check", stacklevel=2)
        diffs = _diff({"items": meta["items"], "taus": len(meta["taus"]),
                       "ctx_dim": meta["ctx_dim"], "cov_dim": meta["cov_dim"]},
                      {"items": b["items"], "taus": len(b["taus"]),
                       "ctx_dim": int(b["ctx"].shape[2]), "cov_dim": int(b["cov"].shape[1])})
    else:
        want, got = meta["spec"], b["spec"]
        diffs = _diff(
            {"items": meta["items"], "taus": [float(t) for t in meta["taus"]],
             "ctx_dim": meta["ctx_dim"], "cov_dim": meta["cov_dim"],
             "context_days": want["context_days"], "holiday_names": want["holiday_names"],
             "weather_kinds": want["weather_kinds"], "spec_hash": meta.get("spec_hash")},
            {"items": b["items"], "taus": [float(t) for t in b["taus"]],
             "ctx_dim": int(b["ctx"].shape[2]), "cov_dim": int(b["cov"].shape[1]),
             "context_days": got["context_days"], "holiday_names": got["holiday_names"],
             "weather_kinds": got["weather_kinds"], "spec_hash": b["spec_hash"]})
    if diffs:
        raise SpecMismatch("checkpoint and features disagree:\n" + "\n".join(diffs))


def _diff(want, got):
    return [f"  {k}: checkpoint {want[k]!r} != features {got[k]!r}"
            for k in want if want[k] is not None and want[k] != got[k]]


def main(argv=None):
    ap = argparse.ArgumentParser(description="build and describe the feature arrays")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--spec", choices=("legacy", "auto"), default=None)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    df = load(args.panel) if args.panel else None
    spec = None
    if args.spec == "legacy":
        spec = legacy_spec()
    elif args.spec == "auto":
        df = load() if df is None else df
        spec = spec_for_panel(df, no_test=args.no_test, allow_short=args.allow_short)
    b = build(df, spec=spec)

    for s in ("train", "val", "test"):
        m = b["split"] == s
        print(s, m.sum(), "rows | censored:", round(b["cens"][m].mean(), 3) if m.any() else "-")
    print("ctx shape", b["ctx"].shape, "| cov shape", b["cov"].shape)
    print(f"span {b['span_days']} days | train_end {b['train_end']} | "
          f"val_start {b['val_start']} | test_start {b['test_start']}")
    print("spec_hash", b["spec_hash"], "| sellout_source", b["sellout_source"],
          "| censoring_known", b["censoring_known"])
    print("cov layout", " ".join(f"{k}[{v[0]}:{v[1]}]" for k, v in b["cov_layout"].items()))
    for e in b["excluded_items"]:
        print(f"excluded {e['item']}: {e['reason']}")
    for item, n in sorted(b["dropped_rows"].items()):
        print(f"dropped {n} non-contiguous windows for {item}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(spec=b["spec"], spec_hash=b["spec_hash"],
                           splits={s: int((b["split"] == s).sum())
                                   for s in ("train", "val", "test")},
                           cov_layout=b["cov_layout"], ctx_channels=b["ctx_channels"],
                           excluded_items=b["excluded_items"],
                           dropped_rows=b["dropped_rows"]), f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
