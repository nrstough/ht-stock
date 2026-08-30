"""Shadow mode: the daily loop, the paper it prints, and the record it leaves behind.

model/backtest.py answers "would this have worked?" against a simulator. This module
answers "is it working?" against a store, one morning at a time, and it exists because
the deliverable of four shadow weeks is not a model -- it is a page a district manager
believes. Believing it requires two things that are easy to lose and impossible to
recover afterwards:

  the forecast provably existed before the day did. Every quantile is appended to
  shadow/predictions.csv BEFORE the sheet is rendered, stamped with when it was made
  and how far the data ran, and every accuracy number in the weekly report is that
  file joined to actuals. Nothing is ever re-forecast. That is the whole difference
  between a proof and a story, and it is what lets the model be retrained mid-pilot
  without rewriting history.

  the day's verdict is frozen when the data arrived. score_day writes once; a later
  corrected export is recorded in _revisions.csv and the original stands. If the
  four-week totals move every time IT re-runs an extract, nobody will believe any of
  them.

The other half is the sheet itself, which someone reads in a walk-in cooler at 5:30am.
It prints the manager's own trailing par next to the model's MAKE, because a number
with nothing to compare it to reads as an instruction rather than an argument; and the
WHY column is generated from calendar and weather facts only, never a model
attribution, because "Fri; rain" can be checked against a window and a wrong
attribution destroys trust faster than a wrong number.

Nothing here can see true demand. Accuracy comes from model/evaluate.py's
observable-only metrics, and the economics are the same measured-or-bounded pair.
"""
import argparse
import collections
import csv
import datetime as dt
import json
import math
import os
import re
import sys
import uuid
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from ht import calendar as ht_calendar
from ht import config as ht_config
from ht import schema

from . import evaluate, features, newsvendor
from .net import DemandNet

MAX_STALENESS_DAYS = 2       # older than this and yesterday's context is a guess
SHEET_WIDTH = 80             # a receipt printer and a back-office PC both do 80
WEEKLY_WIDTH = 100
PAR_WEEKS = 4                # the store's own habit: trailing four same weekdays
TREND_WEEKS = 8              # how far back the WHY clauses look

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

SHADOW_BANNER = "SHADOW MODE - DO NOT CHANGE WHAT YOU MAKE TODAY."
# the printed ITEM column. _item_index registers the truncated name too, because a name the
# sheet prints has to be a name `enter` accepts back.
SHEET_ITEM_WIDTH = 18
NO_SELLOUT_CAVEAT = ("No sellout data: quantities run low on the busiest days "
                     "(size not measured).")

OVERRIDE_COLUMNS = ["date", "item", "rec_qty", "actual_produced", "sold_out_at", "note",
                    "entered_by", "entered_ts", "sellout_source"]
SCORE_COLUMNS = ["for_date", "item", "rec_qty", "par_qty", "p20", "p50", "p90", "sold",
                 "produced", "stockout", "stockout_known", "sellout_source", "wasted",
                 "is_closed", "row_status", "abs_err", "signed_err", "pinball",
                 "waste_actual_units", "waste_model_units", "lost_lower_units", "source",
                 "status"]
REVISION_COLUMNS = ["revised_at", "for_date", "item", "field", "old_value", "new_value"]


def _qname(tau):
    """q_0.05 ... q_0.975 -- two decimals unless the tau needs three."""
    s = f"{float(tau):.3f}".rstrip("0")
    frac = s.split(".")[1]
    return "q_" + s + "0" * max(0, 2 - len(frac))


def quantile_columns(taus):
    return [_qname(t) for t in taus]


PREDICTION_COLUMNS = (
    ["run_id", "made_at", "for_date", "store", "item", "item_name", "dept", "panel_through",
     "model_version", "spec_hash", "items_config_hash", "sellout_source"]
    + quantile_columns(features.TAUS)
    + ["q_star", "rec_qty", "par_qty", "batch", "continuous", "unit", "source",
       "fallback_reason", "why_text", "backfilled"]
)


# ---- the checkpoint's own view of the world ----

def _load_meta(artifacts_dir):
    with open(os.path.join(artifacts_dir, "meta.json")) as f:
        return json.load(f)


def _spec_for(meta, panel, artifacts_dir):
    """Score with the spec the checkpoint was trained on, never one re-derived here.

    A sheet printed this morning must z-score with the same statistics and one-hot the
    same vocabularies as training did; re-deriving them from today's panel is exactly
    the drift shadow mode would hide for four weeks.

    A checkpoint whose meta.json records no spec is the one case where there is nothing
    to score with, only the legacy layout to assume. features.spec_from_meta checks that
    assumption against the panel and refuses when it fails -- on a 2026 panel the legacy
    trend covariate reaches 1.4, a value the network never saw, and the whole sheet's
    quantities would be extrapolation printed as a recommendation.
    """
    return features.spec_from_meta(meta, panel, artifacts_dir)


def _round_batch(q, batch, continuous):
    return newsvendor.quantity(np.array([q, q]), np.array([0.0, 1.0]), 0.5, batch, continuous)


def _prepare(panel, stats):
    """Panel copy with tmax_z attached, holidays normalized, sorted per item."""
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    tmax = pd.to_numeric(df.get("tmax_f"), errors="coerce")
    df["tmax_z"] = ((tmax - stats["tmax_mean"]) / stats["tmax_std"]).fillna(0.0)
    if "holiday" in df:
        df["holiday"] = df["holiday"].fillna("").astype(str)
    for col in ("stockout", "is_closed", "payday", "snow_tomorrow"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.sort_values(["item", "date"]).reset_index(drop=True)


def _day_row(df, for_date, history):
    """Tomorrow's known covariates, as a namespace that has no `sold` to read.

    Preferred source is a panel row dated for_date -- a horizon row, or (when
    replaying) the real day, whose sales columns are deliberately not carried across.
    With no such row the calendar is derived and the weather persists from the last
    observed day, which is a worse forecast and says so in the caveats.
    """
    rows = df[df["date"] == for_date]
    if len(rows):
        r = rows.iloc[0]
        return SimpleNamespace(
            date=for_date, dow=int(r["dow"]), holiday=str(r.get("holiday") or ""),
            payday=float(r.get("payday") or 0.0), weather=str(r.get("weather", "unknown")),
            tmax_f=float(pd.to_numeric(r.get("tmax_f"), errors="coerce")),
            tmax_z=float(r["tmax_z"]), snow_tomorrow=float(r.get("snow_tomorrow") or 0.0),
        ), "panel"

    cal = ht_calendar.annotate(pd.DataFrame({"date": [for_date]})).iloc[0]
    last = history.sort_values("date").iloc[-1]
    return SimpleNamespace(
        date=for_date, dow=int(cal["dow"]), holiday=str(cal["holiday"] or ""),
        payday=float(cal["payday"]), weather=str(last.get("weather", "unknown")),
        tmax_f=float(pd.to_numeric(last.get("tmax_f"), errors="coerce")),
        tmax_z=float(last["tmax_z"]), snow_tomorrow=0.0,
    ), "carried_forward"


def _days_to_holiday(for_date, spec):
    if spec["holiday_countdown"] == "off":
        return 0
    horizon = int(spec["holiday_horizon"])
    end = (for_date + pd.Timedelta(days=horizon)).date()
    hol = sorted(ht_calendar.holiday_map(for_date.date(), end))
    return int((pd.Timestamp(hol[0]) - for_date).days) if hol else horizon


def par_quantity(panel, item, for_date, items):
    """The store's own habit: trailing four same-weekday sales, OPEN DAYS ONLY.

    Deliberately not model.baselines.naive_forecast, which applies no is_closed filter
    and so averages a Christmas zero into the par for four consecutive same-weekday
    targets a year. A printed sheet must not inherit that.
    """
    for_date = pd.Timestamp(for_date).normalize()
    d = pd.to_datetime(panel["date"])
    sold = pd.to_numeric(panel["sold"], errors="coerce")
    keep = ((panel["item"].astype(str) == str(item)) & (d < for_date)
            & (pd.to_numeric(panel.get("is_closed", 0), errors="coerce").fillna(0) == 0)
            & np.isfinite(sold.values) & (d.dt.dayofweek == for_date.dayofweek))
    past = sold[keep.values].values[-PAR_WEEKS:]
    if not len(past):
        return float("nan")
    it = items.get(str(item), {})
    return _round_batch(float(np.mean(past)), float(it.get("batch", 1.0)),
                        bool(it.get("continuous", False)))


def _same_dow_median(history, dow):
    h = history[pd.to_datetime(history["date"]).dt.dayofweek == dow].tail(TREND_WEEKS)
    v = pd.to_numeric(h["sold"], errors="coerce").dropna().values
    return float(np.median(v)) if len(v) else float("nan")


def _dow_rank(history, dow):
    """Where this weekday ranks among the seven for this item over the last 8 weeks."""
    h = history.tail(7 * TREND_WEEKS)
    means = pd.to_numeric(h["sold"], errors="coerce").groupby(
        pd.to_datetime(h["date"]).dt.dayofweek).mean().dropna()
    if dow not in means.index or len(means) < 4:
        return None
    return int((means.sort_values(ascending=False).index.get_loc(dow))) + 1


def why_text(row, history):
    """At most two clauses, from covariates only, in a fixed priority order.

    Never a model attribution: every clause here can be checked against a calendar, a
    window or last week's sheet, and one that cannot be checked is worse than none.
    """
    if str(row.get("source")) == "par_fallback":
        return f"par ({row.get('fallback_reason')})"

    clauses = []
    dow = int(row.get("dow", 0))
    holiday = str(row.get("holiday") or "")
    if holiday:
        clauses.append(holiday.replace("_", " "))

    recent = history.tail(7)
    n_out = int(pd.to_numeric(recent.get("stockout"), errors="coerce").fillna(0).sum()) \
        if "stockout" in recent else 0
    if len(clauses) < 2 and n_out >= 2:
        clauses.append(f"sold out {n_out} of last 7")

    weather = str(row.get("weather", ""))
    if len(clauses) < 2 and weather in ("rain", "snow"):
        med = _same_dow_median(history, dow)
        rec = float(row.get("rec_qty", float("nan")))
        if np.isfinite(med) and med > 0 and np.isfinite(rec):
            clauses.append(f"{weather} ({(rec / med - 1) * 100:+.0f}% vs usual {DAY_ABBR[dow]})")
        else:
            clauses.append(weather)

    if len(clauses) < 2 and float(row.get("snow_tomorrow") or 0.0) > 0:
        clauses.append("snow forecast tomorrow")

    if len(clauses) < 2:
        rank = _dow_rank(history, dow)
        if rank:
            clauses.append(f"{DAY_ABBR[dow]} is #{rank} of 7 for this item")
        else:
            clauses.append(DAY_ABBR[dow])
    return "; ".join(clauses[:2])


# ---- the forecast ----

def forecast(panel, artifacts_dir, items, for_date, allow_backfill=False,
             max_staleness=MAX_STALENESS_DAYS):
    """One row per active item for `for_date`, in print order.

    Context comes from each item's rows STRICTLY BEFORE for_date; the day's covariates
    come from a row dated for_date whose sales columns are never carried across. An
    item the checkpoint never saw, or one without a full context window, gets its
    trailing par with source="par_fallback" rather than a second unvalidated forecast
    printed with the model's authority.
    """
    for_date = pd.Timestamp(for_date).normalize()
    schema.assert_no_truth(panel)      # hard rule 6, structural rather than conventional
    meta = _load_meta(artifacts_dir)
    spec = _spec_for(meta, panel, artifacts_dir)
    stats = meta["stats"]
    ctx_days = int(spec["context_days"])
    taus = np.asarray(meta["taus"], dtype=float)

    df = _prepare(panel, stats)
    observed = df[np.isfinite(pd.to_numeric(df["sold"], errors="coerce").values)]
    if observed.empty:
        raise ValueError("panel carries no rows with an observed `sold` value")

    last = observed["date"].max()
    backfilled = bool(last >= for_date)
    if backfilled and not allow_backfill:
        raise ValueError(
            f"panel carries actuals through {last.date()}, on or after the forecast date "
            f"{for_date.date()}: a morning sheet has to be made before the day it is for. "
            f"Pass --backfill to replay a past date; the prediction log stamps it "
            f"backfilled=1 and every headline number excludes it.")

    history = observed[observed["date"] < for_date]
    if history.empty:
        raise ValueError(f"panel has no rows before {for_date.date()}")
    panel_through = history["date"].max()
    stale = int((for_date - pd.Timedelta(days=1) - panel_through).days)
    if stale > max_staleness:
        raise ValueError(
            f"panel runs through {panel_through.date()}, {stale} days behind the day before "
            f"{for_date.date()}; the trailing {ctx_days}-day context would be a guess. "
            f"Ingest the missing days, or raise --max-staleness deliberately.")

    day, day_source = _day_row(df, for_date, history)
    start_date = pd.Timestamp(meta.get("panel_start") or df["date"].min())
    cov = features.covariate_vector(day, spec, start_date, _days_to_holiday(for_date, spec))
    roster = {k: i for i, k in enumerate(meta["items"])}
    # why an item is not in the roster is recorded at training time; "not in the trained
    # model" is true of the tool and useless to a kitchen manager, while "new item, 31 of 84
    # days" says the item gets a forecast in eight weeks
    why_excluded = {str(e["item"]): str(e.get("reason") or "")
                    for e in meta.get("excluded_items", [])}
    q_star = ht_config.critical_fractiles(items)

    ctxs, idx, keyed, records, warnings = [], [], [], [], []
    for key in sorted(items):
        it = items[key]
        grp = history[history["item"].astype(str) == key]
        base = dict(item=key, item_name=it["name"], dept=it["dept"],
                    batch=float(it["batch"]), continuous=bool(it.get("continuous", False)),
                    unit=str(it.get("unit", "each")),
                    shelf_life_days=int(it.get("shelf_life_days", 1)),
                    q_star=float(q_star[key]), dow=int(day.dow), holiday=day.holiday,
                    weather=day.weather, snow_tomorrow=day.snow_tomorrow,
                    par_qty=par_quantity(history, key, for_date, items))
        if key not in roster:
            records.append(dict(base, source="par_fallback",
                                fallback_reason=why_excluded.get(key)
                                or "not in the trained model"))
            continue
        if len(grp) < ctx_days:
            records.append(dict(base, source="par_fallback",
                                fallback_reason=f"only {len(grp)} of {ctx_days} days"))
            continue
        window = grp.tail(ctx_days)
        span = int((for_date - window["date"].iloc[0]).days)
        if span != ctx_days:
            warnings.append(f"{key}: the {ctx_days} rows behind {for_date.date()} span "
                            f"{span} calendar days")
        st = dict(stats["items"][key], tmax_mean=stats["tmax_mean"], tmax_std=stats["tmax_std"])
        ctxs.append(features.context_matrix(window, st))
        idx.append(roster[key])
        keyed.append(key)
        records.append(dict(base, source="model", fallback_reason=""))

    q_units = {}
    if ctxs:
        model = DemandNet(len(meta["items"]), meta["ctx_dim"], meta["cov_dim"],
                          len(meta["taus"]))
        model.load_state_dict(torch.load(os.path.join(artifacts_dir, "demandnet.pt"),
                                         weights_only=True))
        model.eval()
        with torch.no_grad():
            z = model(torch.tensor(np.array(idx), dtype=torch.int64),
                      torch.tensor(np.stack(ctxs)),
                      torch.tensor(np.tile(cov, (len(ctxs), 1)))).numpy()
        for key, zrow in zip(keyed, z):
            st = stats["items"][key]
            q_units[key] = np.expm1(zrow * st["std"] + st["mean"]).clip(min=0)

    cols = quantile_columns(taus)
    for rec in records:
        key = rec["item"]
        if key in q_units:
            row_q = q_units[key]
            rec.update({c: float(v) for c, v in zip(cols, row_q)})
            rec["rec_qty"] = newsvendor.quantity(row_q, taus, rec["q_star"], rec["batch"],
                                                 rec["continuous"])
        else:
            rec.update({c: float("nan") for c in cols})
            rec["rec_qty"] = rec["par_qty"]
        hist = history[history["item"].astype(str) == key]
        rec["why_text"] = why_text(rec, hist)

    out = pd.DataFrame(records)
    out["for_date"] = for_date
    out["panel_through"] = panel_through
    out["model_version"] = evaluate.model_version(artifacts_dir)
    out["spec_hash"] = meta.get("spec_hash") or features.spec_hash(spec)
    out["sellout_source"] = _panel_sellout_source(df)
    out["backfilled"] = int(backfilled)
    out["value"] = out["rec_qty"].fillna(0) * [float(items[k]["price"]) for k in out["item"]]
    out = out.sort_values(["dept", "value"], ascending=[True, False]).reset_index(drop=True)
    out.attrs.update(day_source=day_source, warnings=warnings, staleness_days=stale,
                     conditions=dict(tmax_f=day.tmax_f, weather=day.weather,
                                     holiday=day.holiday, payday=bool(day.payday),
                                     snow_tomorrow=bool(day.snow_tomorrow)))
    return out


def _panel_sellout_source(df):
    if "sellout_source" not in df or df.empty:
        return "unknown"
    return str(df["sellout_source"].astype(str).mode().iloc[0])


# ---- the sheet ----

def _fmt_qty(v, unit="each"):
    """Pieces are printed as pieces. Only a weighed item ever shows a decimal."""
    if v is None or not np.isfinite(float(v)):
        return "-"
    v = float(v)
    return f"{v:,.1f}" if unit == "lb" else f"{int(round(v)):,d}"


def _fmt_span(lo, hi, unit):
    return f"{_fmt_qty(lo, unit)}..{_fmt_qty(hi, unit)}"


def _conditions_line(conditions):
    tmax = conditions.get("tmax_f")
    bits = ["High " + (f"{tmax:.0f}F" if tmax is not None and np.isfinite(tmax) else "unknown")
            + f", {conditions.get('weather', 'unknown')}."]
    bits.append("Snow forecast tomorrow: " + ("YES." if conditions.get("snow_tomorrow") else "no."))
    holiday = conditions.get("holiday") or ""
    bits.append("Holiday: " + (holiday.replace("_", " ") if holiday else "none") + ".")
    if conditions.get("payday"):
        bits.append("PAYDAY.")
    return " ".join(bits)


def _fallback_reason(row):
    """Why this item has no forecast. A par_fallback row from forecast() carries
    fallback_reason; an excluded_items dict from features.build carries reason. Both mean the
    same thing, and a bare par with no reason is a number a kitchen manager cannot argue with.
    """
    return str(row.get("fallback_reason") or row.get("reason") or "")


def _sheet_rows(recs, multi_day=False):
    """(dept, [row dicts]) in print order, model rows only, one shelf-life class at a time.

    Multi-day items are split out because the MAKE beside them is a one-day newsvendor
    quantity and they carry over, so it is not an order -- and a MAKE column that mixes the
    two teaches the kitchen to distrust the whole page.
    """
    out = []
    live = recs[(recs["source"] == "model")
                & ((pd.to_numeric(recs["shelf_life_days"], errors="coerce").fillna(1) > 1)
                   == bool(multi_day))]
    for dept in sorted(live["dept"].unique()):
        out.append((dept, live[live["dept"] == dept].to_dict("records")))
    return out


def _carry_over_rows(recs):
    """The model rows for items that keep, in print order, flattened across departments."""
    return [r for _, rows in _sheet_rows(recs, multi_day=True) for r in rows]


def _carry_over_why(row):
    """Why this item's MAKE is not an order, in one clause the kitchen can act on."""
    return (f"{int(row['shelf_life_days'])}-day shelf life: MAKE is one day's demand and "
            f"does not subtract what is already on the shelf; {row['why_text']}")


def _wrap(text, width, indent=""):
    words, lines, cur = str(text).split(), [], indent
    for w in words:
        if len(cur) + len(w) + 1 > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent + w + " "
        else:
            cur += w + " "
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def morning_sheet(recs, *, store, for_date, conditions, yesterday=None, excluded=(),
                  caveats=(), fmt="text"):
    """The one artifact a human touches. Plain 80-column ASCII, or the same rows as HTML."""
    if fmt == "html":
        return _morning_html(recs, store=store, for_date=for_date, conditions=conditions,
                             yesterday=yesterday, excluded=excluded, caveats=caveats)
    for_date = pd.Timestamp(for_date).normalize()
    w = SHEET_WIDTH
    L = ["=" * w, "FRESH FORECAST - MORNING SHEET".center(w), "=" * w]
    L.append(f"{store or 'store'}   {DAY_NAMES[for_date.dayofweek]} {for_date.date()}")
    made = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    row0 = recs.iloc[0] if len(recs) else {}
    L.append(f"generated {made}   model {row0.get('model_version', '-')}   "
             f"data through {pd.Timestamp(row0.get('panel_through')).date()}")
    L += ["+" + "-" * (w - 2) + "+",
          "| " + SHADOW_BANNER.ljust(w - 4) + " |",
          "+" + "-" * (w - 2) + "+"]
    stale = recs.attrs.get("staleness_days", 0)
    if stale:
        L.append(f"DATA IS {stale} DAY(S) OLD - TREAT THESE NUMBERS WITH CARE.")
    L += ["", "CONDITIONS  " + _conditions_line(conditions)]

    if yesterday is not None and yesterday.attrs.get("closed"):
        L += ["", "YESTERDAY  the store was closed."]
    elif yesterday is not None and len(yesterday):
        L += ["", "YESTERDAY"]
        for r in yesterday.to_dict("records"):
            tail = ", SOLD OUT." if r.get("sold_out") else "."
            L.append(f"  {r['item_name']}: {_said(r)} {_fmt_qty(r['said'], r.get('unit'))}, "
                     f"you made {_fmt_qty(r['made'], r.get('unit'))}, "
                     f"sold {_fmt_qty(r['sold'], r.get('unit'))}{tail}")

    head = (f"{'ITEM':18s} {'PAR':>5s} {'MAKE':>6s} {'LOW..HIGH':>11s} {'UNIT':>4s} "
            f"{'MADE':<9s} {'SOLD OUT AT':<12s}")
    rule = f"{'-' * 18} {'-' * 5} {'-' * 6} {'-' * 11} {'-' * 4} {'-' * 9} {'-' * 12}"
    for dept, rows in _sheet_rows(recs):
        L += ["", dept.upper(), head, rule]
        for r in rows:
            span = _fmt_span(r.get("q_0.20"), r.get("q_0.90"), r["unit"])
            L.append(f"{r['item_name'][:SHEET_ITEM_WIDTH]:18s} "
                     f"{_fmt_qty(r['par_qty'], r['unit']):>5s} "
                     f"{_fmt_qty(r['rec_qty'], r['unit']):>6s} {span:>11s} "
                     f"{r['unit']:>4s} {'_' * 9:<9s} {'_' * 12:<12s}")
            L += _wrap("why: " + r["why_text"], w - 4, "    ")

    carry = _carry_over_rows(recs)
    if carry:
        L += ["", "CARRY-OVER ITEMS - NOT AN ORDER. CHECK WHAT IS LEFT FIRST.",
              "these keep for more than a day and the model does not know what carried over",
              head, rule]
        for r in carry:
            span = _fmt_span(r.get("q_0.20"), r.get("q_0.90"), r["unit"])
            L.append(f"{r['item_name'][:SHEET_ITEM_WIDTH]:18s} "
                     f"{_fmt_qty(r['par_qty'], r['unit']):>5s} "
                     f"{_fmt_qty(r['rec_qty'], r['unit']):>6s} {span:>11s} "
                     f"{r['unit']:>4s} {'_' * 9:<9s} {'_' * 12:<12s}")
            L += _wrap("why: " + _carry_over_why(r), w - 4, "    ")

    fallback = recs[recs["source"] != "model"].to_dict("records") + list(excluded)
    if fallback:
        L += ["", "NO FORECAST - use your own par (reason given)", rule]
        for r in fallback:
            L.append(f"{str(r.get('item_name', r.get('item')))[:SHEET_ITEM_WIDTH]:18s} "
                     f"{_fmt_qty(r.get('par_qty'), r.get('unit', 'each')):>5s} "
                     f"{'-':>6s} {'-':>11s} {str(r.get('unit', '')):>4s} "
                     f"{'_' * 9:<9s} {'_' * 12:<12s}")
            L += _wrap("why: " + _fallback_reason(r), w - 4, "    ")

    L += ["", "-" * w,
          "Write what you actually made. Circle anything that sold out and write the time.",
          "Hand this to the office at close."]
    for c in caveats:
        L += _wrap("* " + c, w)
    L.append("=" * w)
    return "\n".join(L) + "\n\f"


def _morning_html(recs, *, store, for_date, conditions, yesterday, excluded, caveats):
    """The same rows and the same numbers, for a store that prints from a browser."""
    for_date = pd.Timestamp(for_date).normalize()
    esc = lambda s: (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    row0 = recs.iloc[0] if len(recs) else {}
    P = ["<!doctype html><html><head><meta charset='utf-8'>",
         f"<title>Morning sheet {for_date.date()}</title>",
         "<style>body{font-family:monospace;font-size:12pt;margin:1em}"
         "table{border-collapse:collapse;width:100%}"
         "th,td{border:1px solid #000;padding:3px 5px;text-align:left}"
         "td.write{min-width:110px}.banner{border:2px solid #000;padding:6px;font-weight:bold}"
         "h2{page-break-before:always}h2:first-of-type{page-break-before:auto}"
         "@page{size:letter;margin:12mm}</style></head><body>",
         "<h1>FRESH FORECAST - MORNING SHEET</h1>",
         f"<p>{esc(store or 'store')} &middot; {DAY_NAMES[for_date.dayofweek]} "
         f"{for_date.date()}<br>model {esc(row0.get('model_version', '-'))} &middot; "
         f"data through {pd.Timestamp(row0.get('panel_through')).date()}</p>",
         f"<p class='banner'>{esc(SHADOW_BANNER)}</p>",
         f"<p><b>Conditions</b> {esc(_conditions_line(conditions))}</p>"]
    if yesterday is not None and yesterday.attrs.get("closed"):
        P.append("<p><b>Yesterday</b> the store was closed.</p>")
    elif yesterday is not None and len(yesterday):
        P.append("<p><b>Yesterday</b><br>" + "<br>".join(
            f"{esc(r['item_name'])}: {_said(r)} {_fmt_qty(r['said'], r.get('unit'))}, you made "
            f"{_fmt_qty(r['made'], r.get('unit'))}, sold {_fmt_qty(r['sold'], r.get('unit'))}"
            + (" SOLD OUT" if r.get("sold_out") else "")
            for r in yesterday.to_dict("records")) + "</p>")
    for dept, rows in _sheet_rows(recs):
        P.append(f"<h2>{esc(dept)}</h2><table><tr><th>Item</th><th>Your par</th><th>Make</th>"
                 "<th>Low..High</th><th>Unit</th><th>Why</th><th>Made</th>"
                 "<th>Sold out at</th></tr>")
        for r in rows:
            span = _fmt_span(r.get("q_0.20"), r.get("q_0.90"), r["unit"])
            P.append(f"<tr><td>{esc(r['item_name'])}</td>"
                     f"<td>{_fmt_qty(r['par_qty'], r['unit'])}</td>"
                     f"<td><b>{_fmt_qty(r['rec_qty'], r['unit'])}</b></td><td>{span}</td>"
                     f"<td>{esc(r['unit'])}</td><td>{esc(r['why_text'])}</td>"
                     "<td class='write'></td><td class='write'></td></tr>")
        P.append("</table>")
    carry = _carry_over_rows(recs)
    if carry:
        P.append("<h2>Carry-over items - NOT an order. Check what is left first.</h2>"
                 "<p>These keep for more than a day and the model does not know what carried "
                 "over.</p><table><tr><th>Item</th><th>Your par</th><th>Make</th>"
                 "<th>Low..High</th><th>Unit</th><th>Why</th><th>Made</th>"
                 "<th>Sold out at</th></tr>")
        for r in carry:
            span = _fmt_span(r.get("q_0.20"), r.get("q_0.90"), r["unit"])
            P.append(f"<tr><td>{esc(r['item_name'])}</td>"
                     f"<td>{_fmt_qty(r['par_qty'], r['unit'])}</td>"
                     f"<td><b>{_fmt_qty(r['rec_qty'], r['unit'])}</b></td><td>{span}</td>"
                     f"<td>{esc(r['unit'])}</td><td>{esc(_carry_over_why(r))}</td>"
                     "<td class='write'></td><td class='write'></td></tr>")
        P.append("</table>")
    fallback = recs[recs["source"] != "model"].to_dict("records") + list(excluded)
    if fallback:
        P.append("<h2>No forecast</h2><table><tr><th>Item</th><th>Your par</th>"
                 "<th>Reason</th><th>Made</th><th>Sold out at</th></tr>")
        for r in fallback:
            P.append(f"<tr><td>{esc(r.get('item_name', r.get('item')))}</td>"
                     f"<td>{_fmt_qty(r.get('par_qty'), r.get('unit', 'each'))}</td>"
                     f"<td>{esc(_fallback_reason(r))}</td>"
                     "<td class='write'></td><td class='write'></td></tr>")
        P.append("</table>")
    P.append("<p>Write what you actually made. Circle anything that sold out and write the "
             "time. Hand this to the office at close.</p>")
    if caveats:
        P.append("<ul>" + "".join(f"<li>{esc(c)}</li>" for c in caveats) + "</ul>")
    P.append("</body></html>")
    return "\n".join(P)


def sheet_caveats(recs, items, meta=None, known_share=1.0):
    """The standing caveats that apply to THIS sheet. Silence would be the lie."""
    out = []
    # "unknown" is ht.schema's default for a panel that predates the provenance
    # column, not a store saying it has no signal; only "none" is that declaration.
    if str(recs["sellout_source"].iloc[0]) == "none" or known_share <= 0:
        out.append(NO_SELLOUT_CAVEAT)
    if meta and meta.get("thin_history"):
        out.append("SHORT HISTORY: early stopping is unreliable and the seasonal covariates "
                   "are unidentified -- treat these forecasts as provisional.")
    imputed = sorted(k for k, it in items.items() if it.get("cost_imputed"))
    if imputed:
        out.append("Cost is a department-margin assumption for: " + ", ".join(imputed))
    multi = sorted({r["item_name"] for r in recs.to_dict("records")
                    if r["shelf_life_days"] > 1})
    if multi:
        out.append("Multi-day items (" + ", ".join(multi) + ") print under CARRY-OVER and are "
                   "SHADOW ONLY: a one-day newsvendor does not describe them.")
    if recs.attrs.get("day_source") == "carried_forward":
        out.append("No row for today in the panel: the calendar is derived and the weather "
                   "is yesterday's, carried forward.")
    for w in recs.attrs.get("warnings", []):
        out.append("Context window gap -- " + w)
    return out


def _said(row):
    """Who produced yesterday's number. An item under NO FORECAST never "said" anything.

    The same sheet lists short-history items with a trailing par and the reason they have no
    forecast; calling that par a model prediction twenty lines above contradicts the page and
    hands an untrained number the model's authority.
    """
    return "your par was" if str(row.get("source", "model")) != "model" else "model said"


def _yesterday(panel, shadow_dir, for_date, items):
    """What the sheet said yesterday against what the store did. Empty when there is no log."""
    y = pd.Timestamp(for_date).normalize() - pd.Timedelta(days=1)
    preds = read_predictions(shadow_dir, y, y)
    if preds.empty:
        return pd.DataFrame()
    day = panel[pd.to_datetime(panel["date"]) == y]
    actual = day.set_index(day["item"].astype(str))
    over = read_overrides(shadow_dir, y, y)
    over = over.set_index(over["item"].astype(str)) if len(over) else pd.DataFrame()
    closed = bool(len(actual)) and bool(
        (pd.to_numeric(actual["is_closed"], errors="coerce").fillna(0) > 0).all())
    rows = []
    for r in ([] if closed else preds.to_dict("records")):
        key = str(r["item"])
        if key not in actual.index:
            continue
        a = actual.loc[key]
        made = float(over.loc[key, "actual_produced"]) if key in getattr(over, "index", []) \
            else float(pd.to_numeric(a.get("produced"), errors="coerce"))
        sold = float(pd.to_numeric(a.get("sold"), errors="coerce"))
        rec = float(r["rec_qty"])
        sheet = _sheet_sellout(over.loc[key]) if key in getattr(over, "index", []) else None
        out = bool(sheet[0] > 0) if sheet else bool(float(a.get("stockout") or 0) > 0)
        batch = float(items.get(key, {}).get("batch", 1.0))
        if not out and abs((made if np.isfinite(made) else sold) - rec) <= batch:
            continue
        rows.append(dict(item_name=r["item_name"], said=rec, made=made, sold=sold,
                         sold_out=out, unit=r.get("unit", "each"),
                         source=str(r.get("source", "model"))))
    out_df = pd.DataFrame(rows)
    # a closed day is not nine misses; saying so is the whole point of row_status
    out_df.attrs["closed"] = closed
    return out_df


# ---- the record ----

def log_predictions(recs, shadow_dir, run_id=None, made_at=None, backfilled=None,
                    store="", items_config_hash=""):
    """Append to shadow/predictions.csv. Written BEFORE the sheet is rendered."""
    os.makedirs(shadow_dir, exist_ok=True)
    path = os.path.join(shadow_dir, "predictions.csv")
    run_id = run_id or uuid.uuid4().hex[:12]
    made_at = made_at or dt.datetime.now().astimezone().isoformat(timespec="seconds")
    new = not os.path.exists(path)
    n = 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, PREDICTION_COLUMNS, extrasaction="ignore", lineterminator="\n")
        if new:
            w.writeheader()
        for r in recs.to_dict("records"):
            row = dict(r)
            row.update(run_id=run_id, made_at=made_at, store=store,
                       items_config_hash=items_config_hash,
                       for_date=pd.Timestamp(r["for_date"]).date(),
                       panel_through=pd.Timestamp(r["panel_through"]).date(),
                       continuous=int(bool(r["continuous"])),
                       backfilled=int(r["backfilled"] if backfilled is None else backfilled))
            w.writerow({k: row.get(k, "") for k in PREDICTION_COLUMNS})
            n += 1
    return n


def _read_csv(path, date_cols=(), columns=None):
    if not os.path.exists(path):
        return pd.DataFrame(columns=list(columns or []))
    df = pd.read_csv(path, dtype={"item": str})
    for c in date_cols:
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def _window(df, col, date_from, date_to):
    if df.empty:
        return df
    if date_from is not None:
        df = df[df[col] >= pd.Timestamp(date_from).normalize()]
    if date_to is not None:
        df = df[df[col] <= pd.Timestamp(date_to).normalize()]
    return df


def read_predictions(shadow_dir, date_from=None, date_to=None):
    """The logged forecasts; the LAST run for a (for_date, item) is the live one."""
    df = _read_csv(os.path.join(shadow_dir, "predictions.csv"),
                   ("for_date", "panel_through"), PREDICTION_COLUMNS)
    df = _window(df, "for_date", date_from, date_to) if len(df) else df
    if df.empty:
        return df
    return df.drop_duplicates(["for_date", "item"], keep="last").reset_index(drop=True)


def read_overrides(shadow_dir, date_from=None, date_to=None):
    """What the kitchen really made, hand-keyed from the returned paper sheet.

    Append-only, like the prediction log: the LAST row for a (date, item) wins, so a
    correction is a second entry and the first one stays readable underneath it.
    """
    root = os.path.join(shadow_dir, "overrides")
    if not os.path.isdir(root):
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    frames = [_read_csv(os.path.join(root, f), ("date",), OVERRIDE_COLUMNS)
              for f in sorted(os.listdir(root)) if f.endswith(".csv")]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df = _window(df, "date", date_from, date_to)
    if df.empty:
        return df
    return df.drop_duplicates(["date", "item"], keep="last").reset_index(drop=True)


# ---- the paper sheet, keyed back in ----

# The kitchen's own pen. It is deliberately not one of ht.schema.SELLOUT_SOURCES: those
# name rules a system applied to an export, and this one names a person reading a page.
SHEET_SELLOUT_SOURCE = "sheet"
AFFIRMATIVE = ("y", "yes", "soldout", "out", "circled")


def parse_time(text):
    """The SOLD OUT AT cell as a person writes it: 14:30, 2:30pm, 2pm, 1430 -- or "yes".

    Returns "" for an empty cell or a written negative, "HH:MM" for a time, "yes" for a bare
    affirmative (the sheet asks for a time and somebody will circle the item instead), and
    None for anything it cannot read -- which the caller reports rather than guessing at,
    because a guessed sellout time is a fabricated observation.

    "0", "na", "n/a" and "none" are read as negatives. "0" used to parse as a sellout at
    00:00, which is a fabricated observation: a prepared-foods case is not open at midnight,
    and somebody writing 0 in a box that asks for a time means "it did not". A bare hour
    ("9") is still read on a 24-hour clock and may be off by twelve -- the time is recorded
    and never computed on, and refusing it would throw away the whole returned sheet.
    """
    s = str(text or "").strip().lower().replace(".", "").replace(" ", "")
    if not s or s in ("-", "n", "no", "0", "na", "n/a", "none"):
        return ""
    if s in AFFIRMATIVE:
        return "yes"
    m = re.fullmatch(r"(\d{1,2}):?(\d{2})?(am|pm)?", s)
    if not m:
        return None
    hour, minute, half = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if minute > 59 or (half and not 1 <= hour <= 12) or (not half and hour > 23):
        return None
    if half:
        hour = hour % 12 + (12 if half == "pm" else 0)
    return f"{hour:02d}:{minute:02d}"


def _item_index(items):
    """Config key or printed name -> config key. The person holding the sheet reads a name.

    The TRUNCATED printed name is registered too: the sheet's ITEM column is
    SHEET_ITEM_WIDTH characters, so a store with an ordinary POS description
    ("Rotisserie Chicken Lemon Pepper") reads back a name the config does not contain, and
    refusing it loses the whole sheet -- over a column width.
    """
    idx = {str(k).lower(): str(k) for k in items}
    for k, it in items.items():
        idx.setdefault(str(it["name"]).lower(), str(k))
    # only where the truncation is still unambiguous: two names that shorten to the same
    # 18 characters would silently file one item's production number under the other
    short = collections.Counter(str(it["name"])[:SHEET_ITEM_WIDTH].strip().lower()
                                for it in items.values())
    for k, it in items.items():
        name = str(it["name"])[:SHEET_ITEM_WIDTH].strip().lower()
        if short[name] == 1:
            idx.setdefault(name, str(k))
    return idx


def parse_entries(lines, items):
    """`item, made, sold out at, note` per line -> (rows, errors). One format, both intakes.

    The prompt builds these same lines, so a piped file and a person at a terminal are
    validated by exactly one piece of code. Nothing is written unless every line parses: a
    half-keyed sheet that looks entered is worse than one that obviously is not.
    """
    idx = _item_index(items)
    rows, errors, seen = [], [], set()
    for n, raw in enumerate(lines, 1):
        line = str(raw).lstrip("\ufeff").strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if parts[0].lower() in ("item", "item_name"):
            continue                       # a header row pasted along with the data
        key = idx.get(parts[0].lower())
        if key is None:
            near = [str(k) for k, it in items.items()
                    if parts[0] and str(it["name"]).lower().startswith(parts[0].lower())]
            hint = f" -- did you mean {near[0]}?" if near else ""
            errors.append(f"line {n}: {parts[0]!r} is not an item in the items config{hint}")
            continue
        if key in seen:
            errors.append(f"line {n}: {key} was already entered on an earlier line")
            continue
        made = ""
        if len(parts) > 1 and parts[1]:
            try:
                made = float(parts[1])
            except ValueError:
                made = float("nan")
            if not np.isfinite(made) or made < 0:
                errors.append(f"line {n}: made {parts[1]!r} is not a quantity")
                continue
        sold_out = parse_time(parts[2]) if len(parts) > 2 else ""
        if sold_out is None:
            errors.append(f"line {n}: sold out at {parts[2]!r} is not a time -- write it as "
                          "14:30, 2:30pm or 1430, or 'yes' if it sold out and nobody wrote "
                          "the time, or leave it blank (or 'no') if it did not sell out")
            continue
        seen.add(key)
        rows.append(dict(item=key, actual_produced=made, sold_out_at=sold_out,
                         note=", ".join(parts[3:]) if len(parts) > 3 else ""))
    return rows, errors


def prompt_entries(items, keys, said=None, ask=input, echo=print):
    """One item at a time, in the order the paper printed them, at a back-room terminal."""
    said = said or {}
    echo("Type what the kitchen wrote. Blank = nothing written there. Ctrl-C or Ctrl-D\n"
         "abandons: nothing is written until every line parses.")
    lines = []
    for key in keys:
        it = items[key]
        hint = (f" (sheet said {_fmt_qty(said[key], it.get('unit', 'each'))})"
                if key in said else "")
        made = str(ask(f"{it['name']}{hint} - made: ")).strip()
        out = str(ask("    sold out at (blank if it did not): ")).strip()
        if made or out:
            lines.append(f"{key},{made},{out}")
    return lines


def entry_order(shadow_dir, for_date, items):
    """Item keys in the order the SHEET printed them, then anything the sheet did not have.

    The page is day-fresh departments first (each in _sheet_rows' order), then the CARRY-OVER
    block, then NO FORECAST -- so the prediction log's own order is the sheet's order only
    when every item is day-fresh. Set two items' shelf life truthfully and the log opens with
    an item printed near the foot of the page; keying one item's production number into
    another's is exactly the silent corruption the sheet's layout exists to avoid.
    """
    preds = read_predictions(shadow_dir, for_date, for_date)
    keys = [k for k in preds["item"].astype(str) if k in items] if len(preds) else []
    if not keys:
        return sorted(items)
    pos = {k: i for i, k in enumerate(keys)}
    dept = dict(zip(preds["item"].astype(str), preds["dept"].astype(str))) \
        if "dept" in preds else {}
    forecast = (set(preds.loc[preds["source"] == "model", "item"].astype(str))
                if "source" in preds else set(keys))
    multi = {k for k in keys if int(items[k].get("shelf_life_days", 1)) > 1}
    block = {k: (0 if k in forecast and k not in multi else 1 if k in forecast else 2)
             for k in keys}
    ordered = sorted(keys, key=lambda k: (block[k], dept.get(k, ""), pos[k]))
    return ordered + [k for k in sorted(items) if k not in keys]


def _overrides_columns(path):
    """The columns to append under, migrating a file written against an older set.

    record_actuals appends, so a nine-field row under an eight-field header makes the file
    unparseable -- and read_overrides reads every file in the directory, so one such day
    takes down score, catch-up, weekly and the morning page's YESTERDAY block, with a pandas
    tokenizer message that names no file. Old rows are padded with an empty sellout_source,
    which _sheet_sellout already reads as "says nothing about sellouts", and a column
    somebody added by hand is kept on the end rather than dropped.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or header == OVERRIDE_COLUMNS:
            return OVERRIDE_COLUMNS
        rows = [dict(zip(header, r)) for r in reader]
    cols = OVERRIDE_COLUMNS + [c for c in header if c not in OVERRIDE_COLUMNS]
    # written beside it and renamed over it: this file is a pilot's only record of what the
    # kitchen made, and a rewrite interrupted half way would be the one loss it cannot recover
    tmp = path + ".rewriting"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    os.replace(tmp, path)
    return cols


def record_actuals(shadow_dir, for_date, rows, entered_by=""):
    """Append a returned sheet to shadow/overrides/<date>.csv and return the path.

    Every row is stamped sellout_source="sheet". That stamp is what lets an EMPTY sold_out_at
    cell count as "it did not sell out": the sheet came back with the rest of the row filled
    in, so the blank is an observation. A hand-authored overrides file carries no such
    promise and its blanks stay unknown -- see _sheet_sellout.
    """
    for_date = pd.Timestamp(for_date).normalize()
    preds = read_predictions(shadow_dir, for_date, for_date)
    said = dict(zip(preds["item"].astype(str), preds["rec_qty"])) if len(preds) else {}
    root = os.path.join(shadow_dir, "overrides")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{for_date.date()}.csv")
    ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    new = not os.path.exists(path)
    cols = OVERRIDE_COLUMNS if new else _overrides_columns(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore", lineterminator="\n")
        if new:
            w.writeheader()
        for r in rows:
            row = {c: "" for c in cols}
            row.update(r, date=str(for_date.date()), rec_qty=said.get(str(r["item"]), ""),
                       entered_by=entered_by, entered_ts=ts,
                       sellout_source=SHEET_SELLOUT_SOURCE)
            w.writerow(row)
    return path


def _sheet_sellout(over_row):
    """(stockout, known, source) from a returned sheet's SOLD OUT AT cell, or None.

    None means "this override row says nothing about sellouts", which is the honest reading
    of a hand-authored file that predates this path: it was written to correct a production
    number, and reading its empty sold_out_at as "did not sell out" would invent an
    observation on every row of it.
    """
    if _cell(over_row.get("sellout_source")) != SHEET_SELLOUT_SOURCE:
        return None
    return (1.0 if _cell(over_row.get("sold_out_at")) else 0.0, 1.0, SHEET_SELLOUT_SOURCE)


# ---- scoring, once ----

def _pinball(q_units, taus, sold, cens):
    u = np.asarray(sold, dtype=float)[:, None] - np.asarray(q_units, dtype=float)
    taus = np.asarray(taus, dtype=float)
    full = np.maximum(taus * u, (taus - 1) * u)
    under = taus * np.clip(u, 0, None)
    return np.where(np.asarray(cens, dtype=float)[:, None] > 0, under, full).mean(axis=1)


def _score_rows(panel, items, shadow_dir, for_date):
    for_date = pd.Timestamp(for_date).normalize()
    preds = read_predictions(shadow_dir, for_date, for_date)
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    day = df[df["date"] == for_date].copy()
    day["item"] = day["item"].astype(str)
    day = day.set_index("item")
    over = read_overrides(shadow_dir, for_date, for_date)
    over = over.set_index(over["item"].astype(str)) if len(over) else over

    cols = [c for c in quantile_columns(features.TAUS) if c in preds.columns]
    taus = np.array([float(c.split("_")[1]) for c in cols])
    rows = []
    keys = sorted(set(preds["item"].astype(str)) | set(k for k in items if k in day.index))
    for key in keys:
        p = preds[preds["item"].astype(str) == key]
        a = day.loc[key] if key in day.index else None
        rec = dict(for_date=for_date.date(), item=key, rec_qty="", par_qty="", p20="",
                   p50="", p90="", sold="", produced="", stockout="", stockout_known="",
                   sellout_source="",
                   wasted="", is_closed="", row_status="", abs_err="", signed_err="",
                   pinball="", waste_actual_units="", waste_model_units="",
                   lost_lower_units="", source="", status="")
        if p.empty:
            rec["status"] = "missing_sheet"
            rows.append(rec)
            continue
        p = p.iloc[-1]
        rec.update(rec_qty=float(p["rec_qty"]), par_qty=float(p["par_qty"]),
                   p20=float(p.get("q_0.20", np.nan)), p50=float(p.get("q_0.50", np.nan)),
                   p90=float(p.get("q_0.90", np.nan)), source=str(p["source"]))
        if a is None:
            rec["status"] = "missing_data"
            rows.append(rec)
            continue
        sold = float(pd.to_numeric(a.get("sold"), errors="coerce"))
        produced = float(pd.to_numeric(a.get("produced"), errors="coerce"))
        if key in getattr(over, "index", []):
            o = float(pd.to_numeric(over.loc[key, "actual_produced"], errors="coerce"))
            produced = o if np.isfinite(o) else produced
        stockout = float(pd.to_numeric(a.get("stockout"), errors="coerce") or 0.0)
        known = float(pd.to_numeric(a.get("stockout_known"), errors="coerce")
                      if "stockout_known" in a else 1.0)
        source = _cell(a.get("sellout_source")) or "unknown"
        # the returned paper sheet outranks the export's rule: for a store with no label log
        # it is the only sellout observation that exists, and for one with a rule it is the
        # eyewitness. Its provenance travels into the score row so the weekly page can say so.
        sheet = _sheet_sellout(over.loc[key]) if key in getattr(over, "index", []) else None
        if sheet is not None:
            stockout, known, source = sheet
        cens = stockout * known
        status = str(a.get("row_status", "ok"))
        rec.update(sold=sold, produced=produced, stockout=stockout, stockout_known=known,
                   sellout_source=source,
                   wasted=float(pd.to_numeric(a.get("wasted"), errors="coerce")),
                   is_closed=int(pd.to_numeric(a.get("is_closed"), errors="coerce") or 0),
                   row_status=status)
        if rec["is_closed"] or status == "closed":
            rec["status"] = "excluded_closed"
        elif status == "partial":
            rec["status"] = "excluded_partial"
        elif not np.isfinite(sold):
            rec["status"] = "missing_data"
        elif int(p.get("backfilled", 0) or 0):
            rec["status"] = "backfilled"
        else:
            rec["status"] = "scored"
        q = np.array([[float(p[c]) for c in cols]])
        rec.update(
            abs_err=abs(rec["p50"] - sold), signed_err=rec["p50"] - sold,
            pinball=float(_pinball(q, taus, [sold], [cens])[0]) if np.isfinite(sold) else "",
            waste_actual_units=max(produced - sold, 0.0) if np.isfinite(produced) else "",
            waste_model_units=max(rec["rec_qty"] - sold, 0.0),
            lost_lower_units=max(sold - rec["rec_qty"], 0.0))
        rows.append(rec)
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def score_day(panel, items, shadow_dir, for_date):
    """Freeze the day's verdict. Write-once; a later change is disclosed, not applied."""
    for_date = pd.Timestamp(for_date).normalize()
    path = os.path.join(shadow_dir, "scores", f"{for_date.date()}.csv")
    fresh = _score_rows(panel, items, shadow_dir, for_date)
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"item": str})
        _record_revisions(old, fresh, shadow_dir, for_date)
        return old
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fresh.to_csv(path, index=False, lineterminator="\n")
    return fresh


def _cell(value):
    """One rendering for an absent measure, so re-scoring an unchanged day records nothing.

    A missing_sheet row is written with empty strings and read back from CSV as NaN, so a
    raw str() comparison reports every field as 'nan' -> '' and the page then tells a
    district manager that rows were revised after scoring when nothing moved.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return "" if str(value) in ("nan", "None") else str(value)


def _record_revisions(old, fresh, shadow_dir, for_date):
    """Late and corrected exports are normal; a moving four-week total is not."""
    a = old.set_index(old["item"].astype(str))
    b = fresh.set_index(fresh["item"].astype(str))
    changes = []
    for key in sorted(set(a.index) & set(b.index)):
        for field in ("sold", "produced", "stockout", "wasted", "row_status", "status"):
            ov, nv = _cell(a.loc[key, field]), _cell(b.loc[key, field])
            if ov != nv:
                changes.append(dict(revised_at=dt.datetime.now().astimezone()
                                    .isoformat(timespec="seconds"),
                                    for_date=str(pd.Timestamp(for_date).date()), item=key,
                                    field=field, old_value=ov, new_value=nv))
    if not changes:
        return 0
    path = os.path.join(shadow_dir, "scores", "_revisions.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, REVISION_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(changes)
    return len(changes)


def read_scores(shadow_dir, date_from=None, date_to=None):
    root = os.path.join(shadow_dir, "scores")
    if not os.path.isdir(root):
        return pd.DataFrame(columns=SCORE_COLUMNS)
    frames = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".csv") or name.startswith("_"):
            continue
        d = pd.Timestamp(name[:-4])
        if (date_from is not None and d < pd.Timestamp(date_from)) or \
           (date_to is not None and d > pd.Timestamp(date_to)):
            continue
        frames.append(pd.read_csv(os.path.join(root, name), dtype={"item": str}))
    if not frames:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["for_date"] = pd.to_datetime(out["for_date"])
    return out


def catch_up(panel, items, shadow_dir, since=None):
    """Score every logged day that now has data. How a late export is absorbed."""
    preds = read_predictions(shadow_dir, date_from=since)
    if preds.empty:
        return []
    have = set(read_scores(shadow_dir)["for_date"].astype("datetime64[ns]")) \
        if len(read_scores(shadow_dir)) else set()
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    with_data = set(df["date"].unique())
    done = []
    for d in sorted(set(preds["for_date"])):
        if d in have or np.datetime64(d) not in with_data:
            continue
        score_day(panel, items, shadow_dir, d)
        done.append(str(pd.Timestamp(d).date()))
    return done


# ---- the weekly page ----

def _scored_sellout_source(fc):
    """Where the flags these rows were SCORED with came from, every source named.

    The score row records it because a returned sheet can supply the flag on a panel whose
    own rule is "none"; the prediction's copy only knows what the export said that morning.
    Mixed weeks print every source rather than the majority one -- "produced_vs_sold+sheet"
    is the truth, and a mode would hide the half the reader would want to ask about. Score
    files written before the column existed fall back to the prediction's copy.
    """
    col = fc["sellout_source_sc"] if "sellout_source_sc" in fc else fc.get("sellout_source")
    values = sorted({_cell(v) for v in (col if col is not None else [])} - {""})
    if not values and "sellout_source" in fc:
        values = sorted({_cell(v) for v in fc["sellout_source"]} - {""})
    return "+".join(values) if values else "unknown"


def _point_wape(pred, sold, keep):
    pred, sold = np.asarray(pred, dtype=float), np.asarray(sold, dtype=float)
    ok = keep & np.isfinite(pred) & np.isfinite(sold)
    den = sold[ok].sum()
    return dict(wape_uncensored=float(np.abs(pred[ok] - sold[ok]).sum() / den) if den else
                float("nan"), n=int(ok.sum()))


def weekly_report(shadow_dir, panel, items, week_ending, weeks=1, include_backfilled=False):
    """Accuracy from the LOG joined to the frozen scores. Never a re-forecast.

    Backfilled rows are quarantined by default: a sheet written after the day it is for
    proves nothing about what was known that morning, and a pilot page that silently
    mixed them in would be worthless. include_backfilled=True folds them in ANYWAY and
    stamps the report reconstructed=True with a caveat saying so on the page -- that is
    the dress-rehearsal mode, where every sheet is replayed from history by
    construction, and it is not the mode a real pilot week is read in.
    """
    end = pd.Timestamp(week_ending).normalize()
    start = end - pd.Timedelta(days=7 * weeks - 1)
    preds = read_predictions(shadow_dir, start, end)
    scores = read_scores(shadow_dir, start, end)
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    win = df[(df["date"] >= start) & (df["date"] <= end) & df["item"].astype(str).isin(items)]

    qcols = [c for c in quantile_columns(features.TAUS) if c in preds.columns]
    taus = np.array([float(c.split("_")[1]) for c in qcols])
    joined = preds.merge(scores, on=["for_date", "item"], how="left",
                         suffixes=("", "_sc")) if len(scores) else preds.assign(status="unscored")
    ok = ["scored"] + (["backfilled"] if include_backfilled else [])
    scored = joined[joined["status"].isin(ok)].copy().reset_index(drop=True)

    # Expected counts the days the store should have supplied, not the days it did. An
    # unexplained export gap arrives as row_status 'missing' with is_closed 1, so counting
    # open rows alone drops those item-days out of BOTH sides of the ratio and a four-day
    # outage reads as 100% complete -- on the one gate that exists to catch it.
    status = (win["row_status"].astype(str) if "row_status" in win
              else pd.Series("ok", index=win.index))
    unexplained = status.isin(("missing", "suspect"))
    open_rows = pd.to_numeric(win.get("is_closed", 0), errors="coerce").fillna(0) == 0
    expected = int((open_rows | unexplained).sum())
    outages = [str(pd.Timestamp(d).date()) for d in sorted(win.loc[unexplained, "date"].unique())]
    dates_with_data = sorted(pd.Timestamp(d) for d in win.loc[open_rows, "date"].unique())
    logged_dates = sorted(pd.Timestamp(d) for d in preds["for_date"].unique())
    scored_dates = sorted(pd.Timestamp(d) for d in scored["for_date"].unique())
    settled = sorted(pd.Timestamp(d) for d in scores["for_date"].unique()) if len(scores) else []
    missing_sheets = [str(d.date()) for d in dates_with_data if d not in logged_dates]
    # a closed day has a score file and contributes no rows; that is settled, not pending
    unscored = [str(d.date()) for d in logged_dates if d not in settled]
    backfilled_rows = int((joined["status"] == "backfilled").sum())

    res = dict(
        store=str(preds["store"].iloc[0]) if len(preds) and "store" in preds else "",
        week=f"{end.isocalendar()[0]}-W{end.isocalendar()[1]:02d}",
        week_start=str(start.date()), week_end=str(end.date()), weeks=int(weeks),
        model_version=str(preds["model_version"].iloc[-1]) if len(preds) else "",
        spec_hash=str(preds["spec_hash"].iloc[-1]) if len(preds) else "",
        days_expected=len(dates_with_data), days_covered=len(scored_dates),
        n_rows_expected=expected, n_rows_scored=int(len(scored)),
        completeness=float(len(scored) / expected) if expected else float("nan"),
        missing_sheets=missing_sheets, unscored=unscored, backfilled_rows=backfilled_rows,
        data_gaps=outages,
        reconstructed=bool(include_backfilled and backfilled_rows),
        revisions=int(len(_read_csv(os.path.join(shadow_dir, "scores", "_revisions.csv"),
                                    (), REVISION_COLUMNS))),
        exclusions=(joined["status"].value_counts().to_dict() if len(joined) else {}),
    )
    if scored.empty:
        res.update(accuracy={}, calibration=[], measured={}, bounds={}, by_dept=[], misses=[],
                   overrides={}, censoring={}, skill={}, caveats=["no scored rows this week"])
        res["gates"] = gates(res)
        return res

    # An item the model could not forecast is still logged, with its par, so the sheet never
    # reads "make none" -- but its quantiles are NaN and it is not a model forecast. Those rows
    # belong in the exclusion ledger, not in the model's accuracy, and one NaN would otherwise
    # turn every headline sum into nan. Scoring model and par on this one row set is also what
    # makes the comparison paired, which is the whole point of printing them side by side.
    fc = scored[scored["source"].astype(str) == "model"].copy().reset_index(drop=True)
    res["n_rows_no_forecast"] = int(len(scored) - len(fc))
    if res["n_rows_no_forecast"]:
        res["exclusions"]["no_forecast"] = res["n_rows_no_forecast"]
    if fc.empty:
        res.update(accuracy={}, calibration=[], measured={}, bounds={}, by_dept=[], misses=[],
                   overrides={}, censoring={}, skill={},
                   caveats=["no item had a model forecast this week"])
        res["gates"] = gates(res)
        return res

    item = fc["item"].astype(str).values
    sold = fc["sold"].values.astype(float)
    known = fc["stockout_known"].fillna(1.0).values.astype(float)
    cens = fc["stockout"].values.astype(float) * known
    q_units = fc[qcols].values.astype(float)
    rec = fc["rec_qty"].values.astype(float)
    par = fc["par_qty"].values.astype(float)
    produced = fc["produced"].values.astype(float)
    wasted = fc["wasted"].values.astype(float)
    price = np.array([float(items[k]["price"]) for k in item])
    cost = np.array([float(items[k]["cost"]) for k in item])
    fresh = np.array([int(items[k].get("shelf_life_days", 1)) == 1 for k in item])
    source = _scored_sellout_source(fc)
    known_share = float(known.mean()) if len(known) else 1.0
    censoring_known = bool(known_share > 0 and source != "none")

    acc = evaluate.score_quantiles(q_units, taus, sold, cens, censoring_known, known)
    unc = ~(cens > 0) if censoring_known else np.ones(len(sold), dtype=bool)
    bnd = evaluate.bounds(rec, sold, cens, produced, wasted, cost, price, fresh,
                          censoring_known, item, known)
    pack = evaluate.row_pack(q_units=q_units, taus=taus, sold=sold, cens=cens, rec_qty=rec,
                             produced=produced, wasted=wasted, cost=cost, price=price,
                             day_fresh=fresh, item=item, censoring_known=censoring_known,
                             known=known)

    weekly_pass = []
    iso = fc["for_date"].dt.isocalendar()
    for (year, week), grp in fc.groupby([iso["year"], iso["week"]], sort=True):
        m = fc.index.isin(grp.index)
        wm = _point_wape(evaluate.median_forecast(q_units[m], taus), sold[m], unc[m])
        wp = _point_wape(par[m], sold[m], unc[m])
        weekly_pass.append(dict(week=f"{int(year)}-W{int(week):02d}",
                                start=str(grp["for_date"].min().date()),
                                model=wm["wape_uncensored"], par=wp["wape_uncensored"],
                                n=wm["n"]))
    weekly_pass.sort(key=lambda w: w["start"])

    ov = read_overrides(shadow_dir, start, end)
    # the denominator is the rows this page scores, so the numerator has to be the overrides
    # that landed on one. Now that a returned sheet enters every item every day, dividing the
    # raw count by it printed "131% of scored rows were overridden", which is not a fact
    keyed = set(zip(ov["date"], ov["item"].astype(str))) if len(ov) else set()
    matched = int(sum((d, k) in keyed for d, k in zip(fc["for_date"], fc["item"].astype(str))))
    overrides = dict(n=int(len(ov)), n_matched=matched,
                     share=float(matched / len(fc)) if len(fc) else 0.0,
                     n_closer=None, n_compared=0)
    if len(ov):
        key = ov.set_index([ov["date"], ov["item"].astype(str)])
        closer = compared = 0
        for i, (d, k) in enumerate(zip(fc["for_date"], item)):
            if (d, k) not in key.index or not unc[i]:
                continue
            made = float(pd.to_numeric(key.loc[(d, k), "actual_produced"], errors="coerce"))
            if not np.isfinite(made):
                continue
            compared += 1
            closer += int(abs(made - sold[i]) < abs(rec[i] - sold[i]))
        overrides.update(n_closer=closer, n_compared=compared)

    miss = fc.assign(gap=np.abs(rec - sold)).sort_values("gap", ascending=False).head(5)
    res.update(
        censoring=dict(sellout_source=source, censoring_known=censoring_known,
                       known_share=known_share, sellout_rate=float((cens > 0).mean())),
        accuracy=dict(
            model={k: v for k, v in acc.items() if k != "coverage"},
            par=_point_wape(par, sold, unc),
            skill_vs_par=1.0 - evaluate._ratio(acc["wape_uncensored"],
                                               _point_wape(par, sold, unc)["wape_uncensored"]),
            weekly=weekly_pass),
        calibration=acc["coverage"],
        measured=dict(production_coverage=bnd["production_coverage"],
                      n_rows_measured=bnd["n_rows_measured"],
                      units_produced_actual=float(np.nansum(produced)),
                      units_recommended=float(np.nansum(rec)),
                      waste_observed_units=bnd["waste_observed_units"],
                      waste_observed_cost=bnd["waste_observed_cost"],
                      waste_observed_retail=bnd["waste_observed_retail"]),
        bounds={k: v for k, v in bnd.items() if not k.startswith("waste_observed")},
        by_dept=evaluate.by_group(
            pack, np.array([items[k]["dept"] for k in item])).to_dict("records"),
        misses=[dict(for_date=str(pd.Timestamp(r["for_date"]).date()), item=r["item"],
                     rec_qty=float(r["rec_qty"]), sold=float(r["sold"]),
                     why=str(r.get("why_text", ""))) for r in miss.to_dict("records")],
        overrides=overrides, skill=dict(vs_par=1.0 - evaluate._ratio(
            acc["wape_uncensored"], _point_wape(par, sold, unc)["wape_uncensored"])),
        top_items=[str(k) for k in pd.Series(rec * price, index=item).groupby(level=0)
                   .sum().sort_values(ascending=False).head(10).index],
    )
    res["short_history_items"] = sorted(set(preds.loc[preds["source"] != "model", "item"]
                                            .astype(str)))
    res["caveats"] = _weekly_caveats(res, items, preds)
    res["gates"] = gates(res)
    return res


def _weekly_caveats(res, items, preds):
    out = []
    c = res.get("censoring", {})
    if not c.get("censoring_known", True):
        out.append(NO_SELLOUT_CAVEAT + " Every 'uncensored' figure here is an all-rows figure.")
    elif (c.get("known_share") or 0.0) < 1.0:
        # a mixed week is closer to no signal than to a full one, so it keeps the same warning
        out.append(f"The sellout flag was evaluable on {c['known_share']:.0%} of scored rows. "
                   "On the rest, an 'uncensored' row only means nobody flagged it, and a day "
                   "that ran out unseen pulls the quantities down the same way no sellout "
                   "data does.")
    if res["bounds"].get("excluded_multi_day_items"):
        out.append("Excluded from the waste bound (shelf life > 1 day): "
                   + ", ".join(res["bounds"]["excluded_multi_day_items"]))
    if res.get("reconstructed"):
        out.append(f"RECONSTRUCTED: {res['backfilled_rows']} of these predictions were "
                   "backfilled -- written after the day they are for, from data that ends "
                   "the night before. The arithmetic is honest and the timing is not "
                   "evidence. A real pilot week must be all same-morning sheets.")
    elif res["backfilled_rows"]:
        out.append(f"{res['backfilled_rows']} backfilled predictions are excluded from every "
                   "number above and listed only here.")
    if res["revisions"]:
        out.append(f"{res['revisions']} row(s) were revised after scoring; the original "
                   "scores stand and the changes are in scores/_revisions.csv.")
    out.append("Coverage is reported as an interval because a sellout day's coverage is "
               "genuinely unknown; the interval width IS the censoring rate.")
    out.append("G3 reads cov_point, which is measured on the rows where demand was exactly "
               "observed -- a biased subsample, since sellout days are the busy ones, so a "
               "well-calibrated model can fail it. It reads PENDING when no row was observed.")
    imputed = sorted(k for k, it in items.items() if it.get("cost_imputed"))
    if imputed:
        out.append("Cost imputed from a department margin for: " + ", ".join(imputed))
    return out


def gates(res):
    """G1..G5, fixed in advance and printed from week one. PENDING is not PASS."""
    def verdict(ok, pending=False):
        return "PENDING" if pending else ("PASS" if ok else "FAIL")

    acc, cal = res.get("accuracy", {}), res.get("calibration", [])
    bnd, ms = res.get("bounds", {}), res.get("measured", {})
    out = {}

    # a day the store's export never explained counts against completeness exactly like a
    # morning nobody printed a sheet on: both are days the pilot has no observation for
    gap = _longest_run(set(res.get("missing_sheets", [])) | set(res.get("data_gaps", [])))
    out["G1"] = verdict(res.get("completeness", 0) >= 0.95 and gap <= 1,
                        pending=not res.get("n_rows_scored"))

    if not acc:
        out["G2"] = "PENDING"
    else:
        model, par = acc["model"]["wape_uncensored"], acc["par"]["wape_uncensored"]
        weeks_ok = sum(1 for w in acc.get("weekly", []) if w["model"] <= w["par"])
        ratio_ok = np.isfinite(par) and model <= 0.90 * par
        out["G2"] = verdict(ratio_ok and weeks_ok >= 3,
                            pending=len(acc.get("weekly", [])) < 4 and ratio_ok)

    mid = [c for c in cal if abs(c["tau"] - 0.50) < 1e-9]
    hi = [c for c in cal if abs(c["tau"] - 0.90) < 1e-9]
    observed = mid[0].get("n_observed", 0) if mid else 0
    if not cal or not mid or not hi or not observed:
        # with no row whose demand was exactly observed, cov_lo is 0 and cov_point is nan.
        # That is not a failed calibration, it is an unmeasured one, and PENDING says so.
        out["G3"] = "PENDING"
    else:
        out["G3"] = verdict(abs(mid[0]["cov_point"] - 0.50) <= 0.10
                            and hi[0]["cov_lo"] >= 0.75)

    retail = ms.get("waste_observed_retail", 0.0) or 0.0
    sq = bnd.get("sellout_days_sq")
    if not retail or sq is None:
        out["G4"] = "PENDING"
    else:
        out["G4"] = verdict(bnd["waste_saving_lower_retail"] >= 0.15 * retail
                            and bnd["sellout_days_model_lower"] <= sq + 0.03)

    top = res.get("top_items", [])
    dropped = set(res.get("short_history_items", []))
    out["G5"] = verdict(len(dropped & set(top)) <= 2, pending=not top)
    return out


def _clip(text, width):
    """Trim to width without leaving half a word behind."""
    text = str(text)
    return text if len(text) <= width else text[:width].rsplit(" ", 1)[0]


def _longest_run(iso_dates):
    days = sorted(pd.Timestamp(d) for d in iso_dates)
    best = run = 0
    prev = None
    for d in days:
        run = run + 1 if prev is not None and (d - prev).days == 1 else 1
        best, prev = max(best, run), d
    return best


def format_weekly(res, fmt="text", width=WEEKLY_WIDTH):
    """The district-manager page. Completeness first, then accuracy, then the gates."""
    if fmt == "json":
        return json.dumps(res, indent=1, sort_keys=True, default=evaluate._jsonable)
    pct, num = evaluate._pct, evaluate._num
    L = ["=" * width, f"FRESH FORECAST - WEEKLY SHADOW REPORT  {res['week']}".center(width),
         "=" * width,
         f"{res['store'] or 'store'}   {res['week_start']} .. {res['week_end']}   "
         f"({res['weeks']} week(s))",
         f"days with data {res['days_covered']}/{res['days_expected']}   "
         f"item-days scored {res['n_rows_scored']:,}/{res['n_rows_expected']:,}   "
         f"completeness {pct(res['completeness'])}",
         ("*** RECONSTRUCTED FROM HISTORY - NOT A RECORD OF SAME-MORNING SHEETS ***"
          if res.get("reconstructed") else
          "every number below comes from predictions logged before the day"),
         f"model {res['model_version']}  spec {res['spec_hash']}  "
         f"censoring {res.get('censoring', {}).get('sellout_source', '-')} "
         f"(known {pct(res.get('censoring', {}).get('known_share'))})"]
    acc = res.get("accuracy") or {}
    if not acc:
        L += ["", "NO SCORED ROWS THIS WEEK.", "=" * width]
        return "\n".join(L)

    m = acc["model"]
    # The heading is decided by the rows this number COVERS, not by whether any row anywhere
    # carried a flag. With no sellout signal every "uncensored" figure is an all-rows figure,
    # which is what score_quantiles returns and what the JSON's censoring_known says. And one
    # keyed-in sheet row on a panel whose rule is "none" makes the flag evaluable on 1 row of
    # 48 -- "days where demand was fully served" over all 48 is the same run's two artifacts
    # contradicting each other, on the one number a manager reads first.
    cens = res.get("censoring", {})
    censored_known = cens.get("censoring_known", True)
    known_share = cens.get("known_share")
    known_share = 1.0 if known_share is None else float(known_share)
    n_known = int(round(known_share * m["n_rows"]))
    label = "wape" if not censored_known else ("wape" if known_share >= 1.0 else "wape*")
    if censored_known and known_share >= 1.0:
        L += ["", "1. ACCURACY (median forecast vs sold, on days where demand was fully "
              "served)"]
    elif censored_known:
        L += ["", "1. ACCURACY (median forecast vs sold, on rows nothing flagged as sold "
              "out)",
              f"   * the flag was evaluable on {n_known:,d} of {m['n_rows']:,d} scored rows "
              f"({pct(known_share)}). On the rest, \"not flagged\" only",
              "   means nobody could tell, and those rows sit in this number as if they had "
              "been served in full."]
    else:
        label = "wape_all"
        L += ["", "1. ACCURACY (median forecast vs sold, over EVERY scored row)",
              "   No sellout data, so a day that ran out cannot be told from one that was "
              "fully served.",
              "   Both lines below are all-rows figures and n is every row: on a day that "
              "sold out, sold is",
              "   a floor on demand, so the error against it is not an error."]
    L += [f"   {'':10s} {label:>9s} {'n':>7s}",
          f"   {'model':10s} {pct(m['wape_uncensored']):>9s} {m['n_uncensored']:>7,d}",
          f"   {'your par':10s} {pct(acc['par']['wape_uncensored']):>9s} "
          f"{acc['par']['n']:>7,d}"]
    # only when some row actually was flagged: printing the identical number twice under two
    # different headings is how a page teaches a reader to stop reading it
    if censored_known and m["n_uncensored"] < m["n_rows"]:
        L.append(f"   model wape over all rows including sellouts: {pct(m['wape_all_rows'])} "
                 "(NOT a bound in either direction)")
    L += [f"   skill vs your par {pct(acc['skill_vs_par'])}   mean pinball "
          f"{num(m['pinball_censored'], 3)}   bias {pct(m['bias_pct'])}",
          "   your par is the trailing four same-weekday mean over OPEN days -- the same "
          "computation as the",
          "   naive benchmark, minus the closed-day zeros that make the library version look "
          "weaker than it is."]
    for w in acc.get("weekly", []):
        L.append(f"   {w['week']} (from {w['start']}): model {pct(w['model'])} vs par "
                 f"{pct(w['par'])} (n={w['n']})")

    L += ["", "2. CALIBRATION (cov_lo..cov_hi bracket the truth; the gap is the share of days "
          "whose demand",
          "                we did not see -- a flagged sellout, or a day the rule could not "
          "read)",
          f"   {'tau':>6s} {'cov_lo':>8s} {'cov_hi':>8s} {'cov_pt':>8s} {'n_obs':>7s} "
          f"{'n':>7s}"]
    for c in res["calibration"]:
        L.append(f"   {c['tau']:>6.3f} {pct(c['cov_lo']):>8s} {pct(c['cov_hi']):>8s} "
                 f"{pct(c['cov_point']):>8s} {c.get('n_observed', 0):>7,d} {c['n']:>7,d}")
    L.append("   cov_pt is measured over the n_obs observed rows only, so it sits outside "
             "the bracket.")

    ms, bnd = res["measured"], res["bounds"]
    L += ["", "3. WHAT IT WOULD HAVE CHANGED  (MEASURED, then one-sided BOUNDS)",
          f"   production record on {pct(ms['production_coverage'])} of scored rows; "
          f"{ms['n_rows_measured']:,} rows day-fresh AND recorded",
          f"   units produced (actual) {num(ms['units_produced_actual'])}   "
          f"units recommended {num(ms['units_recommended'])}",
          f"   MEASURED status-quo waste {num(ms['waste_observed_units'])} units, "
          f"${num(ms['waste_observed_cost'])} cost, ${num(ms['waste_observed_retail'])} retail",
          f"   BOUND waste saving >= {num(bnd['waste_saving_lower_units'])} units, "
          f"${num(bnd['waste_saving_lower_cost'])} at cost, "
          f"${num(bnd['waste_saving_lower_retail'])} at retail",
          f"   BOUND units definitely missed >= {num(bnd['lost_units_lower'])}, margin >= "
          f"${num(bnd['lost_margin_lower'])}",
          "   no upper bound on lost margin is given: bounding it needs an upper bound on "
          "demand, which",
          "   nothing observable provides",
          f"   sellout days: store {pct(bnd['sellout_days_sq'])} (of "
          f"{bnd.get('n_flag_evaluable', 0):,d} evaluable rows)   model >= "
          f"{pct(bnd['sellout_days_model_lower'])}, <= "
          f"{pct(bnd['sellout_days_model_upper'])}",
          "", "   BY DEPARTMENT",
          f"   {'dept':16s} {'n':>6s} {label:>9s} {'bias':>9s} {'short':>9s} {'save$':>10s}"]
    for r in res["by_dept"]:
        # by_group returns "n/a (n=k)" strings for thin groups; pass those through
        cell = lambda v, f: v if isinstance(v, str) else f(v)
        L.append(f"   {str(r['group'])[:16]:16s} {r['n']:>6,d} "
                 f"{cell(r['wape_uncensored'], pct):>9s} {cell(r['bias_pct'], pct):>9s} "
                 f"{cell(r['sellout_days_model_lower'], pct):>9s} "
                 f"{cell(r['waste_saving_lower_cost'], num):>10s}")

    L += ["", "4. BIGGEST MISSES"]
    for r in res["misses"]:
        L.append(f"   {r['for_date']}  {r['item']:12s} MAKE {num(r['rec_qty'], 0):>7s}  "
                 f"sold {num(r['sold'], 0):>7s}  why: {_clip(r['why'], 36)}")

    ov = res["overrides"]
    L += ["", "5. OVERRIDES",
          f"   {ov['n']} rows keyed in from returned sheets; {ov.get('n_matched', ov['n'])} "
          f"landed on a scored row ({pct(ov['share'])} of those rows)",
          f"   on {ov['n_compared']} fully-served item-days the manager was closer than the "
          f"sheet "
          + (f"{ov['n_closer']} time(s)" if ov["n_closer"] is not None else "n/a")]

    L += ["", "EXCLUSION LEDGER  " + ", ".join(f"{k}={v}" for k, v in
                                               sorted(res["exclusions"].items()))]
    if res["missing_sheets"]:
        L.append("   no sheet was made on: " + ", ".join(res["missing_sheets"]))
    if res.get("data_gaps"):
        L.append("   the export explained no sales on: " + ", ".join(res["data_gaps"])
                 + " (counted against completeness)")
    if res["unscored"]:
        L.append("   forecast but not yet scored: " + ", ".join(res["unscored"]))
    L += ["", "CAVEATS"] + [line for c in res["caveats"] for line in _wrap("  - " + c, width)]

    g = res["gates"]
    L += ["", "GO / NO-GO", "-" * width]
    for key, text in (("G1", "data completeness >= 95%, no unexplained gap over 1 day"),
                      ("G2", "model wape <= 0.90 x par wape, and <= par in 3 of 4 weeks"),
                      ("G3", "cov_point@0.50 within 0.50 +/- 0.10 and cov_lo@0.90 >= 0.75"),
                      ("G4", "waste saving >= 15% of measured waste, sellout days <= +3pts"),
                      ("G5", "no more than 2 of the top-10 dollar items excluded")):
        L.append(f"  {key} {g.get(key, 'PENDING'):>7s}  {text}")
    L.append("  The store switches to live production only when all five read PASS.")
    L.append("=" * width)
    return "\n".join(L)


# ---- state, and the CLI ----

def status(shadow_dir):
    preds = read_predictions(shadow_dir)
    scores = read_scores(shadow_dir)
    sheets = sorted(f[:-4] for f in os.listdir(os.path.join(shadow_dir, "sheets"))
                    if f.endswith(".txt")) if os.path.isdir(os.path.join(shadow_dir, "sheets")) \
        else []
    logged = sorted(pd.Timestamp(d) for d in preds["for_date"].unique()) if len(preds) else []
    scored = sorted(pd.Timestamp(d) for d in scores["for_date"].unique()) if len(scores) else []
    state = {}
    path = os.path.join(shadow_dir, "state.json")
    if os.path.exists(path):
        with open(path) as f:
            state = json.load(f)
    return dict(
        last_ingested_date=state.get("last_ingested_date"),
        last_sheet_date=sheets[-1] if sheets else None,
        last_scored_date=str(scored[-1].date()) if scored else None,
        missed_sheet_dates=state.get("missed_sheet_dates", []),
        unscored_dates=[str(d.date()) for d in logged if d not in scored],
        current_model_version=str(preds["model_version"].iloc[-1]) if len(preds) else None,
        current_artifacts_dir=state.get("current_artifacts_dir"),
        sellout_source=str(preds["sellout_source"].iloc[-1]) if len(preds) else None,
        last_gates=state.get("last_gates", {}),
    )


def _write_state(shadow_dir, **fields):
    path = os.path.join(shadow_dir, "state.json")
    state = {}
    if os.path.exists(path):
        with open(path) as f:
            state = json.load(f)
    state.update({k: v for k, v in fields.items() if v is not None})
    os.makedirs(shadow_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True, default=str)


def _panel(path):
    df = schema.read_panel(path)
    schema.assert_no_truth(df)
    return df


def _cmd_morning(args):
    panel = _panel(args.panel)
    items = ht_config.load_items(args.items)
    recs = forecast(panel, args.artifacts, items, args.date, allow_backfill=args.backfill,
                    max_staleness=args.max_staleness)
    n = log_predictions(recs, args.out, store=args.store,
                        items_config_hash=ht_config.config_hash(args.items))
    meta = _load_meta(args.artifacts)
    known = pd.to_numeric(panel.get("stockout_known"), errors="coerce")
    caveats = sheet_caveats(recs, items, meta,
                            float(known.mean()) if known is not None else 1.0)
    yest = _yesterday(panel, args.out, args.date, items)
    os.makedirs(os.path.join(args.out, "sheets"), exist_ok=True)
    day = pd.Timestamp(args.date).date()
    text = None
    for fmt in (("text", "html") if args.format == "both" else (args.format,)):
        body = morning_sheet(recs, store=args.store, for_date=args.date,
                             conditions=recs.attrs["conditions"], yesterday=yest,
                             caveats=caveats, fmt=fmt)
        ext = "txt" if fmt == "text" else "html"
        with open(os.path.join(args.out, "sheets", f"{day}.{ext}"), "w") as f:
            f.write(body)
        text = body if fmt == "text" else text
    _write_state(args.out, last_sheet_date=str(day), current_artifacts_dir=args.artifacts,
                 last_ingested_date=str(pd.Timestamp(panel["date"].max()).date()),
                 current_model_version=recs["model_version"].iloc[0],
                 sellout_source=recs["sellout_source"].iloc[0])
    print(text if text is not None else f"wrote {n} predictions and the HTML sheet")
    return 0


def _cmd_enter(args):
    items = ht_config.load_items(args.items)
    for_date = pd.Timestamp(args.date).normalize()
    preds = read_predictions(args.out, for_date, for_date)
    if preds.empty:
        print(f"warning: no sheet was logged for {for_date.date()}; recording it anyway",
              file=sys.stderr)
    if args.file == "-" or (args.file is None and not sys.stdin.isatty()):
        lines = sys.stdin.read().splitlines()
    elif args.file:
        # utf-8-sig: Excel's "CSV UTF-8" writes a byte-order mark, which otherwise becomes
        # part of the first item name and refuses the whole sheet over three invisible bytes
        with open(args.file, encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    else:
        said = dict(zip(preds["item"].astype(str), preds["rec_qty"])) if len(preds) else {}
        lines = prompt_entries(items, entry_order(args.out, for_date, items), said)
    rows, errors = parse_entries(lines, items)
    if errors:
        print("nothing was written. Fix these and run it again:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    if not rows:
        print("nothing to record")
        return 0
    path = record_actuals(args.out, for_date, rows, entered_by=args.by)
    n_out = sum(1 for r in rows if r["sold_out_at"])
    n_made = sum(1 for r in rows if r["actual_produced"] != "")
    print(f"{for_date.date()}: {len(rows)} item(s) -> {path}   "
          f"{n_made} with a production number, {n_out} sold out")
    print(f"now run: score --date {for_date.date()} to fold it into the record")
    return 0


def _cmd_score(args):
    panel = _panel(args.panel)
    items = ht_config.load_items(args.items)
    rows = score_day(panel, items, args.out, args.date)
    counts = rows["status"].value_counts().to_dict()
    _write_state(args.out, last_scored_date=str(pd.Timestamp(args.date).date()))
    print(f"{pd.Timestamp(args.date).date()}: " + ", ".join(f"{k}={v}" for k, v in
                                                            sorted(counts.items())))
    return 0


def _cmd_catch_up(args):
    panel = _panel(args.panel)
    items = ht_config.load_items(args.items)
    done = catch_up(panel, items, args.out, since=args.since)
    st = status(args.out)
    print("scored: " + (", ".join(done) if done else "nothing new"))
    print("still waiting on data: " + (", ".join(st["unscored_dates"]) or "none"))
    return 0


def _cmd_weekly(args):
    panel = _panel(args.panel)
    items = ht_config.load_items(args.items)
    res = weekly_report(args.out, panel, items, args.week_ending, weeks=args.weeks,
                        include_backfilled=args.include_backfilled)
    os.makedirs(os.path.join(args.out, "weekly"), exist_ok=True)
    stem = os.path.join(args.out, "weekly", res["week"])
    if args.format in ("text", "both"):
        with open(stem + ".txt", "w") as f:
            f.write(format_weekly(res, "text", args.width) + "\n")
    if args.format in ("json", "both"):
        with open(stem + ".json", "w") as f:
            f.write(format_weekly(res, "json") + "\n")
    _write_state(args.out, last_gates=res["gates"])
    print(format_weekly(res, "text", args.width))
    return 0


def _cmd_status(args):
    print(json.dumps(status(args.out), indent=1, sort_keys=True, default=str))
    return 0


def _check_args(args):
    """Refuse a mistyped path or date with one line, before anything is read or written.

    The person running this at 5:30am is a store employee at a back-room terminal, and a
    pandas traceback tells them nothing they can do anything about. Same contract as
    ht.ingest's main: one sentence naming the flag, exit 1.
    """
    for flag, what in (("panel", "panel csv"), ("items", "items config"),
                       ("file", "entry file")):
        path = getattr(args, flag, None)
        if path and path != "-" and not os.path.exists(path):
            raise schema.HtError(f"--{flag}: no {what} at {path}")
    art = getattr(args, "artifacts", None)
    if art and not os.path.exists(os.path.join(art, "meta.json")):
        raise schema.HtError(f"--artifacts: {art} has no meta.json -- point it at a trained "
                             f"model directory, e.g. model/artifacts")
    for flag in ("date", "week_ending", "since"):
        value = getattr(args, flag, None)
        if value is None:
            continue
        try:
            pd.Timestamp(value)
        except ValueError as exc:
            raise schema.HtError(f"--{flag.replace('_', '-')}: {value!r} is not a date "
                                 f"({exc}); write it as YYYY-MM-DD")
    # only `morning` creates the shadow directory; for the rest a typo in --out would
    # silently read an empty log and report a pilot that lost its record
    if args.cmd != "morning" and not os.path.isdir(args.out):
        raise schema.HtError(f"--out: no shadow directory at {args.out}; `morning` creates "
                             f"it and every other command works inside it")


def main(argv=None):
    ap = argparse.ArgumentParser(description="shadow mode: morning sheet, log, scores, weekly")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("morning", help="tomorrow's sheet, logged before it is printed")
    m.add_argument("--panel", required=True)
    m.add_argument("--artifacts", required=True)
    m.add_argument("--items", required=True)
    m.add_argument("--date", required=True)
    m.add_argument("--out", default="shadow")
    m.add_argument("--store", default="")
    m.add_argument("--format", choices=("text", "html", "both"), default="both")
    m.add_argument("--backfill", action="store_true")
    m.add_argument("--max-staleness", type=int, default=MAX_STALENESS_DAYS)
    m.set_defaults(fn=_cmd_morning)

    e = sub.add_parser("enter", help="key yesterday's returned paper sheet back in")
    e.add_argument("--items", required=True)
    e.add_argument("--date", required=True, help="the date ON the sheet, not today")
    e.add_argument("--out", default="shadow")
    e.add_argument("--file", default=None,
                   help="'item,made,sold out at[,note]' per line; '-' or a pipe reads stdin. "
                        "Omit it and every item is prompted for in the sheet's own order")
    e.add_argument("--by", default=os.environ.get("USER", ""),
                   help="who keyed it in; recorded on every row")
    e.set_defaults(fn=_cmd_enter)

    s = sub.add_parser("score", help="freeze one day's verdict")
    s.add_argument("--panel", required=True)
    s.add_argument("--items", required=True)
    s.add_argument("--date", required=True)
    s.add_argument("--out", default="shadow")
    s.set_defaults(fn=_cmd_score)

    c = sub.add_parser("catch-up", help="score every logged day that now has data")
    c.add_argument("--panel", required=True)
    c.add_argument("--items", required=True)
    c.add_argument("--since", default=None)
    c.add_argument("--out", default="shadow")
    c.set_defaults(fn=_cmd_catch_up)

    w = sub.add_parser("weekly", help="the district-manager page and the go/no-go gates")
    w.add_argument("--panel", required=True)
    w.add_argument("--items", required=True)
    w.add_argument("--artifacts", default=None)
    w.add_argument("--week-ending", required=True)
    w.add_argument("--weeks", type=int, default=1)
    w.add_argument("--out", default="shadow")
    w.add_argument("--format", choices=("text", "json", "both"), default="both")
    w.add_argument("--width", type=int, default=WEEKLY_WIDTH)
    w.add_argument("--include-backfilled", action="store_true",
                   help="fold replayed sheets into the numbers and stamp the page "
                        "RECONSTRUCTED (the dress rehearsal; never a real pilot week)")
    w.set_defaults(fn=_cmd_weekly)

    st = sub.add_parser("status", help="what is behind, in one screen")
    st.add_argument("--out", default="shadow")
    st.set_defaults(fn=_cmd_status)

    args = ap.parse_args(argv)
    try:
        _check_args(args)
        return args.fn(args)
    except schema.HtError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        # forecast() already raises these with the sentence a person needs ("panel carries
        # actuals through ... pass --backfill"); the traceback above it was the only problem
        print(f"{args.cmd} failed: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        # Ctrl-C and Ctrl-D are both a person leaving the prompt, not a bug. EOFError also
        # arrives when the terminal closes mid-entry, which is a back room at 6am.
        print("\nabandoned; nothing was written", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
