"""One gate that says whether a panel is fit to train on, and what a store would have to fix.

It reports instead of raising, because the person holding a broken export should see every
problem at once rather than discover them one run at a time over an afternoon. The output is
also the document you send back to the store: it names the item, the dates and the count, not
the code path.

Three levels, and the distinction is the whole point. An ERROR means a number computed from
this panel would be wrong -- duplicate item-days, a hole in a date index, sold above produced,
too little history to split. A WARNING means training is fine but the reader has to know --
no sellout signal, a promotion the model will fit as noise, an item that will be dropped. An
INFO records an auto-repair ingest already made, so that a grid-filled zero or a clipped
refund is visible here rather than only in a JSON report nobody opens.
"""
import argparse
import collections
import json
import sys

import numpy as np
import pandas as pd

from . import calendar as ht_calendar
from . import config as ht_config
from . import schema
from .schema import ValidationFailed

Finding = collections.namedtuple("Finding", "level check item message count")

LEVELS = ("error", "warning", "info")

# model/features.py is being parameterized in parallel; read its constants when they are there
# and fall back to the frozen specification's numbers when they are not, so this module is
# runnable on its own. The two must not drift -- tests/test_features_real.py pins them.
try:                                                    # pragma: no cover - import shim
    from model import features as _features
except Exception:                                       # pragma: no cover
    _features = None


def _const(name, default):
    return getattr(_features, name, default) if _features is not None else default


CONTEXT_DAYS = _const("CONTEXT_DAYS", 28)
MIN_PANEL_DAYS = _const("MIN_PANEL_DAYS", 126)
MIN_PANEL_DAYS_NO_TEST = _const("MIN_PANEL_DAYS_NO_TEST", 98)
MIN_PANEL_DAYS_SHORT = _const("MIN_PANEL_DAYS_SHORT", 70)
MIN_ITEM_TRAIN_DAYS = _const("MIN_ITEM_TRAIN_DAYS", 84)
MIN_TRAIN_DAYS = _const("MIN_TRAIN_DAYS", 84)
MIN_TRAIN_DAYS_SHORT = _const("MIN_TRAIN_DAYS_SHORT", 56)

SELLOUT_RATE_LOW = 0.03
SELLOUT_RATE_HIGH = 0.60
GRID_SHARE_WARN = 0.02
PRODUCED_MISSING_WARN = 0.20
PRICE_DIVERGENCE_WARN = 0.05
MAX_SELLOUT_LATENCY_DAYS = 1


def _split_preview(panel, split_opts=None):
    """The boundaries features.build would resolve, from the panel's own range.

    split_opts carries the caller's split mode (no_test, allow_short, val_days, test_days).
    It has to: the history floor is 126 days for a train/val/test split but 98 with --no-test
    and 70 with --allow-short, so a validator that assumed the strictest floor would reject a
    panel for being too short to do something the caller never asked it to do -- while its own
    message told them to pass the flag they had already passed.
    """
    opts = dict(split_opts or {})
    open_rows = panel[panel.is_closed == 0]
    if not len(open_rows):
        return dict(span_days=0, error="no open rows")
    dates = open_rows["date"]
    if _features is not None and hasattr(_features, "resolve_splits"):
        try:
            return dict(_features.resolve_splits(dates.to_numpy(), **opts))
        except TypeError:                               # an older features without the knobs
            return dict(_features.resolve_splits(dates.to_numpy()))
        except Exception as exc:                        # a real InsufficientHistory, reported
            return dict(span_days=int((dates.max() - dates.min()).days) + 1, error=str(exc))
    span = int((dates.max() - dates.min()).days) + 1
    no_test, short = bool(opts.get("no_test")), bool(opts.get("allow_short"))
    test_days = 0 if (no_test or short) else int(min(max(round(0.20 * span), 28), 365))
    val_days = 14 if short else int(min(max(round(0.10 * span), 14), 84))
    train_days = span - test_days - val_days
    test_start = dates.max() - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    out = dict(span_days=span, test_days=test_days, val_days=val_days, train_days=train_days,
               train_end=str((val_start - pd.Timedelta(days=1)).date()),
               val_start=str(val_start.date()), test_start=str(test_start.date()),
               thin=False, source="ht.validate fallback")
    floor = MIN_TRAIN_DAYS_SHORT if short else MIN_TRAIN_DAYS
    if train_days < floor:
        out["error"] = (f"panel covers {span} days ({dates.min().date()}..{dates.max().date()}); "
                        f"a train/val/test split needs {MIN_PANEL_DAYS} days "
                        f"({CONTEXT_DAYS} context + {MIN_TRAIN_DAYS - CONTEXT_DAYS} train targets "
                        f"+ 14 val + 28 test). Ask the store for 104 weeks of item movement, or "
                        f"pass --no-test (needs {MIN_PANEL_DAYS_NO_TEST}) or --allow-short "
                        f"(needs {MIN_PANEL_DAYS_SHORT})")
    return out


def _runs(mask, dates):
    """[(start, end, length)] for each contiguous run where mask is true."""
    out, start = [], None
    values = list(mask)
    for i, flag in enumerate(values):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(values) - 1):
            end = i if (flag and i == len(values) - 1) else i - 1
            out.append((dates.iloc[start], dates.iloc[end], end - start + 1))
            start = None
    return out


def item_census(panel, context_days=CONTEXT_DAYS, train_end=None):
    """Per item: what history there is, and whether it is enough. The table you hand the store."""
    rows = []
    for item, grp in panel.groupby("item", sort=True):
        grp = grp.sort_values("date")
        open_rows = grp[grp.is_closed == 0]
        full = pd.date_range(grp.date.min(), grp.date.max(), freq="D")
        holes = _runs(~full.isin(set(grp.date)), pd.Series(full))
        gaps = _runs((grp.is_closed == 1).to_numpy(), grp["date"].reset_index(drop=True))
        train = open_rows if train_end is None else open_rows[open_rows.date <= train_end]
        seen = open_rows[open_rows.stockout_known == 1]
        rows.append(dict(
            item=item, rows=len(grp), first=grp.date.min().date(), last=grp.date.max().date(),
            open_days=len(open_rows), open_train_days=len(train),
            train_targets=max(len(train) - context_days, 0),
            missing_days=int(sum(h[2] for h in holes)),
            zero_days=int((open_rows.sold == 0).sum()),
            sellout_rate=round(float(seen.stockout.mean()), 3) if len(seen) else None,
            longest_gap=max([h[2] for h in holes] + [g[2] for g in gaps] + [0]),
            status="ok" if len(train) >= MIN_ITEM_TRAIN_DAYS else "short",
            reason="" if len(train) >= MIN_ITEM_TRAIN_DAYS
            else f"{len(train)} of {MIN_ITEM_TRAIN_DAYS} open train days",
        ))
    return pd.DataFrame(rows)


def gap_census(panel):
    """Every run of non-ok rows, per item, with the reason ingest recorded for it."""
    rows = []
    for item, grp in panel.groupby("item", sort=True):
        grp = grp.sort_values("date").reset_index(drop=True)
        for status in [s for s in schema.ROW_STATUS if s != "ok"]:
            for start, end, length in _runs((grp.row_status == status).to_numpy(), grp["date"]):
                rows.append(dict(item=item, row_status=status, start=start.date(),
                                 end=end.date(), days=length))
        full = pd.date_range(grp.date.min(), grp.date.max(), freq="D")
        for start, end, length in _runs(~full.isin(set(grp.date)), pd.Series(full)):
            rows.append(dict(item=item, row_status="absent", start=start.date(),
                             end=end.date(), days=length))
    out = pd.DataFrame(rows, columns=["item", "row_status", "start", "end", "days"])
    return out.sort_values(["item", "start"]).reset_index(drop=True)


# ---- the checks ----

def _structural(panel, items, mapping, add):
    dupes = panel.duplicated(list(schema.KEY), keep=False)
    if dupes.any():
        sample = panel.loc[dupes, list(schema.KEY)].head(3).to_dict("records")
        add("error", "duplicate_key", None,
            f"{int(dupes.sum())} rows share a (store, item, date); every metric downstream "
            f"double-counts them -- {sample}", int(dupes.sum()))

    stray = sorted(set(panel.item) - set(items))
    if stray:
        add("error", "item_not_in_config", None,
            f"item(s) {stray} appear in the panel but not in the items config, so they have no "
            "price, cost or batch and cannot be forecast or costed", len(stray))
    roster = max(len(items), 1)
    if panel.item.nunique() > 3 * roster:
        add("error", "item_explosion", None,
            f"{panel.item.nunique()} distinct items against a config roster of {roster}; that is "
            "the random-weight barcode failure, and it sizes the model's item embedding wrong",
            panel.item.nunique())

    for name, allowed in schema.ENUMS.items():
        off = panel.loc[~panel[name].isin(allowed), name]
        if len(off):
            add("error", f"{name}_vocabulary", None,
                f"{len(off)} row(s) outside {list(allowed)}: "
                f"{off.value_counts().head(3).to_dict()}", len(off))
    for name in ("stockout", "stockout_known", "is_closed", "payday"):
        off = ~panel[name].isin([0, 1])
        if off.any():
            add("error", f"{name}_not_binary", None,
                f"{int(off.sum())} row(s) have {name} outside {{0, 1}}", int(off.sum()))
    if panel["holiday"].isna().any():
        add("error", "holiday_null", None,
            f"{int(panel.holiday.isna().sum())} rows have a null holiday; a conformed panel "
            'writes "" for "no holiday", and NaN != "" silently kills the countdown covariate',
            int(panel.holiday.isna().sum()))

    inv = panel.is_closed != (panel.row_status != "ok").astype(int)
    if inv.any():
        add("error", "is_closed_invariant", None,
            f"{int(inv.sum())} rows break is_closed == int(row_status != 'ok'), so the model "
            "flag and its diagnostic disagree about which days count", int(inv.sum()))


def _quantities(panel, items, mapping, add):
    horizon = panel.date.max()
    bad = panel.sold.isna() & (panel.date < horizon)
    if bad.any():
        add("error", "sold_null", None,
            f"{int(bad.sum())} rows have a null `sold` before the panel's last date; only a "
            "horizon row (tomorrow's covariates, no actuals yet) may be null", int(bad.sum()))
    for name in ("sold", "produced", "wasted", "tmax_f", "unit_price", "unit_cost"):
        inf = np.isinf(panel[name].to_numpy(dtype=float))
        if inf.any():
            add("error", f"{name}_infinite", None,
                f"{int(inf.sum())} rows have a non-finite {name}", int(inf.sum()))
    neg = panel.sold < 0
    if neg.any():
        add("error", "sold_negative", None,
            f"{int(neg.sum())} rows have negative sales; a refund line has to be netted into "
            "the day it belongs to, or clipped and marked row_status='suspect'", int(neg.sum()))
    # only row_status 'closed' -- a partial day is an early close that still sold, and a
    # 'suspect' row is a clipped refund that may legitimately carry sales
    shut = panel[(panel.row_status == "closed") & (panel.sold > 0)]
    if len(shut):
        add("error", "closed_with_sales", None,
            f"{len(shut)} rows are marked closed but sold something; either the closure list is "
            "wrong or the export is dated in a different timezone from the store's day close",
            len(shut))

    prod = (mapping or {}).get("production", {})
    policy = prod.get("overrun_policy", "warn")
    max_share = float(prod.get("max_overrun_share", 0.02))
    for item, grp in panel.groupby("item", sort=True):
        have = grp[grp.produced.notna()]
        if not len(have) or item not in items:
            continue
        if items[item]["shelf_life_days"] > 1:
            # yesterday's tray sells today: sold above produced is normal, not a defect, and
            # nothing derives waste for these items anyway
            continue
        eps = ht_config.resolve_tolerance(items[item])
        over = have.sold > have.produced + eps
        if not over.any():
            continue
        share = float(over.mean())
        worst = float((have.sold - have.produced).max())
        days = ", ".join(str(d.date()) for d in have.loc[over, "date"].head(5))
        # a production count is usually a proxy -- a label log, a clipboard -- and a proxy
        # undercounts. Past the configured share it is not a proxy's error rate any more, it
        # is the wrong column.
        level = "error" if (policy == "error" or share > max_share) else "warning"
        add(level, "sold_above_produced", item,
            f"{int(over.sum())} of {len(have)} days ({share:.1%}) sold more than was produced "
            f"(worst {worst:.1f} units, first {days}); the sellout rule reads those as sellouts "
            f"and the measured waste baseline loses those units. production.max_overrun_share "
            f"is {max_share:.0%}", int(over.sum()))

    for item, grp in panel.groupby("item", sort=True):
        sold = grp.loc[grp.is_closed == 0, "sold"].astype(float)
        if not len(sold):
            continue
        if sold.max() <= 0:
            add("warning", "all_zero_series", item,
                f"every one of {len(sold)} open days sold zero; the item is not carried, is rung "
                "to another department, or its code is mismapped", len(sold))
        elif sold.nunique() == 1:
            add("warning", "constant_series", item,
                f"all {len(sold)} open days sold exactly {sold.iloc[0]:g}; a constant series has "
                "no signal to fit and usually means a par value was exported instead of sales",
                len(sold))
        else:
            ceiling = float(sold.quantile(0.95)) * 10
            wild = sold[sold > max(ceiling, 1.0)]
            if len(wild):
                add("warning", "absurd_quantity", item,
                    f"{len(wild)} day(s) sold more than 10x the item's 95th percentile "
                    f"(max {sold.max():g} against p95 {sold.quantile(0.95):g}); check for a "
                    "decimal point or a case-vs-each units change", len(wild))


def _coverage(panel, items, mapping, add):
    max_gap = int((mapping or {}).get("gaps", {}).get("max_unexplained_gap_days", 3))
    for item, grp in panel.groupby("item", sort=True):
        grp = grp.sort_values("date")
        if not grp.date.is_monotonic_increasing:
            add("error", "unsorted_dates", item, "its date index is not sorted", len(grp))
        full = pd.date_range(grp.date.min(), grp.date.max(), freq="D")
        holes = _runs(~full.isin(set(grp.date)), pd.Series(full))
        if holes:
            longest = max(h[2] for h in holes)
            total = sum(h[2] for h in holes)
            level = "error" if longest > max_gap else "warning"
            add(level, "date_gap", item,
                f"{total} date(s) are absent from its index in {len(holes)} run(s), longest "
                f"{longest} days from {holes[0][0].date()}; ingest fills a complete grid and "
                "records why each inserted day is there, so a hole here means the panel was "
                "assembled some other way", total)

        train = grp[grp.is_closed == 0]
        if len(train) < MIN_ITEM_TRAIN_DAYS:
            add("warning", "short_history", item,
                f"{len(train)} open days against the {MIN_ITEM_TRAIN_DAYS}-day per-item floor; "
                "features.build will exclude it and the morning sheet will print a trailing par "
                "under NO FORECAST instead of a forecast", len(train))

    for item, grp in panel.groupby("item", sort=True):
        grp = grp.sort_values("date")
        runs = _runs((grp.row_status == "missing").to_numpy(), grp.date.reset_index(drop=True))
        long = [r for r in runs if r[2] > max_gap]
        if long:
            add("warning", "unexplained_outage", item,
                f"{len(long)} run(s) of days the export explained nothing about, longest "
                f"{max(r[2] for r in long)} days from {long[0][0].date()}; they are context "
                "only, so they leave training silently unless somebody reads this line. Declare "
                "them in closures.dates, or supply the missing days",
                sum(r[2] for r in long))

    # A store that closes for a day and simply omits it comes back as a genuine zero-sales
    # day for every item at once, which no per-item check can see. mapping.closures.dates is
    # the field that would say so; without it the model learns the collapse.
    open_rows = panel[panel.is_closed == 0]
    if len(open_rows) and open_rows.item.nunique() > 1:
        per_day = open_rows.groupby("date").agg(items=("item", "nunique"),
                                                sold=("sold", "max"))
        dead = per_day[(per_day["sold"] <= 0) & (per_day["items"] == open_rows.item.nunique())]
        if len(dead):
            add("warning", "store_wide_zero_day", None,
                f"{len(dead)} date(s) have every item open and selling zero (first "
                f"{dead.index[0].date()}); that is a closure the export did not declare, and "
                "the model trains on it as real zero demand. Add it to mapping.closures.dates "
                "or supply the store hours file", len(dead))

    inserted = int((panel.row_status == "missing").sum())
    if inserted and inserted / len(panel) > GRID_SHARE_WARN:
        add("warning", "grid_share", None,
            f"{inserted} rows ({inserted / len(panel):.1%}) are grid-inserted 'missing' days "
            "rather than observations; every one of them is a day the export did not explain",
            inserted)
    for status in ("suspect", "partial", "closed", "missing", "not_carried"):
        n = int((panel.row_status == status).sum())
        if n:
            add("info", f"repair_{status}", None,
                f"{n} row(s) carry row_status='{status}'; they are context only and never a "
                "training or scoring target", n)


def _sellout(panel, items, mapping, add):
    source = sorted(set(panel.sellout_source))
    open_rows = panel[panel.is_closed == 0]
    known_share = float(open_rows.stockout_known.mean()) if len(open_rows) else 0.0
    seen = open_rows[open_rows.stockout_known == 1]
    rate = float(seen.stockout.mean()) if len(seen) else None

    if "none" in source or "unknown" in source:
        add("warning", "no_sellout_signal", None,
            f"sellout_source is {source}: the model fits the distribution of CENSORED SALES, "
            "not demand, biasing the recommended quantity 1-8% low on the busiest days. That is "
            "the safe direction and a supported mode -- do not compensate by inflating q*",
            len(panel))
    if known_share < 1.0:
        add("warning", "sellout_coverage", None,
            f"the sellout rule could be evaluated on {known_share:.1%} of open item-days; the "
            "rest are 'we have no idea', not 'did not sell out'",
            int((1 - known_share) * len(open_rows)))
    if rate is not None:
        if rate == 0:
            add("warning", "sellout_rate_zero", None,
                "no open day is flagged as a sellout; a store that never runs out is producing "
                "far too much, so this is almost always a coverage problem", 0)
        elif rate < SELLOUT_RATE_LOW:
            add("warning", "sellout_rate_low", None,
                f"sellout rate is {rate:.1%}, below {SELLOUT_RATE_LOW:.0%}; usually the rule's "
                "recall, not the store's service level", int(seen.stockout.sum()))
        elif rate > SELLOUT_RATE_HIGH:
            add("warning", "sellout_rate_high", None,
                f"sellout rate is {rate:.1%}; above {SELLOUT_RATE_HIGH:.0%} most days are "
                "censored, so the fitted quantiles are a lower bound on demand almost everywhere",
                int(seen.stockout.sum()))

    rule = (mapping or {}).get("sellout", {}).get("rule")
    if rule is not None:
        latency = {"produced_vs_sold": 1, "flag": 1, "none": 0}.get(rule, 99)
        if latency > MAX_SELLOUT_LATENCY_DAYS:
            add("error", "sellout_latency", None,
                f"sellout rule {rule!r} has {latency}-day latency; stockout is an encoder input "
                "across the trailing 28 days, so a flag that is not computable on yesterday's "
                "data is train/serve skew invisible in every offline metric", latency)

    missing_produced = float(panel.produced.isna().mean())
    if missing_produced > PRODUCED_MISSING_WARN:
        add("warning", "produced_missing", None,
            f"{missing_produced:.0%} of item-days have no production record; waste cannot be "
            "measured there and the sellout flag is unknown on those days",
            int(panel.produced.isna().sum()))
    return dict(sources=source, known_share=round(known_share, 4),
                rate=None if rate is None else round(rate, 4),
                by_source={k: int(v) for k, v in panel.sellout_source.value_counts().items()})


def _economics(panel, items, mapping, add):
    tol = float((mapping or {}).get("price_cost", {}).get("tolerance_pct", 0.15))
    for item, grp in panel.groupby("item", sort=True):
        if item not in items:
            continue
        cfg = float(items[item]["price"])
        realized = grp.unit_price.astype(float)
        off = (realized - cfg).abs() / cfg > tol if cfg > 0 else pd.Series(False, grp.index)
        if off.mean() > PRICE_DIVERGENCE_WARN:
            add("warning", "price_divergence", item,
                f"realized unit price differs from the config price ({cfg:.2f}) by more than "
                f"{tol:.0%} on {off.mean():.0%} of days; that is usually the weekly ad, which "
                "the model does not see and will fit as noise", int(off.sum()))
        cfg_cost = float(items[item]["cost"])
        realized_cost = grp.unit_cost.astype(float)
        off_cost = ((realized_cost - cfg_cost).abs() / cfg_cost > tol if cfg_cost > 0
                    else pd.Series(False, grp.index))
        if off_cost.mean() > PRICE_DIVERGENCE_WARN:
            add("warning", "cost_divergence", item,
                f"the panel's unit cost differs from the config cost ({cfg_cost:.2f}) by more "
                f"than {tol:.0%} on {off_cost.mean():.0%} of days (median "
                f"{float(realized_cost.median()):.2f}); check price_cost.cost_basis -- a "
                "per-unit column summed as an extended one multiplies every settlement dollar "
                "by the day's line count", int(off_cost.sum()))
        if items[item]["shelf_life_days"] > 1 and grp.wasted.notna().any():
            add("warning", "multi_day_waste", item,
                f"shelf_life_days is {items[item]['shelf_life_days']} but a wasted column is "
                "present; wasted = produced - sold is false for a multi-day item, and it is "
                "excluded from the waste bound", int(grp.wasted.notna().sum()))
    for warn in ht_config.validate_items(items, panel):
        add("warning", "items_config", None, warn, 1)
    if mapping:
        for warn in ht_config.validate_mapping(mapping, items):
            add("warning", "mapping_config", None, warn, 1)


def _calendar_weather(panel, add):
    names = sorted(set(panel.holiday) - {""})
    stray = [n for n in names if n not in ht_calendar.HOLIDAY_NAMES]
    if stray:
        add("warning", "holiday_vocabulary", None,
            f"holiday name(s) {stray} are outside ht.calendar.HOLIDAY_NAMES; the one-hot "
            "vocabulary has to grow, so the frozen 9-item checkpoint cannot be reused", len(stray))
    unknown = int((panel.weather == "unknown").sum())
    if unknown:
        add("warning", "weather_unknown", None,
            f"{unknown} item-days have weather 'unknown'; the four weather one-hot slots are all "
            "zero there, which the model reads as 'no information' rather than a wrong category",
            unknown)
    if panel.tmax_f.isna().any():
        add("warning", "tmax_missing", None,
            f"{int(panel.tmax_f.isna().sum())} item-days have no high temperature; features "
            "z-scores a missing tmax to 0.0 rather than dropping the row",
            int(panel.tmax_f.isna().sum()))
    if panel.snow_tomorrow.sum() == 0:
        add("info", "snow_tomorrow_flat", None,
            "snow_tomorrow is 0 on every day; either this store sees no snow or the column is a "
            "hindcast with nothing to hindcast", 0)


def validate(panel, items, mapping=None, strict=False, split_opts=None):
    """Every check, run at once. Returns the report; raises only under strict.

    split_opts is the split mode the caller intends to train under -- see _split_preview.
    """
    schema.assert_no_truth(panel)
    findings = []

    def add(level, check, item, message, count):
        findings.append(Finding(level, check, item, message, int(count)))

    try:
        panel = schema.conform(panel, keep_extra=True)
    except schema.SchemaError as exc:
        for line in str(exc).splitlines()[1:]:
            add("error", "schema", None, line.strip(), 1)
        return dict(ok=False, findings=findings, counts={}, coverage={}, date_range=[],
                    item_census=pd.DataFrame(), gap_census=pd.DataFrame(), sellout={},
                    splits_preview={}, excluded_items_preview=[])

    _structural(panel, items, mapping, add)
    _quantities(panel, items, mapping, add)
    _coverage(panel, items, mapping, add)
    sellout = _sellout(panel, items, mapping, add)
    _economics(panel, items, mapping, add)
    _calendar_weather(panel, add)

    splits = _split_preview(panel, split_opts)
    if "error" in splits:
        add("error", "insufficient_history", None, splits["error"], splits.get("span_days", 0))

    census = item_census(panel, train_end=(pd.Timestamp(splits["train_end"])
                                           if splits.get("train_end") else None))
    excluded = [dict(item=r["item"], open_train_days=int(r["open_train_days"]),
                     required=MIN_ITEM_TRAIN_DAYS, reason=r["reason"])
                for _, r in census.iterrows() if r["status"] != "ok"]
    if len(excluded) == len(census) and len(census):
        add("error", "all_items_excluded", None,
            "no item clears the per-item history floor, so there is nothing to train on",
            len(census))

    open_rows = panel[panel.is_closed == 0]
    report = dict(
        ok=not any(f.level == "error" for f in findings),
        findings=findings,
        counts=dict(rows=len(panel), items=int(panel.item.nunique()),
                    dates=int(panel.date.nunique()), open_rows=len(open_rows),
                    row_status={k: int(v) for k, v in panel.row_status.value_counts().items()}),
        coverage=dict(
            open_share=round(float((panel.is_closed == 0).mean()), 4),
            produced_share=round(float(panel.produced.notna().mean()), 4),
            wasted_share=round(float(panel.wasted.notna().mean()), 4),
            zero_share=round(float((open_rows.sold == 0).mean()), 4) if len(open_rows) else None,
        ),
        date_range=[str(panel.date.min().date()), str(panel.date.max().date())],
        item_census=census, gap_census=gap_census(panel), sellout=sellout,
        splits_preview=splits, excluded_items_preview=excluded,
    )
    if strict and not report["ok"]:
        raise ValidationFailed(f"{sum(f.level == 'error' for f in findings)} error-level "
                               "finding(s); the panel is not fit to train on")
    return report


# ---- rendering ----

def _wrap(text, width, indent):
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    return ("\n" + " " * indent).join(out)


def format_report(report, width=100):
    lines = ["VALIDATION REPORT", "=" * width]
    if not report["date_range"]:
        lines.append("panel does not conform; nothing else could be checked")
    else:
        c, cov = report["counts"], report["coverage"]
        lines.append(f"{report['date_range'][0]} .. {report['date_range'][1]}   "
                     f"{c['rows']} rows   {c['items']} items   {c['dates']} dates   "
                     f"{c['open_rows']} open")
        lines.append(f"row_status {c['row_status']}")
        lines.append(f"coverage   open {cov['open_share']:.1%}  production "
                     f"{cov['produced_share']:.1%}  waste {cov['wasted_share']:.1%}  "
                     f"zero-sales days {cov['zero_share']:.1%}"
                     if cov.get("zero_share") is not None else "coverage   n/a")

    for level in LEVELS:
        hits = [f for f in report["findings"] if f.level == level]
        lines.append("")
        lines.append(f"{level.upper()}S ({len(hits)})")
        lines.append("-" * width)
        if not hits:
            lines.append("  none")
        for f in hits:
            head = f"  [{f.check}]" + (f" {f.item}" if f.item else "")
            lines.append(f"{head}: {_wrap(f.message, width - 4, 4)}")

    if report["date_range"]:
        s = report["sellout"]
        lines += ["", "SELLOUT", "-" * width,
                  f"  sources {s['sources']}  known_share {s['known_share']}  rate {s['rate']}",
                  f"  rows by source {s['by_source']}"]
        sp = report["splits_preview"]
        lines += ["", "SPLITS features.build WOULD RESOLVE", "-" * width,
                  "  " + "  ".join(f"{k}={v}" for k, v in sp.items() if k != "error")]
        if sp.get("error"):
            lines.append("  " + _wrap(sp["error"], width - 4, 4))
        lines += ["", "ITEM CENSUS", "-" * width]
        lines.append("  " + report["item_census"].to_string(index=False).replace("\n", "\n  "))
        gaps = report["gap_census"]
        lines += ["", f"GAP CENSUS ({len(gaps)} runs)", "-" * width]
        lines.append("  " + (gaps.head(30).to_string(index=False).replace("\n", "\n  ")
                             if len(gaps) else "none"))
        if report["excluded_items_preview"]:
            lines += ["", "ITEMS features.build WOULD EXCLUDE", "-" * width]
            for e in report["excluded_items_preview"]:
                lines.append(f"  {e['item']:<14s} {e['reason']}")
    lines += ["", "=" * width,
              f"RESULT: {'PASS' if report['ok'] else 'FAIL'} -- "
              f"{sum(f.level == 'error' for f in report['findings'])} error(s), "
              f"{sum(f.level == 'warning' for f in report['findings'])} warning(s)"]
    return "\n".join(lines)


def _jsonable(report):
    out = dict(report)
    out["findings"] = [f._asdict() for f in report["findings"]]
    out["item_census"] = report["item_census"].to_dict("records")
    out["gap_census"] = report["gap_census"].to_dict("records")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ht.validate",
                                description="Is this panel fit to train on?")
    p.add_argument("--panel", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--mapping", default=None)
    p.add_argument("--strict", action="store_true")
    # The history floor depends on the split the caller intends, and the insufficient_history
    # message names these two flags -- so the command has to accept them, or it can only ever
    # tell a store to pass a flag it will not read.
    p.add_argument("--no-test", dest="no_test", action="store_true")
    p.add_argument("--allow-short", dest="allow_short", action="store_true")
    p.add_argument("--json", dest="json_out", default=None)
    p.add_argument("--width", type=int, default=100)
    args = p.parse_args(argv)

    try:
        items = ht_config.load_items(args.items)
        mapping = ht_config.load_mapping(args.mapping, items) if args.mapping else None
        panel = schema.read_panel(args.panel)
    except schema.HtError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.panel}: {exc}", file=sys.stderr)
        return 1

    report = validate(panel, items, mapping,
                      split_opts=dict(no_test=args.no_test, allow_short=args.allow_short))
    print(format_report(report, args.width))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(_jsonable(report), fh, indent=1, sort_keys=True, default=str)

    if not report["ok"]:
        return 1
    if args.strict and any(f.level == "warning" for f in report["findings"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
