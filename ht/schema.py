"""The canonical panel: the one table every module downstream of ingest agrees on.

A panel is one row per store x item x business day. Its columns are exactly the
seventeen a real store can observe -- the same names data/store_synth.csv already uses,
so the simulator is a valid producer of this schema with no adapter code -- plus four
provenance columns (store, row_status, stockout_known, sellout_source) that all carry
defaults, so conform(pd.read_csv("data/store_synth.csv")) passes untouched.

Two things are structural here rather than conventional. The simulator-only columns
(true_demand, true_mean, lost_sales) are dropped by conform and cannot reach a model:
assert_no_truth() is called at the top of ht.ingest.ingest, ht.validate.validate,
model.evaluate.evaluate and model.shadow.morning. And every column name, dtype, default
and allowed value is written down exactly once, in CANONICAL, so an argument about what
`stockout` means is an argument about one line of this file.
"""
import collections
import hashlib

import numpy as np
import pandas as pd

Column = collections.namedtuple("Column", "name dtype required default meaning")

# Text columns use pandas' own default dtype for strings read from a CSV, so a conformed
# panel behaves identically to a freshly-read one: `df.weather == "rain"` stays a plain
# bool array, which is what model/features.py indexes with.
TEXT = "str"

# default=None means "no default: the column must be supplied". dow is the one exception
# -- it is derived from date when absent, because a real POS gives a date and any day
# column it does supply may not be Monday-origin.
CANONICAL = (
    Column("date", "datetime64[ns]", True, None,
           "Business date at midnight -- the store's own day-close, not a UTC day"),
    Column("store", TEXT, True, "default",
           "Store key; v1 panels are single-store but the column costs nothing"),
    Column("item", TEXT, True, None,
           "Canonical item key, and the join key for everything; a key of the items config"),
    Column("item_name", TEXT, True, None,
           "Display name on the morning sheet and in reports"),
    Column("dept", TEXT, True, None,
           "Department, for grouping the sheet and reports. Never a model feature"),
    Column("dow", "int8", True, None,
           "Day of week, 0=Monday, matching date.weekday(). Always recomputed from date"),
    Column("holiday", TEXT, True, "",
           'Primary holiday key for the date, or "" when none. Never null in a panel'),
    Column("payday", "int8", True, 0,
           "1 on the store's payday-adjacent days"),
    Column("is_closed", "int8", True, 0,
           "1 = context only, never a training or scoring target. ctx channel 2"),
    Column("row_status", TEXT, True, "ok",
           "Why the row looks the way it does; the diagnostic behind is_closed"),
    Column("tmax_f", "float32", False, np.nan,
           "Daily high temperature, degrees F. NaN allowed, z-scored to 0 by features"),
    Column("weather", TEXT, True, "unknown",
           "Day's dominant condition, from a closed five-value vocabulary"),
    Column("snow_tomorrow", "int8", True, 0,
           "Snow is expected tomorrow, known today. Drives the bread pantry-load spike"),
    Column("sold", "float32", True, None,
           "Net units sold that business day; pounds for continuous items. The target"),
    Column("produced", "float32", False, np.nan,
           "Units produced or put out. NaN where no production record exists"),
    Column("wasted", "float32", False, np.nan,
           "Units discarded (markout). NaN when unknown. Markdowns are not waste"),
    Column("stockout", "int8", True, 0,
           "1 = ran out of supply, so sold is a lower bound on demand. ctx channel 1"),
    Column("stockout_known", "int8", True, 1,
           "1 = the sellout rule could actually be evaluated for this row"),
    Column("sellout_source", TEXT, True, "unknown",
           "Which sellout rule produced the flag on this row"),
    Column("unit_price", "float32", False, np.nan,
           "Realized net retail per unit that day. Settlement dollars only, never q*"),
    Column("unit_cost", "float32", False, np.nan,
           "Cost of goods per unit. Settlement dollars only"),
)

# Required to be PRESENT, but allowed to be null: a horizon row carries tomorrow's
# covariates so a forecast can be made, and tomorrow has not sold anything yet.
# ht.validate is the layer that decides whether a null sold is acceptable where it sits.
NULLABLE = ("sold",)

NAMES = tuple(c.name for c in CANONICAL)
REQUIRED = tuple(c.name for c in CANONICAL if c.required)
OPTIONAL = tuple(c.name for c in CANONICAL if not c.required)
BY_NAME = {c.name: c for c in CANONICAL}
KEY = ("store", "item", "date")

SIM_ONLY = ("true_demand", "true_mean", "lost_sales")
WEATHER_KINDS = ("sunny", "cloudy", "rain", "snow", "unknown")
ROW_STATUS = ("ok", "closed", "partial", "not_carried", "missing", "suspect")
# What a PANEL column may say. model/shadow.py records a fifth source, "sheet", for a
# sellout a person read off a returned morning sheet; it stays out of this tuple on purpose,
# because it names a person reading a page rather than a rule applied to an export, and it
# never reaches a panel column. Writing it back into a panel means adding it here first.
SELLOUT_SOURCES = ("produced_vs_sold", "flag", "none", "unknown")

# One condition, three layers: ht.ingest refuses a district file, ht.validate errors on a
# district panel, and model.features refuses one that reached build() without either. They
# describe the mechanism in their own terms but must offer the SAME way out, so the way out
# is written once, here beside the key store is part of.
ONE_STORE_REMEDY = (
    "Re-run the item movement report for the one store you are piloting, or map "
    "columns.<role>.store to the raw header carrying the store number and re-run "
    "`python -m ht.ingest --store <number>`; without that column mapped, ingest stamps "
    "mapping.store on every row and the district is summed onto one store's series "
    "invisibly.")

ENUMS = {"weather": WEATHER_KINDS, "row_status": ROW_STATUS,
         "sellout_source": SELLOUT_SOURCES}


class HtError(Exception):
    """Base for every error this project raises on its own behalf."""


class SchemaError(HtError):
    pass


class ConfigError(HtError):
    pass


class MappingError(HtError):
    pass


class IngestError(HtError):
    pass


class ValidationFailed(HtError):
    pass


def dtypes():
    return {c.name: c.dtype for c in CANONICAL}


def empty_panel():
    return pd.DataFrame({c.name: pd.Series([], dtype=c.dtype) for c in CANONICAL})


def _sample(value):
    return repr(value.item() if hasattr(value, "item") else value)


def conform(df, keep_extra=False):
    """Return df as a canonical panel: right columns, right dtypes, right order.

    Inserts every absent column that has a default, casts what is present, fills nulls in
    defaulted columns, sorts by (store, item, date) and resets the index. Unknown columns
    are dropped and listed on the result's .attrs["dropped"]; the simulator-only columns
    are dropped whatever keep_extra says, because nothing downstream of conform may see
    simulator truth. Never mutates its argument.

    Raises SchemaError once, listing every missing required column, then every value that
    would not coerce with its row index, then every value outside a column's vocabulary.
    A null in a required column is a coercion failure, except for the NULLABLE ones.
    """
    out = df.copy()
    missing, coercion, enum, filled = [], [], [], {}
    derived_dow = False

    if "dow" not in out.columns and "date" in out.columns:
        out["dow"] = pd.to_datetime(out["date"], errors="coerce").dt.dayofweek
        derived_dow = True
    if "row_status" not in out.columns and "is_closed" in out.columns:
        # is_closed = int(row_status != "ok"), so inserting a flat "ok" next to an
        # is_closed column would contradict it. "closed" is the only honest guess.
        closed = pd.to_numeric(out["is_closed"], errors="coerce").fillna(0) > 0
        out["row_status"] = np.where(closed, "closed", "ok")

    for col in CANONICAL:
        if col.name in out.columns:
            continue
        if col.default is None:
            missing.append(col.name)
        else:
            out[col.name] = col.default

    for col in CANONICAL:
        if col.name in missing:
            continue
        raw = out[col.name]
        if col.dtype == "datetime64[ns]":
            new = pd.to_datetime(raw, errors="coerce").dt.normalize()
        elif col.dtype == TEXT:
            new = raw.astype(TEXT)
        else:
            new = pd.to_numeric(raw, errors="coerce")
            if col.dtype.startswith("int"):
                # a fractional value in an integer column is a coercion failure, not a
                # rounding job -- 0.5 stockouts mean the caller derived something wrong
                new = new.mask(new.notna() & (new != np.floor(new)))

        bad = new.isna() & raw.notna()
        null = new.isna() & raw.isna()
        if col.default is not None or col.name in NULLABLE:
            null[:] = False
        elif col.name == "dow" and derived_dow:
            null = null & out["date"].notna()   # a null dow here is the date's fault, once
        if bad.any():
            idx = list(raw.index[bad])[:5]
            samples = ", ".join(_sample(raw.loc[i]) for i in idx)
            coercion.append(f"{col.name}: {int(bad.sum())} value(s) will not coerce to "
                            f"{col.dtype} (rows {idx}: {samples})")
        if null.any():
            coercion.append(f"{col.name}: {int(null.sum())} null value(s) in a column that "
                            f"must be supplied (rows {list(raw.index[null])[:5]})")

        if col.default is not None and new.isna().any():
            filled[col.name] = int(new.isna().sum())
            new = new.fillna(col.default)
        if col.dtype.startswith("int"):
            new = new.fillna(0)   # only reachable on a bad column we are about to raise on
        if col.dtype != TEXT:
            new = new.astype(col.dtype)
        out[col.name] = new

    for name, allowed in ENUMS.items():
        if name in missing:
            continue
        off = out.loc[~out[name].isin(allowed), name]
        if len(off):
            counts = off.value_counts().head(5).to_dict()
            enum.append(f"{name}: {len(off)} row(s) outside {list(allowed)} -- {counts} "
                        f"(first at row {off.index[0]})")

    if missing or coercion or enum:
        parts = []
        if missing:
            parts.append("missing required column(s): " + ", ".join(missing))
        parts.extend(coercion)
        parts.extend(enum)
        raise SchemaError("panel does not conform:\n  " + "\n  ".join(parts))

    extra = [c for c in out.columns if c not in NAMES]
    dropped = [c for c in extra if not keep_extra or c in SIM_ONLY]
    out = out[list(NAMES) + [c for c in extra if c not in dropped]]
    out = out.sort_values(list(KEY), kind="stable").reset_index(drop=True)
    out.attrs = {"dropped": dropped, "filled": filled}
    return out


def assert_no_truth(df):
    """Refuse a frame carrying simulator truth. Hard Rule 6, made mechanical."""
    found = [c for c in SIM_ONLY if c in df.columns]
    if found:
        raise SchemaError(
            f"simulator-only column(s) present: {', '.join(found)}. These exist only "
            "inside sim/ and must never reach a model or a metric; a real store has no "
            "such column. Conform the frame first, or use `python -m model.backtest "
            "--settlement sim` if you meant to settle a policy against simulator truth.")


def _csv_bytes(panel):
    return panel.to_csv(index=False, date_format="%Y-%m-%d",
                        lineterminator="\n").encode("utf-8")


def panel_hash(df):
    """sha256 of the conformed panel's CSV bytes -- stable across a round trip to disk."""
    return hashlib.sha256(_csv_bytes(conform(df))).hexdigest()


def read_panel(path):
    head = pd.read_csv(path, nrows=0)
    text = {c.name: TEXT for c in CANONICAL if c.dtype == TEXT and c.name in head.columns}
    df = pd.read_csv(path, dtype=text,
                     parse_dates=["date"] if "date" in head.columns else None)
    return conform(df)


def write_panel(df, path):
    with open(path, "wb") as fh:
        fh.write(_csv_bytes(conform(df)))
