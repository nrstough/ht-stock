"""Raw store export -> the canonical panel.

This is the only module that knows what a real export looks like: cp1252 bytes with a report
title block above the header and a TOTAL row below it, two-digit years, refunds printed as
(12.34), a different barcode on every package of hot bar, a product discontinued under one
item number and re-added under another, and whole days simply absent because nothing sold.
Everything downstream sees one known table instead.

Every repair is authorized by a named field in the mapping. Without that field the same
condition is an error, because the alternative -- guessing -- produces a panel that validates
clean and trains a model on somebody's typo. That is also why nothing here is per-store: a
store's messiness lives in its mapping file, and this module has no branches on which store
it is reading.

Every IngestError reads "<what is wrong> | <which mapping field would authorize the repair>",
so the person holding a broken export is told what to change rather than shown a traceback.

Two inputs come from outside the store's export and are named on the command line rather than
in mapping.files. --logger-backup is the Phase-1 phone logger's own JSON backup, the only
record of this store's waste from before the pilot and the "before" half of the before/after
README's Phase 4 promises; a mapping.files role would need a new entry in ht.config's closed
FILE_ROLES vocabulary, and it is not a file the store's system produced. --store picks one
store's rows out of a movement report that was run for the whole district.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from . import calendar as ht_calendar
from . import config as ht_config
from . import schema
from . import weather as ht_weather
from .schema import IngestError

SELLOUT_RULES = ("produced_vs_sold", "flag", "none")

# Latency is a property of the RULE, not of a store: stockout is an encoder input across the
# trailing 28 days, so it has to be computable every morning on yesterday's data. A rule that
# needs a sheet keyed in three days late trains on a flag that serving time cannot supply.
SELLOUT_LATENCY_DAYS = {"produced_vs_sold": 1, "flag": 1, "none": 0}

# What each raw file role contributes. The key is the canonical measure; the value is the
# mapping.columns.<role> entry that names the raw header carrying it.
ROLE_MEASURES = {
    "sales": {"sold": "units", "dollars": "dollars", "row_cost": "cost"},
    "production": {"produced": "units"},
    "waste": {"wasted": "units"},
}

TRUTHY = ("1", "true", "y", "yes", "sold out", "soldout", "t", "oos")

MEASURES = ("sold", "dollars", "row_cost", "produced", "wasted")


def _samples(values, n=5):
    return ", ".join(repr(v) for v in list(values)[:n])


# ---- reading raw files ----

def read_raw(mapping, role, root="."):
    """Every raw file with this role, read as text and concatenated, with its source recorded.

    Text rather than typed: a movement export writes "1,204" and "(12.34)" and "$3.99", and
    letting pandas guess a dtype per file means two years of the same export parse differently.
    clean_numeric owns the un-mangling.
    """
    entries = [f for f in mapping.get("files", []) if f.get("role") == role]
    if not entries:
        return None
    parts = []
    for entry in entries:
        pattern = os.path.join(root, entry["path"])
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise IngestError(f"role {role!r}: {pattern!r} matches no file | fix the path on "
                              "that mapping.files entry (a glob is fine), or remove the entry")
        for path in paths:
            if os.path.splitext(path)[1].lower() in (".xlsx", ".xlsm", ".xls"):
                raise IngestError(
                    f"{path}: spreadsheet files are not read | save as CSV first. Reading xlsx "
                    "would need a new third-party dependency, which this project does not take")
            try:
                raw = pd.read_csv(path, encoding=entry["encoding"], sep=entry["delimiter"],
                                  skiprows=max(int(entry["header_row"]) - 1, 0),
                                  na_values=entry["na_values"], dtype=str,
                                  keep_default_na=True)
            except UnicodeDecodeError as exc:
                raise IngestError(
                    f"{path}: not readable as {entry['encoding']} ({exc}) | set `encoding` on "
                    "that mapping.files entry; 'cp1252' for anything with AS/400 lineage") from None
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                raise IngestError(
                    f"{path}: will not parse as a {entry['delimiter']!r}-delimited CSV with its "
                    f"header on row {entry['header_row']} ({exc}) | fix header_row, delimiter or "
                    "skip_footer_rows on that mapping.files entry") from None
            if entry["skip_footer_rows"]:
                raw = raw.iloc[:-int(entry["skip_footer_rows"])]
            raw.columns = [str(c).strip() for c in raw.columns]
            _require_headers(mapping, role, path, raw.columns)
            raw["__source"] = path
            parts.append(raw)
    return pd.concat(parts, ignore_index=True)


def _require_headers(mapping, role, path, columns):
    """Every header this role maps has to exist in THIS file, or say so by name.

    Checked per file rather than over the concatenation: a column present in 2023 and
    renamed in 2024 unions away under pd.concat, and the year with the missing header
    silently becomes NaN -- a whole year of sales quietly reading zero.
    """
    want = [h for h in (mapping["columns"].get(role) or {}).values() if h]
    missing = [h for h in want if h not in set(columns)]
    if missing:
        raise IngestError(
            f"{path}: role {role!r} maps header(s) {_samples(missing)} that the file does not "
            f"have | fix mapping.columns.{role}, or header_row on that files entry. The file's "
            f"headers are: {list(columns)}")


def clean_numeric(s, strip=("$", ","), parens_negative=True, decimal="."):
    """Currency text -> float. '$1,204' -> 1204.0; '(12.34)' -> -12.34; '' -> NaN.

    Parentheses are how a retail report prints a refund. Reading them as positive is the one
    mistake here that inflates a demand series instead of breaking it.
    """
    text = s.astype(str).str.strip()
    blank = s.isna() | (text == "") | text.str.lower().isin(["nan", "none"])
    negative = pd.Series(False, index=s.index)
    if parens_negative:
        negative = text.str.match(r"^\(.*\)$", na=False)
        text = text.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    for ch in strip:
        text = text.str.replace(ch, "", regex=False)
    if decimal != ".":
        text = text.str.replace(decimal, ".", regex=False)
    out = pd.to_numeric(text.where(~blank), errors="coerce")
    return out.mask(negative & out.notna(), -out.abs())


def parse_dates(s, fmt):
    """Parse with an EXPLICIT strftime format. Returns (parsed, n_dropped).

    Rows that fail are how a report title line and a DEPT TOTAL footer leave the frame, so a
    handful of failures is expected and silent. Past 0.5% the format is wrong, not the data,
    and guessing a format is how 3/4/25 becomes April.
    """
    if not fmt or fmt == "auto":
        raise IngestError("date.format is not set | give an explicit strftime string such as "
                          "'%m/%d/%y'; 'auto' reads 3/4/25 as either March or April")
    text = s.astype(str).str.strip()
    blank = s.isna() | (text == "") | text.str.lower().isin(["nan", "none"])
    parsed = pd.to_datetime(text.where(~blank), format=fmt, errors="coerce")
    parsed = parsed.dt.normalize()
    bad = parsed.isna()
    n_dropped = int(bad.sum())
    if len(s) and n_dropped / len(s) > 0.005:
        raise IngestError(
            f"{n_dropped} of {len(s)} rows ({n_dropped / len(s):.1%}) have a date that will not "
            f"parse as {fmt!r} -- {_samples(text[bad & ~blank], 3)} | fix mapping.date.format, or "
            "the file's header_row if the header block is being read as data")
    return parsed, n_dropped


BLANK_CODE = "(blank)"


def _code_labels(codes):
    """Raw item codes as report keys: always a string, "(blank)" for an empty cell.

    pandas carries a missing value straight through .astype(str), and one float key among
    strings breaks every sorted() and json.dump(sort_keys=True) that reads these reports --
    a traceback on the page the store is handed, over a department subtotal line.
    """
    s = codes.astype("string").str.strip()
    return s.mask(s.isna() | (s == ""), BLANK_CODE).to_numpy()


def resolve_items(codes, descs, mapping):
    """Raw item codes -> canonical item keys, NaN where nothing maps.

    The chain is barcode -> code map -> description alias, in that order and never reversed:
    a description is re-keyed freely by staff, so it is the fallback, never the identity.
    """
    cfg = mapping["items"]
    raw = codes.astype(str).str.strip()

    plu = cfg.get("random_weight_plu") or {}
    if plu.get("enabled"):
        # A weighed item prints a different barcode per package, so grouping on the raw code
        # turns one hot-bar item into thousands of one-row items. net.py sizes
        # nn.Embedding(n_items), so that is a model-shape failure, not a data quirk.
        try:
            found = raw.str.extract(plu["pattern"], expand=True)
        except re.error as exc:
            raise IngestError(f"items.random_weight_plu.pattern {plu['pattern']!r} is not a "
                              f"regular expression ({exc}) | fix it, or set enabled "
                              "false") from None
        group = int(plu.get("plu_group", 1)) - 1
        if group >= found.shape[1]:
            raise IngestError(
                f"items.random_weight_plu.plu_group is {plu.get('plu_group')} but the pattern "
                f"has {found.shape[1]} capture group(s) | the group must hold the embedded PLU")
        embedded = found.iloc[:, group]
        raw = raw.where(embedded.isna(), embedded)

    excluded = set(str(c) for c in cfg.get("exclude", []))
    keep = ~raw.isin(excluded)

    out = raw.where(keep).map(cfg.get("map", {}))
    aliases = cfg.get("alias_map", {})
    if aliases and descs is not None:
        text = descs.astype(str).str.strip()
        out = out.fillna(text.where(keep).map(aliases))

    # Two different failures, counted separately. RESOLVED keys are what sizes
    # nn.Embedding(n_items), so an explosion there is a model-shape failure. Distinct
    # UNMAPPED codes are just a bigger export than the pilot covers -- a whole prepared-foods
    # movement report is several hundred item numbers of which nine are ours -- so that one is
    # governed by items.max_items alone, which is the remedy the message names.
    roster = set(cfg.get("map", {}).values()) | set(cfg.get("alias_map", {}).values())
    ceiling = max(int(cfg.get("max_items", 400)), 1)
    explosion = max(3 * len(roster), 1)
    resolved = sorted(set(out.dropna()))
    if len(resolved) > min(ceiling, explosion):
        raise IngestError(
            f"{len(resolved)} distinct items resolved against a roster of {len(roster)} "
            f"(ceiling {min(ceiling, explosion)}) -- {_samples(resolved[:5])} | one item per "
            "package would size nn.Embedding(n_items) wrong and break the model, not just the "
            "report. Fix items.map, or raise items.max_items if the roster really is that big")
    unmapped = sorted(set(raw[keep][out.isna()].dropna()))
    if len(unmapped) > ceiling:
        raise IngestError(
            f"{len(unmapped)} distinct item codes match nothing in items.map (ceiling "
            f"{ceiling}) -- {_samples(unmapped[:5])} | if these are random-weight barcodes set "
            "items.random_weight_plu, otherwise raise items.max_items to accept a whole-"
            "department export whose other codes are dropped and counted")

    # a code the mapping deliberately excludes is not the same finding as one nobody has seen,
    # and the caller counts them per row so the two buckets cannot overlap
    out.attrs["excluded_mask"] = (~keep).to_numpy()
    out.attrs["unmapped_codes"] = len(unmapped)
    return out


def aggregate(df, policy="sum", key=None):
    """One row per key. Duplicate export windows are summed; a corrected re-issue is 'last'.

    min_count=1 so a measure absent from every contributing row stays NaN rather than becoming
    a confident zero -- the difference between "no production record" and "produced nothing".
    """
    key = list(key or schema.KEY)
    measures = [c for c in MEASURES if c in df.columns]
    grouped = df.groupby(key, dropna=False, sort=False)
    if policy == "sum":
        out = grouped[measures].sum(min_count=1)
    elif policy == "last":
        out = grouped[measures].last()
    else:
        raise IngestError(f"dedupe.policy is {policy!r} | expected 'sum' (overlapping export "
                          "windows, refunds on their own line) or 'last' (a corrected re-issue)")
    return out.reset_index()


def _spans(df):
    return df.groupby(["store", "item"])["date"].agg(["min", "max"])


def reindex_grid(df, items, mapping):
    """Put every item on a complete daily index across its own carried span.

    A day the export omits is a day nobody can distinguish from a zero, so the absence has to
    be turned into a row with a stated reason. A short hole is a genuine zero-sales day (the
    export writes no line when nothing scanned); a long one is a systems gap, and calling that
    a zero teaches the model that the store shut down.
    """
    max_gap = int(mapping["gaps"]["max_unexplained_gap_days"])
    inserted = collections.Counter()
    frames = []
    for (store, item), grp in df.groupby(["store", "item"], sort=True):
        grp = grp.drop_duplicates("date").set_index("date").sort_index()
        full = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        out = grp.reindex(full)
        out["store"], out["item"] = store, item
        was_missing = ~full.isin(grp.index)
        status = out["row_status"].to_numpy(dtype=object) if "row_status" in out else None
        status = np.array(["ok"] * len(full), dtype=object) if status is None else status
        status[pd.isna(status)] = "ok"
        if was_missing.any():
            # classify by the length of the RUN it belongs to, not the single day
            run_id = np.cumsum(~was_missing)
            for rid, length in collections.Counter(run_id[was_missing]).items():
                if length > max_gap:
                    status[was_missing & (run_id == rid)] = "missing"
            # only the cells this function invented: a row that WAS in the export with a
            # blank units cell is a missing measure, and validate's sold_null must still see it
            out.loc[was_missing, "sold"] = 0.0
            inserted[item] += int(was_missing.sum())
        out["row_status"] = status
        out.index.name = "date"
        frames.append(out.reset_index())
    panel = pd.concat(frames, ignore_index=True)
    panel.attrs["grid_rows_inserted"] = dict(inserted)
    return panel


def _date_format(mapping, role):
    """The date format for one role's files, defaulting to the sales export's.

    A store's hours or weather file usually comes out of a different system than its movement
    report, so a per-file `date_format` overrides mapping.date.format. It is still explicit --
    there is no path here that guesses a format.
    """
    for entry in mapping.get("files", []):
        if entry.get("role") == role and entry.get("date_format"):
            return entry["date_format"]
    return mapping["date"]["format"]


def _hours_closures(mapping, root):
    """Dates the store hours file says had no open interval."""
    raw = read_raw(mapping, "hours", root)
    if raw is None:
        return set()
    cols = (mapping["columns"].get("hours") or {})
    if not cols.get("date"):
        raise IngestError("a file has role 'hours' but columns.hours.date is not mapped | name "
                          "the raw header carrying the date, or drop the file entry")
    dates, _ = parse_dates(raw[cols["date"]], _date_format(mapping, "hours"))
    closed = set()
    open_col, close_col = cols.get("open"), cols.get("close")
    for i, day in enumerate(dates):
        if pd.isna(day):
            continue
        o = str(raw[open_col].iloc[i]).strip() if open_col else ""
        c = str(raw[close_col].iloc[i]).strip() if close_col else ""
        blank = {"", "nan", "none", "closed", "-"}
        if o.lower() in blank or c.lower() in blank or o == c:
            closed.add(day.normalize())
    return closed


def _apply_closures(panel, mapping, hours_closed):
    """Stamp declared closures and partial days. An early close looks exactly like a collapse."""
    applied = collections.Counter()
    status = panel["row_status"].to_numpy(dtype=object)
    dates = panel["date"]

    declared = {pd.Timestamp(d).normalize() for d in mapping["closures"].get("dates", [])}
    for day in sorted(declared | set(hours_closed)):
        hit = (dates == day).to_numpy()
        if hit.any():
            status[hit] = "closed"
            applied["closed"] += int(hit.sum())
    for day in mapping["closures"].get("partial", {}):
        hit = (dates == pd.Timestamp(day).normalize()).to_numpy()
        if hit.any():
            status[hit] = "partial"
            applied["partial"] += int(hit.sum())
    panel = panel.copy()
    panel["row_status"] = status
    panel.attrs["closures_applied"] = dict(applied)
    return panel


def derive_sellout(panel, mapping, items, aux=None):
    """Write stockout, stockout_known and sellout_source. One rule, named in the mapping.

    There is no fallback chain on purpose: a flag whose provenance varies row by row cannot be
    reconstructed six weeks later, when the argument about whether the pilot worked happens.
    """
    rule = mapping["sellout"]["rule"]
    if rule not in SELLOUT_RULES:
        raise IngestError(f"sellout.rule is {rule!r} | expected one of {list(SELLOUT_RULES)}; "
                          "'none' is a valid explicit answer and the right one for most stores")
    panel = panel.copy()
    n = len(panel)
    stockout = np.zeros(n, dtype=np.int8)
    known = np.zeros(n, dtype=np.int8)
    status = panel["row_status"].to_numpy(dtype=object)

    if rule == "produced_vs_sold":
        produced = panel["produced"].to_numpy(dtype=float)
        sold = panel["sold"].to_numpy(dtype=float)
        eps = panel["item"].map(
            {k: ht_config.resolve_tolerance(it) for k, it in items.items()}).fillna(0.0)
        eps = eps.to_numpy(dtype=float)
        have = ~np.isnan(produced)
        zero = have & (produced <= 0)
        real = have & (produced > 0)
        stockout[real] = (sold[real] >= produced[real] - eps[real]).astype(np.int8)
        known[have] = 1
        # zero supply is not a demand observation: sold=0 there says nothing about demand
        status[zero & (status == "ok")] = "missing"
    elif rule == "flag":
        events = (aux or {}).get("oos")
        if events is None or not len(events):
            raise IngestError("sellout.rule is 'flag' but no file has role 'oos' | add the "
                              "out-of-stock log to mapping.files, or use rule 'none'")
        start = pd.Timestamp(mapping["sellout"]["coverage_start"])
        end = pd.Timestamp(mapping["sellout"]["coverage_end"])
        inside = ((panel["date"] >= start) & (panel["date"] <= end)).to_numpy()
        keys = set(zip(events["item"], events["date"]))
        hit = np.array([(i, d) in keys for i, d in zip(panel["item"], panel["date"])])
        known[inside] = 1
        stockout[inside & hit] = 1

    source = np.full(n, rule, dtype=object)
    closed = (status != "ok")
    stockout[closed] = 0                      # a closed day did not run out, it did not open

    panel["stockout"] = stockout
    panel["stockout_known"] = known
    panel["sellout_source"] = source
    panel["row_status"] = status

    scored = ~closed
    per_item = {}
    for item, grp in panel[scored].groupby("item"):
        seen = grp[grp.stockout_known == 1]
        per_item[item] = round(float(seen.stockout.mean()), 4) if len(seen) else None
    seen = panel[scored & (panel.stockout_known == 1)]
    report = dict(
        rule=rule,
        rate=round(float(seen.stockout.mean()), 4) if len(seen) else None,
        known_share=round(float(panel.loc[scored, "stockout_known"].mean()), 4) if scored.any()
        else 0.0,
        unknown_days=int((scored & (panel.stockout_known == 0)).sum()),
        per_item_rate=per_item,
        latency_days=SELLOUT_LATENCY_DAYS[rule],
    )
    return panel, report


def ingest_logger_backup(path, items, seen=None):
    """The Phase-1 logger's JSON backup -> a waste frame keyed by (item, date).

    The BACKUP, never the CSV export: buildCSV drops both the entry id and the item id, so two
    exports of the same log double-count every entry and an item renamed mid-pilot silently
    splits into two series. Both corruptions land straight on the baseline dollar figure.

    `seen` is a shared set of entry ids, so that two backups off two phones -- or the same
    phone's backup handed over twice under different filenames -- deduplicate against each
    other rather than only within one file. Entries carrying no id at all are kept: the logger
    always writes one, and collapsing a hand-edited file's id-less rows into a single entry
    would quietly delete waste.
    """
    with open(path, "rb") as fh:
        try:
            data = json.loads(fh.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IngestError(f"{path}: not valid JSON ({exc}) | export a fresh backup from the "
                              "logger's Setup > Data screen") from None
    if data.get("app") != "ht-stock":
        raise IngestError(f"{path}: app is {data.get('app')!r}, expected 'ht-stock' | this is "
                          "the logger's JSON BACKUP, not its CSV export and not another file")

    # The logger's item id is matched FIRST and against the items config's own keys, because
    # index.html ships those keys as its default item ids; the display name is the fallback
    # for an item somebody added on the phone, exactly as in resolve_items.
    by_name = {}
    for key, it in items.items():
        by_name[key.strip().lower()] = key
        by_name[str(it["name"]).strip().lower()] = key
    by_id = {}
    for rec in data.get("items", []):
        key = (by_name.get(str(rec.get("id", "")).strip().lower())
               or by_name.get(str(rec.get("name", "")).strip().lower()))
        if key is not None:
            by_id[rec.get("id")] = key

    seen = set() if seen is None else seen
    rows, unmapped, dupes, bad_dates = [], collections.Counter(), 0, 0
    for log in data.get("logs", []):
        entry_id = log.get("id")
        if entry_id is not None:
            if entry_id in seen:
                dupes += 1
                continue                      # a second export of the same device's log
            seen.add(entry_id)
        key = (by_id.get(log.get("itemId"))
               or by_name.get(str(log.get("itemId", "")).strip().lower())
               or by_name.get(str(log.get("itemName", "")).strip().lower()))
        if key is None:
            unmapped[str(log.get("itemName"))] += 1
            continue
        # the logger writes an <input type=date>, so this is ISO or it is hand-edited
        day = pd.to_datetime(log.get("date"), errors="coerce")
        if pd.isna(day):
            bad_dates += 1
            continue
        qty = float(log.get("qty") or 0.0)
        if qty < 0:
            # a logged markout is a physical count of what was thrown away; a negative one is
            # a hand edit, and folding it in silently subtracts from the Phase-1 baseline the
            # whole before/after is measured against
            raise IngestError(
                f"{path}: {key} on {day.date()} carries qty {qty} | a logger entry is a count "
                "of what was thrown away, so a negative is a correction somebody has to settle "
                "in the backup before it can be folded into the waste baseline")
        rows.append((key, day.normalize(), qty))
    out = pd.DataFrame(rows, columns=["item", "date", "wasted"])
    out = out.groupby(["item", "date"], as_index=False)["wasted"].sum()
    out.attrs["unmapped_logger_items"] = dict(unmapped)
    out.attrs["entries_used"] = len(rows)
    out.attrs["duplicate_entries"] = dupes
    out.attrs["unparsed_dates"] = bad_dates
    return out


def _logger_waste(paths, items, store, report):
    """Every logger backup, as one (store, item, date, wasted) frame, plus what was lost.

    Several paths because the log lives on a phone: two people logging is two devices and two
    backups, and every entry id is unique per device, so the union deduplicates correctly.

    A backup that matches nothing is refused rather than folded in as zero waste. The whole
    reason to read this file is that README's Phase 4 promises a measured before/after against
    the Phase-1 baseline, and a silently empty baseline is how that promise turns into a
    number nobody can check.
    """
    seen, frames = set(), []
    unmapped, entries, dupes, bad_dates = collections.Counter(), 0, 0, 0
    for path in paths:
        if not os.path.exists(path):
            raise IngestError(f"{path}: no such file | --logger-backup takes the JSON the "
                              "logger's Setup > Data > Download backup button writes")
        frame = ingest_logger_backup(path, items, seen)
        unmapped.update(frame.attrs["unmapped_logger_items"])
        entries += frame.attrs["entries_used"]
        dupes += frame.attrs["duplicate_entries"]
        bad_dates += frame.attrs["unparsed_dates"]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).groupby(
        ["item", "date"], as_index=False)["wasted"].sum()
    if not len(out):
        raise IngestError(
            f"{list(paths)}: none of the {entries + sum(unmapped.values())} logged entries "
            f"matched a configured item -- the names in the backup are "
            f"{_samples(sorted(unmapped)) or '(none)'} | rename the logger's items to the items "
            "config's keys or `name` fields, or drop --logger-backup. Folding in an empty "
            "baseline would leave the Phase-1 before/after measuring nothing")
    out.insert(0, "store", str(store))
    report["logger"] = dict(
        paths=list(paths), entries_used=entries, duplicate_entries=dupes,
        unparsed_dates=bad_dates, unmapped_items=dict(unmapped.most_common(20)),
        unmapped_entries=int(sum(unmapped.values())),
        date_range=[str(out.date.min().date()), str(out.date.max().date())],
        items=sorted(out.item.unique()))
    return out


def _apply_logger_waste(panel, logged, measured, report):
    """Fold the logger's markouts into `wasted`, never adding to the export's own record.

    The logger is a measurement -- somebody stood at the case and typed it in -- so it outranks
    `produced - sold`, which is arithmetic on a label-printer proxy. It does not outrank a waste
    column the export supplied, and it is never added to one: two independent records of the
    same markout, summed, doubles precisely the number the pilot is judged on. Those collisions
    keep the export's figure and are reported with both, for a person to settle.
    """
    key = list(schema.KEY)
    aligned = panel[key].merge(logged.rename(columns={"wasted": "__logged"}),
                               on=key, how="left")["__logged"].to_numpy(dtype=float)
    have = ~np.isnan(aligned)
    fill, conflict = have & ~measured, have & measured

    wasted = panel["wasted"].to_numpy(dtype=float, copy=True)
    info = report["logger"]
    info["item_days_written"] = int(fill.sum())
    info["units_written"] = round(float(aligned[fill].sum()), 3)
    info["conflicts_with_export"] = int(conflict.sum())
    if conflict.any():
        info["conflict_items"] = {k: int(v) for k, v in
                                  panel.loc[conflict, "item"].value_counts().sort_index().items()}
        info["conflict_units"] = dict(logged=round(float(aligned[conflict].sum()), 3),
                                      export=round(float(np.nansum(wasted[conflict])), 3))

    # a logged item-day the export's grid never covers cannot be folded in at all: it would
    # have to invent a panel row with no sales, so it is counted and named instead
    outside = (logged.merge(panel[key], on=key, how="left", indicator=True)["_merge"]
               == "left_only").to_numpy()
    info["item_days_outside_the_export"] = int(outside.sum())
    if outside.any():
        info["units_outside_the_export"] = round(float(logged.loc[outside, "wasted"].sum()), 3)
        info["outside_examples"] = [f"{i} {d.date()}" for i, d in
                                    zip(logged.loc[outside, "item"].head(5),
                                        logged.loc[outside, "date"].head(5))]

    wasted[fill] = aligned[fill]
    panel = panel.copy()
    panel["wasted"] = wasted
    return panel


# ---- the pipeline ----

def _select_store(frame, role, want, report):
    """One store's rows, or a refusal that names the store numbers the file actually holds.

    An item movement report is routinely run for a whole district, and mapping.store stamps a
    single store number on every row it reads, so without this the five stores in a district
    export are summed into one series under one store's name -- and ht.validate's multi_store
    check cannot see it, because by then the panel carries exactly one store. Mapping
    columns.<role>.store is what turns that silent sum into a question, and --store answers it.
    """
    if not len(frame):
        return frame                          # let "no sales rows survived reading" say it
    seen = collections.Counter(frame["store"])
    stores = report["stores"]
    stores["seen"].update({str(k): int(v) for k, v in seen.items()})
    if want is None:
        if len(seen) > 1:
            raise IngestError(
                f"role {role!r}: the export covers {len(seen)} store numbers -- "
                f"{_samples(sorted(seen))} | pass --store <number> to keep one of them. v1 "
                "forecasts one store and every module below ingest keys on item alone, so "
                "leaving them merged would sum the district onto one store's series")
        return frame
    keep = (frame["store"] == str(want).strip()).to_numpy()
    if not keep.any():
        raise IngestError(
            f"role {role!r}: --store {want!r} matches none of this file's {len(seen)} store "
            f"number(s) -- {_samples(sorted(seen))} | check the leading zeros, or map "
            f"columns.{role}.store if the store number lives in a column this mapping does not "
            "name yet")
    stores["rows_dropped"] += int((~keep).sum())
    return frame[keep]


def _tidy(mapping, role, items, root, report, store=None):
    """One raw role -> store/item/date plus that role's measures, deduplicated."""
    raw = read_raw(mapping, role, root)
    if raw is None:
        return None
    cols = mapping["columns"].get(role) or {}
    for need in ("date", "item_code"):
        if not cols.get(need):
            raise IngestError(f"a file has role {role!r} but columns.{role}.{need} is not "
                              f"mapped | name the raw header carrying the {need}")
    rows_in = len(raw)
    key = resolve_items(raw[cols["item_code"]],
                        raw[cols.get("item_desc")] if cols.get("item_desc") else None, mapping)
    dates, n_bad = parse_dates(raw[cols["date"]], _date_format(mapping, role))

    # Each dropped raw line is counted in exactly ONE bucket, so the decomposition the
    # validation page prints adds up to its own total: a department subtotal row carries
    # neither a date nor an item number, and counting it in both made the parts exceed the
    # whole on the one document whose argument is that its numbers are checkable.
    codes = _code_labels(raw[cols["item_code"]])
    excl = key.attrs["excluded_mask"]
    unresolved = key.isna().to_numpy() & ~excl
    datok = dates.notna().to_numpy()
    excluded = dict(collections.Counter(codes[excl & datok]))
    unmapped = collections.Counter(codes[unresolved & datok])
    if unresolved.any() and not mapping["items"]["drop_unmapped"]:
        raise IngestError(
            f"{role}: {int(unresolved.sum())} row(s) carry item codes with no mapping -- "
            f"{_samples(sorted(set(codes[unresolved]))[:5])} | add them to items.map or "
            "items.exclude, or set items.drop_unmapped true to drop them and have the count "
            "reported instead")

    num = mapping["numbers"]
    # A file with no store column is that store's file, so it takes the selected number when
    # there is one: stamping mapping.store there instead would key the production and waste
    # rows to a store the filtered sales rows no longer use, and they would all merge to NaN.
    if cols.get("store"):
        who = raw[cols["store"]].astype(str).str.strip()
    else:
        who = pd.Series(str(store if store is not None else mapping["store"]), index=raw.index)
    frame = pd.DataFrame({"store": who, "item": key, "date": dates})
    for measure, mapped in ROLE_MEASURES[role].items():
        header = cols.get(mapped)
        if header:
            frame[measure] = clean_numeric(raw[header], strip=tuple(num["strip"]),
                                           parens_negative=num["parens_negative"],
                                           decimal=num["decimal"])
    # row_cost is carried as an EXTENDED total so that summing a day's lines is arithmetic
    # rather than a multiplication by the line count; _economics divides it back out by units.
    if "row_cost" in frame and mapping["price_cost"]["cost_basis"] == "per_unit":
        frame["row_cost"] = frame["row_cost"] * frame["sold"]
    frame = frame[frame["item"].notna() & frame["date"].notna()]
    readable = len(frame)
    frame = _select_store(frame, role, store, report)
    before = len(frame)
    frame = aggregate(frame, mapping["dedupe"]["policy"], mapping["dedupe"]["key"])
    report["files"].append(dict(
        path=sorted(set(raw["__source"])), role=role, rows_in=rows_in, rows_kept=int(before),
        rows_bad_date=n_bad, rows_other_store=int(readable - before),
        unmapped_codes=dict(unmapped.most_common(20)), excluded_codes=excluded))
    report["duplicates_collapsed"] += int(before - len(frame))
    return frame


def _oos_frame(mapping, root):
    raw = read_raw(mapping, "oos", root)
    if raw is None:
        return None
    cols = mapping["columns"].get("oos") or {}
    for need in ("date", "item_code"):
        if not cols.get(need):
            raise IngestError(f"a file has role 'oos' but columns.oos.{need} is not mapped | "
                              f"name the raw header carrying the {need}, or drop the file entry")
    key = resolve_items(raw[cols["item_code"]], None, mapping)
    dates, _ = parse_dates(raw[cols["date"]], _date_format(mapping, "oos"))
    flag = cols.get("flag")
    truthy = (pd.Series(True, index=raw.index) if not flag
              else raw[flag].astype(str).str.strip().str.lower().isin(TRUTHY))
    out = pd.DataFrame({"item": key, "date": dates})[truthy.to_numpy()]
    return out[out["item"].notna() & out["date"].notna()].drop_duplicates()


def _apply_negatives(panel, mapping, report):
    policy = mapping["negatives"]["policy"]
    neg = (panel["sold"] < 0).to_numpy()
    share = float(neg.mean()) if len(panel) else 0.0
    if neg.any() and share > float(mapping["negatives"]["max_share"]):
        raise IngestError(
            f"{int(neg.sum())} of {len(panel)} item-days ({share:.1%}) have negative net sales, "
            f"above negatives.max_share | a few refund lines are normal; this many means the "
            "units column is a returns column, or the dedupe policy is dropping the sales half")
    if policy == "clip_zero":
        status = panel["row_status"].to_numpy(dtype=object)
        status[neg] = "suspect"               # clipped, but never silently: the row is marked
        panel["row_status"] = status
        panel["sold"] = panel["sold"].clip(lower=0.0)
    elif policy == "error" and neg.any():
        raise IngestError(f"{int(neg.sum())} item-days have negative net sales -- "
                          f"{_samples(panel.loc[neg, 'item'].unique())} | set "
                          "negatives.policy to 'clip_zero' to clip and count them instead")
    elif policy not in ("clip_zero", "error", "keep"):
        raise IngestError(f"negatives.policy is {policy!r} | expected 'clip_zero', 'error' "
                          "or 'keep'")
    # two numbers, because they are two facts: how many item-days netted out negative, and
    # how many rows this run actually changed. Under policy 'keep' the second is zero, and a
    # field called negatives_clipped that counts untouched rows is a lie a reader cannot see.
    report["negatives_seen"] = int(neg.sum())
    report["negatives_clipped"] = int(neg.sum()) if policy == "clip_zero" else 0
    return panel


def _check_imputed_costs(items, mapping):
    """Re-derive every cost_imputed item's cost from its department gross margin, or refuse.

    ht.config REQUIRES a cost on every item, so a store that cannot give one writes
    price * (1 - margin) into the items file by hand and sets cost_imputed. That made
    items.dept_gross_margin a field nothing read and cost_imputed a flag nothing checked:
    a typed-in 2.10 and a derived 3.36 look identical, and the difference is the whole
    critical fractile. This redoes the arithmetic and refuses a disagreement, and returns
    the margin behind each cost so every dollar figure downstream can print the assumption
    instead of asserting it.
    """
    margins = mapping["items"].get("dept_gross_margin") or {}
    out, problems = {}, []
    for key in sorted(items):
        it = items[key]
        if not it.get("cost_imputed"):
            continue
        margin = margins.get(it["dept"])
        if margin is None:
            problems.append(f"{key} is flagged cost_imputed but items.dept_gross_margin has no "
                            f"entry for department {it['dept']!r}")
            continue
        margin = float(margin)
        if not 0.0 < margin < 1.0:
            problems.append(f"items.dept_gross_margin[{it['dept']!r}] is {margin}: a gross "
                            "margin is a fraction, so 62% is 0.62")
            continue
        implied = round(float(it["price"]) * (1.0 - margin), 2)
        if abs(float(it["cost"]) - implied) > 0.01:
            problems.append(f"{key}: cost {it['cost']} is flagged cost_imputed, but a "
                            f"{margin:.0%} margin on price {it['price']} gives {implied}")
            continue
        out[key] = dict(dept=it["dept"], margin=margin, price=float(it["price"]),
                        cost=float(it["cost"]), derived_cost=implied)
    if problems:
        raise IngestError(
            "; ".join(problems) + " | fix items.dept_gross_margin in the mapping, or the item's "
            "cost, or clear cost_imputed if the store gave you a real cost. An imputed cost no "
            "margin reproduces is a number nobody can check, and it sets that item's "
            "recommended quantity and every dollar quoted for it")
    return out


def _economics(panel, items, mapping, report):
    """item_name, dept, unit_price, unit_cost. Config is the planning authority; the export
    only settles dollars."""
    panel["item_name"] = panel["item"].map({k: it["name"] for k, it in items.items()})
    panel["dept"] = panel["item"].map({k: it["dept"] for k, it in items.items()})

    price = panel["item"].map({k: it["price"] for k, it in items.items()}).astype(float)
    realized = pd.Series(np.nan, index=panel.index)
    if "dollars" in panel.columns:
        sold = panel["sold"].astype(float)
        ok = panel["dollars"].notna() & (sold > 0)
        realized[ok] = panel.loc[ok, "dollars"].astype(float) / sold[ok]
    panel["unit_price"] = realized.fillna(price)

    cost = panel["item"].map({k: it["cost"] for k, it in items.items()}).astype(float)
    if "row_cost" in panel.columns and panel["row_cost"].notna().any():
        sold = panel["sold"].astype(float)
        ok = panel["row_cost"].notna() & (sold > 0)
        realized_cost = pd.Series(np.nan, index=panel.index)
        realized_cost[ok] = panel.loc[ok, "row_cost"].astype(float) / sold[ok]
        panel["unit_cost"] = realized_cost.fillna(cost)
    else:
        panel["unit_cost"] = cost
    imputed = _check_imputed_costs(items, mapping)
    report["cost_imputation"] = imputed
    report["cost_imputed_items"] = sorted(imputed)

    tol = float(mapping["price_cost"]["tolerance_pct"])
    divergences = {}
    for item, grp in panel.groupby("item"):
        cfg = float(items[item]["price"])
        seen = grp.loc[realized.loc[grp.index].notna()]
        if not len(seen) or cfg <= 0:
            continue
        off = (realized.loc[seen.index] - cfg).abs() / cfg > tol
        if off.any():
            divergences[item] = dict(days=int(off.sum()), of=int(len(seen)),
                                     share=round(float(off.mean()), 4), config_price=cfg,
                                     median_realized=round(float(realized.loc[seen.index].median()),
                                                           4))
    report["price_divergences"] = divergences
    return panel


def _overrun_report(panel, items):
    """Item-days where sales exceeded the production record, per item.

    A production count is a proxy, so this is a data-quality figure rather than a fault: it
    tells a store how far its label log is from what actually left the case.
    """
    if "produced" not in panel.columns or not panel["produced"].notna().any():
        return {}
    eps = panel["item"].map(
        {k: ht_config.resolve_tolerance(it) for k, it in items.items()}).fillna(0.0)
    over = panel["produced"].notna() & (panel["sold"] > panel["produced"] + eps)
    return {k: int(v) for k, v in panel.loc[over, "item"].value_counts().sort_index().items()}


def ingest(mapping, items, root=".", weather=None, store=None, logger_backups=()):
    """Raw export -> (canonical panel, report). The whole path, in one fixed order."""
    report = dict(files=[], duplicates_collapsed=0, negatives_seen=0, negatives_clipped=0,
                  items_dropped={}, grid_rows_inserted={}, closures_applied={},
                  price_divergences={}, holiday_collisions=[],
                  stores=dict(selected=store, seen=collections.Counter(), rows_dropped=0))

    # --store filters on a store column. With none mapped, _tidy stamps the number on every
    # row instead, so --store would silently RELABEL this export as a different store: the
    # panel, the checkpoint's meta.json and every dollar figure below would be attributed to a
    # store whose sales they do not contain, and nothing downstream could detect it.
    if store is not None and not any((mapping["columns"].get(role) or {}).get("store")
                                     for role in mapping["columns"]):
        raise IngestError(
            f"--store {store!r} was given but no columns.<role>.store is mapped, so there is "
            f"nothing to filter on and every row would simply be stamped {store!r} | map the "
            f"raw header carrying the store number, or drop --store: this export already "
            f"carries whatever mapping.store names ({mapping['store']!r})")

    sales = _tidy(mapping, "sales", items, root, report, store)
    if sales is None or not len(sales):
        raise IngestError("no sales rows survived reading | check mapping.files paths, "
                          "date.format, and whether items.map covers this export's item codes")
    # one store by construction, and the number the sales rows actually carry rather than the
    # one mapping.store names: every later role has to join on the same key or merge to nothing
    store = str(sales["store"].iloc[0])
    report["stores"]["selected"] = store

    sales["row_status"] = "ok"
    sales = _apply_negatives(sales, mapping, report)
    panel = reindex_grid(sales, items, mapping)
    report["grid_rows_inserted"] = panel.attrs.get("grid_rows_inserted", {})
    panel = _apply_closures(panel, mapping, _hours_closures(mapping, root))
    report["closures_applied"] = panel.attrs.get("closures_applied", {})

    for role in ("production", "waste"):
        side = _tidy(mapping, role, items, root, report, store)
        if side is None:
            continue
        panel = panel.merge(side, on=list(schema.KEY), how="left", suffixes=("", "_side"))
    if "wasted" not in panel.columns:
        panel["wasted"] = np.nan
    if "produced" not in panel.columns:
        panel["produced"] = np.nan

    # which waste cells are the STORE'S OWN record, fixed before anything derives or fills one
    measured = panel["wasted"].notna().to_numpy()

    # wasted = produced - sold is valid ONLY for a day-fresh item; for anything with a shelf
    # life the identity is simply false and deriving it would manufacture the headline number.
    # The precedence is per CELL, as the contract states it: a shrink report that covers one
    # department, or six months, must not switch the identity off for the rest of the panel.
    if panel["produced"].notna().any():
        fresh = panel["item"].map({k: it["shelf_life_days"] == 1 for k, it in items.items()})
        # clipped at zero: a proxy production count (a label log, a clipboard) undercounts,
        # and negative waste would quietly subtract from the measured baseline
        derived = (panel["produced"].astype(float) - panel["sold"].astype(float)).clip(lower=0.0)
        derived = derived.where(fresh.fillna(False) & panel["produced"].notna())
        panel["wasted"] = panel["wasted"].where(measured, derived)

    if logger_backups:
        logged = _logger_waste(logger_backups, items, store, report)
        panel = _apply_logger_waste(panel, logged, measured, report)

    # which record each waste cell came from, because a mixed column is a mixed claim: the
    # export's own cells are measured, the derived ones are arithmetic on a production count
    logged_cells = int(report.get("logger", {}).get("item_days_written", 0))
    report["waste_cells"] = dict(
        export=int(measured.sum()), logger=logged_cells,
        derived=int(panel["wasted"].notna().sum()) - int(measured.sum()) - logged_cells)

    report["production_overruns"] = _overrun_report(panel, items)

    extra = ht_calendar.load_extra_holidays(mapping["calendar"]["extra_holidays_csv"]) \
        if mapping["calendar"].get("extra_holidays_csv") else None
    panel = ht_calendar.annotate(panel, extra=extra,
                                 payday_days=tuple(mapping["calendar"]["payday_days"]))
    report["holiday_collisions"] = [list(c) for c in ht_calendar.holiday_map.collisions]

    provider = weather if weather is not None else ht_weather.make_provider(mapping, root)
    wx = provider.frame(pd.DatetimeIndex(sorted(panel["date"].unique())))
    panel = panel.drop(columns=[c for c in ("tmax_f", "weather", "snow_tomorrow")
                                if c in panel.columns])
    panel = panel.merge(wx, on="date", how="left")
    report["weather"] = provider.report()

    panel, sellout = derive_sellout(panel, mapping, items, aux={"oos": _oos_frame(mapping, root)})
    report["sellout"] = sellout

    panel = _economics(panel, items, mapping, report)
    panel["is_closed"] = (panel["row_status"] != "ok").astype(np.int8)

    panel = schema.conform(panel)
    schema.assert_no_truth(panel)

    report["date_range"] = [str(panel.date.min().date()), str(panel.date.max().date())]
    report["n_days"] = int(panel.date.nunique())
    report["items_kept"] = sorted(panel.item.unique())
    report["items_dropped"] = {k: "no rows in the export" for k in items
                               if k not in set(panel.item)}
    report["row_status_counts"] = {k: int(v) for k, v in
                                   panel.row_status.value_counts().sort_index().items()}
    report["panel_hash"] = schema.panel_hash(panel)
    panel.attrs["ingest_report"] = report
    return panel, report


# ---- CLI ----

def _summary(report):
    lines = ["INGEST", "-" * 100]
    for f in report["files"]:
        lines.append(f"  {f['role']:<11s} rows in {f['rows_in']:>7d}  kept {f['rows_kept']:>7d}"
                     f"  bad dates {f['rows_bad_date']:>4d}  files {len(f['path'])}")
        if f["unmapped_codes"]:
            lines.append(f"    unmapped codes (dropped): {f['unmapped_codes']}")
        if f["excluded_codes"]:
            lines.append(f"    excluded by items.exclude: {f['excluded_codes']}")
    lines.append(f"  dates      {report['date_range'][0]} .. {report['date_range'][1]}  "
                 f"({report['n_days']} days, {len(report['items_kept'])} items)")
    lines.append(f"  repairs    duplicates collapsed {report['duplicates_collapsed']}  "
                 f"negatives {report['negatives_seen']} seen / "
                 f"{report['negatives_clipped']} clipped  "
                 f"grid rows inserted {sum(report['grid_rows_inserted'].values())}")
    w = report.get("waste_cells")
    if w:
        lines.append(f"  waste      {w['export']} cell(s) the export reported, {w['logger']} "
                     f"from the logger, {w['derived']} derived as produced - sold "
                     "(day-fresh items only)")
    lines.append(f"  row_status {report['row_status_counts']}")
    if report["items_dropped"]:
        lines.append(f"  dropped    {report['items_dropped']}")
    st = report["stores"]
    if len(st["seen"]) > 1 or st["rows_dropped"]:
        lines.append(f"  stores     kept {st['selected']} of {dict(st['seen'])}  "
                     f"rows dropped {st['rows_dropped']}")
    g = report.get("logger")
    if g:
        lines.append(f"  logger     {g['entries_used']} entries from {len(g['paths'])} backup(s)"
                     f"  {g['date_range'][0]} .. {g['date_range'][1]}  -> "
                     f"{g['item_days_written']} item-days, {g['units_written']} units of waste")
        if g["unmapped_items"]:
            lines.append(f"    NOT MATCHED to any configured item, {g['unmapped_entries']} "
                         f"entries dropped: {g['unmapped_items']}")
        if g["conflicts_with_export"]:
            lines.append(f"    {g['conflicts_with_export']} item-day(s) the export already "
                         f"reported waste for: kept the export's {g['conflict_units']['export']} "
                         f"units, ignored the logger's {g['conflict_units']['logged']}")
        if g["item_days_outside_the_export"]:
            lines.append(f"    {g['item_days_outside_the_export']} logged item-day(s) are outside "
                         f"the export's grid and fold in nowhere: {g['outside_examples']}")
        if g["duplicate_entries"] or g["unparsed_dates"]:
            lines.append(f"    {g['duplicate_entries']} repeated entry id(s) counted once, "
                         f"{g['unparsed_dates']} entries with an unreadable date dropped")
    for key, c in sorted(report.get("cost_imputation", {}).items()):
        lines.append(f"  COST IMPUTED {key}: cost {c['cost']:.2f} is price {c['price']:.2f} x "
                     f"(1 - {c['margin']:.0%} {c['dept']} margin), not the store's own number -- "
                     "every dollar quoted for it rests on that margin")
    s = report["sellout"]
    lines.append(f"  sellout    rule {s['rule']}  rate {s['rate']}  known_share "
                 f"{s['known_share']}  unknown days {s['unknown_days']}")
    w = report["weather"]
    lines.append(f"  weather    provider {w['provider']}  missing {w['missing_days']}  filled "
                 f"{w['filled_days']}  hindcast snow_tomorrow {w['snow_tomorrow_is_hindcast']}")
    if w["unknown_conditions"]:
        lines.append(f"    unknown conditions: {w['unknown_conditions']}")
    if report["holiday_collisions"]:
        lines.append(f"  holidays   collisions {report['holiday_collisions']}")
    if report["price_divergences"]:
        lines.append(f"  prices     divergent from config: {sorted(report['price_divergences'])}")
    lines.append(f"  panel_hash {report['panel_hash'][:16]}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ht.ingest",
                                description="Raw store export -> canonical panel.")
    p.add_argument("--mapping", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="data/panel.csv")
    p.add_argument("--report", default=None)
    p.add_argument("--sellout-rule", default=None, choices=list(SELLOUT_RULES))
    p.add_argument("--store", default=None,
                   help="keep only this store number, for a report run for the whole district; "
                        "needs columns.<role>.store mapped to the raw header carrying it")
    p.add_argument("--logger-backup", dest="logger_backups", action="append", default=[],
                   metavar="PATH",
                   help="a Phase-1 logger JSON backup (index.html, Setup > Data > Download "
                        "backup), folded into the panel's waste column. Repeatable: one per "
                        "phone that logged")
    p.add_argument("--from", dest="date_from", default=None)
    p.add_argument("--to", dest="date_to", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    from . import validate as ht_validate

    try:
        items = ht_config.load_items(args.items)
        mapping = ht_config.load_mapping(args.mapping, items)
    except schema.HtError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.sellout_rule:
        print(f"sellout rule overridden on the command line: "
              f"{mapping['sellout']['rule']!r} -> {args.sellout_rule!r}")
        mapping["sellout"]["rule"] = args.sellout_rule

    try:
        panel, report = ingest(mapping, items, root=args.root, store=args.store,
                               logger_backups=args.logger_backups)
    except schema.HtError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    if args.date_from or args.date_to:
        before = len(panel)
        if args.date_from:
            panel = panel[panel.date >= pd.Timestamp(args.date_from)]
        if args.date_to:
            panel = panel[panel.date <= pd.Timestamp(args.date_to)]
        panel = panel.reset_index(drop=True)
        # the report above describes everything that was READ; say so rather than rewriting it
        print(f"trimmed to {args.date_from or 'start'}..{args.date_to or 'end'}: "
              f"{before} rows -> {len(panel)}")

    if not args.quiet:
        print(_summary(report))
    # the panel first: everything after this can still fail, and a panel on disk is what the
    # runbook's recovery ("re-run ht.validate against it") needs in order to exist
    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        schema.write_panel(panel, args.out)
        if not args.quiet:
            print(f"\nwrote {args.out}  ({len(panel)} rows)")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, default=str)

    result = ht_validate.validate(panel, items, mapping)
    if not args.quiet:
        print()
        print(ht_validate.format_report(result))
    # the panel is written either way: a person has to be able to look at what went wrong
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
