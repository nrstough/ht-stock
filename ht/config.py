"""Item economics and source mapping: the only place a store's own numbers enter the system.

Everything downstream is a function of this file -- the newsvendor critical fractile, the MAKE
column on the morning sheet, every dollar in every report -- so it validates hard and reports
every problem at once instead of one per run. It is also the artifact a store manager is handed,
which is why an unrecognised item field is an error rather than a shrug: that is exactly how a
simulator latent (base, sigma, dow multipliers) would leak into a document a store has to sign
off on. Those stay in sim/params.py.

Errors are things that make the file unusable; warnings are things worth saying out loud that
are not reasons to refuse it. load_* raises, validate_* returns a list of strings.
"""
import hashlib
import json

from model import newsvendor

from . import schema
from .schema import ConfigError, MappingError

ITEMS_SCHEMA = "ht-items/1"
MAPPING_SCHEMA = "ht-source-mapping/1"

DEFAULT_ITEM = dict(salvage=0.0, batch=1.0, continuous=False, unit="each",
                    shelf_life_days=1, sellout_tolerance=None,
                    cost_imputed=False, active=True, notes="")

REQUIRED_ITEM = ("name", "dept", "price", "cost", "batch")
ITEM_FIELDS = REQUIRED_ITEM + tuple(f for f in DEFAULT_ITEM if f not in REQUIRED_ITEM)
TOP_LEVEL = ("schema", "store", "currency", "defaults", "items", "notes")
UNITS = ("each", "lb")

FILE_ROLES = ("sales", "production", "waste", "oos", "hours", "weather")
SELLOUT_RULES = ("produced_vs_sold", "flag", "none")

# Named here rather than silently unsupported: both were considered and turned down, and a
# store that asks for one deserves the reason instead of "unknown rule".
REJECTED_SELLOUT_RULES = {
    "waste_zero":
        "it is produced_vs_sold rearranged, and a waste sheet the night crew skipped is "
        "indistinguishable from a real sellout on exactly the busy nights that matter -- false "
        "positives correlated with high demand, which is pure added waste. Use produced_vs_sold, "
        "or 'none' and say so",
    "last_sale_gap":
        "it needs a second ingest path over 1-3 GB/year of basket lines plus a per-item x dow x "
        "hour rate model; it is the documented extension point in docs/DATA_CONTRACT.md, not code",
}

COST_BASES = ("per_unit", "extended")
OVERRUN_POLICIES = ("warn", "error")

MAPPING_DEFAULTS = {
    "schema": MAPPING_SCHEMA,
    "store": "default",
    "notes": "",
    "files": [],
    "columns": {},
    "date": {"format": None, "business_day": "unknown", "note": ""},
    "numbers": {"strip": ["$", ","], "parens_negative": True, "decimal": ".",
                "units_are_dollars": False},
    "items": {"drop_unmapped": True, "max_items": 400, "map": {}, "alias_map": {},
              "exclude": [], "dept_gross_margin": {}, "split_history_note": "",
              "random_weight_plu": {"enabled": False, "pattern": r"^2(\d{5})\d{4}$",
                                    "plu_group": 1}},
    "dedupe": {"key": ["store", "item", "date"], "policy": "sum"},
    "negatives": {"policy": "clip_zero", "max_share": 0.01},
    "gaps": {"max_unexplained_gap_days": 3},
    # A production count is usually a proxy -- a label-printer log, a clipboard keyed in --
    # so a handful of days where sales exceed it is expected, not a broken export. The share
    # is what separates "the kitchen ran a second batch nobody printed for" from "this column
    # is not production at all".
    "production": {"overrun_policy": "warn", "max_overrun_share": 0.02},
    "closures": {"dates": [], "partial": {}},
    "sellout": {"rule": None, "validation_labels": None, "coverage_start": None,
                "coverage_end": None, "note": ""},
    "weather": {"provider": "none", "kind_map": {}, "columns": {}},
    "calendar": {"country": "US", "extra_holidays_csv": None,
                 "payday_days": [1, 2, 3, 15, 16, 17]},
    # cost_basis says what one row of the export's cost column means. There is no way to
    # tell a per-unit cost from an extended one by looking at it, and summing the wrong one
    # multiplies every settlement dollar by the day's line count.
    "price_cost": {"authority": "config", "tolerance_pct": 0.15, "cost_basis": "per_unit"},
}

FILE_DEFAULTS = {"role": None, "path": None, "encoding": "utf-8", "delimiter": ",",
                 "header_row": 1, "skip_footer_rows": 0,
                 "na_values": ["", "N/A", "--", "."], "note": ""}


# ---- reading ----

def _read_json(path, expect_schema, error):
    """Parse a config file, refusing duplicate keys -- json.load silently keeps the last."""
    dupes = []

    def pairs(kvs):
        seen = set()
        for k, _ in kvs:
            if k in seen:
                dupes.append(k)
            seen.add(k)
        return dict(kvs)

    try:
        with open(path, "rb") as fh:
            data = json.loads(fh.read().decode("utf-8"), object_pairs_hook=pairs)
    except FileNotFoundError:
        raise error(f"{path}: no such file") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error(f"{path}: not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise error(f"{path}: top level must be a JSON object, got {type(data).__name__}")
    if dupes:
        raise error(f"{path}: duplicate key(s) {sorted(set(dupes))} -- JSON keeps only the last "
                    "one, so a record you wrote is being silently discarded")
    if data.get("schema") != expect_schema:
        raise error(f"{path}: schema is {data.get('schema')!r}, expected {expect_schema!r}")
    return data


def _problems(path, problems):
    return "\n".join([f"{path}: {len(problems)} problem(s)"] + [f"  - {p}" for p in problems])


def config_hash(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


# ---- items ----

def _as_str(v):
    if not isinstance(v, str):
        raise TypeError(f"expected text, got {type(v).__name__} {v!r}")
    return v


def _as_float(v):
    if isinstance(v, bool):
        raise TypeError(f"expected a number, got {v!r}")
    return float(v)


def _as_int(v):
    f = _as_float(v)
    if f != int(f):
        raise ValueError(f"expected a whole number, got {v!r}")
    return int(f)


def _as_bool(v):
    if not isinstance(v, bool):
        raise TypeError(f"expected true or false, got {v!r}")
    return v


_COERCE = {"name": _as_str, "dept": _as_str, "unit": _as_str, "notes": _as_str,
           "price": _as_float, "cost": _as_float, "salvage": _as_float, "batch": _as_float,
           "shelf_life_days": _as_int,
           "continuous": _as_bool, "cost_imputed": _as_bool, "active": _as_bool}


def _coerce_item(key, item, problems):
    ok = True
    for field, coerce in _COERCE.items():
        try:
            item[field] = coerce(item[field])
        except (TypeError, ValueError) as exc:
            problems.append(f"{key}.{field}: {exc}")
            ok = False
    if item["sellout_tolerance"] is not None:
        try:
            item["sellout_tolerance"] = _as_float(item["sellout_tolerance"])
        except (TypeError, ValueError) as exc:
            problems.append(f"{key}.sellout_tolerance: {exc}; null means 'use the unit default'")
            ok = False
    return ok


def _check_item(key, it):
    bad = []
    if it["price"] <= 0:
        bad.append(f"{key}.price is {it['price']}: must be > 0")
    if it["cost"] < 0:
        bad.append(f"{key}.cost is {it['cost']}: must be >= 0")
    if it["cost"] >= it["price"]:
        bad.append(f"{key}: cost {it['cost']} >= price {it['price']}, so the critical fractile "
                   "is non-positive and the item should not be produced at all")
    if it["salvage"] < 0 or (it["cost"] > 0 and it["salvage"] >= it["cost"]):
        bad.append(f"{key}.salvage is {it['salvage']}: must be >= 0 and below cost "
                   f"({it['cost']}); recovering the full cost would make waste free")
    if it["batch"] <= 0:
        bad.append(f"{key}.batch is {it['batch']}: must be > 0 -- it is the production rounding "
                   "unit (one tray, one bake, one pan), never zero")
    if it["shelf_life_days"] < 1:
        bad.append(f"{key}.shelf_life_days is {it['shelf_life_days']}: must be >= 1")
    if it["unit"] not in UNITS:
        bad.append(f"{key}.unit is {it['unit']!r}: expected one of {list(UNITS)}")
    if it["continuous"] and it["unit"] == "each":
        bad.append(f"{key}: continuous is true but unit is 'each'; a weighed item is sold in "
                   "pounds, and the two settings disagree about how batch rounds")
    return bad


def load_items(path, include_inactive=False):
    """Load the item economics. Merge order: DEFAULT_ITEM, the file's `defaults`, the item."""
    data = _read_json(path, ITEMS_SCHEMA, ConfigError)
    problems = []

    unknown = sorted(k for k in data if k not in TOP_LEVEL)
    if unknown:
        problems.append(f"unknown top-level key(s) {unknown}; expected {list(TOP_LEVEL)}")

    file_defaults = data.get("defaults") or {}
    if not isinstance(file_defaults, dict):
        problems.append(f"`defaults` must be an object, got {type(file_defaults).__name__}")
        file_defaults = {}
    for field in sorted(f for f in file_defaults if f not in ITEM_FIELDS):
        problems.append(f"defaults.{field}: unknown field; expected {list(ITEM_FIELDS)}")

    raw = data.get("items")
    if not isinstance(raw, dict) or not raw:
        problems.append("no `items` block: the file must map at least one canonical item key to "
                        "its economics, and the keys must match the panel's `item` column")
        raise ConfigError(_problems(path, problems))

    out = {}
    # sorted, not file order: model/backtest.py builds results.json's q_star block straight from
    # this dict and dumps it with indent=1 and no sort_keys, so insertion order here is
    # load-bearing on a frozen provenance file.
    for key in sorted(raw):
        rec = raw[key]
        if not isinstance(rec, dict):
            problems.append(f"{key}: must be an object, got {type(rec).__name__}")
            continue
        strays = sorted(f for f in rec if f not in ITEM_FIELDS)
        if strays:
            problems.append(f"{key}: unknown field(s) {strays}; expected {list(ITEM_FIELDS)}. "
                            "Simulator latents (base, dow, sigma, seas_amp, trend_per_year) "
                            "belong in sim/params.py -- a store manager cannot answer for them")
        supplied = dict(file_defaults)
        supplied.update(rec)
        missing = [f for f in REQUIRED_ITEM if supplied.get(f) is None]
        if missing:
            problems.append(f"{key}: missing required field(s) {missing}")
            continue
        item = dict(DEFAULT_ITEM)
        item.update({f: v for f, v in supplied.items() if f in ITEM_FIELDS})
        if _coerce_item(key, item, problems):
            problems.extend(_check_item(key, item))
            out[key] = item

    if problems:
        raise ConfigError(_problems(path, problems))
    if not include_inactive:
        out = {k: v for k, v in out.items() if v["active"]}
    if not out:
        raise ConfigError(f"{path}: every item is inactive, so there is nothing to forecast; "
                          "pass include_inactive=True to load them for context anyway")
    return out


def resolve_tolerance(item):
    """eps for the produced_vs_sold rule: produced pounds and sold pounds never match exactly."""
    tol = item.get("sellout_tolerance")
    if tol is None:
        return 0.5 if item.get("unit") == "lb" else 0.0
    return float(tol)


def critical_fractiles(items):
    """One shared q* definition, so backtest, evaluate and shadow cannot drift apart."""
    return {k: newsvendor.critical_fractile(it["price"], it["cost"], it.get("salvage", 0.0))
            for k, it in items.items()}


def validate_items(items, panel=None):
    warns = []
    for key, q in sorted(critical_fractiles(items).items()):
        if not 0.05 <= q <= 0.99:
            warns.append(f"{key}: critical fractile {q:.3f} falls outside the model's tau grid "
                         "[0.05, 0.99]; newsvendor.quantity extrapolates flat, so the "
                         "recommendation is the edge quantile, not the one the economics ask for")
    for key in sorted(items):
        it = items[key]
        if it["shelf_life_days"] > 1:
            warns.append(f"{key}: shelf_life_days={it['shelf_life_days']}, so the single-period "
                         "newsvendor and wasted = produced - sold are both false for it. It is "
                         "SHADOW ONLY on the sheet and excluded from the waste bound")
        if it["cost_imputed"]:
            warns.append(f"{key}: cost {it['cost']} came from a department gross margin, not from "
                         "the store; every report quoting its dollars must print the assumption")
    if panel is None:
        return warns

    present = set(panel["item"])
    for key in sorted(set(items) - present):
        warns.append(f"{key}: in the items config but absent from the panel")
    for key in sorted(set(items) & present):
        sold = [float(v) for v in panel.loc[panel["item"] == key, "sold"] if v == v]
        frac = sum(1 for v in sold if v != int(v))
        if items[key]["unit"] == "each" and frac:
            warns.append(f"{key}: unit is 'each' but {frac} of {len(sold)} days have fractional "
                         "sales; it is probably weighed, and batch rounding is wrong if so")
        if items[key]["unit"] == "lb" and sold and not frac:
            warns.append(f"{key}: unit is 'lb' but all {len(sold)} days are whole numbers; either "
                         "it is not weighed, or the export rounded the pounds away")
    return warns


# ---- source mapping ----

def _merge_mapping(data):
    out = {}
    for key, default in MAPPING_DEFAULTS.items():
        got = data.get(key)
        if not isinstance(default, dict):
            out[key] = default if got is None else got
            continue
        merged = dict(default)
        if isinstance(got, dict):
            merged.update(got)
            plu = default.get("random_weight_plu")
            if plu and isinstance(got.get("random_weight_plu"), dict):
                merged["random_weight_plu"] = {**plu, **got["random_weight_plu"]}
        out[key] = merged
    for key in data:                      # carry sections we do not know about rather than drop
        out.setdefault(key, data[key])
    out["files"] = [{**FILE_DEFAULTS, **f} for f in out["files"] if isinstance(f, dict)]
    return out


def load_mapping(path, items=None):
    """Load a source mapping with every documented default filled in.

    `items` is optional so a mapping can be read without the item catalog; supply it and the
    items.map targets are checked against the real roster.
    """
    mapping = _merge_mapping(_read_json(path, MAPPING_SCHEMA, MappingError))
    problems = []

    for i, f in enumerate(mapping["files"]):
        if f["role"] not in FILE_ROLES:
            problems.append(f"files[{i}].role is {f['role']!r} | expected one of "
                            f"{list(FILE_ROLES)}")
        if not f["path"]:
            problems.append(f"files[{i}] (role {f['role']!r}) has no path | every file entry "
                            "needs one, a glob is fine")
    if not any(f["role"] == "sales" for f in mapping["files"]):
        problems.append("no file with role 'sales' | mapping.files must include the item "
                        "movement export; nothing else can stand in for it")

    sales_cols = mapping["columns"].get("sales") or {}
    for need in ("date", "item_code", "units"):
        if not sales_cols.get(need):
            problems.append(f"columns.sales.{need} is not mapped | name the raw header that "
                            f"carries the {need}")

    fmt = mapping["date"].get("format")
    if not fmt or fmt == "auto":
        problems.append(f"date.format is {fmt!r} | give an explicit strftime string such as "
                        "'%m/%d/%y'; a guess silently reads 3/4/25 as either March or April")

    if mapping["numbers"].get("units_are_dollars"):
        problems.append("numbers.units_are_dollars is true | dollars are net of markdowns, so "
                        "units-derived-from-dollars are wrong precisely on promotion days. "
                        "Export a units column")

    rule = mapping["sellout"].get("rule")
    if rule is None:
        problems.append("sellout.rule is missing | it is required, and \"none\" is a valid "
                        "explicit answer; the code must never pick this for you")
    elif rule in REJECTED_SELLOUT_RULES:
        problems.append(f"sellout.rule {rule!r} is not implemented | "
                        f"{REJECTED_SELLOUT_RULES[rule]}")
    elif rule not in SELLOUT_RULES:
        problems.append(f"sellout.rule {rule!r} is unknown | expected one of "
                        f"{list(SELLOUT_RULES)}")
    elif rule == "flag":
        for bound in ("coverage_start", "coverage_end"):
            if not mapping["sellout"].get(bound):
                problems.append(f"sellout.{bound} is missing | rule 'flag' can only tell 'did "
                                "not sell out' from 'nobody was looking' inside a stated window")

    key = list(mapping["dedupe"].get("key") or [])
    if set(key) != set(schema.KEY):
        problems.append(f"dedupe.key is {key} | it must be {list(schema.KEY)}. Trimming it is "
                        "a natural edit for a single-store pilot and it drops the column the "
                        "panel is grouped by downstream; 'store' costs nothing on one store "
                        "and is what lets a second one join later without a schema change")

    if mapping["sellout"].get("validation_labels"):
        problems.append("sellout.validation_labels names a file nothing reads | there is no "
                        "confusion-matrix check yet; map the out-of-stock log as a files entry "
                        "with role 'oos' instead, and set it to null so the config does not "
                        "claim a measurement that is not happening")

    basis = mapping["price_cost"].get("cost_basis")
    if basis not in COST_BASES:
        problems.append(f"price_cost.cost_basis is {basis!r} | expected 'per_unit' (the column "
                        "is a cost each) or 'extended' (it is already cost x units); they differ "
                        "by the day's line count and nothing in the file says which")

    policy = mapping["production"].get("overrun_policy")
    if policy not in OVERRUN_POLICIES:
        problems.append(f"production.overrun_policy is {policy!r} | expected "
                        f"{list(OVERRUN_POLICIES)}")

    if items is not None:
        stray = sorted({v for v in mapping["items"]["map"].values() if v not in items})
        if stray:
            problems.append(f"items.map resolves to {stray}, which are not keys of the items "
                            "config | add them there, or fix the mapping")

    if problems:
        raise MappingError(_problems(path, problems))
    return mapping


def validate_mapping(mapping, items):
    warns = []
    roles = {f["role"] for f in mapping["files"]}
    if not mapping["items"].get("drop_unmapped"):
        warns.append("items.drop_unmapped is false: a single unrecognised item code anywhere in "
                     "the export aborts the whole ingest")
    if "production" not in roles:
        warns.append("no file with role 'production': waste cannot be measured, and the only "
                     "definitionally-correct sellout rule is unavailable. Asking for a production "
                     "count -- even a photographed clipboard keyed in -- is the single most "
                     "valuable question in the permission conversation")
    if not (mapping["columns"].get("sales") or {}).get("cost"):
        imputed = sorted(k for k, it in items.items() if it.get("cost_imputed"))
        warns.append("no cost column in the sales export: settlement dollars use the items "
                     "config's planning cost"
                     + (f", and {imputed} are imputed from a department margin" if imputed else ""))
    if mapping["date"].get("business_day") == "unknown":
        warns.append("date.business_day is 'unknown': a cloud BI export in UTC shifts a US store "
                     "4-5 hours and pushes evening sales into tomorrow, scrambling day of week. "
                     "Confirm the store's day-close cutoff in writing")
    return warns
