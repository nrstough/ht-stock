"""Turn the synthetic store into a raw, real-shaped, deliberately dirty store export.

The whole real-data layer is untested until something that is not a canonical panel goes
through it. A merely renamed copy of data/store_synth.csv would prove the column mapping and
nothing else, so this writes what a store actually hands over: a movement report per year in
cp1252 with a title block above the header and a totals row below it, two-digit years,
parenthesised refunds, truncated all-caps descriptions, a separate scale label-printer log
standing in for a production count, a weather file in somebody else's vocabulary, and a store
hours sheet -- with the simulator's columns stripped out entirely and the rest renamed.

What is deliberately NOT exported is the point. `true_demand`, `true_mean` and `lost_sales`
are never read from the source file at all (usecols keeps them out of memory). Neither are
dow, holiday or payday -- a real POS hands over a date and the pipeline re-derives the rest --
nor stockout, wasted, item_name, dept or unit_cost, which come from a rule or from the items
config. Exporting any of those would let the rehearsal skip the work it exists to prove.

Run:  python tools/make_mock_export.py --src data/store_synth.csv --out .rehearsal/raw
"""
import argparse
import csv
import datetime as dt
import json
import os
import shutil

import numpy as np
import pandas as pd

SIM_ONLY = ("true_demand", "true_mean", "lost_sales")
OBSERVABLE = ("date", "item", "is_closed", "tmax_f", "weather", "snow_tomorrow",
              "produced", "sold", "unit_price")

# Strings no emitted DATA file may contain. The canonical names are on the list because a raw
# export that already speaks our vocabulary tests nothing: ht.ingest would have no mapping work
# to do. The two mapping JSONs are exempt (they must name sellout.rule "produced_vs_sold").
FORBIDDEN = SIM_ONLY + ("sold", "produced", "stockout", "is_closed", "snow_tomorrow",
                        "unit_price")

# item key -> (item number, movement-report description, department)
CATALOG = {
    "pizza-whole": ("771002", "PIZZA 16IN CHS WHL", "PIZZA"),
    "pizza-slice": ("771015", "PIZZA SLICE CHEESE", "PIZZA"),
    "doughnut": ("330118", "DOUGHNUT GLAZED EA", "BAKERY"),
    "cake": ("330442", "CAKE 8IN RND ICED", "BAKERY"),
    "bread": ("330901", "BREAD ITALIAN LOAF", "BAKERY"),
    "rotisserie": ("884213", "ROTIS CHKN ORIGINL", "HOT FOODS"),
    "hotbar-lb": ("00451", "HOT BAR SELF SERVE", "HOT FOODS"),
    "sushi": ("550012", "SUSHI ROLL CALIF 8", "FRESH FOODS"),
    "sub": ("612050", "SUB ITALIAN 12IN", "FRESH FOODS"),
}

# The messiness, each piece exercising one named ingest repair or validator finding.
GAP_START = "2024-03-10"        # a four-day systems outage: no sales AND no labels
GAP_DAYS = 4
DESC_CHANGE = ("doughnut", "2024-06-01", "DONUT GLZ EACH")      # staff re-key a description
CODE_CHANGE = ("rotisserie", "2025-04-02", "884219")            # pack change, new item number
CODE_BLANK_DAYS = 12            # trailing rotisserie days carrying only a description
SHORT_ITEM = ("sub", 30)        # kept only for its last 30 days: the short-history guard
OVERLAP_DAYS = 3                # re-exported window duplicated into the next year's file
N_REFUNDS = 5
UNMAPPED_CODE = ("777777", "SEASONAL PLATTER", "FRESH FOODS")
EXCLUDED_CODE = ("999999", "DEPT TRANSFER OUT", "HOT FOODS")
PRODUCTION_MISSING_SHARE = 0.08
UNKNOWN_CONDITION = "BLOWING DUST"      # in nobody's alias table, so it must be counted

# One synthetic weather kind, several ways a real feed writes it. Every one of these resolves
# through the mapping's kind_map or ht.weather.KIND_ALIASES; UNKNOWN_CONDITION does not.
CONDITIONS = {
    "sunny": ["CLEAR", "SUNNY", "FAIR", "MOSTLY SUNNY"],
    "cloudy": ["PARTLY CLOUDY", "MOSTLY CLOUDY", "CLOUDY", "OVERCAST", "FOG"],
    "rain": ["RAIN", "SHOWERS", "DRIZZLE", "T-STORM", "THUNDERSTORM"],
    "snow": ["SNOW", "FLURRIES", "WINTRY MIX", "SLEET"],
}

MOVEMENT_HEADER = ["BUS DT", "ITEM NBR", "ITEM DESC", "DEPT", "UNITS", "NET SALES $"]

# The same movement report as the district office runs it: one more column, one more store,
# and no other difference. It is emitted alongside the single-store files rather than instead
# of them, because both are things a store hands over and the chain has to survive either.
# The second store's units are the first store's scaled, which is enough to make a merged
# panel obviously wrong and a filtered one exactly right.
DISTRICT_HEADER = ["STORE"] + MOVEMENT_HEADER
DISTRICT_STORES = ("0123", "0456")
DISTRICT_SCALE = 1.37
TITLE_BLOCK = [
    ["RETAIL PRO 9 - ITEM MOVEMENT BY DAY"],
    ["STORE 0123   PREPARED FOODS   RUN {run}"],
    ["ALL FIGURES NET OF RETURNS. UNITS ARE SELLING UNITS."],
]


def _money(v, messy):
    """A retail report writes a refund as ($12.34), which is the one mangling that flips a sign."""
    if not messy:
        return f"{v:.2f}"
    return f"(${abs(v):,.2f})" if v < 0 else f"${v:,.2f}"


def _units(v, messy):
    text = f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"
    if not messy:
        return f"{v:g}"
    return f"({text.lstrip('-')})" if v < 0 else text


def _date(ts, messy):
    return ts.strftime("%m/%d/%y" if messy else "%Y-%m-%d")


def _write(path, rows, header, *, encoding, newline, title=None, footer=None):
    with open(path, "w", encoding=encoding, newline="") as fh:
        w = csv.writer(fh, lineterminator=newline)
        for line in (title or []):
            w.writerow(line)
        w.writerow(header)
        w.writerows(rows)
        if footer:
            w.writerow(footer)
    return len(rows)


def _codes_and_descs(sales, rng, messy, broken):
    """Item identity as the export writes it: barcodes, a renumbering and a re-keyed name."""
    def field(i):
        return sales["item"].map(lambda k: CATALOG[k][i] if k in CATALOG else None)

    # Rows whose item is not one of ours (an unmapped or excluded code) carry their own.
    code = field(0).fillna(sales["raw_code"]).to_numpy(dtype=object)
    desc = field(1).fillna(sales["raw_desc"]).to_numpy(dtype=object)
    dept = field(2).fillna(sales["raw_dept"]).to_numpy(dtype=object)

    if messy:
        # A weighed item prints a different barcode per package: 2 + 5-digit PLU + 4 digits.
        hot = (sales["item"] == "hotbar-lb").to_numpy()
        tail = rng.integers(0, 10000, size=int(hot.sum()))
        code[hot] = ["2" + CATALOG["hotbar-lb"][0] + f"{t:04d}" for t in tail]
    if not broken:
        return code, desc, dept

    item, since, new_desc = DESC_CHANGE
    hit = (sales["item"] == item).to_numpy() & (sales["date"] >= since).to_numpy()
    desc[hit] = new_desc

    item, since, new_code = CODE_CHANGE
    hit = (sales["item"] == item).to_numpy() & (sales["date"] >= since).to_numpy()
    code[hit] = new_code
    desc[hit] = "ROTIS CHKN LEM PEP"          # the mapping's alias_map entry
    # The last few days lose the number entirely -- only the alias can identify them.
    days = sorted(sales.loc[hit, "date"].unique())[-CODE_BLANK_DAYS:]
    code[hit & sales["date"].isin(days).to_numpy()] = ""
    return code, desc, dept


def _sales_frame(df, rng, broken, manifest):
    sales = df.loc[df["sold"] > 0, ["date", "item", "sold", "unit_price"]].copy()
    manifest["zero_rows_omitted"] = int(len(df) - len(sales))
    sales["raw_code"] = None
    sales["raw_desc"] = None
    sales["raw_dept"] = None
    sales["units"] = sales["sold"].to_numpy()
    sales["dollars"] = (sales["sold"] * sales["unit_price"]).round(2).to_numpy()
    sales = sales.drop(columns=["sold", "unit_price"])
    if not broken:
        return sales.sort_values(["date", "item"], kind="stable").reset_index(drop=True)

    gap = pd.date_range(GAP_START, periods=GAP_DAYS, freq="D")
    before = len(sales)
    sales = sales[~sales["date"].isin(gap)]
    manifest["gap_rows_dropped"] = int(before - len(sales))

    item, keep_days = SHORT_ITEM
    cut = sales["date"].max() - pd.Timedelta(days=keep_days - 1)
    before = len(sales)
    sales = sales[(sales["item"] != item) | (sales["date"] >= cut)]
    manifest["short_history_rows_dropped"] = int(before - len(sales))

    # Refunds print as their own line, so the day's true net is the SUM of its rows.
    room = sales[(sales["units"] >= 20) & (sales["item"] != item)]
    picks = rng.choice(len(room), size=N_REFUNDS, replace=False)
    refunds = room.iloc[sorted(picks)].copy()
    refunds["units"] = -np.ceil(refunds["units"] * 0.1)
    refunds["dollars"] = (refunds["dollars"] * refunds["units"] / room.iloc[sorted(picks)]
                          ["units"].to_numpy()).round(2)
    manifest["refund_rows"] = len(refunds)

    extra = []
    for code, desc, dept in (UNMAPPED_CODE, EXCLUDED_CODE):
        days = pd.to_datetime(sorted(sales["date"].unique()))[::200][:6]
        extra.append(pd.DataFrame(dict(date=days, item=None, raw_code=code, raw_desc=desc,
                                       raw_dept=dept, units=np.arange(1, len(days) + 1) * 1.0,
                                       dollars=np.arange(1, len(days) + 1) * 4.5)))
    manifest["unmapped_rows"] = int(len(extra[0]))
    manifest["excluded_rows"] = int(len(extra[1]))

    sales = pd.concat([sales, refunds] + extra, ignore_index=True)
    return sales.sort_values(["date", "item"], kind="stable").reset_index(drop=True)


def _production_frame(df, rng, broken, manifest):
    prod = df.loc[df["produced"] > 0, ["date", "item", "produced"]].copy()
    prod = prod.rename(columns={"produced": "labels"})
    if not broken:
        return prod.sort_values(["date", "item"], kind="stable").reset_index(drop=True)

    gap = pd.date_range(GAP_START, periods=GAP_DAYS, freq="D")
    prod = prod[~prod["date"].isin(gap)]
    item, keep_days = SHORT_ITEM
    cut = prod["date"].max() - pd.Timedelta(days=keep_days - 1)
    prod = prod[(prod["item"] != item) | (prod["date"] >= cut)]

    # Nobody prints labels every single day; the missing days are what makes produced NaN and
    # stockout_known 0, which is the branch a real store will actually spend time in.
    keep = rng.random(len(prod)) >= PRODUCTION_MISSING_SHARE
    manifest["production_days_missing"] = int((~keep).sum())
    prod = prod[keep]
    return prod.sort_values(["date", "item"], kind="stable").reset_index(drop=True)


def _overlap(frame, years, manifest, key):
    """The last few days of one year re-exported at the top of the next: a real chain does this."""
    dupes = []
    for year in years[:-1]:
        end = pd.Timestamp(f"{year}-12-31")
        window = pd.date_range(end - pd.Timedelta(days=OVERLAP_DAYS - 1), end, freq="D")
        block = frame[frame["date"].isin(window)].copy()
        block["__year"] = year + 1
        dupes.append(block)
    out = pd.concat(dupes, ignore_index=True) if dupes else frame.iloc[:0].copy()
    manifest[key] = int(len(out))
    return out


def _weather_rows(df, rng, broken, manifest):
    wx = df.groupby("date", as_index=False).agg(tmax_f=("tmax_f", "first"),
                                                weather=("weather", "first"))
    wx = wx.sort_values("date")
    pick = [CONDITIONS[k][int(i)] for k, i in
            zip(wx["weather"], rng.integers(0, 99, size=len(wx)) % [len(CONDITIONS[k])
                                                                    for k in wx["weather"]])]
    wx["condition"] = pick
    if broken:
        # One condition no alias table knows. It must be counted, never guessed into a category.
        target = len(wx) // 3
        wx.iloc[target, wx.columns.get_loc("condition")] = UNKNOWN_CONDITION
        manifest["unknown_conditions"] = 1
    rows = [[d.strftime("%Y-%m-%d"), f"{t:.1f}", c]
            for d, t, c in zip(wx["date"], wx["tmax_f"], wx["condition"])]
    return rows


def _hours_rows(df, messy):
    days = df.groupby("date", as_index=False)["is_closed"].max().sort_values("date")
    rows = []
    for d, closed in zip(days["date"], days["is_closed"]):
        if closed:
            rows.append([_date(d, messy), "", ""])
        else:
            rows.append([_date(d, messy), "07:00", "21:00" if d.weekday() < 6 else "20:00"])
    return rows


def _mapping(out_dir, dirt, sellout, district=False):
    """A source mapping that fits exactly the files just written, and nothing else."""
    messy = dirt in ("light", "full")
    fmt = "%m/%d/%y" if messy else "%Y-%m-%d"
    entries = [
        dict(role="sales", path=os.path.join(out_dir, "MOVEMENT_*.CSV"),
             encoding="cp1252" if messy else "utf-8", delimiter=",",
             header_row=4 if messy else 1, skip_footer_rows=1 if messy else 0,
             na_values=["", "N/A", "--", "."],
             note="rows 1-3 are a report title block; the last row is a DEPT TOTAL"),
        dict(role="production", path=os.path.join(out_dir, "SCALE_LABELS.CSV"),
             encoding="utf-8", delimiter=",", header_row=1, skip_footer_rows=0,
             na_values=["", "N/A"],
             note="labels printed per item per day: the best production proxy this store has"),
        dict(role="hours", path=os.path.join(out_dir, "STORE_HOURS.CSV"), encoding="utf-8",
             delimiter=",", header_row=1, skip_footer_rows=0, na_values=[""]),
        dict(role="weather", path=os.path.join(out_dir, "WEATHER.CSV"), encoding="utf-8",
             delimiter=",", header_row=1, skip_footer_rows=0, na_values=[""],
             date_format="%Y-%m-%d",
             note="a different system from the movement report, hence its own date format"),
    ]
    if sellout == "none":
        entries = [e for e in entries if e["role"] != "production"]
    if district:
        # only the sales role changes: the scale log, hours and weather files carry no store
        # column, so they join to whichever store --store selected
        entries[0] = dict(entries[0], path=os.path.join(out_dir, "DISTRICT_MOVEMENT.CSV"),
                          header_row=1, skip_footer_rows=0,
                          note="run for the whole district: one row per store per item-day")

    mapping = {
        "schema": "ht-source-mapping/1",
        "store": "0123",
        "notes": f"Mock 'Retail Pro' movement export written by tools/make_mock_export.py at "
                 f"dirt={dirt}. Not a real store.",
        "files": entries,
        "columns": {
            "sales": {"date": "BUS DT", "item_code": "ITEM NBR", "item_desc": "ITEM DESC",
                      "units": "UNITS", "dollars": "NET SALES $", "cost": None,
                      "store": "STORE" if district else None},
            "production": {"date": "PRINT DT", "item_code": "ITEM NBR",
                           "units": "LABELS PRINTED"},
            "hours": {"date": "DATE", "open": "OPEN", "close": "CLOSE"},
            "weather": {"date": "DATE", "tmax_f": "TMAX", "kind": "CONDITION",
                        "snow_tomorrow": None},
        },
        "date": {"format": fmt, "business_day": "store_close",
                 "note": "the simulator's day is a calendar day; a real store must confirm its "
                         "day-close cutoff in writing"},
        "numbers": {"strip": ["$", ",", "%", " "], "parens_negative": True,
                    "decimal": ".", "units_are_dollars": False},
        "items": {
            "drop_unmapped": True,
            "max_items": 400,
            "random_weight_plu": {"enabled": bool(messy), "pattern": r"^2(\d{5})\d{4}$",
                                  "plu_group": 1},
            "map": {"771002": "pizza-whole", "771015": "pizza-slice", "330118": "doughnut",
                    "330442": "cake", "330901": "bread", "884213": "rotisserie",
                    "884219": "rotisserie", "00451": "hotbar-lb", "550012": "sushi",
                    "612050": "sub"},
            "alias_map": {"ROTIS CHKN LEM PEP": "rotisserie"},
            "exclude": [EXCLUDED_CODE[0]],
            "dept_gross_margin": {"Bakery": 0.62, "Hot Foods": 0.58, "Fresh Foods": 0.55,
                                  "Pizza": 0.66},
            "split_history_note": f"{CODE_CHANGE[2]} replaced {CATALOG['rotisserie'][0]} on "
                                  f"{CODE_CHANGE[1]} after a pack change; both map to rotisserie "
                                  "so the series is not split in half and excluded twice",
        },
        "dedupe": {"key": ["store", "item", "date"], "policy": "sum"},
        "negatives": {"policy": "clip_zero", "max_share": 0.01},
        "gaps": {"max_unexplained_gap_days": 3},
        "closures": {"dates": ["2023-12-25", "2024-12-25", "2025-12-25"],
                     "partial": {"2025-12-24": 0.60, "2025-11-27": 0.55}},
        "sellout": {"rule": sellout, "validation_labels": None,
                    "note": "waste_zero is not implemented: a shrink sheet cannot distinguish a "
                            "missing row from a real zero"},
        "weather": {
            "provider": "csv",
            "kind_map": {"CLEAR": "sunny", "SUNNY": "sunny", "PARTLY CLOUDY": "cloudy",
                         "MOSTLY CLOUDY": "cloudy", "CLOUDY": "cloudy", "OVERCAST": "cloudy",
                         "FOG": "cloudy", "RAIN": "rain", "SHOWERS": "rain", "DRIZZLE": "rain",
                         "THUNDERSTORM": "rain", "T-STORM": "rain", "SNOW": "snow",
                         "SLEET": "snow", "WINTRY MIX": "snow", "FLURRIES": "snow"},
        },
        "calendar": {"country": "US", "extra_holidays_csv": None,
                     "payday_days": [1, 2, 3, 15, 16, 17]},
        "price_cost": {"authority": "config", "tolerance_pct": 0.15},
    }
    if sellout == "none":
        del mapping["columns"]["production"]
    return mapping


def make_export(src_csv, out_dir, *, seed=7, dirt="full"):
    """Write the raw export, the two mappings and the item config; return the manifest."""
    if dirt not in ("none", "light", "full"):
        raise ValueError(f"dirt is {dirt!r} | expected 'none', 'light' or 'full'")
    messy, broken = dirt in ("light", "full"), dirt == "full"
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # usecols is the guarantee: the simulator's three truth columns are never even read.
    df = pd.read_csv(src_csv, usecols=list(OBSERVABLE), parse_dates=["date"])
    manifest = dict(src=src_csv, out_dir=out_dir, seed=seed, dirt=dirt,
                    source_rows=len(df), files={}, applied={})

    sales = _sales_frame(df, rng, broken, manifest["applied"])
    prod = _production_frame(df, rng, broken, manifest["applied"])
    years = sorted({d.year for d in df["date"]})
    sales["__year"] = sales["date"].dt.year
    if broken:
        sales = pd.concat([sales, _overlap(sales, years, manifest["applied"], "duplicate_rows")],
                          ignore_index=True)
        prod = pd.concat([prod, _overlap(prod, years, manifest["applied"],
                                         "duplicate_label_rows").drop(columns="__year")],
                         ignore_index=True)
        sales = sales.sort_values(["__year", "date", "item"], kind="stable")
        prod = prod.sort_values(["date", "item"], kind="stable")
    # The code/desc arrays below are positional, so the frame's index must be too.
    sales = sales.reset_index(drop=True)
    prod = prod.reset_index(drop=True)

    code, desc, dept = _codes_and_descs(sales, rng, messy, broken)
    run = dt.datetime(2026, 1, 2, 6, 14).strftime("%m/%d/%y %H:%M")
    for year, grp in sales.groupby("__year", sort=True):
        rows = [[_date(d, messy), c, s, p, _units(u, messy), _money(v, messy)]
                for d, c, s, p, u, v in zip(grp["date"], code[grp.index], desc[grp.index],
                                            dept[grp.index], grp["units"], grp["dollars"])]
        title = [[t[0].format(run=run)] for t in TITLE_BLOCK] if messy else None
        footer = (["DEPT TOTAL", "", "", "", _units(grp["units"].sum(), messy),
                   _money(grp["dollars"].sum(), messy)] if messy else None)
        name = f"MOVEMENT_{year}.CSV"
        manifest["files"][name] = _write(
            os.path.join(out_dir, name), rows, MOVEMENT_HEADER,
            encoding="cp1252" if messy else "utf-8", newline="\r\n" if messy else "\n",
            title=title, footer=footer)

    district = []
    for d, c, s, p, u, v in zip(sales["date"], code, desc, dept, sales["units"],
                                sales["dollars"]):
        district.append([DISTRICT_STORES[0], _date(d, messy), c, s, p,
                         _units(u, messy), _money(v, messy)])
        district.append([DISTRICT_STORES[1], _date(d, messy), c, s, p,
                         _units(round(u * DISTRICT_SCALE, 2), messy),
                         _money(round(v * DISTRICT_SCALE, 2), messy)])
    manifest["files"]["DISTRICT_MOVEMENT.CSV"] = _write(
        os.path.join(out_dir, "DISTRICT_MOVEMENT.CSV"), district, DISTRICT_HEADER,
        encoding="cp1252" if messy else "utf-8", newline="\r\n" if messy else "\n")

    # The scale log carries the PLU, not the package barcode, and for a weighed item the
    # printed label records the weight -- so this column is pounds for hotbar-lb and pieces
    # elsewhere, on the same basis as UNITS. That is what makes it comparable at all.
    pcode = prod["item"].map(lambda k: CATALOG[k][0]).to_numpy(dtype=object)
    if broken:
        item, since, new_code = CODE_CHANGE
        pcode[(prod["item"] == item).to_numpy() & (prod["date"] >= since).to_numpy()] = new_code
    manifest["files"]["SCALE_LABELS.CSV"] = _write(
        os.path.join(out_dir, "SCALE_LABELS.CSV"),
        [[_date(d, messy), c, _units(v, messy)]
         for d, c, v in zip(prod["date"], pcode, prod["labels"])],
        ["PRINT DT", "ITEM NBR", "LABELS PRINTED"], encoding="utf-8", newline="\n")

    manifest["files"]["STORE_HOURS.CSV"] = _write(
        os.path.join(out_dir, "STORE_HOURS.CSV"), _hours_rows(df, messy),
        ["DATE", "OPEN", "CLOSE"], encoding="utf-8", newline="\n")
    manifest["files"]["WEATHER.CSV"] = _write(
        os.path.join(out_dir, "WEATHER.CSV"), _weather_rows(df, rng, broken, manifest["applied"]),
        ["DATE", "TMAX", "CONDITION"], encoding="utf-8", newline="\n")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.copyfile(os.path.join(root, "config", "items.example.json"),
                    os.path.join(out_dir, "items.json"))
    for name, rule in (("mapping.json", "produced_vs_sold"), ("mapping_nosellout.json", "none")):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(_mapping(out_dir, dirt, rule), fh, indent=1)
            fh.write("\n")
    with open(os.path.join(out_dir, "mapping_district.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(_mapping(out_dir, dirt, "produced_vs_sold", district=True), fh, indent=1)
        fh.write("\n")

    _assert_clean(out_dir)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return manifest


def _assert_clean(out_dir):
    """No emitted data file may leak a simulator column or speak the canonical vocabulary."""
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            text = fh.read().decode("cp1252", errors="replace").lower()
        # The mappings must name sellout.rule "produced_vs_sold"; only truth may not leak there.
        bad = [w for w in (SIM_ONLY if name.endswith(".json") else FORBIDDEN) if w in text]
        if bad:
            raise AssertionError(f"{path} contains {bad} -- the export would let the rehearsal "
                                 "skip the work it exists to prove")


def main(argv=None):
    ap = argparse.ArgumentParser(description="synthetic store -> raw real-shaped export")
    ap.add_argument("--src", default="data/store_synth.csv")
    ap.add_argument("--out", default=".rehearsal/raw")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dirt", choices=("none", "light", "full"), default="full")
    args = ap.parse_args(argv)

    m = make_export(args.src, args.out, seed=args.seed, dirt=args.dirt)
    print(f"wrote {args.out}  (dirt={args.dirt}, seed={args.seed})")
    for name, n in m["files"].items():
        print(f"  {name:22s} {n:6d} rows")
    print("  mapping.json, mapping_nosellout.json, mapping_district.json, items.json, "
          "manifest.json")
    if m["applied"]:
        print("dirt applied:")
        for k, v in sorted(m["applied"].items()):
            print(f"  {k:28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
