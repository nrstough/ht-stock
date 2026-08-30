"""Ingest is where a store's mess becomes one known table, and it is a correlated risk.

Every repair lives in this one mapping-driven module, so a wrong date format or a wrong
dedupe policy corrupts every downstream number at once and uniformly, and the model fits
the corruption happily. These tests therefore point ingest at an export that is wrong in
every way a real one is -- cp1252 with CRLF, a report title block, a TOTAL footer,
MM/DD/YY, thousands separators, parenthesised refunds, a duplicated export window, an
omitted day, a random-weight barcode per package, and an item number that changes
mid-history -- and check the panel that comes out the far side, row by row.
"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import pytest

from ht import config, ingest, schema

ITEMS = {
    "schema": "ht-items/1",
    "items": {
        "bread": {"name": "Bread Loaf", "dept": "Bakery", "price": 3.99, "cost": 1.10,
                  "batch": 1},
        "rotisserie": {"name": "Rotisserie", "dept": "Hot Foods", "price": 7.99,
                       "cost": 3.20, "batch": 4},
        "hotbar-lb": {"name": "Hot Bar", "dept": "Hot Foods", "price": 9.99, "cost": 3.50,
                      "batch": 5, "continuous": True, "unit": "lb",
                      "sellout_tolerance": 0.0},
    },
}

START = dt.date(2025, 1, 1)
DAYS = 160
MISSING_DAY = dt.date(2025, 2, 10)                 # one hole: a genuine zero-sales day
GAP = [dt.date(2025, 3, 3) + dt.timedelta(days=i) for i in range(5)]   # a systems outage
REFUND_DAY = dt.date(2025, 4, 7)
# an overlapping export window duplicates a row. Deliberately placed on a day the
# production sheet is missing: under dedupe policy "sum" a duplicate on a day that HAS a
# production record doubles the sales against it and the validator rightly errors, which
# is a fact about the policy rather than about ingest.
DUP_DAY = dt.date(2025, 1, 18)


def _units(item, day):
    """Deterministic, and never zero, so an absent row means an absent row."""
    base = {"bread": 40.0, "rotisserie": 30.0, "hotbar-lb": 55.0}[item]
    return round(base + 6.0 * np.sin(day.toordinal() / 7.0 * 2 * np.pi) + (day.day % 5), 1)


def _movement_rows():
    rows = []
    for i in range(DAYS):
        day = START + dt.timedelta(days=i)
        if day == MISSING_DAY or day in GAP:
            continue
        for item in ("bread", "rotisserie", "hotbar-lb"):
            units = _units(item, day)
            if item == "bread":
                code, desc = "330901", "BREAD WHITE 20OZ"
            elif item == "rotisserie":
                # a pack change discontinues one item number and re-adds the product
                code = "884213" if day < dt.date(2025, 3, 20) else "884219"
                desc = "ROTIS CHKN"
            else:
                # a weighed item prints a different barcode per package: 2 + PLU + cents
                code = f"2004510{int(units * 10) % 1000:03d}"
                desc = "HOT BAR SELF SERVE"
            dollars = units * 4.5
            rows.append((f"{day:%m/%d/%y}", code, desc,
                         f"{units:,.1f}", f"${dollars:,.2f}"))
        if day == REFUND_DAY:
            rows.append((f"{day:%m/%d/%y}", "330901", "BREAD WHITE 20OZ",
                         "(12.0)", "($54.00)"))          # a refund prints in parentheses
        if day == DUP_DAY:
            rows.append(rows[-1])                        # an overlapping export window
        if i == 0:
            rows.append((f"{day:%m/%d/%y}", "777777", "MYSTERY ITEM", "5.0", "$20.00"))
    return rows


def _write_export(tmp_path):
    lines = ["FRESH FOODS MOVEMENT REPORT", "STORE 0123", "RUN 06/01/25",
             "BUS DT,ITEM NBR,ITEM DESC,UNITS,NET SALES $"]
    lines += [",".join(r) for r in _movement_rows()]
    lines.append("DEPT TOTAL,,,999999,$0.00")
    (tmp_path / "MOVEMENT.CSV").write_bytes(
        ("\r\n".join(lines) + "\r\n").encode("cp1252"))

    prod = ["PRINT DT,ITEM NBR,LABELS PRINTED"]
    for i in range(DAYS):
        day = START + dt.timedelta(days=i)
        if day == MISSING_DAY or day in GAP or i % 12 == 5:   # ~8% of days have no sheet
            continue
        for item, code in (("bread", "330901"), ("rotisserie", "884213"),
                           ("hotbar-lb", "00451")):
            prod.append(f"{day:%m/%d/%y},{code},{_units(item, day) + 4:.1f}")
    (tmp_path / "SCALE_LABELS.CSV").write_text("\n".join(prod) + "\n", encoding="utf-8")

    wx = ["DATE,TMAX,CONDITION"]
    for i in range(DAYS):
        day = START + dt.timedelta(days=i)
        kind = ["CLEAR", "PARTLY CLOUDY", "T-STORM", "WINTRY MIX", "FOG"][i % 5]
        if i == 3:
            kind = "BLOWING DUST"                    # nothing can resolve this, by design
        wx.append(f"{day:%m/%d/%y},{40 + i % 30},{kind}")
    (tmp_path / "WEATHER.CSV").write_text("\n".join(wx) + "\n", encoding="utf-8")


MAPPING = {
    "schema": "ht-source-mapping/1",
    "store": "0123",
    "files": [
        {"role": "sales", "path": "MOVEMENT.CSV", "encoding": "cp1252",
         "header_row": 4, "skip_footer_rows": 1},
        {"role": "production", "path": "SCALE_LABELS.CSV"},
        {"role": "weather", "path": "WEATHER.CSV"},
    ],
    "columns": {
        "sales": {"date": "BUS DT", "item_code": "ITEM NBR", "item_desc": "ITEM DESC",
                  "units": "UNITS", "dollars": "NET SALES $"},
        "production": {"date": "PRINT DT", "item_code": "ITEM NBR",
                       "units": "LABELS PRINTED"},
        "weather": {"date": "DATE", "tmax_f": "TMAX", "kind": "CONDITION"},
    },
    "date": {"format": "%m/%d/%y", "business_day": "store_close"},
    "numbers": {"strip": ["$", ",", " "], "parens_negative": True, "decimal": ".",
                "units_are_dollars": False},
    "items": {
        "drop_unmapped": True,
        "random_weight_plu": {"enabled": True, "pattern": r"^2(\d{5})\d{4}$",
                              "plu_group": 1},
        "map": {"330901": "bread", "884213": "rotisserie", "884219": "rotisserie",
                "00451": "hotbar-lb"},
    },
    "dedupe": {"key": ["store", "item", "date"], "policy": "sum"},
    "negatives": {"policy": "clip_zero", "max_share": 0.01},
    "gaps": {"max_unexplained_gap_days": 3},
    "sellout": {"rule": "produced_vs_sold"},
    "weather": {"provider": "csv", "date_format": "%m/%d/%y"},
    "calendar": {"payday_days": [1, 2, 3, 15, 16, 17]},
    "price_cost": {"authority": "config", "tolerance_pct": 0.15},
}


@pytest.fixture
def export(tmp_path):
    """A raw store export that is wrong in every way a real one is."""
    _write_export(tmp_path)
    (tmp_path / "items.json").write_text(json.dumps(ITEMS), encoding="utf-8")
    (tmp_path / "mapping.json").write_text(json.dumps(MAPPING), encoding="utf-8")
    items = config.load_items(str(tmp_path / "items.json"))
    mapping = config.load_mapping(str(tmp_path / "mapping.json"), items)
    return dict(root=str(tmp_path), mapping=mapping, items=items,
                mapping_path=str(tmp_path / "mapping.json"),
                items_path=str(tmp_path / "items.json"))


@pytest.fixture
def ingested(export):
    panel, report = ingest.ingest(export["mapping"], export["items"], root=export["root"])
    return panel, report, export


# ---- the unit pieces ----

@pytest.mark.parametrize("text,value", [
    ("$1,204", 1204.0), ("(12.34)", -12.34), ("7", 7.0), (" 3.5 ", 3.5),
])
def test_clean_numeric_unmangles_currency_text(text, value):
    got = ingest.clean_numeric(pd.Series([text]))
    assert float(got.iloc[0]) == pytest.approx(value)


def test_clean_numeric_blanks_become_nan_not_zero():
    got = ingest.clean_numeric(pd.Series(["", None, "N/A"]))
    assert got.isna().all()                     # a missing measure is not a confident zero


def test_parse_dates_needs_an_explicit_format():
    with pytest.raises(schema.IngestError):
        ingest.parse_dates(pd.Series(["03/04/25"]), "auto")


def test_a_two_digit_date_is_read_the_way_the_mapping_says():
    parsed, dropped = ingest.parse_dates(pd.Series(["03/04/25"]), "%m/%d/%y")
    assert parsed.iloc[0] == pd.Timestamp("2025-03-04")     # March, never April
    assert dropped == 0


def test_title_and_total_rows_leave_by_failing_to_parse():
    rows = pd.Series(["01/01/25"] * 400 + ["DEPT TOTAL"])
    parsed, dropped = ingest.parse_dates(rows, "%m/%d/%y")
    assert dropped == 1
    assert parsed.isna().sum() == 1


def test_a_wrong_format_raises_rather_than_dropping_the_file():
    with pytest.raises(schema.IngestError) as exc:
        ingest.parse_dates(pd.Series(["2025-01-01"] * 100), "%m/%d/%y")
    assert "date.format" in str(exc.value) or "%m/%d/%y" in str(exc.value)


def test_random_weight_barcodes_collapse_to_one_item():
    codes = pd.Series([f"20045100{i:02d}" for i in range(50)])
    resolved = ingest.resolve_items(codes, None, MAPPING)
    assert set(resolved.dropna()) == {"hotbar-lb"}


def test_a_barcode_explosion_raises_because_it_is_a_model_shape_failure():
    mapping = json.loads(json.dumps(MAPPING))
    mapping["items"]["random_weight_plu"]["enabled"] = False
    mapping["items"]["max_items"] = 10          # a three-item store, not a whole department
    codes = pd.Series([f"20045100{i:02d}" for i in range(60)])
    with pytest.raises(schema.IngestError) as exc:
        ingest.resolve_items(codes, None, mapping)
    assert "item codes" in str(exc.value) and "random_weight_plu" in str(exc.value)


def test_a_whole_department_export_ingests_instead_of_tripping_the_ceiling():
    """The normal shape of a real export: hundreds of item numbers, nine of them ours.

    items.drop_unmapped says to drop the rest and report the count, so the ceiling has to
    count what BECOMES an item -- the thing that sizes nn.Embedding -- not what the store
    happened to put in the file.
    """
    codes = pd.Series(["330901"] * 5 + [f"5{i:05d}" for i in range(200)])
    resolved = ingest.resolve_items(codes, None, MAPPING)
    assert set(resolved.dropna()) == {"bread"}
    assert int(resolved.attrs["unmapped_codes"]) == 200


def test_two_item_numbers_stitch_into_one_series():
    codes = pd.Series(["884213", "884219"])
    assert list(ingest.resolve_items(codes, None, MAPPING)) == ["rotisserie", "rotisserie"]


def test_aggregate_keeps_an_absent_measure_null():
    frame = pd.DataFrame({"store": ["a", "a"], "item": ["b", "b"],
                          "date": pd.to_datetime(["2026-01-01"] * 2),
                          "sold": [3.0, 4.0], "produced": [np.nan, np.nan]})
    out = ingest.aggregate(frame, "sum")
    assert float(out["sold"].iloc[0]) == 7.0             # refunds net into their own day
    assert out["produced"].isna().all()                  # not 0.0


def test_aggregate_rejects_an_unknown_policy():
    frame = pd.DataFrame({"store": ["a"], "item": ["b"],
                          "date": pd.to_datetime(["2026-01-01"]), "sold": [1.0]})
    with pytest.raises(schema.IngestError):
        ingest.aggregate(frame, "median")


def test_read_raw_refuses_a_spreadsheet(tmp_path):
    (tmp_path / "S.xlsx").write_bytes(b"PK\x03\x04")
    mapping = {"files": [{"role": "sales", "path": "S.xlsx"}]}
    with pytest.raises(schema.IngestError) as exc:
        ingest.read_raw(mapping, "sales", root=str(tmp_path))
    assert "CSV" in str(exc.value)


# ---- the whole path ----

def test_the_panel_is_canonical_and_carries_no_simulator_truth(ingested):
    panel, _, _ = ingested
    assert list(panel.columns) == list(schema.NAMES)
    schema.assert_no_truth(panel)
    assert set(panel["item"]) == {"bread", "rotisserie", "hotbar-lb"}


def test_the_export_is_read_through_its_encoding_and_header_block(ingested):
    panel, report, _ = ingested
    assert report["date_range"] == ["2025-01-01", str(START + dt.timedelta(days=DAYS - 1))]
    # skip_footer_rows removes the DEPT TOTAL line before the date parser ever sees it,
    # so nothing should fail to parse; the 999999 units on that row never reach the panel
    assert report["files"][0]["rows_bad_date"] == 0
    assert float(panel["sold"].max()) < 1000


def test_an_unmapped_code_is_dropped_and_counted(ingested):
    _, report, _ = ingested
    assert report["files"][0]["unmapped_codes"] == {"777777": 1}


def test_a_duplicate_export_row_is_collapsed_and_counted(ingested):
    panel, report, _ = ingested
    assert report["duplicates_collapsed"] >= 1
    assert not panel.duplicated(subset=["store", "item", "date"]).any()


def test_a_refund_is_netted_then_clipped_and_the_row_is_marked(ingested):
    panel, report, _ = ingested
    assert report["negatives_clipped"] >= 0
    row = panel[(panel.item == "bread") & (panel.date == pd.Timestamp(REFUND_DAY))]
    assert len(row) == 1
    # netted against the day's sale, so it is smaller than the neighbouring days
    before = panel[(panel.item == "bread")
                   & (panel.date == pd.Timestamp(REFUND_DAY) - pd.Timedelta(days=1))]
    assert float(row.sold.iloc[0]) < float(before.sold.iloc[0])
    assert float(row.sold.iloc[0]) >= 0.0


def test_a_one_day_hole_is_a_genuine_zero_and_a_long_one_is_not(ingested):
    panel, _, _ = ingested
    hole = panel[panel.date == pd.Timestamp(MISSING_DAY)]
    assert len(hole) == 3
    assert (hole.sold == 0).all()
    assert set(hole.row_status) == {"ok"}          # the export writes no line for a zero day

    gap = panel[panel.date.isin([pd.Timestamp(d) for d in GAP])]
    assert len(gap) == 15
    assert set(gap.row_status) == {"missing"}      # five days is a systems outage
    assert (gap.is_closed == 1).all()


def test_dow_holiday_and_payday_are_re_derived_from_the_date(ingested):
    panel, _, _ = ingested
    assert (panel["dow"] == panel["date"].dt.dayofweek).all()
    assert (panel["payday"] == panel["date"].dt.day.isin([1, 2, 3, 15, 16, 17])
            .astype(int)).all()
    assert panel["holiday"].isna().sum() == 0
    assert "easter" in set(panel["holiday"])       # 2025-04-20, derived not tabulated


def test_weather_is_normalized_and_the_unresolvable_day_is_counted(ingested):
    panel, report, _ = ingested
    assert set(panel["weather"]) <= set(schema.WEATHER_KINDS)
    assert report["weather"]["unknown_conditions"] == {"BLOWING DUST": 1}


def test_item_name_dept_and_cost_come_from_the_config_not_the_export(ingested):
    panel, _, export = ingested
    bread = panel[panel.item == "bread"].iloc[0]
    assert bread["item_name"] == "Bread Loaf"      # not "BREAD WHITE 20OZ"
    assert bread["dept"] == "Bakery"
    assert float(bread["unit_cost"]) == pytest.approx(1.10)


def test_ingest_is_idempotent(export):
    first, _ = ingest.ingest(export["mapping"], export["items"], root=export["root"])
    second, _ = ingest.ingest(export["mapping"], export["items"], root=export["root"])
    assert schema.panel_hash(first) == schema.panel_hash(second)


def test_the_report_carries_the_documented_blocks(ingested):
    _, report, _ = ingested
    for key in ("files", "date_range", "n_days", "items_kept", "items_dropped",
                "duplicates_collapsed", "negatives_clipped", "grid_rows_inserted",
                "row_status_counts", "closures_applied", "weather",
                "holiday_collisions", "sellout", "panel_hash"):
        assert key in report, key
    json.dumps(report, default=str)               # the CLI writes it, so it must serialize


def test_a_closure_date_marks_the_day_and_zeroes_the_sellout(export):
    mapping = json.loads(json.dumps(MAPPING))
    mapping["closures"] = {"dates": ["2025-02-20"]}
    mapping = config.load_mapping(_dump(export, mapping), export["items"])
    panel, _ = ingest.ingest(mapping, export["items"], root=export["root"])
    closed = panel[panel.date == pd.Timestamp("2025-02-20")]
    assert set(closed.row_status) == {"closed"}
    assert (closed.is_closed == 1).all()
    assert (closed.stockout == 0).all()           # a closed day did not run out, it shut


def _dump(export, mapping):
    path = export["root"] + "/mapping2.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh)
    return path


def test_the_logger_json_backup_dedupes_by_entry_id(tmp_path, export):
    backup = {"app": "ht-stock", "v": 1,
              "items": [{"id": "i1", "name": "Bread Loaf"}],
              "logs": [
                  {"id": "e1", "itemId": "i1", "date": "2025-02-01", "qty": 4},
                  {"id": "e1", "itemId": "i1", "date": "2025-02-01", "qty": 4},
                  {"id": "e2", "itemId": "i1", "date": "2025-02-01", "qty": 2},
              ]}
    path = tmp_path / "backup.json"
    path.write_text(json.dumps(backup), encoding="utf-8")
    frame = ingest.ingest_logger_backup(str(path), export["items"])
    row = frame[frame.item == "bread"]
    assert float(row.wasted.sum()) == 6.0         # the repeat counts once


def test_the_logger_backup_refuses_a_foreign_file(tmp_path, export):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"app": "something-else", "logs": []}), encoding="utf-8")
    with pytest.raises(schema.IngestError):
        ingest.ingest_logger_backup(str(path), export["items"])


def test_a_mapped_header_the_file_does_not_have_names_itself(export, tmp_path):
    """The most likely mapping mistake there is, and it used to be a bare pandas KeyError."""
    mapping = json.loads(json.dumps(MAPPING))
    mapping["columns"]["sales"]["units"] = "QTY SOLD"
    mapping = config.load_mapping(_dump(export, mapping), export["items"])
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(mapping, export["items"], root=export["root"])
    message = str(exc.value)
    assert "QTY SOLD" in message and "MOVEMENT.CSV" in message
    assert "UNITS" in message                      # the file's real headers, so it is fixable


def test_a_header_present_in_only_some_files_is_caught_per_file(export, tmp_path):
    """Concatenation unions columns, so a column renamed in one year of a multi-year export
    turns that year into NaN and the whole pipeline stays green."""
    path = os.path.join(export["root"], "SECOND.CSV")
    with open(path, "w", encoding="cp1252", newline="") as fh:
        fh.write("A,B,C\r\nHEADER\r\n01/01/26,330901,5.0\r\n")
    mapping = json.loads(json.dumps(MAPPING))
    mapping["files"].append({"role": "sales", "path": "SECOND.CSV", "encoding": "cp1252",
                             "header_row": 1, "skip_footer_rows": 0})
    mapping = config.load_mapping(_dump(export, mapping), export["items"])
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(mapping, export["items"], root=export["root"])
    assert "SECOND.CSV" in str(exc.value)


def test_a_per_unit_cost_column_is_not_multiplied_by_the_days_line_count(tmp_path):
    """A per-unit COST at line grain summed as an extended total inflates every settlement
    dollar by the number of register lines, and nothing downstream would notice."""
    frame = pd.DataFrame({"store": ["s"] * 3, "item": ["bread"] * 3,
                          "date": pd.to_datetime(["2026-01-01"] * 3),
                          "sold": [4.0, 4.0, 4.0], "row_cost": [1.10 * 4] * 3})
    out = ingest.aggregate(frame, "sum")             # what _tidy hands aggregate, extended
    panel = out.assign(row_status="ok")
    items = {"bread": {"name": "B", "dept": "Bakery", "price": 3.99, "cost": 1.10,
                       "shelf_life_days": 1}}
    mapping = {"price_cost": {"tolerance_pct": 0.15}}
    priced = ingest._economics(panel, items, {**MAPPING, **mapping}, {"files": []})
    assert float(priced["unit_cost"].iloc[0]) == pytest.approx(1.10)


def test_a_blank_units_cell_stays_null_whether_or_not_the_item_has_a_hole():
    """Identical blank cells must not be an error in one panel and a fabricated zero in the
    next, decided by an unrelated property of the item."""
    def grid(hole):
        days = pd.date_range("2026-01-01", periods=10)
        if hole:
            days = days.delete(4)
        df = pd.DataFrame({"store": "s", "item": "bread", "date": days,
                           "sold": np.arange(len(days), dtype=float), "row_status": "ok"})
        df.loc[df.date == pd.Timestamp("2026-01-02"), "sold"] = np.nan
        out = ingest.reindex_grid(df, {}, {"gaps": {"max_unexplained_gap_days": 3}})
        return out.set_index("date")

    for hole in (False, True):
        out = grid(hole)
        assert pd.isna(out.loc[pd.Timestamp("2026-01-02"), "sold"]), hole
    assert float(grid(True).loc[pd.Timestamp("2026-01-05"), "sold"]) == 0.0   # inserted, zero


# ---- report furniture in the middle of the file ----

# two shapes of report furniture, both of which DATA_CONTRACT tells the store are fine:
# a subtotal with no date at all, and one the report dates and leaves the item number blank
SUBTOTAL_ROWS = ['BAKERY SUBTOTAL,,,999,"$9,999.00"',
                 '01/20/25,,BAKERY SUBTOTAL,999,"$9,999.00"']


def _insert_subtotal(root):
    path = os.path.join(root, "MOVEMENT.CSV")
    with open(path, "rb") as fh:
        lines = fh.read().decode("cp1252").split("\r\n")
    lines[20:20] = SUBTOTAL_ROWS
    with open(path, "wb") as fh:
        fh.write("\r\n".join(lines).encode("cp1252"))


def test_a_blank_item_number_never_reaches_a_report_as_a_float(export):
    """A float NaN key beside string codes breaks sorted() and json.dump(sort_keys=True).

    Both crash sites are on the store's own path: the --report write, and the repair INFO
    the validation page prints. The panel was fine; the page describing it was a traceback.
    """
    _insert_subtotal(export["root"])
    panel, report = ingest.ingest(export["mapping"], export["items"], root=export["root"])

    codes = report["files"][0]["unmapped_codes"]
    assert all(isinstance(k, str) for k in codes)
    assert ingest.BLANK_CODE in codes                  # named, not printed as "nan"
    json.dumps(report, sort_keys=True, default=str)    # this used to raise TypeError
    assert len(panel)


def test_the_ingest_command_writes_the_panel_before_anything_that_can_fail(export, tmp_path):
    out, rep = str(tmp_path / "p.csv"), str(tmp_path / "r.json")
    _insert_subtotal(export["root"])
    rc = ingest.main(["--mapping", export["mapping_path"], "--items", export["items_path"],
                      "--root", export["root"], "--out", out, "--report", rep, "--quiet"])
    assert rc in (0, 2)
    assert os.path.exists(out)                         # the documented recovery needs it
    assert json.load(open(rep))["files"][0]["rows_in"] > 0


def test_the_dropped_line_buckets_add_up_to_their_own_total(export):
    """Parts that exceed the whole, on the page a category manager reconciles against."""
    from ht import validate as ht_validate

    _insert_subtotal(export["root"])
    panel, report = ingest.ingest(export["mapping"], export["items"], root=export["root"])
    entry = report["files"][0]
    parts = (int(entry["rows_bad_date"]) + int(sum(entry["unmapped_codes"].values()))
             + int(sum(entry["excluded_codes"].values())) + int(entry["rows_other_store"]))
    assert parts == int(entry["rows_in"]) - int(entry["rows_kept"])

    result = ht_validate.validate(panel, export["items"], ingest_report=report)
    line = [f for f in result["findings"] if f.check == "repair_rows_dropped"][0]
    assert str(parts) in line.message


def test_store_without_a_store_column_would_relabel_rather_than_filter(export):
    """--store on a file with no store column stamps the number on every row.

    The panel is then internally consistent and attributed to a store whose sales it does
    not contain, and nothing downstream can detect it.
    """
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(export["mapping"], export["items"], root=export["root"], store="0456")
    assert "columns.<role>.store" in str(exc.value) and "0123" in str(exc.value)


# ---- the waste ladder is per cell, as the contract states it ----

def _add_partial_waste_file(root, mapping, days=6):
    """A shrink report that covers the first few days only -- a department report, or a
    six-month one, is the ordinary shape."""
    rows = ["WASTE DT,ITEM NBR,UNITS"]
    for i in range(days):
        day = START + dt.timedelta(days=i)
        rows.append(f"{day:%m/%d/%y},330901,3")
    with open(os.path.join(root, "SHRINK.CSV"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    mapping = json.loads(json.dumps(mapping, default=str))
    mapping["files"].append({"role": "waste", "path": "SHRINK.CSV"})
    mapping["columns"]["waste"] = {"date": "WASTE DT", "item_code": "ITEM NBR",
                                   "units": "UNITS"}
    return mapping


def test_a_partial_waste_report_does_not_switch_off_the_rest_of_the_panel(export, tmp_path):
    raw = json.loads(open(export["mapping_path"]).read())
    raw = _add_partial_waste_file(export["root"], raw)
    (tmp_path / "mapping2.json").write_text(json.dumps(raw), encoding="utf-8")
    mapping = config.load_mapping(str(tmp_path / "mapping2.json"), export["items"])

    panel, report = ingest.ingest(mapping, export["items"], root=export["root"])
    base, _ = ingest.ingest(export["mapping"], export["items"], root=export["root"])

    # the export's own six cells win where it has them, and produced - sold still fills the
    # rest: one department's shrink report used to leave the panel with six waste numbers
    assert int(panel["wasted"].notna().sum()) >= int(base["wasted"].notna().sum())
    day = panel[(panel.item == "bread") & (panel.date == pd.Timestamp(START))]
    assert float(day["wasted"].iloc[0]) == 3.0
    assert report["waste_cells"]["export"] == 6
    assert report["waste_cells"]["derived"] > 100
