"""The three inputs ingest used to accept and then quietly ignore.

Each one fails the same way: the run is green, the panel validates, and a number in the pitch
is wrong. A district movement report was summed onto one store's series under one store's
name, so the panel that came out carried a single store number and ht.validate's multi_store
check could not see it. The Phase-1 logger backup -- the only record of this store's waste
before the pilot, and the "before" half of the before/after README's Phase 4 promises -- was
read by nothing. And an imputed cost was a hand-typed number wearing a flag: a 2.10 and a
derived 3.36 look identical in the items file, and the difference is the critical fractile.

So these tests check the numbers, not the plumbing: what the panel's waste column holds after
a backup is folded in, whose units survive a collision with the store's own markout report,
and which store's sales the panel is made of.
"""
import datetime as dt
import json

import pandas as pd
import pytest

from ht import config, ingest, schema

START = dt.date(2025, 3, 1)
DAYS = 40
STORES = ("0123", "0456")
MARKOUT_DAYS = 10                       # the export's own waste report covers the first 10 days

ITEMS = {
    "schema": "ht-items/1",
    "items": {
        "bread": {"name": "Bread Loaf", "dept": "Bakery", "price": 3.99, "cost": 1.10,
                  "batch": 1},
        "rotisserie": {"name": "Rotisserie Chicken", "dept": "Hot Foods", "price": 7.99,
                       "cost": 3.20, "batch": 4},
    },
}

MAPPING = {
    "schema": "ht-source-mapping/1",
    "store": "0123",
    "files": [
        {"role": "sales", "path": "MOVEMENT.CSV"},
        {"role": "production", "path": "SCALE_LABELS.CSV"},
    ],
    "columns": {
        # the production sheet carries no store number, which is the normal case: it comes off
        # a scale in one department of one store
        "sales": {"date": "BUS DT", "store": "STORE", "item_code": "ITEM NBR", "units": "UNITS",
                  "dollars": "NET SALES $"},
        "production": {"date": "PRINT DT", "item_code": "ITEM NBR", "units": "LABELS PRINTED"},
        "waste": {"date": "MARKOUT DT", "store": "STORE", "item_code": "ITEM NBR",
                  "units": "UNITS"},
    },
    "date": {"format": "%m/%d/%y", "business_day": "store_close"},
    "items": {"drop_unmapped": True, "map": {"330901": "bread", "884213": "rotisserie"},
              "dept_gross_margin": {"Bakery": 0.62, "Hot Foods": 0.58}},
    "sellout": {"rule": "none"},
    "weather": {"provider": "none"},
    "calendar": {"payday_days": [1, 2, 3, 15, 16, 17]},
    "price_cost": {"authority": "config", "tolerance_pct": 0.15},
}

CODES = {"bread": "330901", "rotisserie": "884213"}


def _units(item, day, store):
    """Deterministic, and different per store, so a summed panel is visible in one number."""
    base = {"bread": 40.0, "rotisserie": 30.0}[item] + (7.0 if store == "0456" else 0.0)
    return round(base + (day.day % 5), 1)


def _write_export(root, stores=STORES, waste=False):
    sales = ["STORE,BUS DT,ITEM NBR,UNITS,NET SALES $"]
    prod = ["PRINT DT,ITEM NBR,LABELS PRINTED"]
    mark = ["STORE,MARKOUT DT,ITEM NBR,UNITS"]
    for i in range(DAYS):
        day = START + dt.timedelta(days=i)
        for store in stores:
            for item, code in CODES.items():
                units = _units(item, day, store)
                sales.append(f"{store},{day:%m/%d/%y},{code},{units:.1f},${units * 4.5:.2f}")
                if store == stores[-1]:
                    prod.append(f"{day:%m/%d/%y},{code},{units + 4:.1f}")
                if waste and i < MARKOUT_DAYS:
                    mark.append(f"{store},{day:%m/%d/%y},{code},2.0")
    (root / "MOVEMENT.CSV").write_text("\n".join(sales) + "\n", encoding="utf-8")
    (root / "SCALE_LABELS.CSV").write_text("\n".join(prod) + "\n", encoding="utf-8")
    if waste:
        (root / "MARKOUT.CSV").write_text("\n".join(mark) + "\n", encoding="utf-8")


def _backup(logs, items=None):
    """Exactly the shape index.html's buildBackup() writes."""
    return {"app": "ht-stock", "v": 1,
            "items": items if items is not None else
            [{"id": "bread", "name": "Bread Loaf", "dept": "Bakery", "price": 3.99},
             {"id": "rotis-new", "name": "Rotisserie Chicken", "dept": "Hot Foods",
              "price": 7.99}],
            "logs": logs, "settings": {"storeName": "Store 0456", "stores": 20}}


def _entry(n, day, item_id, name, qty):
    return {"id": f"e{n:04d}", "date": day, "time": "20:15", "itemId": item_id,
            "itemName": name, "dept": "Bakery", "qty": qty, "price": 3.99,
            "reason": "End of day", "note": ""}


@pytest.fixture
def export(tmp_path):
    def build(stores=STORES, waste=False, items=ITEMS, mapping=MAPPING):
        _write_export(tmp_path, stores, waste)
        mapping = json.loads(json.dumps(mapping))
        if waste:
            mapping["files"].append({"role": "waste", "path": "MARKOUT.CSV"})
        (tmp_path / "items.json").write_text(json.dumps(items), encoding="utf-8")
        (tmp_path / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
        loaded = config.load_items(str(tmp_path / "items.json"))
        return dict(root=str(tmp_path),
                    items=loaded,
                    mapping=config.load_mapping(str(tmp_path / "mapping.json"), loaded))
    return build


@pytest.fixture
def write_backup(tmp_path):
    def write(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return str(path)
    return write


def _waste(panel, item, day):
    row = panel[(panel.item == item) & (panel.date == pd.Timestamp(day))]
    return float(row.wasted.iloc[0])


# ---- the Phase-1 logger ----

def test_a_logger_backup_reaches_the_panels_waste_column(export, write_backup):
    """The whole point: nothing called ingest_logger_backup, so Phase 1 measured nothing."""
    e = export(stores=("0456",))
    path = write_backup("backup.json", _backup([
        _entry(1, "2025-03-04", "bread", "Bread Loaf", 3),
        _entry(2, "2025-03-04", "bread", "Bread Loaf", 1),      # two markouts, one day
        _entry(3, "2025-03-06", "rotis-new", "Rotisserie Chicken", 2),
    ]))
    panel, report = ingest.ingest(e["mapping"], e["items"], root=e["root"],
                                  logger_backups=[path])
    assert _waste(panel, "bread", "2025-03-04") == 4.0
    assert _waste(panel, "rotisserie", "2025-03-06") == 2.0
    assert report["logger"]["item_days_written"] == 2
    assert report["logger"]["units_written"] == 6.0


def test_a_logged_item_that_maps_to_nothing_is_reported_by_name(export, write_backup):
    """A store logs "Mac Salad" and the items config has never heard of it. Losing those rows
    silently is losing part of the baseline the pilot is measured against."""
    e = export(stores=("0456",))
    path = write_backup("backup.json", _backup([
        _entry(1, "2025-03-04", "bread", "Bread Loaf", 3),
        _entry(2, "2025-03-05", "cust-1", "Mac Salad", 6),
        _entry(3, "2025-03-06", "cust-1", "Mac Salad", 4),
    ]))
    _, report = ingest.ingest(e["mapping"], e["items"], root=e["root"], logger_backups=[path])
    assert report["logger"]["unmapped_items"] == {"Mac Salad": 2}
    assert report["logger"]["unmapped_entries"] == 2
    assert "Mac Salad" in ingest._summary(report)      # loudly, not only in the JSON


def test_the_logger_never_adds_to_the_exports_own_waste_record(export, write_backup):
    """Two independent records of one markout, summed, doubles the headline number."""
    e = export(stores=("0456",), waste=True)
    path = write_backup("backup.json", _backup([
        _entry(1, "2025-03-02", "bread", "Bread Loaf", 9),      # inside the markout report
        _entry(2, "2025-03-20", "bread", "Bread Loaf", 5),      # after it stops
    ]))
    panel, report = ingest.ingest(e["mapping"], e["items"], root=e["root"],
                                  logger_backups=[path])
    assert _waste(panel, "bread", "2025-03-02") == 2.0          # the export's, not 9 or 11
    assert _waste(panel, "bread", "2025-03-20") == 5.0
    logger = report["logger"]
    assert logger["conflicts_with_export"] == 1
    assert logger["conflict_units"] == {"logged": 9.0, "export": 2.0}


def test_the_logger_outranks_produced_minus_sold(export, write_backup):
    """A measured markout beats arithmetic on a label-printer count, and only on its own days."""
    e = export(stores=("0456",))
    path = write_backup("backup.json",
                        _backup([_entry(1, "2025-03-04", "bread", "Bread Loaf", 9)]))
    plain, _ = ingest.ingest(e["mapping"], e["items"], root=e["root"])
    panel, _ = ingest.ingest(e["mapping"], e["items"], root=e["root"], logger_backups=[path])
    assert _waste(plain, "bread", "2025-03-04") == 4.0          # produced - sold, the proxy
    assert _waste(panel, "bread", "2025-03-04") == 9.0          # what somebody actually counted
    assert _waste(panel, "bread", "2025-03-05") == _waste(plain, "bread", "2025-03-05")


def test_two_phones_dedupe_against_each_other(export, write_backup):
    """The likeliest operator error is handing over the same phone's backup twice."""
    e = export(stores=("0456",))
    logs = [_entry(1, "2025-03-04", "bread", "Bread Loaf", 3)]
    first = write_backup("one.json", _backup(logs))
    again = write_backup("one-copy.json", _backup(logs))
    other = write_backup("two.json",
                         _backup([_entry(9, "2025-03-04", "bread", "Bread Loaf", 2)]))
    panel, report = ingest.ingest(e["mapping"], e["items"], root=e["root"],
                                  logger_backups=[first, again, other])
    assert _waste(panel, "bread", "2025-03-04") == 5.0          # 3 + 2, not 3 + 3 + 2
    assert report["logger"]["duplicate_entries"] == 1


def test_a_logger_item_id_matches_the_items_config_key(export, write_backup):
    """index.html ships the items config's own keys as its default item ids, and that id is
    the only thing that survives somebody renaming an item mid-pilot."""
    e = export(stores=("0456",))
    path = write_backup("backup.json", _backup(
        [_entry(1, "2025-03-04", "bread", "Renamed On The Phone", 3)],
        items=[{"id": "bread", "name": "Renamed On The Phone"}]))
    panel, _ = ingest.ingest(e["mapping"], e["items"], root=e["root"], logger_backups=[path])
    assert _waste(panel, "bread", "2025-03-04") == 3.0


def test_a_backup_that_matches_nothing_is_refused(export, write_backup):
    """Folding in an empty baseline would leave Phase 4's before/after measuring nothing."""
    e = export(stores=("0456",))
    path = write_backup("backup.json", _backup(
        [_entry(1, "2025-03-04", "x-1", "Mac Salad", 3)], items=[{"id": "x-1",
                                                                 "name": "Mac Salad"}]))
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(e["mapping"], e["items"], root=e["root"], logger_backups=[path])
    assert "Mac Salad" in str(exc.value)


def test_logged_days_outside_the_export_are_counted_not_invented(export, write_backup):
    """The logger usually starts before the export the store finally sends. Those entries
    cannot be folded in without inventing a panel row that has no sales, so they are named."""
    e = export(stores=("0456",))
    path = write_backup("backup.json", _backup([
        _entry(1, "2025-03-04", "bread", "Bread Loaf", 3),
        _entry(2, "2024-11-30", "bread", "Bread Loaf", 8),
    ]))
    panel, report = ingest.ingest(e["mapping"], e["items"], root=e["root"],
                                  logger_backups=[path])
    assert report["logger"]["item_days_outside_the_export"] == 1
    assert report["logger"]["units_outside_the_export"] == 8.0
    assert panel.date.min() == pd.Timestamp(START)


def test_a_missing_backup_file_says_so_instead_of_a_traceback(export):
    e = export(stores=("0456",))
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(e["mapping"], e["items"], root=e["root"],
                      logger_backups=[e["root"] + "/nope.json"])
    assert "no such file" in str(exc.value)


def test_an_entry_with_an_unreadable_date_is_dropped_and_counted(export, write_backup):
    e = export(stores=("0456",))
    path = write_backup("backup.json", _backup([
        _entry(1, "2025-03-04", "bread", "Bread Loaf", 3),
        _entry(2, "sometime last week", "bread", "Bread Loaf", 3),
    ]))
    _, report = ingest.ingest(e["mapping"], e["items"], root=e["root"], logger_backups=[path])
    assert report["logger"]["unparsed_dates"] == 1
    assert report["logger"]["units_written"] == 3.0


# ---- one store out of a district export ----

def test_a_district_export_is_refused_and_names_the_flag(export):
    """It used to be summed onto one store's series under mapping.store's name, which is
    invisible to ht.validate's multi_store check: the panel it sees carries one store."""
    e = export()
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(e["mapping"], e["items"], root=e["root"])
    assert "--store" in str(exc.value)
    assert "'0123'" in str(exc.value) and "'0456'" in str(exc.value)


def test_the_store_filter_keeps_one_stores_numbers(export):
    e = export()
    panel, report = ingest.ingest(e["mapping"], e["items"], root=e["root"], store="0456")
    day = pd.Timestamp("2025-03-04")
    assert set(panel.store) == {"0456"}
    assert float(panel[(panel.item == "bread") & (panel.date == day)].sold.iloc[0]) == \
        _units("bread", day.date(), "0456")
    assert report["stores"]["rows_dropped"] == DAYS * len(CODES)


def test_a_store_number_that_is_not_in_the_file_lists_the_ones_that_are(export):
    """The leading zero on a store number is the single likeliest way to get this wrong."""
    e = export()
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(e["mapping"], e["items"], root=e["root"], store="456")
    assert "'0123'" in str(exc.value) and "'0456'" in str(exc.value)


def test_a_file_with_no_store_column_joins_to_the_selected_store(export):
    """The scale log has no store number on it. Stamping mapping.store there instead would
    key it to a store the filtered sales rows no longer use, and produced would be all NaN."""
    e = export()
    panel, _ = ingest.ingest(e["mapping"], e["items"], root=e["root"], store="0456")
    assert panel["produced"].notna().all()


def test_an_export_with_no_store_column_is_unchanged(export):
    """Nothing about a single-store pilot changes: mapping.store still stamps every row."""
    mapping = json.loads(json.dumps(MAPPING))
    mapping["columns"]["sales"].pop("store")
    e = export(stores=("0456",), mapping=mapping)
    panel, report = ingest.ingest(e["mapping"], e["items"], root=e["root"])
    assert set(panel.store) == {"0123"}                # mapping.store, as before
    assert report["stores"]["rows_dropped"] == 0


# ---- an imputed cost ----

def _imputed(cost, margins=None):
    items = json.loads(json.dumps(ITEMS))
    items["items"]["rotisserie"]["cost"] = cost
    items["items"]["rotisserie"]["cost_imputed"] = True
    mapping = json.loads(json.dumps(MAPPING))
    if margins is not None:
        mapping["items"]["dept_gross_margin"] = margins
    return items, mapping


def test_an_imputed_cost_records_the_margin_it_came_from(export):
    """3.36 and a typed-in 2.10 are the same number in the items file. The report has to say
    which one this is, because every dollar quoted for the item is a function of it."""
    items, mapping = _imputed(round(7.99 * (1 - 0.58), 2))
    e = export(stores=("0456",), items=items, mapping=mapping)
    _, report = ingest.ingest(e["mapping"], e["items"], root=e["root"])
    assert report["cost_imputed_items"] == ["rotisserie"]
    assert report["cost_imputation"]["rotisserie"] == dict(
        dept="Hot Foods", margin=0.58, price=7.99, cost=3.36, derived_cost=3.36)
    assert "COST IMPUTED rotisserie" in ingest._summary(report)


def test_an_imputed_cost_no_margin_reproduces_is_refused(export):
    items, mapping = _imputed(2.10)
    e = export(stores=("0456",), items=items, mapping=mapping)
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(e["mapping"], e["items"], root=e["root"])
    assert "3.36" in str(exc.value) and "cost_imputed" in str(exc.value)


def test_an_imputed_cost_with_no_margin_for_its_department_is_refused(export):
    items, mapping = _imputed(3.36, margins={"Bakery": 0.62})
    e = export(stores=("0456",), items=items, mapping=mapping)
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(e["mapping"], e["items"], root=e["root"])
    assert "dept_gross_margin" in str(exc.value) and "Hot Foods" in str(exc.value)


def test_a_margin_written_as_a_percentage_is_refused(export):
    """0.58 and 58 are both plausible keystrokes and only one of them is a margin."""
    items, mapping = _imputed(3.36, margins={"Bakery": 62, "Hot Foods": 58})
    e = export(stores=("0456",), items=items, mapping=mapping)
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(e["mapping"], e["items"], root=e["root"])
    assert "0.62" in str(exc.value)


def test_an_item_with_a_real_cost_is_left_alone(export):
    """dept_gross_margin is in the mapping either way; it only speaks for flagged items."""
    e = export(stores=("0456",))
    _, report = ingest.ingest(e["mapping"], e["items"], root=e["root"])
    assert report["cost_imputation"] == {}
    assert report["cost_imputed_items"] == []


def test_a_partial_markout_report_still_leaves_the_rest_derived(export):
    """The export's own waste report covers the first 10 days of 40. The other 30 are
    `produced - sold` per the contract's ladder -- one non-null cell used to switch the
    derivation off for the entire panel, and validation said PASS with one digit changed."""
    e = export(stores=("0456",), waste=True)
    with_report, report = ingest.ingest(e["mapping"], e["items"], root=e["root"])
    assert report["waste_cells"]["export"] == MARKOUT_DAYS * 2
    assert report["waste_cells"]["derived"] > 0
    assert int(with_report["wasted"].notna().sum()) == len(with_report)
    # the store's own number is untouched where it has one
    assert _waste(with_report, "bread", "2025-03-02") == 2.0


def test_a_negative_logged_quantity_is_refused_rather_than_folded_in(export, write_backup):
    """A markout is a count of what was thrown away; a negative subtracts from the baseline
    the whole Phase-1 before/after is measured against, and nothing downstream reads it."""
    e = export(stores=("0456",))
    path = write_backup("backup.json",
                        _backup([_entry(1, "2025-03-04", "bread", "Bread Loaf", -4)]))
    with pytest.raises(schema.IngestError) as exc:
        ingest.ingest(e["mapping"], e["items"], root=e["root"], logger_backups=[path])
    assert "bread" in str(exc.value) and "-4" in str(exc.value)
