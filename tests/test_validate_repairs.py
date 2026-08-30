"""Two things the validator has to say out loud: whose store this is, and what was repaired.

A chain runs its item-movement report for a district, so the first export a store hands over
plausibly carries several store numbers. Nothing below ht.validate keys on store, so a panel
like that used to pass here and fail days later as "not enough history" -- a misdiagnosis that
sends somebody back to ask for history they already gave.

And the INFO block is the whole readable account of what ingest changed. Half of those repairs
leave no trace in the panel: a collapsed duplicate is a row that is gone, a short hole filled
with a zero keeps row_status "ok". They are readable only from the ingest report, so this file
pins that they are read from it when it is there -- and that their absence is stated, rather
than implied to be zero, when it is not.
"""
import json

import pandas as pd

from ht import schema, validate


ITEMS = {
    "bread": {"name": "Bread", "dept": "Bakery", "price": 4.0, "cost": 1.0, "batch": 1.0,
              "salvage": 0.0, "continuous": False, "unit": "each", "shelf_life_days": 1,
              "sellout_tolerance": None, "cost_imputed": False, "active": True, "notes": ""},
}
ITEMS["cake"] = dict(ITEMS["bread"], name="Cake")


def _levels(report, level):
    return {f.check for f in report["findings"] if f.level == level}


def _finding(report, check):
    hits = [f for f in report["findings"] if f.check == check]
    assert len(hits) >= 1, f"no {check} finding in {[f.check for f in report['findings']]}"
    return hits[0]


def _district(make_panel, stores, days=200):
    """The same movement report run for a district: one block of rows per store number."""
    one = make_panel(["bread", "cake"], start="2025-01-01", days=days)
    frames = []
    for store in stores:
        part = one.copy()
        part["store"] = store
        frames.append(part)
    return schema.conform(pd.concat(frames, ignore_index=True))


# ---- (a) more than one store ----

def test_a_district_panel_is_an_error(make_panel):
    report = validate.validate(_district(make_panel, ["0123", "0456", "0789"]), ITEMS)
    assert "multi_store" in _levels(report, "error")
    assert report["ok"] is False


def test_the_error_names_the_stores_it_found(make_panel):
    """The store fixes this by re-running one report; it has to be told which numbers came."""
    finding = _finding(validate.validate(_district(make_panel, ["0123", "0456"]), ITEMS),
                       "multi_store")
    assert "0123" in finding.message and "0456" in finding.message
    assert finding.count == 2


def test_a_single_store_panel_says_nothing_about_stores(make_panel):
    panel = make_panel(["bread", "cake"], start="2025-01-01", days=200)
    assert "multi_store" not in {f.check for f in validate.validate(panel, ITEMS)["findings"]}


def test_a_short_district_panel_is_diagnosed_as_a_district_not_as_short_history(make_panel):
    """The defect this replaces: 3 stores x 60 days reported only as 'covers 60 days'.

    The history finding is still true and still fires. What matters is that the reason the
    panel looks short is now on the same page, because 'ask the store for more history' is
    the wrong instruction for an export that already carries three stores' worth.
    """
    report = validate.validate(_district(make_panel, ["0123", "0456", "0789"], days=60), ITEMS)
    errors = _levels(report, "error")
    assert "multi_store" in errors
    assert "insufficient_history" in errors


def test_the_header_counts_the_stores_it_read(make_panel):
    """The one line above the findings has to disagree with "this is my store's export"."""
    report = validate.validate(_district(make_panel, ["0123", "0456", "0789"]), ITEMS)
    assert report["counts"]["stores"] == 3
    assert "3 STORES" in validate.format_report(report).splitlines()[2]


def test_a_duplicate_key_check_cannot_stand_in_for_it(make_panel):
    """store is part of schema.KEY, so a district panel has no duplicate keys at all."""
    panel = _district(make_panel, ["0123", "0456"])
    assert not panel.duplicated(list(schema.KEY)).any()
    assert "duplicate_key" not in _levels(validate.validate(panel, ITEMS), "error")


# ---- (b) repairs ----

def _report(**over):
    """An ht.ingest report, in the shape ingest() actually writes."""
    out = dict(files=[dict(role="sales", path=["S.CSV"], rows_in=1000, rows_kept=1000,
                           rows_bad_date=0, unmapped_codes={}, excluded_codes={})],
               duplicates_collapsed=0, negatives_clipped=0, grid_rows_inserted={},
               closures_applied={}, weather=dict(provider="csv", filled_days=0,
                                                 missing_days=0))
    out.update(over)
    return out


def test_without_a_report_the_validator_says_what_it_cannot_see(make_panel):
    """Silence would read as "no repairs happened", which is the claim it cannot make."""
    panel = make_panel(["bread"], start="2025-01-01", days=200)
    finding = _finding(validate.validate(panel, ITEMS), "repair_report_absent")
    assert finding.level == "info"
    assert "--ingest-report" in finding.message


def test_with_a_report_it_stops_saying_so(make_panel):
    panel = make_panel(["bread"], start="2025-01-01", days=200)
    report = validate.validate(panel, ITEMS, ingest_report=_report())
    assert "repair_report_absent" not in _levels(report, "info")


def test_grid_filled_days_are_counted_including_the_ones_the_panel_hides(make_panel):
    """57 inserted, 32 of them flagged: the other 25 are the defect this check exists for."""
    panel = make_panel(["bread"], start="2025-01-01", days=200).copy()
    panel.loc[panel.index[:32], ["row_status", "is_closed", "sold"]] = ["missing", 1, 0.0]
    report = validate.validate(panel, ITEMS,
                               ingest_report=_report(grid_rows_inserted={"bread": 57}))
    finding = _finding(report, "repair_grid_filled")
    assert finding.count == 57
    assert "57" in finding.message and "32" in finding.message and "25" in finding.message


def test_a_grid_fill_with_nothing_left_over_does_not_invent_a_remainder(make_panel):
    panel = make_panel(["bread"], start="2025-01-01", days=200).copy()
    panel.loc[panel.index[:9], ["row_status", "is_closed", "sold"]] = ["missing", 1, 0.0]
    finding = _finding(validate.validate(panel, ITEMS,
                                         ingest_report=_report(grid_rows_inserted={"bread": 9})),
                       "repair_grid_filled")
    assert "the other" not in finding.message


def test_collapsed_duplicates_are_reported_though_no_row_carries_them(make_panel):
    panel = make_panel(["bread"], start="2025-01-01", days=200)
    finding = _finding(validate.validate(panel, ITEMS,
                                         ingest_report=_report(duplicates_collapsed=96)),
                       "repair_duplicates_collapsed")
    assert finding.level == "info" and finding.count == 96


def test_clipped_negatives_are_reported_with_the_rows_they_marked(make_panel):
    panel = make_panel(["bread"], start="2025-01-01", days=200).copy()
    panel.loc[panel.index[:4], ["row_status", "is_closed"]] = ["suspect", 1]
    finding = _finding(validate.validate(panel, ITEMS,
                                         ingest_report=_report(negatives_clipped=4)),
                       "repair_negatives_clipped")
    assert finding.count == 4
    assert "4 row(s) here carry row_status='suspect'" in finding.message


def test_lines_dropped_before_the_panel_existed_are_reported(make_panel):
    panel = make_panel(["bread"], start="2025-01-01", days=200)
    files = [dict(role="sales", path=["S.CSV"], rows_in=8806, rows_kept=8794, rows_bad_date=0,
                  unmapped_codes={"777777": 6}, excluded_codes={"999999": 6})]
    finding = _finding(validate.validate(panel, ITEMS, ingest_report=_report(files=files)),
                       "repair_rows_dropped")
    assert finding.count == 12
    assert "777777" in finding.message and "999999" in finding.message


def test_declared_closures_and_filled_weather_are_reported(make_panel):
    panel = make_panel(["bread"], start="2025-01-01", days=200)
    report = validate.validate(panel, ITEMS, ingest_report=_report(
        closures_applied={"closed": 25, "partial": 17},
        weather=dict(provider="csv", filled_days=3, missing_days=0)))
    assert _finding(report, "repair_closures_applied").count == 42
    assert _finding(report, "repair_weather_filled").count == 3


def test_no_repair_is_invented_when_the_report_counts_none(make_panel):
    panel = make_panel(["bread"], start="2025-01-01", days=200)
    infos = _levels(validate.validate(panel, ITEMS, ingest_report=_report()), "info")
    assert not {i for i in infos if i.startswith("repair_")} - {"snow_tomorrow_flat"}


def test_an_info_never_changes_the_verdict(make_panel):
    panel = make_panel(["bread"], start="2025-01-01", days=200)
    report = validate.validate(panel, ITEMS, ingest_report=_report(
        duplicates_collapsed=96, grid_rows_inserted={"bread": 57}))
    assert report["ok"] is True


def test_the_report_rides_along_on_the_panels_attrs(make_panel):
    """ht.ingest hangs its report on the panel, so its own CLI needs no extra wiring."""
    panel = make_panel(["bread"], start="2025-01-01", days=200)
    panel.attrs["ingest_report"] = _report(duplicates_collapsed=7)
    finding = _finding(validate.validate(panel, ITEMS), "repair_duplicates_collapsed")
    assert finding.count == 7


# ---- the CLI ----

def test_the_cli_reads_the_report_beside_the_panel(make_panel, items_path, tmp_path,
                                                  capsys):
    panel = make_panel(["bread"], start="2025-01-01", days=200)
    panel_path = tmp_path / "panel.csv"
    schema.write_panel(panel, str(panel_path))
    report_path = tmp_path / "ingest_report.json"
    report_path.write_text(json.dumps(_report(duplicates_collapsed=96)), encoding="utf-8")

    rc = validate.main(["--panel", str(panel_path), "--items", items_path,
                        "--ingest-report", str(report_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "repair_duplicates_collapsed" in out
    assert "repair_report_absent" not in out


def test_the_cli_refuses_an_unreadable_report(make_panel, items_path, tmp_path, capsys):
    panel = make_panel(["bread"], start="2025-01-01", days=200)
    panel_path = tmp_path / "panel.csv"
    schema.write_panel(panel, str(panel_path))
    rc = validate.main(["--panel", str(panel_path), "--items", items_path,
                        "--ingest-report", str(tmp_path / "nope.json")])
    assert rc == 1
    assert "nope.json" in capsys.readouterr().err
