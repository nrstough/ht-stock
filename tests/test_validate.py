"""The validator is the document you hand back when a store's export needs fixing.

Its job is the classification, not the count: a problem that blocks training is an ERROR, a
problem that changes how a number must be read is a WARNING, and a repair ingest already
made is an INFO. Getting that wrong in either direction is expensive -- an error that
should be a warning stops a pilot on its first morning, and a warning that should be an
error trains a model on a corrupt panel and says nothing.
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from ht import config, schema, validate


def _levels(report, level):
    return {f.check for f in report["findings"] if f.level == level}


def _clean(make_panel):
    return make_panel(["bread", "cake"], start="2025-01-01", days=200)


ITEMS = {
    "bread": {"name": "Bread", "dept": "Bakery", "price": 4.0, "cost": 1.0, "batch": 1.0,
              "salvage": 0.0, "continuous": False, "unit": "each", "shelf_life_days": 1,
              "sellout_tolerance": None, "cost_imputed": False, "active": True, "notes": ""},
}
ITEMS["cake"] = dict(ITEMS["bread"], name="Cake")


def test_a_clean_panel_passes(make_panel):
    report = validate.validate(_clean(make_panel), ITEMS)
    assert report["ok"] is True
    assert _levels(report, "error") == set()


def test_the_report_carries_every_documented_block(make_panel):
    report = validate.validate(_clean(make_panel), ITEMS)
    for key in ("ok", "findings", "counts", "coverage", "date_range", "item_census",
                "gap_census", "sellout", "splits_preview", "excluded_items_preview"):
        assert key in report, key


def test_simulator_truth_is_refused_before_any_check_runs(make_panel):
    panel = _clean(make_panel).copy()
    panel["true_demand"] = 1.0
    with pytest.raises(schema.SchemaError):
        validate.validate(panel, ITEMS)


def test_a_duplicate_item_day_is_an_error(make_panel):
    panel = _clean(make_panel)
    panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    assert "duplicate_key" in _levels(validate.validate(panel, ITEMS), "error")


def test_an_item_missing_from_the_config_is_an_error(make_panel):
    panel = _clean(make_panel)
    assert "item_not_in_config" in _levels(
        validate.validate(panel, {"bread": ITEMS["bread"]}), "error")


def test_a_hole_in_an_items_date_index_is_an_error(make_panel):
    panel = _clean(make_panel)
    drop = panel[(panel.item == "bread")
                 & panel.date.between("2025-03-01", "2025-03-09")].index
    assert "date_gap" in _levels(validate.validate(panel.drop(drop), ITEMS), "error")


def test_negative_sales_are_an_error(make_panel):
    panel = _clean(make_panel)
    panel.loc[0, "sold"] = -3.0
    assert "sold_negative" in _levels(validate.validate(panel, ITEMS), "error")


def test_a_closed_day_that_sold_something_is_an_error(make_panel):
    panel = _clean(make_panel)
    panel.loc[0, ["is_closed", "row_status"]] = [1, "closed"]
    assert "closed_with_sales" in _levels(validate.validate(panel, ITEMS), "error")


def test_a_few_days_above_the_production_count_warn_but_do_not_block(make_panel):
    """A label-printer log is a proxy and a proxy undercounts.

    A second bake nobody printed for is a fact about the store's paperwork, not a reason to
    refuse to train on two years of movement; the share is what separates the two.
    """
    panel = _clean(make_panel)
    panel.loc[0, "produced"] = float(panel.loc[0, "sold"]) - 40.0
    report = validate.validate(panel, ITEMS)
    assert "sold_above_produced" in _levels(report, "warning")
    assert "sold_above_produced" not in _levels(report, "error")


def test_selling_more_than_was_produced_all_week_is_an_error(make_panel):
    panel = _clean(make_panel)
    bread = panel.index[panel.item == "bread"][:40]
    panel.loc[bread, "produced"] = (panel.loc[bread, "sold"].astype(float) - 40.0) \
        .astype("float32")
    assert "sold_above_produced" in _levels(validate.validate(panel, ITEMS), "error")


def test_the_overrun_threshold_is_a_mapping_field(make_panel):
    panel = _clean(make_panel)
    panel.loc[0, "produced"] = float(panel.loc[0, "sold"]) - 40.0
    mapping = copy.deepcopy(config.MAPPING_DEFAULTS)
    mapping["sellout"]["rule"] = "produced_vs_sold"
    mapping["production"]["overrun_policy"] = "error"
    assert "sold_above_produced" in _levels(
        validate.validate(panel, ITEMS, mapping=mapping), "error")


def test_a_panel_too_short_to_split_is_an_error(make_panel):
    panel = make_panel(["bread"], start="2026-06-01", days=90)
    report = validate.validate(panel, {"bread": ITEMS["bread"]})
    assert "insufficient_history" in _levels(report, "error")
    message = " ".join(f.message for f in report["findings"] if f.level == "error")
    assert "126" in message and "90" in message      # the numbers, not just a complaint


def test_no_sellout_signal_is_a_warning_not_an_error(make_panel):
    panel = _clean(make_panel).copy()
    panel["stockout"] = 0
    panel["stockout_known"] = 0
    panel["sellout_source"] = "none"
    report = validate.validate(panel, ITEMS)
    assert report["ok"] is True                      # a first-class mode, training proceeds
    warnings = _levels(report, "warning")
    assert "no_sellout_signal" in warnings
    assert "sellout_coverage" in warnings


def test_a_short_history_item_is_a_warning_and_a_named_exclusion(make_panel):
    panel = _clean(make_panel)
    keep = ~((panel.item == "cake") & (panel.date < "2025-06-01"))
    report = validate.validate(panel[keep], ITEMS)
    assert "short_history" in _levels(report, "warning")
    assert [e["item"] for e in report["excluded_items_preview"]] == ["cake"]
    assert report["excluded_items_preview"][0]["required"] == validate.MIN_ITEM_TRAIN_DAYS


def _loaded_mapping(tmp_path):
    """validate() takes a mapping that has been through load_mapping, defaults and all."""
    doc = {"schema": "ht-source-mapping/1", "store": "0123",
           "files": [{"role": "sales", "path": "S.CSV"}],
           "columns": {"sales": {"date": "D", "item_code": "I", "units": "U"}},
           "date": {"format": "%m/%d/%y"},
           "sellout": {"rule": "produced_vs_sold"},
           "price_cost": {"authority": "config", "tolerance_pct": 0.15}}
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return config.load_mapping(str(path))


def test_a_price_far_from_the_config_is_a_promotion_detector(make_panel, tmp_path):
    panel = _clean(make_panel).copy()
    panel["unit_price"] = 1.0                        # config price is 4.00
    report = validate.validate(panel, ITEMS, _loaded_mapping(tmp_path))
    assert "price_divergence" in _levels(report, "warning")


def test_missing_production_is_a_warning(make_panel):
    panel = _clean(make_panel).copy()
    panel["produced"] = np.nan
    panel["wasted"] = np.nan
    assert "produced_missing" in _levels(validate.validate(panel, ITEMS), "warning")


def test_a_holiday_outside_the_frozen_vocabulary_warns(make_panel):
    panel = _clean(make_panel).copy()
    panel.loc[panel.index[:2], "holiday"] = "store_anniversary"
    assert "holiday_vocabulary" in _levels(validate.validate(panel, ITEMS), "warning")


def test_unknown_weather_warns_rather_than_being_guessed(make_panel):
    panel = _clean(make_panel).copy()
    panel.loc[panel.index[:5], "weather"] = "unknown"
    assert "weather_unknown" in _levels(validate.validate(panel, ITEMS), "warning")


def test_a_repair_ingest_made_is_recorded_as_info(make_panel):
    panel = _clean(make_panel).copy()
    panel.loc[panel.index[:3], ["row_status", "is_closed", "sold"]] = ["missing", 1, 0.0]
    report = validate.validate(panel, ITEMS)
    assert "repair_missing" in _levels(report, "info")
    assert report["ok"] is True                      # an explained gap does not block training


def test_strict_raises_only_on_an_error(make_panel):
    panel = _clean(make_panel)
    validate.validate(panel, ITEMS, strict=True)
    panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(schema.ValidationFailed):
        validate.validate(panel, ITEMS, strict=True)


def test_the_item_census_is_the_table_you_hand_the_store(make_panel):
    census = validate.item_census(_clean(make_panel))
    assert set(census["item"]) == {"bread", "cake"}
    for column in ("rows", "first", "last", "open_days", "open_train_days", "missing_days",
                   "zero_days", "sellout_rate", "longest_gap", "status", "reason"):
        assert column in census.columns, column


def test_the_gap_census_names_every_run_of_non_ok_rows(make_panel):
    panel = _clean(make_panel).copy()
    mask = (panel.item == "bread") & panel.date.between("2025-02-01", "2025-02-04")
    panel.loc[mask, ["row_status", "is_closed", "sold"]] = ["missing", 1, 0.0]
    gaps = validate.gap_census(panel)
    run = gaps[(gaps["item"] == "bread") & (gaps["row_status"] == "missing")]
    assert len(run) == 1
    assert int(run.iloc[0]["days"]) == 4


def test_format_report_prints_the_verdict_and_the_sections(make_panel):
    panel = _clean(make_panel)
    panel.loc[0, "sold"] = -3.0
    text = validate.format_report(validate.validate(panel, ITEMS))
    assert "FAIL" in text
    assert "ERRORS" in text and "sold_negative" in text
    assert "ITEM CENSUS" in text


# ---- conditions that look alike from a distance but have different remedies ----

def test_a_few_blank_store_cells_are_not_diagnosed_as_a_district_export(make_panel):
    """Both remedies for multi_store -- re-run for one store, re-ingest with --store -- are
    useless against an empty cell, and the message's arithmetic ("the context window reaches
    back 14 days") is false for a panel that is 8,793 rows of one store and 5 of nothing."""
    panel = _clean(make_panel).copy()
    panel.loc[panel.index[:5], "store"] = "default"
    report = validate.validate(panel, ITEMS)
    assert "store_blank" in _levels(report, "error")
    assert "multi_store" not in _levels(report, "error")
    finding = [f for f in report["findings"] if f.check == "store_blank"][0]
    assert finding.count == 5 and "5 row(s)" in finding.message


def test_a_real_district_panel_is_still_a_district_panel(make_panel):
    one = _clean(make_panel).copy()
    two = one.copy()
    two["store"] = "0456"
    report = validate.validate(pd.concat([one, two], ignore_index=True), ITEMS)
    assert "multi_store" in _levels(report, "error")


def test_negative_waste_is_an_error_from_whatever_path_wrote_it(make_panel):
    panel = _clean(make_panel).copy()
    panel.loc[panel.index[:2], "wasted"] = -4.0
    report = validate.validate(panel, ITEMS)
    assert "wasted_negative" in _levels(report, "error")


def test_a_day_fresh_row_with_production_and_no_waste_is_named(make_panel):
    panel = _clean(make_panel).copy()
    panel["produced"] = panel["sold"] + 3.0
    panel["wasted"] = np.nan
    report = validate.validate(panel, ITEMS)
    assert "waste_not_derived" in _levels(report, "warning")


# ---- the floors move with the split mode the caller asked for ----

def test_allow_short_relaxes_the_per_item_floor_the_trainer_will_use(make_panel):
    """The runbook's floors table offers --allow-short at 70 panel days / 56 item days, and
    model.train's docstring promises the validator checks the floor this run will use. With
    the 84-day floor hard-coded, `model.train --allow-short` refused every panel the trainer
    itself would have accepted, and the documented escape hatch never fired."""
    panel = make_panel(["bread"], start="2025-01-01", days=90)
    strict = validate.validate(panel, ITEMS)
    short = validate.validate(panel, ITEMS, split_opts=dict(allow_short=True))

    assert "insufficient_history" in _levels(strict, "error")     # 90 days, 126-day floor
    assert short["ok"] is True                                    # the documented short mode
    assert _levels(short, "error") == set()
    assert short["excluded_items_preview"] == []
    # the per-item floor moved with it: 62 open train days clears 56 and not 84
    train_end = pd.Timestamp(short["splits_preview"]["train_end"])
    assert validate.item_census(panel, train_end=train_end,
                                min_item_days=validate.MIN_TRAIN_DAYS_SHORT)["status"].eq(
        "ok").all()
    assert validate.item_census(panel, train_end=train_end)["status"].eq("short").all()
