"""The daily loop, and the reasons its record is worth anything.

Four shadow weeks produce one piece of paper that has to convince a district manager, and
it is worth nothing unless every number traces to a forecast that provably existed before
the day did. So: the prediction log is append-only, a day's score is written once and any
later change is disclosed rather than applied, a backfilled forecast is stamped as such,
and the forecaster refuses outright to see a date it is forecasting. Those four properties
are the difference between a proof and a story, and they are what these tests check.
"""
import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest

from ht import config, schema
from model import features, shadow

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "model", "artifacts")


@pytest.fixture(scope="module")
def panel():
    from tests.conftest import SYNTH_CSV
    return schema.conform(pd.read_csv(SYNTH_CSV, parse_dates=["date"]))


@pytest.fixture(scope="module")
def catalog():
    from tests.conftest import ITEMS_JSON
    return config.load_items(ITEMS_JSON)


@pytest.fixture(scope="module")
def recs(panel, catalog):
    """A forecast for the day after the panel ends -- the 5:30am case."""
    return shadow.forecast(panel, ARTIFACTS, catalog, "2026-01-01")


def test_every_active_item_gets_a_row(recs, catalog):
    assert set(recs["item"]) == set(catalog)
    assert (recs["rec_qty"] >= 0).all()
    assert set(recs["source"]) <= {"model", "par_fallback"}


def test_the_recommendation_is_batch_rounded(recs, catalog):
    for _, row in recs.iterrows():
        batch = catalog[row["item"]]["batch"]
        assert float(row["rec_qty"]) % batch == pytest.approx(0.0, abs=1e-9)


def test_the_sheet_shows_the_stores_own_par_beside_the_recommendation(recs):
    # a manager will not act on a number they cannot compare to their own habit
    assert "par_qty" in recs.columns
    assert np.isfinite(recs["par_qty"]).all()


def test_the_why_string_is_checkable_against_a_calendar(recs):
    for text in recs["why_text"]:
        assert text                                  # never empty: a blank WHY reads as none
        assert len(text.split(";")) <= 2             # at most two clauses


def test_forecast_refuses_to_see_the_day_it_is_forecasting(panel, catalog):
    with pytest.raises(ValueError) as exc:
        shadow.forecast(panel, ARTIFACTS, catalog, "2025-12-31")   # the panel's last day
    assert "2025-12-31" in str(exc.value) or "panel" in str(exc.value)


def test_forecast_refuses_a_stale_panel(panel, catalog):
    with pytest.raises(ValueError) as exc:
        shadow.forecast(panel, ARTIFACTS, catalog, "2026-02-01")
    assert "2025-12-31" in str(exc.value)            # names the panel's actual last date


def test_forecast_refuses_a_past_date_unless_backfill_is_asked_for(panel, catalog):
    with pytest.raises(ValueError):
        shadow.forecast(panel, ARTIFACTS, catalog, "2025-06-01")
    back = shadow.forecast(panel, ARTIFACTS, catalog, "2025-06-01", allow_backfill=True)
    assert int(back["backfilled"].iloc[0]) == 1      # quarantined out of every headline


def test_the_prediction_log_is_append_only(recs, tmp_path):
    out = str(tmp_path / "shadow")
    assert shadow.log_predictions(recs, out) == len(recs)
    shadow.log_predictions(recs, out)
    raw = open(os.path.join(out, "predictions.csv"), encoding="utf-8").read().splitlines()
    assert raw.count(raw[0]) == 1                    # one header, ever
    assert len(raw) == 1 + 2 * len(recs)             # nothing was rewritten
    live = shadow.read_predictions(out)
    assert len(live) == len(recs)                    # the last run for a day is the live one


def test_the_log_records_what_makes_it_a_proof(recs, tmp_path):
    out = str(tmp_path / "shadow")
    shadow.log_predictions(recs, out, store="0123", items_config_hash="deadbeef")
    row = shadow.read_predictions(out).iloc[0]
    assert pd.Timestamp(row["panel_through"]) < pd.Timestamp(row["for_date"])
    for field in ("run_id", "made_at", "model_version", "spec_hash", "sellout_source",
                  "rec_qty", "par_qty", "q_0.50", "q_star", "source", "backfilled"):
        assert field in row.index, field
    assert row["items_config_hash"] == "deadbeef"


def test_the_quantiles_are_logged_in_units_not_z_space(recs, tmp_path):
    out = str(tmp_path / "shadow")
    shadow.log_predictions(recs, out)
    row = shadow.read_predictions(out).iloc[0]
    assert float(row["q_0.05"]) <= float(row["q_0.50"]) <= float(row["q_0.95"])
    assert float(row["q_0.50"]) > 1.0                # units, not a z-score around zero


def test_the_text_and_html_sheets_carry_identical_numbers(recs):
    conditions = dict(tmax_f=48.0, weather="rain", snow_tomorrow=0, holiday="new_years_day",
                      payday=1)
    common = dict(store="Example Store", for_date=pd.Timestamp("2026-01-01"),
                  conditions=conditions, yesterday=None, excluded=[], caveats=[])
    text = shadow.morning_sheet(recs, fmt="text", **common)
    html = shadow.morning_sheet(recs, fmt="html", **common)
    for _, row in recs.iterrows():
        for value in (row["rec_qty"], row["par_qty"]):
            rendered = shadow._fmt_qty(value, row.get("unit", "each"))
            assert rendered in text
            assert rendered in html


def test_the_text_sheet_prints_in_a_walk_in_cooler(recs):
    text = shadow.morning_sheet(
        recs, store="Example Store", for_date=pd.Timestamp("2026-01-01"),
        conditions=dict(tmax_f=48.0, weather="rain", snow_tomorrow=0, holiday="", payday=0),
        yesterday=None, excluded=[], caveats=[shadow.NO_SELLOUT_CAVEAT])
    assert max(len(line) for line in text.splitlines()) <= shadow.SHEET_WIDTH
    assert text.isascii()                            # a receipt printer, not a browser
    assert "MAKE" in text and "PAR" in text
    assert shadow.NO_SELLOUT_CAVEAT in text


def test_an_excluded_item_still_appears_under_no_forecast_with_its_reason(recs):
    """An item missing from the sheet reads to a kitchen manager as "make none".

    The spec asks for the excluded item's par AND its reason -- "new item, 31 of 84 days"
    -- because a bare par with no explanation is a number nobody can act on or argue with.
    The par prints; the reason does not.
    """
    text = shadow.morning_sheet(
        recs, store="S", for_date=pd.Timestamp("2026-01-01"),
        conditions=dict(tmax_f=48.0, weather="sunny", snow_tomorrow=0, holiday="", payday=0),
        yesterday=None, caveats=[],
        excluded=[dict(item="newthing", open_train_days=31, required=84,
                       reason="new item, 31 of 84 days", par_qty=6.0)])
    assert "NO FORECAST" in text
    assert "newthing" in text and "31 of 84" in text


def test_par_quantity_ignores_closed_days(catalog):
    """model/baselines.naive_forecast averages a Christmas zero into the par for four
    consecutive same-weekday targets a year. A printed sheet must not inherit that."""
    dates = pd.date_range("2025-01-06", periods=29, freq="7D")   # 29 Mondays
    frame = pd.DataFrame({
        "store": "0123", "date": dates, "item": "bread", "item_name": "Bread Loaf",
        "dept": "Bakery", "sold": 20.0, "is_closed": 0, "row_status": "ok",
    })
    frame.loc[frame.index[-2], ["sold", "is_closed", "row_status"]] = [0.0, 1, "closed"]
    panel = schema.conform(frame)
    for_date = dates[-1] + pd.Timedelta(days=7)
    assert shadow.par_quantity(panel, "bread", for_date, catalog) == pytest.approx(20.0)


def test_score_day_is_write_once_and_discloses_a_revision(panel, catalog, tmp_path):
    out = str(tmp_path / "shadow")
    for_date = pd.Timestamp("2025-06-01")
    recs = shadow.forecast(panel, ARTIFACTS, catalog, for_date, allow_backfill=True)
    shadow.log_predictions(recs, out)

    first = shadow.score_day(panel, catalog, out, for_date)
    assert len(first) == len(recs)
    path = os.path.join(out, "scores", "2025-06-01.csv")
    frozen = open(path, encoding="utf-8").read()

    revised = panel.copy()
    mask = (revised.date == for_date) & (revised.item == "bread")
    revised.loc[mask, "sold"] = revised.loc[mask, "sold"] + 25.0   # a corrected export
    shadow.score_day(revised, catalog, out, for_date)
    assert open(path, encoding="utf-8").read() == frozen            # the verdict stands
    revisions = pd.read_csv(os.path.join(out, "scores", "_revisions.csv"))
    assert len(revisions) >= 1
    assert set(revisions["item"]) == {"bread"}


def test_a_day_with_no_logged_prediction_is_counted_not_silently_passed(panel, catalog,
                                                                        tmp_path):
    out = str(tmp_path / "shadow")
    rows = shadow.score_day(panel, catalog, out, "2025-06-02")
    # an empty frame IS the silent pass this test exists to rule out: the missed morning has
    # to appear as a row so it counts against completeness
    assert len(rows) == len(catalog)
    assert set(rows["status"]) == {"missing_sheet"}


def test_gates_are_printed_from_week_one_as_pass_fail_or_pending():
    verdicts = shadow.gates({})
    assert set(verdicts) == {"G1", "G2", "G3", "G4", "G5"}
    assert set(verdicts.values()) == {"PENDING"}      # no data yet is never a PASS

    good = dict(
        completeness=1.0, n_rows_scored=100, missing_sheets=[],
        accuracy=dict(model=dict(wape_uncensored=0.10), par=dict(wape_uncensored=0.20),
                      weekly=[dict(model=0.1, par=0.2)] * 4),
        calibration=[dict(tau=0.50, cov_point=0.50, cov_lo=0.45, n_observed=90),
                     dict(tau=0.90, cov_point=0.88, cov_lo=0.80, n_observed=90)],
        measured=dict(waste_observed_retail=1000.0),
        bounds=dict(waste_saving_lower_retail=300.0, sellout_days_model_lower=0.10,
                    sellout_days_sq=0.12),
        top_items=["bread"], short_history_items=[],
    )
    assert shadow.gates(good) == {g: "PASS" for g in ("G1", "G2", "G3", "G4", "G5")}

    # a panel with no sellout signal cannot measure calibration at all: cov_lo is zero by
    # construction there, and reading a verdict off it is how a quantile head that
    # under-covers gets a PASS
    blind = dict(good, calibration=[dict(c, n_observed=0, cov_lo=0.0)
                                    for c in good["calibration"]])
    assert shadow.gates(blind)["G3"] == "PENDING"

    bad = dict(good, completeness=0.80,
               bounds=dict(good["bounds"], waste_saving_lower_retail=10.0))
    verdicts = shadow.gates(bad)
    assert verdicts["G1"] == "FAIL" and verdicts["G4"] == "FAIL"


def test_status_answers_what_is_behind(tmp_path, recs):
    out = str(tmp_path / "shadow")
    shadow.log_predictions(recs, out)
    state = shadow.status(out)
    # the keys are literals in status(); "what is behind" is in the values. A logged
    # prediction with no sheet and no score has to read as both.
    day = str(pd.Timestamp(recs["for_date"].iloc[0]).date())
    assert state["unscored_dates"] == [day]
    assert state["last_scored_date"] is None and state["last_sheet_date"] is None
    assert state["sellout_source"] == str(recs["sellout_source"].iloc[0])


def test_an_unforecast_item_does_not_turn_the_weekly_headline_into_nan(panel, catalog,
                                                                      tmp_path):
    """A par_fallback row carries NaN quantiles, and one NaN poisons a plain sum.

    An item the model cannot forecast is still logged, with its par, so the sheet never reads
    "make none". Those rows are not model forecasts: they belong in the exclusion ledger, not
    in the model's accuracy. Before this was fixed the whole ACCURACY table printed n/a while
    the per-week lines beneath it printed real numbers -- on the one page the pilot exists to
    produce. Model and par must also land on the SAME rows, or the comparison is not paired.
    """
    out = str(tmp_path / "shadow")
    dates = pd.date_range("2025-06-02", periods=7)
    for d in dates:
        recs = shadow.forecast(panel, ARTIFACTS, catalog, d, allow_backfill=True)
        # a short-history item reaches the log exactly this way
        recs.loc[recs["item"] == "cake", ["source", "fallback_reason"]] = \
            ["par_fallback", "new item, 31 of 84 days"]
        recs.loc[recs["item"] == "cake", shadow.quantile_columns(features.TAUS)] = np.nan
        recs.loc[recs["item"] == "cake", "rec_qty"] = np.nan
        shadow.log_predictions(recs, out)
        shadow.score_day(panel, catalog, out, d)

    res = shadow.weekly_report(out, panel, catalog, dates[-1], weeks=1,
                               include_backfilled=True)
    model, par = res["accuracy"]["model"], res["accuracy"]["par"]
    assert np.isfinite(model["wape_uncensored"]), "the headline accuracy is nan"
    assert np.isfinite(model["wape_all_rows"])
    assert model["n_uncensored"] == par["n"], "model and par are not on the same rows"
    assert res["n_rows_no_forecast"] == len(dates)          # one cake row per day
    assert res["exclusions"]["no_forecast"] == len(dates)   # and it is disclosed
    assert "cake" in res["short_history_items"]


def test_a_par_row_is_not_reported_as_something_the_model_said(recs):
    """The same sheet lists short-history items under NO FORECAST with a par and a reason.
    Calling that par a model prediction twenty lines above contradicts the page."""
    y = pd.DataFrame([dict(item_name="Sub / Sandwich", said=17.0, made=21.0, sold=20.0,
                           sold_out=False, unit="each", source="par_fallback"),
                      dict(item_name="Bread Loaf", said=30.0, made=28.0, sold=28.0,
                           sold_out=True, unit="each", source="model")])
    y.attrs["closed"] = False
    text = shadow.morning_sheet(recs, store="s", for_date=pd.Timestamp("2026-01-01"),
                                conditions={}, yesterday=y, excluded=[], caveats=[])
    html = shadow.morning_sheet(recs, store="s", for_date=pd.Timestamp("2026-01-01"),
                                conditions={}, yesterday=y, excluded=[], caveats=[],
                                fmt="html")
    for page in (text, html):
        assert "Sub / Sandwich: your par was" in page
        assert "Bread Loaf: model said" in page
        assert "Sub / Sandwich: model said" not in page


def test_an_unexplained_export_gap_counts_against_completeness(panel, catalog, tmp_path):
    """G1 exists to catch exactly this, and an outage used to leave both sides of the ratio."""
    out = str(tmp_path / "shadow")
    end = pd.Timestamp(panel["date"].max())
    holed = panel.copy()
    gap = pd.date_range(end - pd.Timedelta(days=5), end - pd.Timedelta(days=2))
    hit = pd.to_datetime(holed["date"]).isin(gap)
    holed.loc[hit, ["row_status", "is_closed"]] = ["missing", 1]
    res = shadow.weekly_report(out, holed, catalog, end, weeks=1)
    assert res["completeness"] < 0.95         # the outage is in the denominator, not dropped
    assert len(res["data_gaps"]) == 4
    # and the gate reads it: a four-day hole is an unexplained gap over one day
    scored = dict(res, n_rows_scored=100, completeness=1.0, missing_sheets=[])
    assert shadow.gates(scored)["G1"] == "FAIL"
