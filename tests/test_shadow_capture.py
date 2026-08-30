"""Getting the paper sheet back, and saying exactly what came back on it.

Four shadow weeks produce two kinds of evidence: the forecast that provably existed before
the day, which tests/test_shadow.py guards, and the MADE / SOLD OUT AT columns a kitchen
fills in by hand. For a store with no label log those two columns are the ONLY production
and sellout data the pilot will ever have -- gate G4 and the whole "this would have cut
waste by $X" sentence rest on them -- so the return path is tested here to the same
standard as the outbound one:

  a sheet that cannot be read writes NOTHING, because a half-keyed day that looks entered
  is worse than one that obviously is not;
  a blank SOLD OUT AT cell counts as "did not sell out" only on a row that came in through
  `enter`, and a hand-authored overrides file's blanks stay unknown;
  the flag's provenance says "sheet" in the record, so a weekly page can say the sellout
  data came from a person and not from a rule the export never ran.

And two things the page itself must not overstate: a report heading that says "on days where
demand was fully served" over an all-rows WAPE, and a confident MAKE for an item that keeps.
"""
import os

import numpy as np
import pandas as pd
import pytest

from ht import config, schema
from model import features, shadow

QCOLS = shadow.quantile_columns(features.TAUS)
DATES = pd.date_range("2025-03-03", periods=7, freq="D")


@pytest.fixture
def items():
    from tests.conftest import ITEMS_JSON
    return config.load_items(ITEMS_JSON)


@pytest.fixture
def panel(make_panel):
    """Two real item keys, and a store whose export carries no sellout signal at all."""
    return make_panel(["bread", "doughnut"], start="2025-02-01", days=40,
                      stockout=0, stockout_known=0, sellout_source="none")


def _sold(panel, date, key):
    hit = panel[(panel["date"] == date) & (panel["item"] == key)]
    return float(hit["sold"].iloc[0])


def _log_sheets(shadow_dir, panel, items, dates=DATES, keys=("bread", "doughnut")):
    """A prediction log for `dates`, shaped like morning's, with no checkpoint involved."""
    rows = []
    for d in dates:
        for key in keys:
            sold = _sold(panel, d, key)
            q = {c: sold * (0.6 + float(c.split("_")[1])) for c in QCOLS}
            rows.append(dict(item=key, item_name=items[key]["name"], dept=items[key]["dept"],
                             for_date=d, panel_through=d - pd.Timedelta(days=1),
                             model_version="test", spec_hash="testspec", sellout_source="none",
                             batch=1.0, continuous=False, unit="each", source="model",
                             fallback_reason="", why_text="test", backfilled=0, q_star=0.7,
                             par_qty=round(sold), rec_qty=round(sold), **q))
    shadow.log_predictions(pd.DataFrame(rows), shadow_dir, store="Test Store")


def _enter(shadow_dir, items, date, lines, entered_by="tester"):
    rows, errors = shadow.parse_entries(lines, items)
    assert not errors, errors
    return shadow.record_actuals(shadow_dir, date, rows, entered_by=entered_by)


# ---- (a) the return path ----

@pytest.mark.parametrize("written,expected", [
    ("", ""), ("  ", ""), ("-", ""),
    ("14:30", "14:30"), ("2:30pm", "14:30"), ("2:30 PM", "14:30"), ("1430", "14:30"),
    ("930am", "09:30"), ("2pm", "14:00"), ("12am", "00:00"), ("9", "09:00"),
    ("yes", "yes"), ("Y", "yes"), ("circled", "yes"),
    ("elevenish", None), ("25:00", None), ("14:75", None), ("13pm", None),
])
def test_the_sold_out_at_cell_is_read_the_way_a_person_writes_it(written, expected):
    # a guessed sellout time is a fabricated observation, so anything unreadable is refused
    # by name rather than coerced into a number
    assert shadow.parse_time(written) == expected


def test_a_sheet_that_cannot_be_read_writes_nothing(tmp_path, items):
    lines = ["doughnut,96,2:30pm", "Croissant,12,", "bread,-4,", "cake,4,elevenish"]
    rows, errors = shadow.parse_entries(lines, items)
    assert [r["item"] for r in rows] == ["doughnut"]
    assert len(errors) == 3
    assert "'Croissant' is not an item" in errors[0] and "line 2" in errors[0]

    (tmp_path / "shadow").mkdir()
    (tmp_path / "sheet.csv").write_text("\n".join(lines), encoding="utf-8")
    from tests.conftest import ITEMS_JSON
    rc = shadow.main(["enter", "--items", ITEMS_JSON, "--date", "2025-03-03",
                      "--out", str(tmp_path / "shadow"), "--file", str(tmp_path / "sheet.csv")])
    assert rc == 1
    assert not os.path.isdir(tmp_path / "shadow" / "overrides")


def test_the_prompt_and_the_pipe_are_validated_by_the_same_code(items):
    typed = iter(["96", "2:30pm", "", "", "30", ""])
    lines = shadow.prompt_entries(items, ["doughnut", "cake", "bread"],
                                  said={"doughnut": 108.0}, ask=lambda p: next(typed),
                                  echo=lambda *a: None)
    # cake was left blank on both lines: nothing was written there, so nothing is recorded
    assert lines == ["doughnut,96,2:30pm", "bread,30,"]
    rows, errors = shadow.parse_entries(lines, items)
    assert not errors
    assert [(r["item"], r["actual_produced"], r["sold_out_at"]) for r in rows] == [
        ("doughnut", 96.0, "14:30"), ("bread", 30.0, "")]


def test_the_record_keeps_what_the_sheet_said_next_to_what_the_kitchen_did(tmp_path, panel,
                                                                          items):
    shadow_dir = str(tmp_path / "shadow")
    _log_sheets(shadow_dir, panel, items)
    _enter(shadow_dir, items, DATES[0], ["bread,30,17:15", "doughnut,96,"])
    ov = shadow.read_overrides(shadow_dir)
    assert list(ov["sellout_source"]) == ["sheet", "sheet"]
    assert list(ov["entered_by"]) == ["tester", "tester"]
    # rec_qty is filled from the prediction log, so the row carries both numbers
    assert float(ov.set_index("item").loc["bread", "rec_qty"]) == round(_sold(panel, DATES[0],
                                                                             "bread"))

    # a correction is a second entry: the file is append-only and the last row wins
    _enter(shadow_dir, items, DATES[0], ["bread,34,17:15"])
    raw = (tmp_path / "shadow" / "overrides" / f"{DATES[0].date()}.csv").read_text()
    assert raw.count("bread") == 2
    assert float(shadow.read_overrides(shadow_dir).set_index("item")
                 .loc["bread", "actual_produced"]) == 34.0


def test_sold_out_at_becomes_the_sellout_flag_the_panel_never_had(tmp_path, panel, items):
    shadow_dir = str(tmp_path / "shadow")
    _log_sheets(shadow_dir, panel, items)
    _enter(shadow_dir, items, DATES[0], ["bread,30,17:15", "doughnut,96,"])
    scored = shadow.score_day(panel, items, shadow_dir, DATES[0]).set_index("item")

    assert float(scored.loc["bread", "stockout"]) == 1.0        # a time was written
    assert float(scored.loc["doughnut", "stockout"]) == 0.0     # the cell came back blank
    assert list(scored["stockout_known"]) == [1.0, 1.0]         # the panel's column is 0
    assert list(scored["sellout_source"]) == ["sheet", "sheet"]
    # the production number the kitchen wrote outranks the export's
    assert float(scored.loc["bread", "produced"]) == 30.0


def test_a_hand_authored_override_does_not_invent_a_sellout_observation(tmp_path, panel,
                                                                       items):
    """The pre-existing intake: a file someone typed to correct a production number.

    Its empty sold_out_at cells were never an answer to a question, and reading them as
    "did not sell out" would manufacture an observation on every row.
    """
    shadow_dir = str(tmp_path / "shadow")
    _log_sheets(shadow_dir, panel, items)
    root = os.path.join(shadow_dir, "overrides")
    os.makedirs(root)
    with open(os.path.join(root, "legacy.csv"), "w", encoding="utf-8") as f:
        f.write("date,item,actual_produced,note\n"
                f"{DATES[0].date()},bread,30,typo in the export\n")
    scored = shadow.score_day(panel, items, shadow_dir, DATES[0]).set_index("item")
    assert float(scored.loc["bread", "produced"]) == 30.0
    assert float(scored.loc["bread", "stockout_known"]) == 0.0
    assert scored.loc["bread", "sellout_source"] == "none"


def test_the_weekly_page_credits_the_sheet_for_the_censoring(tmp_path, panel, items):
    shadow_dir = str(tmp_path / "shadow")
    _log_sheets(shadow_dir, panel, items)
    for i, d in enumerate(DATES):
        _enter(shadow_dir, items, d, [f"bread,30,{'17:15' if i % 3 == 0 else ''}",
                                      "doughnut,96,"])
        shadow.score_day(panel, items, shadow_dir, d)
    # an item this store does not stock a panel row for: keyed in, but nothing scores it
    _enter(shadow_dir, items, DATES[0], ["cake,4,"])
    res = shadow.weekly_report(shadow_dir, panel, items, DATES[-1], weeks=1)

    # the panel's own rule is "none"; without the sheet this week would be uncensorable
    assert res["censoring"]["sellout_source"] == "sheet"
    assert res["censoring"]["censoring_known"] is True
    assert res["censoring"]["known_share"] == 1.0
    assert res["censoring"]["sellout_rate"] == pytest.approx(3 / 14)
    assert res["bounds"]["sellout_days_sq"] == pytest.approx(3 / 14)
    # a share of scored rows counts overrides that landed on one: with a sheet coming back
    # every day the raw count divided by that denominator reads over 100%, which is not a fact
    assert res["overrides"]["n"] == 15 and res["overrides"]["n_matched"] == 14
    assert res["overrides"]["share"] == 1.0
    assert "1. ACCURACY (median forecast vs sold, on days where demand was fully served)" \
        in shadow.format_weekly(res)


# ---- (b) the heading in the censoring-unknown case ----

def test_the_accuracy_heading_does_not_promise_uncensored_rows_it_has_not_got(tmp_path,
                                                                              panel, items):
    shadow_dir = str(tmp_path / "shadow")
    _log_sheets(shadow_dir, panel, items)
    for d in DATES:
        shadow.score_day(panel, items, shadow_dir, d)
    res = shadow.weekly_report(shadow_dir, panel, items, DATES[-1], weeks=1)
    page = shadow.format_weekly(res)

    assert res["censoring"]["censoring_known"] is False
    # the same run's JSON says every uncensored figure is an all-rows figure; the page must
    # not say the opposite two lines above the number
    assert "on days where demand was fully served" not in page
    assert "1. ACCURACY (median forecast vs sold, over EVERY scored row)" in page
    assert "wape_all" in page
    assert res["accuracy"]["model"]["n_uncensored"] == res["accuracy"]["model"]["n_rows"]
    # and the per-department table underneath stops naming an uncensored subset that is
    # really the whole group
    assert "n_unc=" not in page


# ---- (c) items that carry over ----

def _recs(shelf_life):
    """Two model rows and the attrs morning_sheet reads, without loading a checkpoint."""
    rows = []
    for key, name, life in (("doughnut", "Doughnut", 1), ("bread", "Bread Loaf", shelf_life)):
        row = dict(item=key, item_name=name, dept="Bakery", unit="each", batch=1.0,
                   continuous=False, shelf_life_days=life, par_qty=30.0, rec_qty=37.0,
                   why_text="Wed is #3 of 7 for this item", source="model",
                   fallback_reason="", model_version="test", sellout_source="none",
                   panel_through=pd.Timestamp("2025-03-02"), for_date=pd.Timestamp("2025-03-03"))
        row.update({c: 30.0 for c in QCOLS})
        rows.append(row)
    recs = pd.DataFrame(rows)
    recs.attrs.update(day_source="panel", warnings=[], staleness_days=0)
    return recs


def test_a_carry_over_item_is_marked_and_not_in_the_make_block():
    sheet = shadow.morning_sheet(_recs(3), store="Test", for_date="2025-03-03",
                                 conditions=dict(tmax_f=50.0, weather="rain"))
    bakery = sheet.split("BAKERY")[1].split("CARRY-OVER")[0]
    assert "Doughnut" in bakery and "Bread Loaf" not in bakery
    carry = sheet.split("CARRY-OVER")[1]
    assert "Bread Loaf" in carry
    # the reason has to be readable in one glance, not inferred from a caveat at the bottom
    assert "NOT AN ORDER" in sheet
    assert "3-day shelf life" in carry
    assert "does not subtract" in carry
    # and the name is not truncated to make room for a marker
    assert "Bread  (multi-day)" not in sheet


def test_a_day_fresh_sheet_has_no_carry_over_block():
    sheet = shadow.morning_sheet(_recs(1), store="Test", for_date="2025-03-03",
                                 conditions=dict(tmax_f=50.0, weather="rain"))
    assert "CARRY-OVER" not in sheet
    assert "Bread Loaf" in sheet.split("BAKERY")[1]


def test_the_html_sheet_marks_the_same_rows():
    html = shadow.morning_sheet(_recs(3), store="Test", for_date="2025-03-03",
                                conditions=dict(tmax_f=50.0, weather="rain"), fmt="html")
    assert "Carry-over items - NOT an order" in html
    assert "3-day shelf life" in html
    bakery = html.split("<h2>Bakery</h2>")[1].split("<h2>Carry-over")[0]
    assert "Doughnut" in bakery and "Bread Loaf" not in bakery


# ---- (d) the person at the terminal at 5:30am ----

@pytest.mark.parametrize("argv,needle", [
    (["morning", "--panel", "/nope/panel.csv", "--artifacts", "model/artifacts",
      "--items", "config/items.example.json", "--date", "2025-03-03"],
     "--panel: no panel csv"),
    (["status", "--out", "/nope/shadow"], "--out: no shadow directory"),
    (["weekly", "--panel", "data/store_synth.csv", "--items", "config/items.example.json",
      "--week-ending", "next tuesday", "--out", "."], "--week-ending: 'next tuesday' is not"),
])
def test_a_mistyped_path_or_date_is_one_line_and_exit_1(argv, needle, capsys, repo,
                                                        monkeypatch):
    monkeypatch.chdir(repo)
    assert shadow.main(argv) == 1
    err = capsys.readouterr().err
    assert needle in err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1


def test_evaluate_refuses_a_bad_artifacts_directory_without_a_traceback(capsys, repo,
                                                                       monkeypatch):
    from model import evaluate
    monkeypatch.chdir(repo)
    assert evaluate.main(["--panel", "data/store_synth.csv", "--artifacts", "model",
                          "--items", "config/items.example.json"]) == 1
    err = capsys.readouterr().err
    assert "--artifacts: model has no meta.json" in err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1


def test_the_score_row_carries_the_flag_it_was_scored_with(tmp_path, panel, items):
    """SCORE_COLUMNS is the frozen per-day verdict; the flag's provenance belongs in it."""
    shadow_dir = str(tmp_path / "shadow")
    _log_sheets(shadow_dir, panel, items)
    shadow.score_day(panel, items, shadow_dir, DATES[0])
    written = pd.read_csv(os.path.join(shadow_dir, "scores", f"{DATES[0].date()}.csv"))
    assert list(written.columns) == shadow.SCORE_COLUMNS
    assert set(written["sellout_source"]) == {"none"}
    assert np.isfinite(pd.to_numeric(written["sold"], errors="coerce")).all()
    assert schema.SELLOUT_SOURCES  # the panel vocabulary; "sheet" is deliberately not in it


# ---- (d) the returned sheet must not break what is already on disk ----

LEGACY_OVERRIDE_COLUMNS = ["date", "item", "rec_qty", "actual_produced", "sold_out_at",
                           "note", "entered_by", "entered_ts"]


def test_appending_to_an_older_overrides_file_keeps_the_whole_directory_readable(tmp_path,
                                                                                items):
    """A pilot's hand-keyed corrections predate sellout_source, and `enter` appends to them.

    Nine fields under an eight-field header is a pandas ParserError, and read_overrides
    parses every file in the directory before windowing, so one such day takes down score,
    catch-up, weekly and the morning page's YESTERDAY block.
    """
    shadow_dir = str(tmp_path / "shadow")
    root = os.path.join(shadow_dir, "overrides")
    os.makedirs(root)
    path = os.path.join(root, f"{DATES[0].date()}.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(LEGACY_OVERRIDE_COLUMNS) + "\n")
        fh.write(f"{DATES[0].date()},bread,53,60,,corrected production,kmurphy,"
                 "2025-03-04T06:00:00+00:00\n")

    _enter(shadow_dir, items, DATES[0], ["bread,44,2:30pm", "doughnut,6,"])

    ov = shadow.read_overrides(shadow_dir)
    assert list(ov.columns)[:len(shadow.OVERRIDE_COLUMNS)] == shadow.OVERRIDE_COLUMNS
    assert len(ov) == 2                       # the correction, superseded by the new row
    old = pd.read_csv(path).iloc[0]
    # the migrated row keeps its meaning: a hand-authored blank says nothing about sellouts
    assert old["note"] == "corrected production"
    assert shadow._sheet_sellout(old.to_dict()) is None


def test_a_hand_authored_column_survives_the_migration(tmp_path, items):
    shadow_dir = str(tmp_path / "shadow")
    root = os.path.join(shadow_dir, "overrides")
    os.makedirs(root)
    path = os.path.join(root, f"{DATES[0].date()}.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("date,item,actual_produced,who_asked\n")
        fh.write(f"{DATES[0].date()},bread,60,the baker\n")

    _enter(shadow_dir, items, DATES[0], ["doughnut,6,"])
    back = pd.read_csv(path)
    assert "who_asked" in back.columns and back["who_asked"].iloc[0] == "the baker"
    assert list(back.columns)[:len(shadow.OVERRIDE_COLUMNS)] == shadow.OVERRIDE_COLUMNS


# ---- (e) the heading follows the rows the number covers ----

def test_one_keyed_in_row_does_not_relabel_a_whole_week_as_fully_served(tmp_path, panel,
                                                                       items):
    """The panel's rule is "none"; one returned sheet row makes the flag evaluable on 1 of 14.

    censoring_known went True on bare presence, which put "on days where demand was fully
    served" over an all-rows WAPE -- the exact claim the all-rows heading exists to remove.
    """
    shadow_dir = str(tmp_path / "shadow")
    _log_sheets(shadow_dir, panel, items)
    _enter(shadow_dir, items, DATES[1], ["bread,30,"])
    for d in DATES:
        shadow.score_day(panel, items, shadow_dir, d)
    res = shadow.weekly_report(shadow_dir, panel, items, DATES[-1], weeks=1)
    page = shadow.format_weekly(res)

    assert 0 < res["censoring"]["known_share"] < 1
    assert "on days where demand was fully served" not in page
    assert "on rows nothing flagged as sold out" in page
    assert "the flag was evaluable on 1 of 14 scored rows" in page
    # and the caveat keeps its force: it used to shrink to "evaluable on 7% of rows"
    assert any("only means nobody flagged it" in c and "pulls the quantities down" in c
               for c in res["caveats"])
    # the store's sellout rate says how many rows it was measured over
    assert "sellout days: store 0.0% (of 1 evaluable rows)" in page


def test_a_fully_flagged_week_still_reads_as_fully_served(tmp_path, panel, items):
    shadow_dir = str(tmp_path / "shadow")
    _log_sheets(shadow_dir, panel, items)
    for d in DATES:
        _enter(shadow_dir, items, d, ["bread,30,", "doughnut,96,"])
        shadow.score_day(panel, items, shadow_dir, d)
    res = shadow.weekly_report(shadow_dir, panel, items, DATES[-1], weeks=1)
    page = shadow.format_weekly(res)
    assert res["censoring"]["known_share"] == 1.0
    assert "1. ACCURACY (median forecast vs sold, on days where demand was fully served)" \
        in page
    # nothing sold out, so the all-rows line would restate the number under a second heading
    assert "wape over all rows including sellouts" not in page


# ---- (f) a name the sheet printed is a name `enter` accepts ----

def test_a_name_the_sheet_truncated_is_still_keyed_back_in(items):
    long_name = "Rotisserie Chicken Lemon Pepper"
    cfg = {k: dict(v) for k, v in items.items()}
    cfg["rotisserie"]["name"] = long_name
    printed = long_name[:shadow.SHEET_ITEM_WIDTH]
    assert printed != long_name                       # the sheet cannot print all of it

    rows, errors = shadow.parse_entries([f"{printed}, 40, 2:30pm"], cfg)
    assert not errors and rows[0]["item"] == "rotisserie"


def test_a_name_that_is_not_an_item_says_what_it_might_have_been(items):
    rows, errors = shadow.parse_entries(["Whole, 3,"], items)
    assert not rows and "did you mean pizza-whole?" in errors[0]


def test_a_sheet_saved_from_excel_is_read_despite_the_byte_order_mark(tmp_path, items,
                                                                     panel):
    from tests.conftest import ITEMS_JSON

    shadow_dir = str(tmp_path / "shadow")
    _log_sheets(shadow_dir, panel, items)
    sheet = tmp_path / "sheet.csv"
    sheet.write_bytes("﻿bread,44,2:30pm\r\ndoughnut,6,\r\n".encode("utf-8"))
    rc = shadow.main(["enter", "--items", ITEMS_JSON, "--date", str(DATES[0].date()),
                      "--out", shadow_dir, "--file", str(sheet), "--by", "rev"])
    assert rc == 0
    ov = shadow.read_overrides(shadow_dir)
    assert sorted(ov["item"]) == ["bread", "doughnut"]


def test_a_zero_in_the_sold_out_cell_is_not_a_sellout_at_midnight():
    assert shadow.parse_time("0") == ""
    assert shadow.parse_time("none") == "" and shadow.parse_time("n/a") == ""
    assert shadow._sheet_sellout(dict(sellout_source="sheet", sold_out_at="")) == (0.0, 1.0,
                                                                                  "sheet")


# ---- (g) the prompt walks the page ----

def test_the_prompt_order_is_the_sheet_order_when_an_item_carries_over(tmp_path, panel,
                                                                      items):
    shadow_dir = str(tmp_path / "shadow")
    cfg = {k: dict(v) for k, v in items.items()}
    cfg["bread"]["shelf_life_days"] = 2               # printed in the CARRY-OVER block
    _log_sheets(shadow_dir, panel, cfg)
    order = shadow.entry_order(shadow_dir, DATES[0], cfg)
    assert order.index("doughnut") < order.index("bread")
