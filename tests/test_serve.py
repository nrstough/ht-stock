"""The daily loop behind HTTP, and the two ways a browser can wreck a record a CLI cannot.

model/shadow.py's own guards are tested in test_shadow.py and test_shadow_capture.py. What
is tested here is that putting a request layer in front of them neither softens a refusal
nor introduces one of its own, plus the two hazards that only exist once there is a button:

  a GET that writes. log_predictions appends and read_predictions keeps the LAST row per
  (for_date, item), so a refresh, a prefetch or a double-click would replace the morning's
  forecast with a later one and leave nothing in the record saying so;

  a Score button pressed before the store's export lands. score_day is write-once by file
  existence, so that freezes an all-missing verdict which catch_up cannot repair and which
  counts against G1 completeness for the rest of the pilot.

Every test drives ht.serve.Api.handle directly. That is not a shortcut around the server:
conftest.py patches socket.socket to raise for every test, so binding a port is impossible
here by design, and the handler is a pure function of (method, path, query, body) precisely
so that the whole route surface stays reachable without one.
"""
import datetime as dt
import json
import os
import threading

import numpy as np
import pandas as pd
import pytest

from ht import config as ht_config, schema, serve
from model import shadow

from tests.conftest import ARTIFACTS, ITEMS_JSON, SYNTH_CSV

# The panel ends 2025-12-31, so this is the first date that can be forecast without
# --backfill, exactly as in test_shadow.py.
TOMORROW = "2026-01-01"
COVERED = "2025-12-30"


@pytest.fixture(scope="module")
def panel_path(tmp_path_factory):
    """The synthetic store as a canonical panel on disk, written once."""
    out = tmp_path_factory.mktemp("panel") / "panel.csv"
    schema.write_panel(schema.conform(pd.read_csv(SYNTH_CSV, parse_dates=["date"])), str(out))
    return str(out)


@pytest.fixture
def settings(panel_path, tmp_path):
    return serve.Settings(panel=panel_path, items=ITEMS_JSON, artifacts=ARTIFACTS,
                          out=str(tmp_path / "shadow"), store="0123",
                          timezone="America/Chicago", entered_by="kmurphy")


@pytest.fixture
def api(settings):
    return serve.Api(settings)


@pytest.fixture
def started(api):
    """An api whose shadow directory exists and carries one morning sheet."""
    status, payload = api.handle("POST", "/api/morning", body={"date": TOMORROW})
    assert status == 200, payload
    return api


def _pred_bytes(api):
    path = os.path.join(api.s.out_dir, "predictions.csv")
    return open(path, "rb").read() if os.path.exists(path) else b""


# ---- no GET writes, and a second run is refused ----

def test_get_morning_never_writes(started):
    before = _pred_bytes(started)
    for _ in range(3):
        status, _ = started.handle("GET", "/api/morning", query={"date": TOMORROW})
        assert status == 200
    assert _pred_bytes(started) == before


def test_get_morning_404s_before_any_run(api):
    api.handle("POST", "/api/morning", body={"date": TOMORROW})
    status, payload = api.handle("GET", "/api/morning", query={"date": "2025-11-04"})
    assert status == 404
    assert "no morning sheet" in payload["error"]


def test_a_second_post_for_a_date_is_refused(started):
    before = _pred_bytes(started)
    status, payload = started.handle("POST", "/api/morning", body={"date": TOMORROW})
    assert status == 409
    assert _pred_bytes(started) == before, "the refusal must not have logged anything"


def test_the_refusal_names_the_existing_run(started):
    _, payload = started.handle("POST", "/api/morning", body={"date": TOMORROW})
    # str(): run_id is uuid4().hex[:12], which is all digits about one run in three hundred,
    # and pandas then reads the column back as int64. The server already stringifies it; a
    # test that did not would fail on those runs and look like flake.
    logged = str(shadow.read_predictions(started.s.out_dir)["run_id"].iloc[0])
    assert logged in payload["error"]
    assert str(payload["runs"][-1]["run_id"]) == logged


def test_an_all_digit_run_id_is_still_reported_as_a_string(started, monkeypatch):
    """The one-in-three-hundred case, made deterministic."""
    monkeypatch.setattr(shadow.uuid, "uuid4",
                        lambda: type("U", (), {"hex": "123456789012ab"})())
    started.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})
    _, payload = started.handle("GET", "/api/morning", query={"date": COVERED})
    assert payload["runs"][0]["run_id"] == "123456789012"
    assert isinstance(payload["live_run_id"], str)


def test_a_deliberate_reforecast_is_disclosed(started):
    status, payload = started.handle("POST", "/api/morning",
                                     body={"date": TOMORROW, "reforecast": True})
    assert status == 200
    # PREDICTION_COLUMNS has no field for "this replaced an earlier sheet" and the printed
    # page carries no marker, so the honest evidence is the one the log already keeps
    assert payload["superseded"] is True
    assert len({r["run_id"] for r in payload["runs"]}) == 2
    _, got = started.handle("GET", "/api/morning", query={"date": TOMORROW})
    assert got["superseded"] is True
    assert got["live_run_id"] == payload["runs"][-1]["run_id"]


def test_one_run_is_not_reported_as_superseded(started):
    _, payload = started.handle("GET", "/api/morning", query={"date": TOMORROW})
    assert payload["superseded"] is False


# ---- refusals the CLI makes, reproduced rather than softened ----

def test_a_covered_date_is_refused_in_the_forecasters_own_words(api):
    api.handle("POST", "/api/morning", body={"date": TOMORROW})
    status, payload = api.handle("POST", "/api/morning", body={"date": COVERED})
    assert status == 400
    assert "--backfill" in payload["error"]
    assert "before the day it is for" in payload["error"]


def test_backfill_is_allowed_deliberately_and_stamped(api):
    api.handle("POST", "/api/morning", body={"date": TOMORROW})
    status, payload = api.handle("POST", "/api/morning",
                                 body={"date": COVERED, "backfill": True})
    assert status == 200
    assert payload["backfilled"] == 1
    assert all(r["backfilled"] == 1 for r in payload["rows"])


def test_a_stale_panel_is_refused(api):
    status, payload = api.handle("POST", "/api/morning", body={"date": "2026-06-01"})
    assert status == 400
    assert "days behind" in payload["error"]


def test_every_route_but_morning_refuses_a_missing_shadow_dir(api):
    # a typo in --out must not read an empty log and report a pilot that lost its record
    for method, path, kw in (("GET", "/api/status", {}),
                             ("GET", "/api/scores", {}),
                             ("GET", "/api/predictions", {}),
                             ("POST", "/api/score", {"body": {"date": TOMORROW}}),
                             ("POST", "/api/catch-up", {"body": {}}),
                             ("GET", "/api/entry-order", {"query": {"date": TOMORROW}})):
        status, payload = api.handle(method, path, **kw)
        assert status == 400, (path, payload)
        assert "no shadow directory" in payload["error"], path


def test_morning_creates_the_shadow_dir(api):
    assert not os.path.isdir(api.s.out_dir)
    assert api.handle("POST", "/api/morning", body={"date": TOMORROW})[0] == 200
    assert os.path.isdir(api.s.out_dir)


def test_the_simulator_only_columns_cannot_reach_a_forecast(tmp_path, settings):
    """conform strips them, so they are gone rather than refused -- either way, absent.

    schema.read_panel conforms what it reads and conform drops those columns whatever it is
    asked, so a panel file carrying them is silently cleaned. That is worth pinning: the
    refusal a reader might assume is there is not, and the property that matters is that
    nothing downstream can see them.
    """
    raw = pd.read_csv(SYNTH_CSV, parse_dates=["date"])
    assert set(schema.SIM_ONLY) & set(raw.columns), "the fixture must carry them to matter"
    dirty = str(tmp_path / "dirty.csv")
    raw.to_csv(dirty, index=False)
    settings.panel_path = dirty
    api = serve.Api(settings)
    status, payload = api.handle("POST", "/api/morning", body={"date": TOMORROW})
    assert status == 200
    assert not set(schema.SIM_ONLY) & set(api._panel().columns)
    assert not set(schema.SIM_ONLY) & set(payload["rows"][0])


def test_ht_serve_names_no_simulator_column():
    """test_no_sim_import.py enforces this over every file in ht/, including comments.

    Pinned here too so the failure names this module rather than arriving as a parametrised
    surprise in a file nobody editing serve.py is looking at.
    """
    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "ht", "serve.py"), encoding="utf-8").read()
    for name in schema.SIM_ONLY:
        assert name not in source


# ---- the returned paper sheet ----

def _enter(api, lines, date=TOMORROW, **kw):
    return api.handle("POST", "/api/enter", body=dict(date=date, lines=lines, **kw))


def test_one_unreadable_line_writes_nothing(started):
    status, payload = _enter(started, ["bread,20,14:30", "cake,not-a-number,"])
    assert status == 400
    assert payload["errors"] and "not a quantity" in payload["errors"][0]
    assert not os.path.isdir(os.path.join(started.s.out_dir, "overrides")), \
        "a half-keyed day that looks entered is worse than one that obviously is not"


def test_a_good_sheet_is_written_once_and_stamped(started):
    status, payload = _enter(started, ["bread,20,14:30", "cake,4,"])
    assert status == 200 and payload["written"] == 2
    over = shadow.read_overrides(started.s.out_dir)
    assert set(over["sellout_source"]) == {shadow.SHEET_SELLOUT_SOURCE}
    assert set(over["entered_by"]) == {"kmurphy"}


def test_entered_by_is_recorded_from_the_request(started):
    _enter(started, ["bread,20,"], by="dspencer")
    assert set(shadow.read_overrides(started.s.out_dir)["entered_by"]) == {"dspencer"}


def test_a_blank_sold_out_cell_is_a_negative_on_a_sheet_row(started):
    _enter(started, ["bread,20,"])
    over = shadow.read_overrides(started.s.out_dir).set_index("item")
    assert shadow._sheet_sellout(over.loc["bread"]) == (0.0, 1.0, shadow.SHEET_SELLOUT_SOURCE)


def test_zero_is_a_negative_not_a_sellout_at_midnight(started):
    _enter(started, ["bread,20,0"])
    over = shadow.read_overrides(started.s.out_dir).set_index("item")
    assert serve.json_safe(over.loc["bread", "sold_out_at"]) in ("", None)


def test_an_unreadable_time_is_named_rather_than_guessed(started):
    status, payload = _enter(started, ["bread,20,half past two"])
    assert status == 400
    assert "is not a time" in payload["errors"][0]


def test_a_truncated_printed_name_is_accepted(started, items):
    name = str(items["rotisserie"]["name"])[:shadow.SHEET_ITEM_WIDTH].strip()
    status, payload = _enter(started, [f"{name},8,"])
    assert status == 200, payload
    assert list(shadow.read_overrides(started.s.out_dir)["item"]) == ["rotisserie"]


def test_an_item_that_is_not_in_the_config_is_refused(started):
    status, payload = _enter(started, ["kombucha,3,"])
    assert status == 400
    assert "is not an item in the items config" in payload["errors"][0]


def test_lines_may_arrive_as_one_string(started):
    status, payload = _enter(started, "bread,20,\ncake,4,")
    assert status == 200 and payload["written"] == 2


def test_missing_lines_is_one_sentence(started):
    status, payload = started.handle("POST", "/api/enter", body={"date": TOMORROW})
    assert status == 400 and "lines is required" in payload["error"]


# ---- keying a sheet in under the wrong date ----

def test_entering_a_date_with_no_sheet_returns_the_warning(started):
    status, payload = _enter(started, ["bread,20,"], date="2025-11-04")
    assert status == 200
    assert payload["warning"] and "no sheet was logged" in payload["warning"]


def test_entry_order_says_when_it_fell_back_to_alphabetical(started):
    _, payload = started.handle("GET", "/api/entry-order", query={"date": "2025-11-04"})
    assert payload["no_sheet_logged"] is True
    assert "alphabetical" in payload["warning"]


def test_entry_order_is_the_order_the_sheet_printed(started, items):
    _, payload = started.handle("GET", "/api/entry-order", query={"date": TOMORROW})
    assert payload["no_sheet_logged"] is False
    assert [r["item"] for r in payload["rows"]] == \
        shadow.entry_order(started.s.out_dir, pd.Timestamp(TOMORROW), items)
    assert all(r["said"] is not None for r in payload["rows"])


# ---- scoring ----

def test_scoring_a_day_with_no_sales_data_is_refused(started):
    """The worst thing a button can do that a command rarely does."""
    status, payload = started.handle("POST", "/api/score", body={"date": TOMORROW})
    assert status == 409
    assert "not fully in the panel yet" in payload["error"]
    assert "frozen once" in payload["error"]
    assert "catch-up" in payload["error"]
    assert not os.path.exists(os.path.join(started.s.out_dir, "scores",
                                           f"{TOMORROW}.csv")), \
        "an empty verdict, once frozen, cannot be re-scored"


def test_the_refusal_is_what_stops_the_freeze(started):
    """Counterfactual: without the precondition, score_day writes an all-missing verdict."""
    panel, items = started._panel(), started._items()
    rows = shadow.score_day(panel, items, started.s.out_dir, TOMORROW)
    assert set(rows["status"].astype(str)) <= serve.UNSETTLED
    assert os.path.exists(os.path.join(started.s.out_dir, "scores", f"{TOMORROW}.csv"))
    # and catch_up cannot undo it: same function, sees the file, records revisions only
    assert TOMORROW not in shadow.catch_up(panel, items, started.s.out_dir)


def test_a_day_with_data_scores(api):
    api.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})
    status, payload = api.handle("POST", "/api/score", body={"date": COVERED})
    assert status == 200
    assert payload["rows"] > 0
    assert not set(payload["counts"]) <= serve.UNSETTLED


def test_rescoring_an_unchanged_day_records_no_revision(api):
    api.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})
    api.handle("POST", "/api/score", body={"date": COVERED})
    api.handle("POST", "/api/score", body={"date": COVERED})
    revisions = os.path.join(api.s.out_dir, "scores", "_revisions.csv")
    assert not os.path.exists(revisions) or len(pd.read_csv(revisions)) == 0


def test_a_later_sheet_is_disclosed_rather_than_applied(api):
    api.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})
    api.handle("POST", "/api/score", body={"date": COVERED})
    frozen = pd.read_csv(os.path.join(api.s.out_dir, "scores", f"{COVERED}.csv"))
    _enter(api, ["bread,999,"], date=COVERED)
    api.handle("POST", "/api/score", body={"date": COVERED})
    after = pd.read_csv(os.path.join(api.s.out_dir, "scores", f"{COVERED}.csv"))
    pd.testing.assert_frame_equal(frozen, after)
    assert len(pd.read_csv(os.path.join(api.s.out_dir, "scores", "_revisions.csv"))) > 0


def test_catch_up_skips_days_with_no_data(started):
    status, payload = started.handle("POST", "/api/catch-up", body={})
    assert status == 200
    assert TOMORROW not in payload["scored"]
    assert TOMORROW in payload["unscored"]
    assert not os.path.exists(os.path.join(started.s.out_dir, "scores", f"{TOMORROW}.csv"))


# ---- the weekly page ----

@pytest.fixture
def a_scored_week(api):
    for day in pd.date_range("2025-12-25", "2025-12-31"):
        api.handle("POST", "/api/morning",
                   body={"date": str(day.date()), "backfill": True})
    api.handle("POST", "/api/catch-up", body={})
    return api


def test_weekly_quarantines_backfilled_rows_by_default(a_scored_week):
    status, res = a_scored_week.handle("POST", "/api/weekly",
                                       body={"week_ending": "2025-12-31"})
    assert status == 200
    assert res["backfilled_rows"] > 0
    assert res["n_rows_scored"] == 0, "a sheet made after the day proves nothing about it"
    assert res["reconstructed"] is False


def test_include_backfilled_stamps_the_page_reconstructed(a_scored_week):
    _, res = a_scored_week.handle("POST", "/api/weekly",
                                  body={"week_ending": "2025-12-31",
                                        "include_backfilled": True})
    assert res["reconstructed"] is True
    assert res["n_rows_scored"] > 0


def test_pending_is_never_a_pass(a_scored_week):
    _, res = a_scored_week.handle("POST", "/api/weekly", body={"week_ending": "2025-12-31"})
    assert set(res["gates"]) == {"G1", "G2", "G3", "G4", "G5"}
    assert set(res["gates"].values()) <= {"PASS", "FAIL", "PENDING"}
    assert "PENDING" in res["gates"].values()
    # the encoded form must not let a renderer read PENDING as truthy-and-green
    assert all(isinstance(v, str) for v in res["gates"].values())


def test_weekly_writes_the_page_and_the_gates(a_scored_week):
    _, res = a_scored_week.handle("POST", "/api/weekly", body={"week_ending": "2025-12-31"})
    root = os.path.join(a_scored_week.s.out_dir, "weekly")
    assert os.path.exists(os.path.join(root, res["week"] + ".txt"))
    assert os.path.exists(os.path.join(root, res["week"] + ".json"))
    _, st = a_scored_week.handle("GET", "/api/status")
    assert st["last_gates"] == res["gates"]


def test_weekly_get_never_writes(a_scored_week):
    status, payload = a_scored_week.handle("GET", "/api/weekly")
    assert status == 200 and payload["weeks"] == []
    a_scored_week.handle("POST", "/api/weekly", body={"week_ending": "2025-12-31"})
    _, payload = a_scored_week.handle("GET", "/api/weekly")
    assert len(payload["weeks"]) == 1
    _, one = a_scored_week.handle("GET", "/api/weekly", query={"week": payload["weeks"][0]})
    assert one["gates"]


def test_rows_without_a_model_forecast_do_not_turn_sums_into_null(a_scored_week):
    _, res = a_scored_week.handle("POST", "/api/weekly",
                                  body={"week_ending": "2025-12-31",
                                        "include_backfilled": True})
    assert res["accuracy"]["model"]["wape_uncensored"] is not None


# ---- state, freshness, identity ----

def test_status_reports_what_the_cli_would(started):
    status, payload = started.handle("GET", "/api/status")
    assert status == 200
    assert payload["last_sheet_date"] == TOMORROW
    assert payload["current_artifacts_dir"] == ARTIFACTS
    assert payload["last_ingested_date"] == "2025-12-31"
    assert payload["today"] and payload["store"] == "0123"


def test_the_panel_is_reread_when_it_changes(api, tmp_path, panel_path):
    local = str(tmp_path / "panel.csv")
    frame = schema.read_panel(panel_path)
    schema.write_panel(frame[frame["date"] < "2025-12-01"], local)
    api.s.panel_path = local
    assert api._panel()["date"].max() < pd.Timestamp("2025-12-01")
    schema.write_panel(frame, local)
    assert api._panel()["date"].max() == pd.Timestamp("2025-12-31")


def test_the_items_hash_is_recomputed_not_captured_at_boot(api, tmp_path, items_doc):
    local = str(tmp_path / "items.json")
    with open(local, "w", encoding="utf-8") as fh:
        json.dump(items_doc, fh)
    api.s.items_path = local
    first = api._items_hash()
    items_doc["items"]["bread"]["price"] = 4.49
    with open(local, "w", encoding="utf-8") as fh:
        json.dump(items_doc, fh)
    assert api._items_hash() != first


def test_the_model_version_reaches_the_log(started):
    logged = shadow.read_predictions(started.s.out_dir)["model_version"].iloc[0]
    _, cfg = started.handle("GET", "/api/config")
    assert logged and cfg["model_version"] == logged


def test_the_same_inputs_forecast_the_same_numbers(api, settings):
    api.handle("POST", "/api/morning", body={"date": TOMORROW})
    first = shadow.read_predictions(api.s.out_dir)
    second_dir = api.s.out_dir + "-again"
    other = serve.Api(serve.Settings(panel=api.s.panel_path, items=api.s.items_path,
                                     artifacts=ARTIFACTS, out=second_dir,
                                     timezone="America/Chicago"))
    other.handle("POST", "/api/morning", body={"date": TOMORROW})
    second = shadow.read_predictions(second_dir)
    pd.testing.assert_series_equal(first["rec_qty"], second["rec_qty"])
    pd.testing.assert_series_equal(first["q_0.50"], second["q_0.50"])


def test_settings_defaults_match_the_morning_subparser(panel_path):
    """Read the parser, do not copy it.

    The point of this criterion is to catch drift, so asserting the same literals the server
    hard-codes would pass whatever the CLI did. shadow's parser is built inside main(), so the
    defaults are recovered by parsing a minimal `morning` argv and comparing field by field.
    """
    argv = ["morning", "--panel", "p", "--artifacts", "a", "--items", "i",
            "--date", "2026-01-01"]
    parsed = {}
    real_check = shadow._check_args

    def capture(args):
        parsed.update(vars(args))
        raise KeyboardInterrupt          # stop before the command runs

    shadow._check_args = capture
    try:
        shadow.main(argv)
    finally:
        shadow._check_args = real_check

    assert parsed, "the morning subparser did not parse"
    settings = serve.Settings(panel=panel_path, items=ITEMS_JSON, artifacts=ARTIFACTS,
                              timezone="UTC")
    assert settings.out_dir == parsed["out"]
    assert settings.store == parsed["store"]
    assert settings.max_staleness == parsed["max_staleness"]


def test_the_store_timezone_decides_today(panel_path):
    kiritimati = serve.Settings(panel=panel_path, items=ITEMS_JSON, artifacts=ARTIFACTS,
                                timezone="Pacific/Kiritimati").today()
    midway = serve.Settings(panel=panel_path, items=ITEMS_JSON, artifacts=ARTIFACTS,
                            timezone="Pacific/Midway").today()
    # 25 hours apart: a server guessing its own zone would forecast the wrong day, and a
    # FUTURE date is never backfilled, so no existing guard would refuse it
    assert kiritimati >= midway
    assert (pd.Timestamp(kiritimati) - pd.Timestamp(midway)).days in (0, 1)


# ---- the sheet files ----

def test_both_sheet_formats_are_written(started):
    for ext in ("txt", "html"):
        status, payload = started.handle("GET", f"/api/sheet/{TOMORROW}.{ext}")
        assert status == 200, ext
        assert payload["body"]
    assert "<" in started.handle("GET", f"/api/sheet/{TOMORROW}.html")[1]["body"]


def test_the_sheet_meta_carries_what_attrs_would_have_lost(started):
    """recs.attrs survives neither to_dict("records") nor the CSV."""
    _, payload = started.handle("GET", "/api/morning", query={"date": TOMORROW})
    assert payload["conditions"], "morning_sheet requires conditions and cannot re-derive them"
    assert "weather" in payload["conditions"]
    assert isinstance(payload["caveats"], list)


def test_an_unwritten_sheet_is_a_plain_404(started):
    status, payload = started.handle("GET", "/api/sheet/2025-11-04.txt")
    assert status == 404 and "no txt sheet" in payload["error"]


def test_a_sheet_path_cannot_escape_the_shadow_dir(started):
    for name in ("../../../etc/passwd", "..%2f..%2fetc%2fpasswd", "../state.json"):
        status, _ = started.handle("GET", f"/api/sheet/{name}")
        assert status == 404, name


# ---- request hygiene ----

def test_an_unknown_route_is_a_json_404(api):
    status, payload = api.handle("GET", "/api/nope")
    assert status == 404 and payload["error"].startswith("no route")


def test_a_wrong_method_is_405(api):
    status, payload = api.handle("POST", "/api/status")
    assert status == 405 and "not allowed" in payload["error"]


def test_a_missing_date_is_one_sentence(api):
    status, payload = api.handle("POST", "/api/morning", body={})
    assert status == 400 and payload["error"] == "date is required, as YYYY-MM-DD"


@pytest.mark.parametrize("bad", ["3/4/25", "2026-1-1", "tomorrow", "2026-13-40"])
def test_a_bad_date_is_named_not_guessed(api, bad):
    status, payload = api.handle("POST", "/api/morning", body={"date": bad})
    assert status == 400
    assert "YYYY-MM-DD" in payload["error"]


def test_no_error_carries_a_traceback_or_an_absolute_path(api):
    for method, path, kw in (("GET", "/api/status", {}),
                             ("POST", "/api/morning", {"body": {"date": "nope"}}),
                             ("GET", "/api/nope", {})):
        _, payload = api.handle(method, path, **kw)
        assert "Traceback" not in payload["error"]
        assert "/home/" not in payload["error"]


def test_nan_is_serialised_as_null_never_zero():
    # a par_fallback row's quantiles are NaN by design; a zero there prints "make none"
    assert serve.json_safe(float("nan")) is None
    assert serve.json_safe({"q": float("nan"), "r": 0.0}) == {"q": None, "r": 0.0}
    frame = pd.DataFrame({"a": [1.0, float("nan")], "b": ["x", None]})
    assert serve.json_safe(frame) == [{"a": 1.0, "b": "x"}, {"a": None, "b": None}]


# ---- concurrency ----

def test_a_reader_never_sees_a_half_written_row(started):
    """A GET runs beside a POST under ThreadingHTTPServer; pd.read_csv must never tear."""
    errors = []

    def writer():
        for day in pd.date_range("2025-12-20", "2025-12-27"):
            started.handle("POST", "/api/morning",
                           body={"date": str(day.date()), "backfill": True})

    def reader():
        for _ in range(40):
            try:
                status, payload = started.handle("GET", "/api/predictions")
                if status != 200:
                    errors.append(payload)
            except Exception as exc:                      # noqa: BLE001 - the point
                errors.append(repr(exc))

    threads = [threading.Thread(target=writer)] + \
              [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(shadow.read_predictions(started.s.out_dir)) > 0


def test_concurrent_posts_for_one_date_log_exactly_one_run(api):
    api.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})
    results = []

    def post():
        results.append(api.handle("POST", "/api/morning", body={"date": TOMORROW})[0])

    threads = [threading.Thread(target=post) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [200, 409, 409, 409]
    assert len(api._runs_for(pd.Timestamp(TOMORROW))) == 1


# ---- the lock file ----

def test_a_second_server_on_one_directory_is_refused(tmp_path):
    out = str(tmp_path / "shadow")
    with serve.hold_lock(out):
        with pytest.raises(schema.HtError) as exc:
            with serve.hold_lock(out):
                pass
    assert "already serving" in str(exc.value)
    # no flag overrides a live holder; the escape hatch for a recycled pid is named instead
    assert serve.LOCK_NAME in str(exc.value)


def test_a_lock_left_by_a_dead_process_is_reclaimed(tmp_path):
    """A power cut must not lock the operator out of tomorrow morning."""
    out = str(tmp_path / "shadow")
    os.makedirs(out)
    with open(os.path.join(out, serve.LOCK_NAME), "w", encoding="utf-8") as fh:
        json.dump({"pid": 999999, "start_time": "1"}, fh)
    with serve.hold_lock(out):
        with open(os.path.join(out, serve.LOCK_NAME), encoding="utf-8") as fh:
            assert json.load(fh)["pid"] == os.getpid()


def test_a_live_holder_is_never_overridden(tmp_path):
    """There is no flag for it: two servers on one append-only record is the hazard."""
    out = str(tmp_path / "shadow")
    with serve.hold_lock(out):
        with pytest.raises(schema.HtError) as exc:
            with serve.hold_lock(out):
                pass
    assert "Stop that process first" in str(exc.value)
    assert serve.LOCK_NAME in str(exc.value)


def test_the_lock_is_keyed_on_the_real_path(tmp_path):
    out = str(tmp_path / "shadow")
    os.makedirs(out)
    with serve.hold_lock(out):
        with pytest.raises(schema.HtError):
            with serve.hold_lock(os.path.join(str(tmp_path), ".", "shadow")):
                pass


def test_the_lock_is_released_on_the_way_out(tmp_path):
    out = str(tmp_path / "shadow")
    with serve.hold_lock(out):
        pass
    assert not os.path.exists(os.path.join(out, serve.LOCK_NAME))


# ---- the process boundary ----

def test_the_default_bind_is_loopback():
    args = serve.build_parser().parse_args(
        ["--panel", "p", "--items", "i", "--artifacts", "a", "--timezone", "UTC"])
    assert args.host == serve.LOOPBACK
    assert args.allow_remote is False


def test_binding_the_network_requires_saying_so(tmp_path, panel_path):
    args = serve.build_parser().parse_args(
        ["--panel", panel_path, "--items", ITEMS_JSON, "--artifacts", ARTIFACTS,
         "--timezone", "UTC", "--host", "0.0.0.0"])
    with pytest.raises(schema.HtError) as exc:
        serve.check_args(args)
    assert "no authentication" in str(exc.value)
    assert "--allow-remote" in str(exc.value)
    args.allow_remote = True
    serve.check_args(args)


def test_a_mistyped_path_is_one_line_naming_the_flag(panel_path):
    args = serve.build_parser().parse_args(
        ["--panel", "/nope/panel.csv", "--items", ITEMS_JSON, "--artifacts", ARTIFACTS,
         "--timezone", "UTC"])
    with pytest.raises(schema.HtError) as exc:
        serve.check_args(args)
    assert str(exc.value) == "--panel: no panel csv at /nope/panel.csv"


def test_an_artifacts_dir_with_no_checkpoint_is_named(tmp_path, panel_path):
    args = serve.build_parser().parse_args(
        ["--panel", panel_path, "--items", ITEMS_JSON, "--artifacts", str(tmp_path),
         "--timezone", "UTC"])
    with pytest.raises(schema.HtError) as exc:
        serve.check_args(args)
    assert "has no meta.json" in str(exc.value)


def test_a_timezone_that_is_not_one_is_refused(panel_path):
    args = serve.build_parser().parse_args(
        ["--panel", panel_path, "--items", ITEMS_JSON, "--artifacts", ARTIFACTS,
         "--timezone", "Middle/Earth"])
    with pytest.raises(schema.HtError) as exc:
        serve.check_args(args)
    assert "is not a timezone" in str(exc.value)


# ---- static files ----

def test_an_unbuilt_frontend_names_the_build_command(tmp_path):
    status, _, body = serve.static_response(str(tmp_path / "dist"), "/")
    assert status == 503
    assert "npm --prefix web run build" in body


@pytest.mark.parametrize("attack", [
    "/../secret.txt", "/../../secret.txt", "/..%2fsecret.txt", "/%2e%2e/secret.txt",
    "/./../secret.txt", "/subdir/../../secret.txt", "/%2e%2e%2fsecret.txt",
])
def test_static_files_cannot_escape_the_build_dir(tmp_path, attack):
    """The secret's own bytes must never come back, whatever the status code says."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>SPA", encoding="utf-8")
    secret = "SUPERSECRETVALUE"
    (tmp_path / "secret.txt").write_text(secret, encoding="utf-8")
    status, _, body = serve.static_response(str(dist), attack)
    raw = body if isinstance(body, bytes) else body.encode()
    assert secret.encode() not in raw, attack
    assert status in (200, 404), attack        # 200 is the SPA entry, which is fine


def test_static_serving_cannot_follow_a_symlink_out(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>SPA", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("SUPERSECRETVALUE", encoding="utf-8")
    os.symlink(str(tmp_path / "secret.txt"), str(dist / "escape.txt"))
    _, _, body = serve.static_response(str(dist), "/escape.txt")
    raw = body if isinstance(body, bytes) else body.encode()
    assert b"SUPERSECRETVALUE" not in raw


def test_an_unknown_path_falls_back_to_the_spa_entry(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    status, ctype, body = serve.static_response(str(dist), "/weekly")
    assert status == 200 and b"doctype" in body
    assert "html" in ctype


# ---- the repo's own promises ----

def test_serving_added_no_python_dependency():
    """requirements.txt says ht/ and model/ import numpy, pandas and torch and nothing else."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "requirements.txt"), encoding="utf-8") as fh:
        pinned = [ln.split("==")[0] for ln in fh if ln.strip() and not ln.startswith("#")]
    assert pinned == ["numpy", "pandas", "torch"]


def test_nothing_is_written_inside_the_repo_by_default():
    """The served default --out is relative, so it must be ignored by git."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, ".gitignore"), encoding="utf-8") as fh:
        assert "/shadow/" in fh.read().split("\n")


# ---- the guard the server cannot keep, and says so instead ----

def test_taking_the_lock_reports_whether_it_created_the_directory(tmp_path):
    """Only `morning` creates the shadow dir -- a rule a server that must lock cannot keep.

    The CLI refuses a missing --out everywhere but `morning`, because a typo would otherwise
    read an empty log and report a pilot that lost its record. A server has to take a lock
    inside that directory before it serves anything, so it cannot refuse. It reports instead.
    """
    out = str(tmp_path / "typo")
    with serve.hold_lock(out) as fresh:
        assert fresh is True
    with serve.hold_lock(out) as fresh:
        assert fresh is False


def test_status_says_when_the_record_is_a_brand_new_directory(settings):
    api = serve.Api(settings, fresh_out=True)
    os.makedirs(settings.out_dir, exist_ok=True)
    _, payload = api.handle("GET", "/api/status")
    assert payload["new_shadow_dir"] is True
    assert payload["shadow_dir"] == os.path.realpath(settings.out_dir)


def test_a_real_record_is_not_flagged_as_new(settings):
    api = serve.Api(settings, fresh_out=True)
    api.handle("POST", "/api/morning", body={"date": TOMORROW})
    _, payload = api.handle("GET", "/api/status")
    assert payload["new_shadow_dir"] is False


# ---- the vector file the browser is held to ----

def test_the_shared_entry_vectors_still_describe_the_parser(items):
    """The browser parses the returned sheet too, and two parsers drift.

    tests/fixtures/entry_vectors.json is generated from parse_entries and asserted against by
    web/src/lib/parseEntries.test.ts. If parse_entries changes and the fixture is not
    regenerated, this fails here rather than in a browser six weeks into a pilot.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "tests", "fixtures", "entry_vectors.json"),
              encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["items"].keys() == items.keys(), "regenerate: the items config moved"
    for case in doc["cases"]:
        # a case may carry its own catalogue: no shipped item name is longer than the
        # sheet's 18-character ITEM column, so truncation needs one to be exercised at all
        catalogue = case.get("items", items)
        rows, errors = shadow.parse_entries(case["lines"], catalogue)
        assert rows == case["rows"], case["label"]
        assert errors == case["errors"], case["label"]


def test_the_vectors_cover_both_outcomes():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "tests", "fixtures", "entry_vectors.json"),
              encoding="utf-8") as fh:
        doc = json.load(fh)
    assert len(doc["cases"]) >= 20
    assert any(c["errors"] for c in doc["cases"])
    assert any(c["rows"] for c in doc["cases"])
    # the guard that decides whether a returned sheet is read or lost over a column width
    assert any("items" in c for c in doc["cases"]), "no case exercises truncation"
    assert any("truncate alike" in c["label"] for c in doc["cases"])


# ---- what the post-commit audit found ----

def test_the_weekly_route_cannot_read_a_file_of_the_callers_choosing(started, tmp_path):
    """`week` named a path. os.path.join also drops the prefix on an absolute one.

    weekly_report writes weekly/<ISO year-week>.json, so anything that is not a week cannot
    name a file either -- checked before the path is built, as the sheet route does with a
    date. Reachable over HTTP: parse_qs unquotes, so ..%2f arrives already decoded.
    """
    outside = tmp_path / "secret.json"
    outside.write_text('{"aws_key": "SECRET"}', encoding="utf-8")
    os.makedirs(os.path.join(started.s.out_dir, "weekly"), exist_ok=True)
    for attack in ("../state", "../../secret", f"..{os.sep}..{os.sep}secret",
                   str(tmp_path / "secret"), "/etc/passwd", "2026-W01/../../state"):
        status, payload = started.handle("GET", "/api/weekly", query={"week": attack})
        assert status == 404, attack
        assert "aws_key" not in json.dumps(payload), attack
        assert "last_gates" not in payload, attack


def test_a_partly_landed_export_does_not_freeze_the_items_still_missing(api, tmp_path,
                                                                       panel_path):
    """The half of the Score guard that was missing.

    score_day freezes the whole file at once, so a day where eight items have landed and one
    has not would record that one as missing_data for the rest of the pilot -- catch_up calls
    the same write-once function and only disclosures follow. Refusing needs ANY row waiting,
    not all of them.
    """
    frame = schema.read_panel(panel_path)
    day = pd.Timestamp(COVERED)
    # drop one item's row for that date: the export landed for everything else
    partial = frame[~((frame["date"] == day) & (frame["item"].astype(str) == "cake"))]
    local = str(tmp_path / "partial.csv")
    schema.write_panel(partial, local)
    api.s.panel_path = local

    api.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})
    status, payload = api.handle("POST", "/api/score", body={"date": COVERED})
    assert status == 409
    assert "cake" in payload["error"]
    assert "frozen once" in payload["error"]
    assert not os.path.exists(os.path.join(api.s.out_dir, "scores", f"{COVERED}.csv"))


def test_a_day_whose_only_gap_is_a_missing_sheet_can_still_be_scored(api):
    """Waiting cannot produce a sheet that was never printed, so it is not "not yet"."""
    api.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})
    # a second date with panel data but no sheet of its own
    status, _ = api.handle("POST", "/api/score", body={"date": "2025-12-29"})
    assert status in (200, 409)
    if status == 409:
        # if it refuses, it must not claim the panel is missing data it actually has
        _, payload = api.handle("POST", "/api/score", body={"date": "2025-12-29"})
        assert "no sales data" not in payload["error"]


def test_status_reads_under_the_lock_like_every_other_reader(started):
    """status goes through read_predictions too, and it is the route the page polls."""
    errors = []

    def writer():
        for d in pd.date_range("2025-12-20", "2025-12-24"):
            started.handle("POST", "/api/morning",
                           body={"date": str(d.date()), "backfill": True})

    def reader():
        for _ in range(60):
            status, payload = started.handle("GET", "/api/status")
            if status != 200:
                errors.append(payload)

    threads = [threading.Thread(target=writer)] + \
              [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:2]


def test_concurrent_writes_across_dates_keep_every_row(api):
    """Criterion 4 as written: the exact row count, not just a parseable file."""
    dates = [str(d.date()) for d in pd.date_range("2025-12-20", "2025-12-27")]
    results = {}

    def post(day):
        results[day] = api.handle("POST", "/api/morning",
                                  body={"date": day, "backfill": True})[0]

    threads = [threading.Thread(target=post, args=(d,)) for d in dates]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(results.values()) == {200}
    logged = shadow.read_predictions(api.s.out_dir)
    items = len(api._items())
    assert len(logged) == len(dates) * items
    assert sorted(str(pd.Timestamp(d).date())
                  for d in logged["for_date"].unique()) == sorted(dates)


def test_a_live_holders_lock_survives_the_refusal(tmp_path):
    """The refused caller must not remove the lock it failed to take."""
    out = str(tmp_path / "shadow")
    with serve.hold_lock(out):
        with pytest.raises(schema.HtError):
            with serve.hold_lock(out):
                pass
        assert os.path.exists(os.path.join(os.path.realpath(out), serve.LOCK_NAME))


def test_two_simultaneous_starts_cannot_both_take_the_lock(tmp_path):
    """exists() then open() is not atomic; O_EXCL is."""
    out = str(tmp_path / "shadow")
    barrier = threading.Barrier(2)
    held = []

    def take():
        barrier.wait()
        try:
            with serve.hold_lock(out):
                held.append(1)
                pd.Timestamp.now()          # hold it briefly
                import time as _t
                _t.sleep(0.2)
        except schema.HtError:
            pass

    threads = [threading.Thread(target=take) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(held) == 1


def test_a_writer_is_not_starved_by_a_stream_of_readers(started):
    """The 5:30am sheet must not be refused because the page keeps polling."""
    stop = threading.Event()
    lock = serve.RWLock()

    def reader():
        while not stop.is_set():
            with lock.read():
                pass

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(6)]
    for t in readers:
        t.start()
    try:
        with lock.write(timeout=5.0):
            pass                            # acquiring at all is the assertion
    finally:
        stop.set()
        for t in readers:
            t.join(timeout=2)


@pytest.mark.parametrize("query,field", [
    ({"weeks": "lots"}, "weeks"),
    ({"weeks": 10 ** 6}, "weeks"),
])
def test_a_bad_weeks_value_is_a_sentence_not_a_pandas_message(started, query, field):
    status, payload = started.handle("POST", "/api/weekly",
                                     body=dict(week_ending="2026-01-04", **query))
    assert status == 400
    assert payload["error"].startswith(field)
    for leak in ("invalid literal", "nanoseconds", "Traceback", "int() with base"):
        assert leak not in payload["error"]


@pytest.mark.parametrize("route", ["/api/scores", "/api/predictions"])
def test_a_bad_date_window_is_a_sentence(started, route):
    status, payload = started.handle("GET", route, query={"to": "lunchtime"})
    assert status == 400
    assert "YYYY-MM-DD" in payload["error"]
    assert "Unknown datetime string" not in payload["error"]


def test_every_route_that_reads_the_record_guards_the_shadow_dir(api):
    """The earlier version of this test checked six of them and was named for all.

    /api/config is deliberately absent: it reports the panel, the items file and the model,
    none of which live in the shadow directory, so it has nothing to guard.
    """
    guarded = [("GET", "/api/status", {}),
               ("GET", "/api/scores", {}),
               ("GET", "/api/predictions", {}),
               ("GET", "/api/morning", {"query": {"date": TOMORROW}}),
               ("GET", "/api/entry-order", {"query": {"date": TOMORROW}}),
               ("GET", "/api/sheet/2026-01-01.txt", {}),
               ("GET", "/api/weekly", {}),
               ("POST", "/api/weekly", {"body": {"week_ending": "2026-01-04"}}),
               ("POST", "/api/enter", {"body": {"date": TOMORROW, "lines": ["bread,1,"]}}),
               ("POST", "/api/score", {"body": {"date": TOMORROW}}),
               ("POST", "/api/catch-up", {"body": {}})]
    for method, path, kw in guarded:
        status, payload = api.handle(method, path, **kw)
        assert status == 400, (path, status, payload)
        assert "no shadow directory" in payload["error"], path
    # and the one that legitimately does not need it
    assert api.handle("GET", "/api/config")[0] == 200


def test_json_safe_renders_a_type_it_does_not_know_rather_than_passing_it_on(started):
    """A value json.dumps cannot take would drop the connection at the socket."""
    import decimal
    for odd in (np.datetime64("2026-01-01"), pd.Timedelta("1D"), decimal.Decimal("1.5"),
                b"bytes", complex(1, 2), dt.time(5, 30)):
        json.dumps(serve.json_safe(odd))
    assert serve.json_safe(float("inf")) is None
    assert serve.json_safe(10 ** 30) == 10 ** 30


# ---- what the SECOND audit found ----

def _panel_without(panel_path, tmp_path, item, since=None, on=None):
    """A panel with one item's rows removed, either from a date onward or on one day."""
    frame = schema.read_panel(panel_path)
    mask = frame["item"].astype(str) == item
    if since is not None:
        mask &= frame["date"] >= pd.Timestamp(since)
    if on is not None:
        mask &= frame["date"] == pd.Timestamp(on)
    local = str(tmp_path / f"panel-no-{item}.csv")
    schema.write_panel(frame[~mask], local)
    return local


def test_catch_up_holds_back_the_day_score_refuses(api, tmp_path, panel_path):
    """The refusal names catch-up as the safe route, so catch-up must not freeze it.

    shadow.catch_up skips a date only when the panel has NO rows for it. On its own it would
    freeze exactly the partly-landed day POST /api/score refuses -- through the button the
    refusal recommends.
    """
    api.s.panel_path = _panel_without(panel_path, tmp_path, "cake", on=COVERED)
    api.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})

    status, refusal = api.handle("POST", "/api/score", body={"date": COVERED})
    assert status == 409 and "cake" in refusal["error"]

    status, payload = api.handle("POST", "/api/catch-up", body={})
    assert status == 200
    assert COVERED not in payload["scored"]
    assert payload["waiting_on_data"].get(COVERED) == ["cake"]
    assert not os.path.exists(os.path.join(api.s.out_dir, "scores", f"{COVERED}.csv")), \
        "catch-up must not freeze what score refused"


def test_an_item_the_panel_will_never_carry_does_not_block_the_day_forever(api, tmp_path,
                                                                          panel_path):
    """The new guard's own failure mode: a permanent refusal with no way out.

    An item discontinued mid-pilot, or added to the items file before the export carries it,
    has no panel row and never will -- while forecast() keeps logging a prediction for it.
    Treating that as "not landed yet" refuses the day on every attempt, forever.
    """
    api.s.panel_path = _panel_without(panel_path, tmp_path, "cake", since="2025-01-01")
    api.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})
    status, payload = api.handle("POST", "/api/score", body={"date": COVERED})
    assert status == 200, payload
    assert payload["counts"].get("missing_data") == 1, "and it is recorded, not hidden"


def test_catch_up_scores_a_day_whose_only_gap_is_an_item_that_is_gone(api, tmp_path,
                                                                     panel_path):
    api.s.panel_path = _panel_without(panel_path, tmp_path, "cake", since="2025-01-01")
    api.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})
    _, payload = api.handle("POST", "/api/catch-up", body={})
    assert COVERED in payload["scored"]
    assert not payload["waiting_on_data"]


def test_a_day_with_no_sheet_but_full_data_scores(api):
    """Deterministic, not `assert status in (200, 409)`.

    missing_sheet is permanent by construction -- no waiting produces a sheet nobody printed
    -- so such a day is as complete as it will ever be.
    """
    api.handle("POST", "/api/morning", body={"date": COVERED, "backfill": True})
    status, payload = api.handle("POST", "/api/score", body={"date": "2025-12-29"})
    assert status == 200, payload
    assert payload["counts"] == {"missing_sheet": len(api._items())}


def test_a_half_written_lock_is_never_mistaken_for_a_dead_one(tmp_path):
    """The race the first fix left open.

    O_EXCL creates an empty file and the payload is a second write, so a racer could read
    nothing, parse no pid out of it, conclude the owner was dead and delete a LIVE lock. The
    file is now linked into place complete or not at all.
    """
    out = os.path.realpath(str(tmp_path / "shadow"))
    os.makedirs(out)
    open(os.path.join(out, serve.LOCK_NAME), "w").close()          # empty: mid-write
    with pytest.raises(schema.HtError) as exc:
        with serve.hold_lock(out):
            pass
    assert "unreadable" in str(exc.value)


def test_a_corrupt_lock_is_a_sentence_not_a_traceback(tmp_path):
    out = os.path.realpath(str(tmp_path / "shadow"))
    os.makedirs(out)
    with open(os.path.join(out, serve.LOCK_NAME), "w", encoding="utf-8") as fh:
        fh.write('{"pid": "not-a-number"}')
    with pytest.raises(schema.HtError):          # not ValueError, not a traceback
        with serve.hold_lock(out):
            pass


def test_an_unreclaimable_stale_lock_says_so_instead_of_spinning(tmp_path, monkeypatch):
    out = os.path.realpath(str(tmp_path / "shadow"))
    os.makedirs(out)
    with open(os.path.join(out, serve.LOCK_NAME), "w", encoding="utf-8") as fh:
        json.dump({"pid": 999999, "start_time": "1"}, fh)

    def refuse(path):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(serve.os, "remove", refuse)
    with pytest.raises(schema.HtError) as exc:
        with serve.hold_lock(out):
            pass
    assert "by hand" in str(exc.value)
