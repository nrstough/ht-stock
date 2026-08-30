"""Three sellout rules, one chosen explicitly, no fallback chain.

The design tunes for PRECISION because of the asymmetry the loss has by construction, which
is argued rather than measured: missing a sellout leaves the model fitting censored sales, so
production runs low and fails safe, while inventing one widens the fitted distribution and
inflates production -- pure added waste, and invisible, because no sellouts are then observed.
Neither side's size is pinned by a test here; docs/REAL_DATA_READINESS.md reports the one
experiment that measured the first direction. So NaN production must yield stockout_known=0
and never stockout=1, and the two rejected rules must be refused by name rather than quietly
approximated.
"""
import json

import numpy as np
import pandas as pd
import pytest

from ht import config, ingest, schema

ITEM_EACH = {"name": "B", "dept": "D", "price": 4.0, "cost": 1.0, "batch": 1.0,
             "unit": "each", "sellout_tolerance": None, "shelf_life_days": 1}
ITEM_LB = dict(ITEM_EACH, unit="lb", continuous=True)


def _frame(rows):
    """rows: (date, item, sold, produced) or (date, item, sold, produced, row_status)."""
    out = []
    for row in rows:
        date, item, sold, produced = row[:4]
        status = row[4] if len(row) > 4 else "ok"
        out.append(dict(store="0123", date=pd.Timestamp(date), item=item, sold=sold,
                        produced=produced, row_status=status))
    return pd.DataFrame(out)


def _mapping(rule="produced_vs_sold", **sellout):
    return {"sellout": dict(rule=rule, **sellout)}


def test_produced_vs_sold_flags_a_sellout_and_spares_a_served_day():
    panel, report = ingest.derive_sellout(
        _frame([("2026-01-01", "b", 40.0, 40.0), ("2026-01-02", "b", 30.0, 40.0)]),
        _mapping(), {"b": ITEM_EACH})
    assert list(panel.stockout) == [1, 0]
    assert list(panel.stockout_known) == [1, 1]
    assert set(panel.sellout_source) == {"produced_vs_sold"}
    assert report["rule"] == "produced_vs_sold"
    assert report["latency_days"] == 1               # computable on yesterday's data


def test_nan_production_is_unknown_never_a_sellout():
    panel, report = ingest.derive_sellout(
        _frame([("2026-01-01", "b", 40.0, np.nan)]), _mapping(), {"b": ITEM_EACH})
    assert list(panel.stockout) == [0]
    assert list(panel.stockout_known) == [0]         # "we have no idea", not "did not sell out"
    assert report["unknown_days"] == 1


def test_zero_production_is_not_a_demand_observation():
    panel, _ = ingest.derive_sellout(
        _frame([("2026-01-01", "b", 0.0, 0.0)]), _mapping(), {"b": ITEM_EACH})
    assert list(panel.row_status) == ["missing"]     # sold=0 with no supply says nothing
    assert list(panel.stockout) == [0]
    assert list(panel.stockout_known) == [1]


def test_a_weighed_item_gets_a_tolerance_because_pounds_never_match_exactly():
    rows = [("2026-01-01", "h", 49.7, 50.0), ("2026-01-02", "h", 44.0, 50.0)]
    panel, _ = ingest.derive_sellout(_frame(rows), _mapping(),
                                     {"h": dict(ITEM_LB, sellout_tolerance=None)})
    assert list(panel.stockout) == [1, 0]
    strict, _ = ingest.derive_sellout(_frame(rows), _mapping(),
                                      {"h": dict(ITEM_LB, sellout_tolerance=0.0)})
    assert list(strict.stockout) == [0, 0]


def test_a_closed_day_is_never_a_sellout():
    panel, _ = ingest.derive_sellout(
        _frame([("2026-01-01", "b", 0.0, 0.0, "closed")]), _mapping(), {"b": ITEM_EACH})
    assert list(panel.stockout) == [0]


def test_rule_none_is_a_supported_mode_not_a_broken_state():
    panel, report = ingest.derive_sellout(
        _frame([("2026-01-01", "b", 40.0, 40.0), ("2026-01-02", "b", 30.0, np.nan)]),
        _mapping("none"), {"b": ITEM_EACH})
    assert list(panel.stockout) == [0, 0]
    assert list(panel.stockout_known) == [0, 0]
    assert set(panel.sellout_source) == {"none"}
    assert report["known_share"] == 0.0
    assert report["latency_days"] == 0
    # cens = stockout * stockout_known is then zero everywhere and the loss degrades to
    # ordinary pinball with no code change
    assert float((panel.stockout * panel.stockout_known).sum()) == 0.0


def test_the_flag_rule_separates_coverage_from_accuracy():
    rows = _frame([("2026-01-01", "b", 40.0, np.nan), ("2026-01-05", "b", 40.0, np.nan),
                   ("2026-02-01", "b", 40.0, np.nan)])
    oos = pd.DataFrame({"item": ["b"], "date": [pd.Timestamp("2026-01-05")]})
    panel, report = ingest.derive_sellout(
        rows, _mapping("flag", coverage_start="2026-01-01", coverage_end="2026-01-31"),
        {"b": ITEM_EACH}, aux={"oos": oos})
    assert list(panel.stockout) == [0, 1, 0]
    assert list(panel.stockout_known) == [1, 1, 0]   # outside the window nobody was looking
    assert report["rule"] == "flag"


def test_the_flag_rule_without_an_oos_log_raises():
    with pytest.raises(schema.IngestError):
        ingest.derive_sellout(_frame([("2026-01-01", "b", 1.0, np.nan)]),
                              _mapping("flag", coverage_start="2026-01-01",
                                       coverage_end="2026-01-31"),
                              {"b": ITEM_EACH}, aux={"oos": None})


@pytest.mark.parametrize("rule", ["waste_zero", "last_sale_gap"])
def test_the_rejected_rules_are_refused_by_name(rule, tmp_path):
    doc = {"schema": "ht-source-mapping/1", "store": "0123",
           "files": [{"role": "sales", "path": "S.CSV"}],
           "columns": {"sales": {"date": "D", "item_code": "I", "units": "U"}},
           "date": {"format": "%m/%d/%y"}, "sellout": {"rule": rule}}
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(schema.MappingError) as exc:
        config.load_mapping(str(path))
    assert rule in str(exc.value)


def test_produced_vs_sold_reproduces_the_simulator_stockout_column(synth_panel):
    """The rehearsal's ground truth: with eps=0 the rule and the simulator agree on every
    open row. It agrees because sim/generate.py defines them as the same event, which is
    exactly why a green rehearsal licenses nothing about which rule fits a real store."""
    frame = synth_panel[["store", "date", "item", "sold", "produced", "row_status"]].copy()
    items = {key: dict(ITEM_EACH, sellout_tolerance=0.0)
             for key in synth_panel["item"].unique()}
    panel, report = ingest.derive_sellout(frame, _mapping(), items)
    open_rows = synth_panel["row_status"] == "ok"
    assert int(open_rows.sum()) == 9837
    disagreements = int((panel.loc[open_rows, "stockout"].to_numpy()
                         != synth_panel.loc[open_rows, "stockout"].to_numpy()).sum())
    assert disagreements == 0
    assert report["known_share"] == 1.0


def test_the_shipped_lb_tolerance_moves_the_flag_and_the_test_must_say_so(synth_panel, items):
    """config/items.example.json leaves hotbar-lb's tolerance null, which resolves to 0.5 lb
    -- right for a real scale, and NOT the simulator's `sold >= produced`. The disagreement
    is small, one-sided and on one item; it is recorded here so nobody reads the zero above
    as a claim about the shipped config."""
    frame = synth_panel[["store", "date", "item", "sold", "produced", "row_status"]].copy()
    panel, _ = ingest.derive_sellout(frame, _mapping(), items)
    open_rows = synth_panel["row_status"] == "ok"
    diff = (panel.loc[open_rows, "stockout"].to_numpy()
            != synth_panel.loc[open_rows, "stockout"].to_numpy())
    assert set(panel.loc[open_rows][diff]["item"]) == {"hotbar-lb"}
    assert int(diff.sum()) < 20                 # extra flags only, and only on the lb item
