"""The dress rehearsal in miniature: raw export -> panel -> gate -> features -> scoring.

scripts/rehearse.sh runs this chain for real, on three synthetic years, with training. This
runs the same chain on 160 days in a couple of seconds and without torch, so a break in a
seam between two agents' modules shows up in CI rather than in the rehearsal log. What it
proves is narrow and worth stating: the handoffs line up. It says nothing about whether the
model is any good, and nothing at all about which sellout rule fits a real store.
"""
import numpy as np
import pandas as pd
import pytest

from ht import ingest, schema, validate
from model import features
from tests.test_ingest import export, ingested            # noqa: F401  (fixtures)


def test_the_ingested_panel_passes_the_validator_gate(ingested):
    panel, _, exp = ingested
    report = validate.validate(panel, exp["items"], exp["mapping"])
    errors = sorted({f.check for f in report["findings"] if f.level == "error"})
    assert errors == [], errors


def test_the_rehearsal_finds_the_dirt_it_planted(ingested):
    """A rehearsal where the validator finds nothing has proved nothing."""
    panel, _, exp = ingested
    report = validate.validate(panel, exp["items"], exp["mapping"])
    warnings = {f.check for f in report["findings"] if f.level == "warning"}
    assert "weather_unknown" in warnings          # the unresolvable condition
    infos = {f.check for f in report["findings"] if f.level == "info"}
    assert "repair_missing" in infos              # the five-day export gap


def test_features_build_on_the_ingested_panel(ingested):
    panel, _, _ = ingested
    spec = features.spec_for_panel(panel)
    b = features.build(panel, spec=spec)
    assert b["ctx"].shape[1:] == (28, 6)          # the frozen encoder shape, unchanged
    assert b["cov"].shape[1] == len(_cov_width(b))
    assert set(b["items"]) <= {"bread", "rotisserie", "hotbar-lb"}
    assert b["sellout_source"] == "produced_vs_sold"
    assert b["censoring_known"] is True
    for split in ("train", "val", "test"):
        assert int((b["split"] == split).sum()) > 0


def _cov_width(b):
    return np.zeros(max(hi for _, hi in b["cov_layout"].values()))


def test_a_short_panel_fails_at_the_data_layer_with_a_data_message(ingested):
    """A bad panel must fail with a data message, not twelve epochs later inside torch."""
    panel, _, exp = ingested
    short = panel[panel.date < "2025-03-01"]
    report = validate.validate(short, exp["items"], exp["mapping"])
    assert report["ok"] is False
    assert "insufficient_history" in {f.check for f in report["findings"]
                                      if f.level == "error"}
    with pytest.raises(features.InsufficientHistory):
        features.build(short, spec=features.spec_for_panel(short))


def test_the_degraded_no_sellout_path_runs_end_to_end(export):
    """The likely real case: sales only, no production sheet, rule 'none'."""
    import json
    mapping = json.loads(json.dumps(export["mapping"]))
    mapping["files"] = [f for f in mapping["files"] if f["role"] != "production"]
    mapping["sellout"] = dict(mapping["sellout"], rule="none")

    panel, report = ingest.ingest(mapping, export["items"], root=export["root"])
    assert report["sellout"]["known_share"] == 0.0
    assert set(panel["sellout_source"]) == {"none"}

    gate = validate.validate(panel, export["items"], mapping)
    assert gate["ok"] is True                     # a supported mode, not a broken state
    assert "no_sellout_signal" in {f.check for f in gate["findings"]
                                   if f.level == "warning"}

    b = features.build(panel, spec=features.spec_for_panel(panel))
    assert float(b["cens"].max()) == 0.0          # the loss degrades to ordinary pinball
    assert b["censoring_known"] is False


def test_the_panel_round_trips_through_the_csv_handoff(ingested, tmp_path):
    panel, _, _ = ingested
    path = str(tmp_path / "panel.csv")
    schema.write_panel(panel, path)
    back = schema.read_panel(path)
    assert schema.panel_hash(back) == schema.panel_hash(panel)
    assert list(back.columns) == list(schema.NAMES)
