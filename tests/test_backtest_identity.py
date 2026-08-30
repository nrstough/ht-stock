"""results/results.json must still be reproducible, byte for byte.

Every dollar figure in proposal/ and poc/ is settled by model/backtest.py against the
frozen checkpoint, and the whole real-data layer was built on the promise that
parameterizing features.py did not move any of it. The path is fully deterministic -- a
frozen checkpoint under torch.no_grad and a closed-form ridge solve, no RNG anywhere --
which is what makes byte-identity a testable claim rather than an aspiration.

Marked slow because it loads torch and replays three years; it runs in CI as its own step.
It writes into tmp_path, never into results/.
"""
import hashlib
import json
import os

import pytest

from tests.conftest import ARTIFACTS, RESULTS_JSON, SYNTH_CSV

FROZEN = (RESULTS_JSON, os.path.join(ARTIFACTS, "demandnet.pt"),
          os.path.join(ARTIFACTS, "meta.json"), SYNTH_CSV)


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


@pytest.fixture
def frozen_unchanged():
    """The provenance guard: nothing in this test may move a frozen artifact."""
    before = {p: _md5(p) for p in FROZEN}
    yield
    assert {p: _md5(p) for p in FROZEN} == before


@pytest.mark.slow
def test_backtest_with_default_arguments_reproduces_results_json(tmp_path,
                                                                 frozen_unchanged):
    from model import backtest

    out = str(tmp_path / "results.json")
    backtest.main(["--out", out])
    with open(out, "rb") as fh:
        produced = fh.read()
    with open(RESULTS_JSON, "rb") as fh:
        published = fh.read()
    assert produced == published, (
        f"model.backtest wrote {len(produced)} bytes against the published "
        f"{len(published)}; the proposal's figures are no longer reproducible")


@pytest.mark.slow
def test_the_backtest_is_deterministic_across_two_runs(tmp_path, frozen_unchanged):
    from model import backtest

    first, second = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    backtest.main(["--out", first])
    backtest.main(["--out", second])
    assert open(first, "rb").read() == open(second, "rb").read()


@pytest.mark.slow
def test_evaluate_scores_the_frozen_checkpoint_without_simulator_truth(tmp_path,
                                                                       synth_raw,
                                                                       items_path,
                                                                       frozen_unchanged):
    """The real-data path, run against the one panel whose answers are already published."""
    from ht import schema
    from model import evaluate

    panel = str(tmp_path / "panel.csv")
    schema.write_panel(schema.conform(synth_raw), panel)
    res = evaluate.evaluate(panel, ARTIFACTS, items_path, split="test",
                            out=str(tmp_path / "evaluate.json"))

    with open(RESULTS_JSON, encoding="utf-8") as fh:
        published = json.load(fh)["summary"]["status_quo"]
    # MEASURED, from produced - sold, with no simulator column read anywhere
    assert res["measured"]["waste_observed_units"] == pytest.approx(
        published["waste_units"], rel=1e-4)
    assert res["measured"]["waste_observed_cost"] == pytest.approx(
        published["waste_cost"], rel=1e-4)

    cov = res["coverage_of_scoring"]
    assert cov["n_rows_scored"] == 3276
    assert res["bounds"]["lost_margin_upper"] is None
    assert res["bounds"]["waste_saving_lower_cost"] > 0
    text = evaluate.format_report(res)
    assert "MEASURED" in text
