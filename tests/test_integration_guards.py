"""The guards that only exist between two files, and the honesty claims that span them.

Every fix in this repo lives in one module, but four of them are only true if a SECOND module
agrees: a --spec refusal that model.train makes and model.backtest does not is not a policy,
it is a gap with a test in front of it. So these tests are deliberately cross-file. They check
that the two CLIs refuse the same thing in the same words, that the three layers guarding a
district export offer the same way out, that a checkpoint with no recorded spec cannot be
quietly scored on the simulator's boundaries by either scoring path, and that ingest's report
does not name a repair it did not make.
"""
import os
import subprocess

import pandas as pd
import pytest

from ht import schema
from model import backtest, features

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, ".venv", "bin", "python")


def _run(*args):
    return subprocess.run([PY, *args], capture_output=True, text=True, cwd=REPO, timeout=300)


# ---- the split spec: one refusal, two commands ----

def test_backtest_refuses_a_supplied_panel_with_no_spec(make_panel, tmp_path):
    """The worst instance of the trap: this command's output is the dollar saving."""
    panel = tmp_path / "panel.csv"
    schema.write_panel(make_panel(days=400, start="2026-01-01"), str(panel))
    with pytest.raises(SystemExit) as exc:
        backtest.main(["--panel", str(panel), "--settlement", "observed",
                       "--out", str(tmp_path / "out.json")])
    msg = str(exc.value)
    assert "--spec auto" in msg and "--spec legacy" in msg
    assert "2024-12-31" in msg and "2024-11-04" in msg


def test_train_and_backtest_refuse_in_the_same_words(make_panel, tmp_path):
    """Two copies of the legacy paragraph would eventually disagree about the frozen dates."""
    from model import train

    panel = tmp_path / "panel.csv"
    schema.write_panel(make_panel(days=400, start="2026-01-01"), str(panel))
    with pytest.raises(SystemExit) as a:
        train.main(["--panel", str(panel), "--artifacts", str(tmp_path / "art")])
    with pytest.raises(SystemExit) as b:
        backtest.main(["--panel", str(panel), "--settlement", "observed",
                       "--out", str(tmp_path / "out.json")])
    shared = features.spec_refusal(str(panel), "")
    legacy_paragraph = shared.split("--spec legacy", 1)[1]
    assert legacy_paragraph in str(a.value)
    assert legacy_paragraph in str(b.value)


def test_the_settlement_refusal_comes_before_the_spec_refusal(synth_panel, tmp_path):
    """A panel that cannot be settled at all is not helped by first choosing a split for it."""
    panel = tmp_path / "panel.csv"
    schema.write_panel(synth_panel, str(panel))
    with pytest.raises(SystemExit) as exc:
        backtest.main(["--panel", str(panel), "--settlement", "sim",
                       "--out", str(tmp_path / "out.json")])
    assert "model.evaluate" in str(exc.value)


def test_backtest_with_no_panel_still_means_legacy():
    """`python -m model.backtest` is the provenance of results/results.json."""
    assert backtest._guard_spec(None, None) is None


def test_spec_legacy_is_still_one_flag_away(make_panel, tmp_path):
    panel = tmp_path / "panel.csv"
    schema.write_panel(make_panel(days=400, start="2026-01-01"), str(panel))
    assert backtest._guard_spec(str(panel), "legacy") is None
    assert backtest._guard_spec(str(panel), "auto") is None


def test_backtest_auto_honours_the_split_flags(make_panel):
    """--no-test was unreachable here, so a store needing it could not backtest at all."""
    df = make_panel(days=110, start="2026-01-01")
    spec = features.spec_for_panel(df, no_test=True)
    assert spec["test_start"] is None
    args = backtest._parse_args(["--panel", "p.csv", "--spec", "auto", "--no-test"])
    assert args.no_test is True and args.allow_short is False


# ---- a checkpoint with no recorded spec ----

def test_a_recorded_spec_is_returned_verbatim(make_panel):
    spec = features.spec_for_panel(make_panel(days=400, start="2026-01-01"))
    got = features.spec_from_meta({"spec": spec}, make_panel(days=400, start="2026-01-01"))
    assert got is spec


def test_no_recorded_spec_still_works_on_the_frame_legacy_describes(synth_panel):
    """The frozen checkpoint on the simulator's own panel: the assumption holds, so it stands."""
    got = features.spec_from_meta({}, synth_panel)
    assert got["train_end"] == features.TRAIN_END


def test_no_recorded_spec_is_refused_on_a_panel_legacy_does_not_describe(make_panel):
    """A 2026 export fed to the frozen checkpoint: the trend covariate leaves [0, 1]."""
    with pytest.raises(features.SpecMismatch) as exc:
        features.spec_from_meta({}, make_panel(days=400, start="2026-06-01"), "model/artifacts")
    msg = str(exc.value)
    assert "records no feature spec" in msg and "ASSUMING" in msg
    assert "trend covariate" in msg
    assert "--spec auto" in msg


def test_spec_fit_problems_names_an_unknown_weather_kind(make_panel):
    df = make_panel(days=200, start="2023-06-01", weather="unknown")
    problems = features.spec_fit_problems(features.legacy_spec(), df)
    assert any("weather kind" in p for p in problems)


def test_spec_fit_problems_is_silent_where_the_spec_fits(synth_panel):
    assert features.spec_fit_problems(features.legacy_spec(), synth_panel) == []


def test_evaluate_refuses_the_frozen_checkpoint_on_a_panel_it_cannot_describe(
        make_panel, tmp_path, items_path):
    """End to end: one line to stderr and exit 1, not a plausible-looking accuracy table."""
    panel = tmp_path / "panel.csv"
    schema.write_panel(make_panel(item_keys=("bread",), days=400, start="2026-06-01"),
                       str(panel))
    r = _run("-m", "model.evaluate", "--panel", str(panel),
             "--artifacts", os.path.join(REPO, "model", "artifacts"),
             "--items", items_path, "--split", "test")
    assert r.returncode == 1, r.stdout
    assert "records no feature spec" in r.stderr
    assert "\n" in r.stderr and "Traceback" not in r.stderr


def test_shadow_refuses_the_frozen_checkpoint_on_a_panel_it_cannot_describe(
        make_panel, tmp_path, items_path):
    panel = tmp_path / "panel.csv"
    schema.write_panel(make_panel(item_keys=("bread",), days=400, start="2026-06-01"),
                       str(panel))
    r = _run("-m", "model.shadow", "morning", "--panel", str(panel),
             "--artifacts", os.path.join(REPO, "model", "artifacts"),
             "--items", items_path, "--date", "2027-01-01",
             "--out", str(tmp_path / "shadow"), "--backfill")
    assert r.returncode == 1, r.stdout
    assert "records no feature spec" in r.stderr
    assert "Traceback" not in r.stderr


def test_evaluate_says_in_its_caveats_that_the_spec_was_assumed(synth_panel, tmp_path,
                                                               items_path):
    """It scores -- the assumption holds here -- but the report must not present it as a record."""
    from model import evaluate

    panel = tmp_path / "panel.csv"
    schema.write_panel(synth_panel, str(panel))
    res = evaluate.evaluate(str(panel), os.path.join(REPO, "model", "artifacts"), items_path,
                            split="test")
    assert any("records NO feature spec" in c for c in res["caveats"])
    assert any("simulator's own dates" in c for c in res["caveats"])


# ---- one condition, three layers ----

def test_all_three_multi_store_guards_offer_the_same_way_out(make_panel, tmp_path, items):
    from ht import validate as ht_validate

    two = pd.concat([make_panel(days=140), make_panel(days=140).assign(store="0456")],
                    ignore_index=True)
    report = ht_validate.validate(schema.conform(two), items)
    finding = [f for f in report["findings"] if f.check == "multi_store"]
    assert finding and finding[0].level == "error"
    assert schema.ONE_STORE_REMEDY in finding[0].message

    with pytest.raises(features.MultiStorePanel) as exc:
        features.build(schema.conform(two), spec=features.spec_for_panel(make_panel(days=140)))
    assert schema.ONE_STORE_REMEDY in str(exc.value)


def test_the_shared_remedy_names_the_flag_and_the_mapping_key_that_make_it_work():
    """--store is useless without columns.<role>.store mapped; saying one without the other
    sends somebody to a flag that silently does nothing."""
    assert "--store" in schema.ONE_STORE_REMEDY
    assert "columns.<role>.store" in schema.ONE_STORE_REMEDY


# ---- ingest's report must not name a repair it did not make ----

def _negatives_report(policy):
    from ht import ingest

    panel = pd.DataFrame({"sold": [-3.0, 1.0, 2.0, -1.0], "row_status": ["ok"] * 4})
    mapping = {"negatives": {"policy": policy, "max_share": 0.9}}
    report = {}
    out = ingest._apply_negatives(panel, mapping, report)
    return report, out


def test_negatives_clipped_counts_only_rows_that_were_clipped():
    report, out = _negatives_report("keep")
    assert report["negatives_seen"] == 2
    assert report["negatives_clipped"] == 0
    assert (out["sold"] < 0).sum() == 2      # the panel still carries them, so 'clipped' is 0


def test_negatives_clipped_and_seen_agree_when_clipping_happened():
    report, out = _negatives_report("clip_zero")
    assert report["negatives_seen"] == report["negatives_clipped"] == 2
    assert (out["sold"] < 0).sum() == 0


def test_the_validation_page_says_nothing_was_clipped_when_nothing_was(make_panel, items):
    from ht import validate as ht_validate

    report = ht_validate.validate(
        make_panel(days=140), items,
        ingest_report=dict(negatives_seen=4, negatives_clipped=0, files=[],
                           duplicates_collapsed=0, grid_rows_inserted={}, closures_applied={},
                           weather={}))
    info = [f for f in report["findings"] if f.check == "repair_negatives_clipped"]
    assert info and "nothing was clipped" in info[0].message


# ---- a mistyped --panel is a person's mistake, not a traceback ----

@pytest.mark.parametrize("argv", [
    ["-m", "model.features", "--panel", "/nope/panel.csv"],
    ["-m", "model.train", "--panel", "/nope/panel.csv", "--spec", "auto",
     "--artifacts", "/tmp/ht-nope-artifacts"],
    ["-m", "model.backtest", "--panel", "/nope/panel.csv", "--spec", "auto",
     "--settlement", "observed", "--out", "/tmp/ht-nope.json"],
])
def test_a_missing_panel_is_one_line_naming_the_path(argv):
    r = _run(*argv)
    assert r.returncode == 1
    assert "no panel csv at /nope/panel.csv" in (r.stdout + r.stderr)
    assert "Traceback" not in (r.stdout + r.stderr)


def test_load_raises_a_named_error_rather_than_a_pandas_one():
    with pytest.raises(features.PanelNotFound) as exc:
        features.load("/nope/panel.csv")
    assert "/nope/panel.csv" in str(exc.value)


# ---- the one generator that can destroy a frozen artifact ----

def test_sim_generate_refuses_to_overwrite_the_frozen_csv():
    """`python -m sim.generate` used to write data/store_synth.csv with no way to redirect."""
    from sim import generate

    with pytest.raises(SystemExit) as exc:
        generate.main([])
    assert "provenance" in str(exc.value) and "--out" in str(exc.value)


def test_sim_generate_still_reproduces_the_frozen_csv(tmp_path):
    """The md5 guard says the file has not moved; this says the generator still produces it."""
    import filecmp

    from sim import generate

    out = tmp_path / "store_synth.csv"
    generate.main(["--out", str(out)])
    frozen = os.path.join(REPO, "data", "store_synth.csv")
    assert filecmp.cmp(str(out), frozen, shallow=False)


class _ClosedTerminal:
    """A tty whose reader is at end of input -- what Ctrl-D, or closing the window, leaves."""

    def isatty(self):
        return True

    def readline(self, *a):
        return ""

    def fileno(self):
        raise OSError("not a real terminal")


def test_ctrl_d_at_the_entry_prompt_is_not_a_traceback(tmp_path, items_path, monkeypatch,
                                                       capsys):
    """Ctrl-C was handled and Ctrl-D was not; both are a person leaving a 6am terminal."""
    from model import shadow

    monkeypatch.setattr("sys.stdin", _ClosedTerminal())
    assert shadow.main(["enter", "--items", items_path, "--date", "2025-12-30",
                        "--out", str(tmp_path)]) == 1
    assert "abandoned; nothing was written" in capsys.readouterr().err
    assert not (tmp_path / "overrides").exists()


# ---- results/results.json: reproducible by the frozen run, and only by it ----

def test_only_the_frozen_configuration_may_write_results_json(tmp_path):
    """README said all three frozen writers refuse; this one wrote on any flag but --out.

    `python -m model.backtest --policies dl,naive` replaced a six-policy file with a
    two-policy one and printed nothing but "wrote". The plain command has to keep working --
    it is the provenance of the file -- so the refusal is on the configuration, not the path.
    """
    frozen = os.path.join(REPO, "results", "results.json")
    before = open(frozen, "rb").read()

    # --no-test is refused one step earlier, by the guard below, so it is tested there
    for argv in (["--policies", "dl,naive"], ["--settlement", "observed"],
                 ["--calib-window", "90"], ["--allow-short"],
                 ["--artifacts", str(tmp_path)], ["--test-days", "30"]):
        with pytest.raises(SystemExit) as exc:
            backtest.main(argv)
        assert "results/results.json" in str(exc.value), argv
        assert argv[0] in str(exc.value), argv
    assert open(frozen, "rb").read() == before


def test_the_frozen_run_itself_is_not_refused():
    args = backtest._parse_args([])
    assert backtest._not_the_frozen_run(args) == []
    assert backtest._guard_frozen_out(args) is None


def test_any_configuration_may_write_somewhere_else(tmp_path):
    args = backtest._parse_args(["--policies", "dl", "--out", str(tmp_path / "x.json")])
    assert backtest._guard_frozen_out(args) is None


def test_backtest_refuses_no_test_before_it_replays_anything(make_panel, tmp_path):
    """--no-test says there is no held-out window; a backtest is a replay OVER one.

    It used to reach _chart_window with an empty test mask and die on
    "NaTType does not support strftime" after minutes of forecasting.
    """
    panel = tmp_path / "panel.csv"
    schema.write_panel(make_panel(days=140, start="2026-01-01"), str(panel))
    with pytest.raises(SystemExit) as exc:
        backtest.main(["--panel", str(panel), "--spec", "auto", "--no-test",
                       "--settlement", "observed", "--out", str(tmp_path / "out.json")])
    msg = str(exc.value)
    assert "--no-test" in msg and "--test-days" in msg
    assert "NaT" not in msg


def test_the_frozen_file_is_not_replaced_even_by_the_frozen_command(tmp_path):
    """The configuration guard cannot see a checkpoint retrained with --force-frozen.

    After that, the plain command's own numbers change and no flag says so, so the last
    check is the bytes: reproducing them writes, and not reproducing them refuses.
    """
    frozen = open(os.path.join(REPO, "results", "results.json")).read()
    assert backtest._guard_frozen_bytes(backtest.DEFAULT_OUT, frozen) is None

    with pytest.raises(SystemExit) as exc:
        backtest._guard_frozen_bytes(backtest.DEFAULT_OUT, frozen.replace("1", "2", 1))
    assert "no longer reproduces" in str(exc.value)
    # anywhere else is the caller's own file and is written without comment
    assert backtest._guard_frozen_bytes(str(tmp_path / "mine.json"), "{}") is None
