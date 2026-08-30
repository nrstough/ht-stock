"""The split a training run used has to be one somebody chose.

`python -m model.train --panel store.csv --items items.json` used to default to
features.legacy_spec(), whose boundaries are the frozen simulator's: train_end
2024-12-31, val_start 2024-11-04. On a real 2024-2026 export that trained ten
months, validated on eight weeks the normalizer had already seen, parked two
years in "test" and exited 0. Nothing printed said so.

These tests pin the three paths that matter: a panel with no --spec is refused,
each --spec is honoured, and the no-argument command -- the one behind
results/results.json -- still resolves the frozen legacy spec unchanged.
"""
import pytest

from model import features, train

LEGACY_HASH = features.spec_hash(features.legacy_spec())


@pytest.fixture
def panel_csv(make_panel, tmp_path):
    """make_panel written to disk, because --panel takes a path, not a frame."""
    def write(start="2024-01-01", days=500, name="panel.csv"):
        path = tmp_path / name
        make_panel(("bread", "rotisserie"), start=start, days=days).to_csv(path, index=False)
        return str(path)
    return write


def test_a_supplied_panel_without_spec_is_refused(panel_csv, items_path):
    with pytest.raises(SystemExit) as exc:
        train.main(["--panel", panel_csv(), "--items", items_path])
    message = str(exc.value)
    assert "--spec auto" in message and "--spec legacy" in message
    assert "2024-12-31" in message and "2024-11-04" in message   # whose dates they are


def test_the_refusal_lands_before_the_panel_is_even_read(items_path, tmp_path):
    """A guard that runs after features.load() would report the wrong problem first."""
    missing = str(tmp_path / "not-an-export.csv")
    with pytest.raises(SystemExit) as exc:
        train.main(["--panel", missing, "--items", items_path])
    assert "--spec" in str(exc.value)


def test_dry_run_is_refused_too(panel_csv, items_path):
    """--dry-run prints the split; a dry run on the wrong one is exactly as misleading."""
    with pytest.raises(SystemExit):
        train.main(["--panel", panel_csv(), "--items", items_path, "--dry-run"])


def test_spec_auto_derives_the_boundaries_from_the_panel(panel_csv, items_path, capsys):
    path = panel_csv(start="2026-01-01", days=420)
    assert train.main(["--panel", path, "--items", items_path,
                       "--spec", "auto", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "train_end 2026-" in out and "val_start 2026-" in out
    assert LEGACY_HASH not in out


def test_spec_legacy_is_still_one_flag_away(panel_csv, items_path, capsys):
    """Selectable on purpose, and it says out loud whose dates it is using."""
    path = panel_csv(start="2024-01-01", days=500)
    assert train.main(["--panel", path, "--items", items_path,
                       "--spec", "legacy", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "train_end 2024-12-31 | val_start 2024-11-04" in out
    assert f"spec_hash {LEGACY_HASH}" in out
    assert "the simulator's fixed dates, not derived from this panel" in out
    assert "2024-01-01.." in out                  # the panel's own range, for comparison


def test_no_arguments_still_means_the_frozen_legacy_spec(capsys):
    """The provenance path: this is the command results/results.json is settled against."""
    assert train._parse_args([]).spec is None
    assert train.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "train_end 2024-12-31 | val_start 2024-11-04 | test_start 2025-01-01" in out
    assert f"spec_hash {LEGACY_HASH}" in out
    assert "supplied panel" not in out            # no panel, so nothing to warn about
