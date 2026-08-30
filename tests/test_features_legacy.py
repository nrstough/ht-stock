"""THE GUARD that keeps results/results.json reachable.

model/artifacts/demandnet.pt was trained on one exact set of arrays, and every dollar
figure in the proposal is settled against it. features.py had to be parameterized to accept
a real store's panel; these hashes are how we know the parameterization did not move a
single float on the frozen path. A failure here is not a style problem -- it means the
published numbers can no longer be reproduced.

The hashes are of the C-contiguous bytes of each array, recorded from git HEAD's features.py
before the refactor.
"""
import hashlib

import numpy as np
import pytest

from model import features

GOLDEN = {
    "ctx": "6a5f4a1d885c5272",
    "cov": "f0ab7c0b733ff39e",
    "y": "f6c16e3e5c3a6442",
    "cens": "00560133617a55b3",
}
SPLITS = {"train": 5796, "val": 513, "test": 3276}


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def legacy():
    return features.build()          # no arguments: exactly what backtest.py runs


def test_default_build_is_the_legacy_spec(legacy):
    assert legacy["spec"] == features.legacy_spec()


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_arrays_are_byte_identical_to_the_published_run(legacy, key):
    assert _hash(legacy[key]) == GOLDEN[key]


def test_shapes_and_split_counts(legacy):
    assert legacy["ctx"].shape == (9585, 28, 6)
    assert legacy["cov"].shape == (9585, 35)
    assert {s: int((legacy["split"] == s).sum()) for s in SPLITS} == SPLITS
    assert legacy["cens"].mean() == pytest.approx(0.19593113660812378)


def test_the_holiday_countdown_covariate_is_dead_on_the_frozen_path(legacy):
    """cov column 26 is identically zero in the published model.

    features.py filters cal[cal.holiday != ""] before the fillna(""), and NaN != "" is
    True, so every date read as a holiday and the countdown collapsed to 0. The checkpoint
    was trained that way. legacy_spec pins holiday_countdown="off" so the behaviour is
    reproduced deliberately rather than by a NaN accident in one CSV -- which matters,
    because ht.schema.conform fills that NaN and would otherwise wake the covariate up.
    """
    assert legacy["spec"]["holiday_countdown"] == "off"
    assert np.unique(legacy["cov"][:, 26]).tolist() == [0.0]


def test_the_legacy_spec_pins_every_frozen_value():
    spec = features.legacy_spec()
    assert spec["context_days"] == 28
    assert spec["train_end"] == "2024-12-31"
    assert spec["val_start"] == "2024-11-04"
    assert spec["test_start"] == "2025-01-01"
    assert spec["stats_scope"] == "train_val"     # the documented normalization leak
    assert spec["stats_end"] == "2024-12-31"
    assert spec["trend_days"] == 1095.0           # == the panel's own (max-min).days
    assert spec["fourier_harmonics"] == 2
    assert spec["require_contiguous_context"] is False
    assert spec["min_item_train_days"] == 0
    assert spec["unknown_vocab"] == "raise"
    assert list(spec["holiday_names"]) == features.HOLIDAY_NAMES
    assert list(spec["weather_kinds"]) == features.WEATHER_KINDS


def test_the_trend_denominator_is_the_panels_own_span(legacy, synth_panel):
    # 1095 is not a magic number: it is (max - min).days on this panel, so the derived
    # formula max(H - 1, 1) would give the identical value here
    span = (synth_panel.date.max() - synth_panel.date.min()).days
    assert span == 1095 == int(features.legacy_spec()["trend_days"])


def test_legacy_is_the_default_for_any_frame_carrying_simulator_columns(synth_raw):
    b = features.build(synth_raw)
    assert b["spec"] == features.legacy_spec()
    assert _hash(b["cov"]) == GOLDEN["cov"]


def test_the_spec_hash_is_stable():
    assert features.spec_hash(features.legacy_spec()) == "792530c0480b"


def test_lags_are_attached_by_build(legacy):
    # baselines.fit_predict_ridge reads b["lags"], which build() never used to produce
    assert legacy["lags"].shape == (9585, 4)
    assert legacy["lags"].dtype == np.float32


def test_the_covariate_layout_is_the_frozen_block_order(legacy):
    layout = legacy["cov_layout"]
    assert layout["dow"] == [0, 7]
    assert layout["holiday"] == [11, 25]
    assert layout["weather"] == [28, 32]
    assert layout["trend"] == [34, 35]


def test_the_context_channels_are_the_frozen_six(legacy):
    assert legacy["ctx_channels"] == ["sales_z", "stockout", "is_closed", "tmax_z",
                                      "rain", "snow"]


def test_assert_compatible_accepts_the_frozen_meta_json(legacy, repo):
    import json
    import os
    with open(os.path.join(repo, "model", "artifacts", "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert "spec" not in meta                     # the frozen artifact predates the spec key
    features.assert_compatible(meta, legacy)      # falls back to the dimension-only check


def test_assert_compatible_catches_a_permuted_holiday_vocabulary(legacy):
    meta = dict(items=legacy["items"], taus=list(legacy["taus"]),
                ctx_dim=6, cov_dim=35, spec=dict(legacy["spec"]),
                spec_hash=legacy["spec_hash"])
    meta["spec"] = dict(meta["spec"])
    meta["spec"]["holiday_names"] = list(reversed(legacy["spec"]["holiday_names"]))
    meta["spec_hash"] = features.spec_hash(meta["spec"])
    with pytest.raises(features.SpecMismatch) as exc:
        features.assert_compatible(meta, legacy)
    assert "holiday_names" in str(exc.value)
