"""The split policy, and what happens when a store's panel is short.

Today's hardcoded 2024/2025 boundaries do one of two things to a pilot panel: a bare
KeyError from stats["items"][item] if the panel sits entirely after TRAIN_END, or a
zero-row validation split that does not fail at the split -- the val loss is NaN, best_state
stays None, and the run dies twelve epochs later inside load_state_dict with an error that
names nothing about the cause. Both happen on the morning a store first points the tool at
its export, which is the worst possible morning for either. These tests pin the arithmetic
that replaces them, and the messages that must carry the numbers.
"""
import json

import numpy as np
import pandas as pd
import pytest

from model import features


def _dates(days, start="2026-01-01"):
    return pd.date_range(start, periods=days, freq="D").values


def test_the_documented_arithmetic_at_the_exact_floor():
    # H=126 yields 28/14/84 with every floor met and none to spare -- which is why
    # MIN_PANEL_DAYS is 126 and not a round number
    split = features.resolve_splits(_dates(126))
    assert split["span_days"] == 126
    assert split["test_start"] == "2026-04-09"
    assert split["val_start"] == "2026-03-26"
    assert split["train_end"] == "2026-03-25"
    assert (pd.Timestamp("2026-05-06") - pd.Timestamp(split["test_start"])).days + 1 == 28
    assert (pd.Timestamp(split["test_start"])
            - pd.Timestamp(split["val_start"])).days == 14


def test_one_day_short_of_the_floor_raises_with_the_numbers_in_the_message():
    with pytest.raises(features.InsufficientHistory) as exc:
        features.resolve_splits(_dates(125))
    message = str(exc.value)
    assert "125" in message and "126" in message
    assert "104 weeks" in message                     # the remedy, not just the complaint
    assert "--no-test" in message and "--allow-short" in message


def test_the_windows_are_contiguous_with_no_gap_or_overlap():
    split = features.resolve_splits(_dates(1096, start="2023-01-01"))
    assert split["span_days"] == 1096
    day = pd.Timedelta(days=1)
    assert pd.Timestamp(split["train_end"]) + day == pd.Timestamp(split["val_start"])
    assert (pd.Timestamp(split["test_start"])
            - pd.Timestamp(split["val_start"])).days == 84       # the 12-week cap
    assert (pd.Timestamp("2025-12-31")
            - pd.Timestamp(split["test_start"])).days + 1 == 219  # 20% of 1096


@pytest.mark.parametrize("days,ok", [(98, True), (97, False)])
def test_no_test_mode_has_its_own_floor(days, ok):
    if ok:
        split = features.resolve_splits(_dates(days), no_test=True)
        assert split["test_start"] is None
        assert split["thin"] is False
    else:
        with pytest.raises(features.InsufficientHistory):
            features.resolve_splits(_dates(days), no_test=True)


@pytest.mark.parametrize("days,ok", [(70, True), (69, False)])
def test_allow_short_is_an_escape_hatch_that_stamps_itself(days, ok):
    if ok:
        split = features.resolve_splits(_dates(days), allow_short=True)
        assert split["test_start"] is None
        assert split["thin"] is True                  # never engages silently
    else:
        with pytest.raises(features.InsufficientHistory):
            features.resolve_splits(_dates(days), allow_short=True)


def test_spec_for_panel_degrades_the_seasonal_terms_with_the_history(make_panel):
    # you cannot identify an annual cosine from under a year, and a second harmonic
    # fitted on one year is fitting that year
    assert features.spec_for_panel(make_panel(days=200))["fourier_harmonics"] == 0
    assert features.spec_for_panel(make_panel(days=400))["fourier_harmonics"] == 1
    assert features.spec_for_panel(make_panel(days=600))["fourier_harmonics"] == 2
    assert features.spec_for_panel(make_panel(days=200))["include_trend"] is True
    # a 110-day panel needs the escape hatch before it can be split at all
    assert features.spec_for_panel(make_panel(days=110),
                                   allow_short=True)["include_trend"] is False


def test_the_derived_spec_fixes_the_three_legacy_defects(make_panel):
    spec = features.spec_for_panel(make_panel(days=600))
    assert spec["holiday_countdown"] == "days"
    assert spec["stats_scope"] == "train"
    assert spec["stats_end"] == spec["train_end"]     # no validation leak into the z-scoring
    assert spec["trend_days"] == 599.0
    assert spec["require_contiguous_context"] is True
    assert spec["min_item_train_days"] == features.MIN_ITEM_TRAIN_DAYS
    assert spec["unknown_vocab"] == "zero"


def test_the_holiday_vocabulary_only_ever_grows_at_the_end(make_panel):
    panel = make_panel(days=600)
    panel.loc[panel.index[:5], "holiday"] = "store_anniversary"
    spec = features.spec_for_panel(panel)
    assert spec["holiday_names"][:14] == features.HOLIDAY_NAMES     # slots stay put
    assert "store_anniversary" in spec["holiday_names"]


def test_a_short_history_item_is_excluded_not_crashed_on(make_panel):
    panel = make_panel(["bread", "cake", "newthing"], start="2025-01-01", days=200)
    # newthing opened last month: 30 of the 200 days, against the 84-day per-item floor
    keep = ~((panel.item == "newthing") & (panel.date < "2025-06-20"))
    b = features.build(panel[keep], spec=features.spec_for_panel(panel[keep]))
    assert "newthing" not in b["items"]
    excluded = {e["item"]: e for e in b["excluded_items"]}
    assert "newthing" in excluded
    assert excluded["newthing"]["required"] == features.MIN_ITEM_TRAIN_DAYS
    assert excluded["newthing"]["reason"]


def test_all_items_excluded_raises_rather_than_training_on_nothing(make_panel):
    panel = make_panel(["bread"], days=200)
    spec = features.spec_for_panel(panel)
    spec["min_item_train_days"] = 10_000
    with pytest.raises(features.InsufficientHistory):
        features.build(panel, spec=spec)


def test_the_contiguity_guard_drops_a_window_that_spans_a_gap(make_panel):
    """The one failure mode that produces plausible-looking output: a POS outage turns a
    28-day window into 59 calendar days with no exception and unchanged aggregate metrics."""
    panel = make_panel(["bread"], start="2025-01-01", days=400)
    gap = panel.date.between("2025-06-01", "2025-06-09")
    with_gap = panel[~gap]
    spec = features.spec_for_panel(with_gap)
    loose = features.build(with_gap, spec=dict(spec, require_contiguous_context=False))
    strict = features.build(with_gap, spec=spec)
    assert strict["dropped_rows"].get("bread", 0) > 0
    assert len(strict["y"]) < len(loose["y"])


def test_an_unseen_vocabulary_value_leaves_an_all_zero_one_hot(make_panel):
    panel = make_panel(days=600).copy()
    panel["weather"] = "unknown"
    spec = features.spec_for_panel(panel)
    b = features.build(panel, spec=spec)
    lo, hi = b["cov_layout"]["weather"]
    # "unknown" is appended to the vocabulary rather than miscoded as sunny
    assert "unknown" in spec["weather_kinds"]
    assert b["cov"][:, lo:hi].sum(axis=1).max() == 1.0


def test_the_legacy_vocabulary_still_raises_on_an_unseen_value(make_panel):
    panel = make_panel(days=600).copy()
    panel["weather"] = "unknown"
    with pytest.raises(ValueError):
        features.build(panel, spec=features.legacy_spec())


def test_empty_split_names_the_boundaries_and_the_row_counts(make_panel):
    panel = make_panel(days=600)
    spec = dict(features.spec_for_panel(panel), val_start="2027-08-20",
                test_start="2027-08-21")
    with pytest.raises(features.EmptySplit) as exc:
        features.build(panel, spec=spec)
    message = str(exc.value)
    assert "val_start" in message and "rows" in message


def test_censoring_is_the_product_of_the_flag_and_whether_it_was_evaluable(make_panel):
    panel = make_panel(days=200).copy()
    panel["stockout"] = 1
    panel["stockout_known"] = 0
    panel["sellout_source"] = "none"
    b = features.build(panel, spec=features.spec_for_panel(panel))
    assert float(b["cens"].max()) == 0.0
    assert b["censoring_known"] is False
    assert b["sellout_source"] == "none"


def test_an_absent_stockout_known_column_is_treated_as_all_ones(make_panel):
    panel = make_panel(days=200).copy()
    panel["stockout"] = 1
    b = features.build(panel.drop(columns=["stockout_known"]),
                       spec=features.spec_for_panel(panel))
    assert float(b["cens"].min()) == 1.0


def test_a_missing_required_column_is_named_rather_than_crashing_late(make_panel):
    panel = make_panel(days=200).drop(columns=["payday"])
    with pytest.raises(ValueError) as exc:
        features.build(panel, spec=features.spec_for_panel(make_panel(days=200)))
    assert "payday" in str(exc.value)


def test_a_store_with_no_weather_feed_still_builds(make_panel):
    panel = make_panel(days=600).copy()
    panel["tmax_f"] = np.nan
    panel["weather"] = "unknown"
    b = features.build(panel, spec=features.spec_for_panel(panel))
    lo, hi = b["cov_layout"]["tmax_z"]
    assert np.unique(b["cov"][:, lo:hi]).tolist() == [0.0]      # z-scored to zero, not NaN
    assert np.unique(b["ctx"][:, :, 3]).tolist() == [0.0]


def test_the_trend_origin_travels_in_the_spec_not_in_the_frame(make_panel):
    """A re-export that starts later must not shift a covariate the checkpoint was fitted on.

    The trend covariate is (date - origin) / trend_days. With the origin read off whatever
    frame build() is handed, scoring the same checkpoint against a shorter panel of the same
    store feeds the model a different number for the same day, with spec_hash unchanged and
    assert_compatible silent.
    """
    panel = make_panel(days=600)
    spec = features.spec_for_panel(panel)
    assert spec["trend_start"] == str(pd.Timestamp(panel.date.min()).date())
    assert "trend_start" in json.dumps(spec)          # so it lands in spec_hash and meta.json

    trimmed = panel[panel.date >= panel.date.min() + pd.Timedelta(days=60)]
    a = features.build(panel, spec=spec)
    b = features.build(trimmed, spec=spec)
    lo = a["cov_layout"]["trend"][0]
    shared = pd.Timestamp(trimmed.date.min()) + pd.Timedelta(days=90)
    ia = int(np.where(a["date"] == np.datetime64(shared))[0][0])
    ib = int(np.where(b["date"] == np.datetime64(shared))[0][0])
    assert a["cov"][ia, lo] == pytest.approx(b["cov"][ib, lo])


def test_build_can_be_handed_the_checkpoints_own_normalizers(make_panel):
    """Scoring must not refit the z-scoring from the frame it happens to be scoring."""
    panel = make_panel(days=600)
    spec = features.spec_for_panel(panel)
    pinned = features.build(panel, spec=spec)["stats"]
    trimmed = panel[panel.date >= panel.date.min() + pd.Timedelta(days=60)]
    b = features.build(trimmed, spec=spec, stats=pinned)
    assert b["stats"]["items"] == pinned["items"]
    assert b["stats"]["tmax_mean"] == pinned["tmax_mean"]


def test_the_holiday_countdown_sees_past_the_end_of_the_panel(make_panel):
    """The last three weeks of any panel are the test window's tail and the days before the
    first served morning; reading them as 'no holiday within 21 days' is a train/serve skew."""
    panel = make_panel(days=600, start="2024-04-15")          # ends 2025-12-05, before Christmas
    spec = features.spec_for_panel(panel)
    d2h = features._days_to_holiday_map(panel, spec)
    last = max(d2h)
    assert d2h[last] < spec["holiday_horizon"]


def test_allow_short_can_actually_succeed_at_the_floor_it_advertises(make_panel):
    """--allow-short relaxes the panel floor to 70 days; an 84-day per-item floor would then
    exclude every item and the flag the error message points at could never work."""
    panel = make_panel(["bread"], days=70)
    b = features.build(panel, spec=features.spec_for_panel(panel, allow_short=True))
    assert b["items"] == ["bread"] and len(b["y"]) > 0
