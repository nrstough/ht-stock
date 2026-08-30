"""The panels build() must refuse, and the two floors it is easiest to be wrong about.

Three things are pinned here. A district movement report reaches features.build() as a
legal panel -- ht.validate calls it an error, but nothing forces a panel through
ht.validate -- and grouping by item alone silently interleaves the stores, so the refusal
has to live in build() itself. The split policy has to work on dates nowhere near the
simulator's 2023-2025, because the legacy boundaries are dates rather than offsets. And the
--allow-short floor is two floors counting two different things, which is why the span that
actually works is not the span the table advertises.
"""
import numpy as np
import pandas as pd
import pytest

from ht import schema
from model import features


def _partial_day(panel, date):
    """One reduced-hours day: context only, never a target. What a Thanksgiving looks like."""
    out = panel.copy()
    day = out.date == pd.Timestamp(date)
    assert day.any()
    out.loc[day, "is_closed"] = 1
    out.loc[day, "row_status"] = "partial"
    return out


# ---- a panel carrying more than one store ----

def test_a_two_store_panel_is_refused_by_name_rather_than_interleaved(make_panel):
    """Sorted by (store, item, date), the two stores' rows stack into one item series.

    Only the 28 windows straddling the seam fail the contiguity guard, so the build
    succeeds with twice the rows, two different sales figures on every date, and a context
    window that is 28 rows of two stores rather than 28 days of one.
    """
    both = pd.concat([make_panel(["bread"], days=400, conform=False),
                      make_panel(["bread"], days=400, conform=False, store="0456")],
                     ignore_index=True)
    panel = schema.conform(both)
    assert len(panel) == 800 and panel.store.nunique() == 2

    with pytest.raises(features.MultiStorePanel) as exc:
        features.build(panel, spec=features.spec_for_panel(panel))
    message = str(exc.value)
    assert "'0123'" in message and "'0456'" in message      # names them, does not just count
    assert "--store" in message                             # and names the ingest filter
    assert "14 calendar days" in message                    # what the context window becomes


def test_the_refusal_lists_the_first_stores_and_counts_the_rest(make_panel):
    frames = [make_panel(["bread"], days=140, conform=False, store=f"{n:04d}")
              for n in range(1, 9)]
    panel = schema.conform(pd.concat(frames, ignore_index=True))
    with pytest.raises(features.MultiStorePanel) as exc:
        features.build(panel, spec=features.spec_for_panel(panel))
    assert "8 store numbers" in str(exc.value)
    assert "and 2 more" in str(exc.value)


def test_one_store_and_a_panel_with_no_store_column_both_build(make_panel):
    """The guard must not cost the legacy path anything: store_synth.csv has no such column."""
    panel = make_panel(["bread"], days=200)
    assert len(features.build(panel, spec=features.spec_for_panel(panel))["y"]) > 0
    bare = panel.drop(columns=["store"])
    assert len(features.build(bare, spec=features.spec_for_panel(bare))["y"]) > 0


# ---- a panel whose dates are nowhere near the simulator's ----

@pytest.mark.parametrize("start", ["2016-03-01", "2031-07-14"])
def test_the_split_policy_holds_on_dates_far_outside_the_simulators_range(make_panel, start):
    """Fifteen years either side of 2023-2025, because the legacy boundaries are dates.

    legacy_spec's train_end is the literal 2024-12-31, so on a panel that ends in 2019 every
    row is train and the val split is empty, and on one that starts in 2031 no row is, and
    the failure is a KeyError out of stats["items"]. The derived spec reads the panel's own
    range, so both must land three occupied splits inside the panel's own dates.
    """
    panel = make_panel(["bread", "cake"], start=start, days=500)
    b = features.build(panel)                       # spec=None: the derived path must be chosen
    assert b["spec"]["train_end"] != features.TRAIN_END

    first, last = panel.date.min(), panel.date.max()
    for key in ("train_end", "val_start", "test_start"):
        assert first <= pd.Timestamp(b[key]) <= last, key
    # 20% of 500 into [28, 365] is 100 test days; 10% into [14, 84] is 50 val days
    assert (last - pd.Timestamp(b["test_start"])).days + 1 == 100
    assert (pd.Timestamp(b["test_start"]) - pd.Timestamp(b["val_start"])).days == 50
    assert pd.Timestamp(b["train_end"]) + pd.Timedelta(days=1) == pd.Timestamp(b["val_start"])
    counts = {s: int((b["split"] == s).sum()) for s in ("train", "val", "test")}
    assert min(counts.values()) > 0, counts


def test_the_derived_trend_origin_follows_the_panel_not_the_simulator(make_panel):
    """trend is (date - origin)/trend_days; a 2031 panel against a 2023 origin is off by 8."""
    panel = make_panel(["bread"], start="2031-07-14", days=500)
    spec = features.spec_for_panel(panel)
    assert spec["trend_start"] == "2031-07-14"
    b = features.build(panel, spec=spec)
    lo, hi = b["cov_layout"]["trend"]
    assert 0.0 <= b["cov"][:, lo:hi].min() and b["cov"][:, lo:hi].max() <= 1.0


# ---- the --allow-short floor, at the exact span where it starts working ----
#
# The panel floor counts calendar days and the per-item floor counts OPEN days, and at
# --allow-short the train window is the whole panel minus the 14 val days, so at 70 days it
# is exactly 56 -- with nothing to spare against the 56-day per-item floor. One closed day
# inside it costs every item the same day at once, so the failure is store-wide.

CLOSED = "2025-11-27"          # a real Thanksgiving; the rehearsal panel closes here too
LAST = "2025-12-31"


def _tail(make_panel, items, days):
    start = pd.Timestamp(LAST) - pd.Timedelta(days=days - 1)
    return make_panel(items, start=str(start.date()), days=days)


def test_seventy_days_with_no_closure_is_the_advertised_floor_and_it_works(make_panel):
    panel = _tail(make_panel, ["bread", "cake"], 70)
    b = features.build(panel, spec=features.spec_for_panel(panel, allow_short=True))
    assert b["items"] == ["bread", "cake"] and len(b["y"]) > 0


def test_one_closed_day_fails_every_item_at_seventy_days_and_seventy_one_works(make_panel):
    """The true boundary, and it is store-wide rather than one unlucky item.

    docs/REAL_DATA_READINESS.md says a 70-day --allow-short panel fails with
    "sushi: 55 of 56 days" and that 72 days works. Both halves are wrong: every item in the
    panel is at 55 of 56, because the closed day is the store's and not the item's, and 71
    days is the first span that clears the floor.
    """
    short = _partial_day(_tail(make_panel, ["bread", "cake"], 70), CLOSED)
    spec = features.spec_for_panel(short, allow_short=True)
    assert spec["min_item_train_days"] == features.MIN_TRAIN_DAYS_SHORT == 56
    assert (pd.Timestamp(spec["train_end"])
            - short.date.min()).days + 1 == 56           # the whole panel minus 14 val days

    with pytest.raises(features.InsufficientHistory) as exc:
        features.build(short, spec=spec)
    message = str(exc.value)
    assert "bread: 55 of 56 days" in message
    assert "cake: 55 of 56 days" in message              # not one item, all of them

    longer = _partial_day(_tail(make_panel, ["bread", "cake"], 71), CLOSED)
    b = features.build(longer, spec=features.spec_for_panel(longer, allow_short=True))
    assert b["items"] == ["bread", "cake"] and len(b["y"]) > 0
    assert b["excluded_items"] == []


def test_the_panel_floor_still_counts_calendar_days_so_a_closure_does_not_move_it(make_panel):
    """69 days is refused for its span; the closed day is invisible to resolve_splits."""
    with pytest.raises(features.InsufficientHistory) as exc:
        features.spec_for_panel(_partial_day(_tail(make_panel, ["bread"], 69), CLOSED),
                                allow_short=True)
    assert "69 days" in str(exc.value)
    # ...and 70 calendar days does clear it, which is what makes the per-item floor the
    # binding one and the effective floor "70 plus the closed days in the train window"
    spec = features.spec_for_panel(_partial_day(_tail(make_panel, ["bread"], 70), CLOSED),
                                   allow_short=True)
    assert spec["train_end"] == "2025-12-17"


def test_a_closed_day_is_context_but_never_a_target(make_panel):
    """Why the two floors differ at all: the row is kept, the target is not."""
    panel = _partial_day(_tail(make_panel, ["bread"], 200), CLOSED)
    b = features.build(panel, spec=features.spec_for_panel(panel))
    assert np.datetime64(CLOSED) not in set(b["date"])
    assert float(b["ctx"][:, :, 2].max()) == 1.0        # is_closed is encoder channel 2
