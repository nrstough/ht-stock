"""Every metric, on six hand-built rows whose answers are worked out in the comments.

The point of evaluate.py is that a store's pilot can be scored without true demand, so
these tests do two things. First, arithmetic: six rows, two of them censored, with every
expected number computed by hand beside the assertion -- because a metric nobody can
recompute on paper is a metric nobody will defend in a meeting. Second, direction: on a
frame where the true demand IS known (constructed here, never read from a panel), every
bound must fall on the safe side of it. A bound that is occasionally optimistic is not a
bound, and this is the only place that can be checked.
"""
import json

import numpy as np
import pandas as pd
import pytest

from model import evaluate

TAUS = np.array([0.2, 0.5, 0.8])

# rows 2 and 4 are sellout days, so `sold` there is a lower bound on demand
SOLD = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
CENS = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
Q = np.array([
    [8.0, 12.0, 16.0],
    [15.0, 18.0, 25.0],
    [20.0, 35.0, 40.0],
    [35.0, 44.0, 50.0],
    [40.0, 45.0, 52.0],
    [50.0, 60.0, 70.0],
])
# p50 - sold = [+2, -2, +5, +4, -5, 0]
ITEM = np.array(["a", "a", "a", "b", "b", "multi"])
REC = np.array([15.0, 15.0, 40.0, 40.0, 40.0, 55.0])
PRODUCED = np.array([20.0, 25.0, np.nan, 45.0, 60.0, 70.0])
WASTED = np.array([10.0, 5.0, np.nan, 5.0, 10.0, 10.0])
COST = np.full(6, 1.0)
PRICE = np.full(6, 4.0)
DAY_FRESH = np.array([True, True, True, True, True, False])   # "multi" has a shelf life


@pytest.fixture(scope="module")
def scored():
    return evaluate.score_quantiles(Q, TAUS, SOLD, CENS)


@pytest.fixture(scope="module")
def bounded():
    return evaluate.bounds(REC, SOLD, CENS, PRODUCED, WASTED, COST, PRICE, DAY_FRESH,
                           item=ITEM)


def test_median_forecast_is_the_tau_half_column():
    assert list(evaluate.median_forecast(Q, TAUS)) == [12.0, 18.0, 35.0, 44.0, 45.0, 60.0]


def test_censoring_coverage_is_reported_first_and_plainly(scored):
    assert scored["n_rows"] == 6
    assert scored["n_uncensored"] == 4
    assert scored["censored_share"] == pytest.approx(2 / 6)
    assert scored["sellout_rate"] == pytest.approx(2 / 6)
    assert scored["censoring_known"] is True


def test_wape_uncensored_drops_the_sellout_days(scored):
    # |err| on rows 0,1,3,5 = 2+2+4+0 = 8; sold there = 10+20+40+60 = 130
    assert scored["wape_uncensored"] == pytest.approx(8 / 130)


def test_wape_all_rows_charges_for_the_censored_rows_too(scored):
    # |err| over all six = 2+2+5+4+5+0 = 18; sold = 210
    assert scored["wape_all_rows"] == pytest.approx(18 / 210)


def test_bias_is_signed_and_computed_on_uncensored_rows(scored):
    # err on 0,1,3,5 = +2 -2 +4 +0 = 4
    assert scored["bias_pct"] == pytest.approx(4 / 130)
    assert scored["bias_units"] == pytest.approx(4 / 4)


def test_bias_lower_bound_counts_only_the_certain_half(scored):
    # min(err, 0) over all six = 0, -2, 0, 0, -5, 0 -> -7/6. An under-prediction is
    # definitely an error because demand is at least sold; an over-prediction on a
    # sellout day may be no error at all.
    assert scored["bias_lower_bound"] == pytest.approx(-7 / 6)


def test_pinball_matches_the_censored_loss_by_hand(scored):
    # tau=0.5: u = sold - q = [-2, +2, -5, -4, +5, 0]
    # uncensored rows use 0.5*|u| -> 1.0, 1.0, 2.0, 0.0
    # censored rows use tau*max(u,0) -> row2 0.0, row4 2.5
    assert scored["pinball_per_tau"]["0.5"] == pytest.approx(6.5 / 6)


def test_coverage_is_an_interval_whose_width_is_the_censoring_rate(scored):
    row = [c for c in scored["coverage"] if c["tau"] == 0.5][0]
    # hit = sold <= q_0.5 -> [T, F, T, T, F, T]; row 2 is a hit AND censored
    assert row["cov_lo"] == pytest.approx(3 / 6)     # unknown rows counted as misses
    assert row["cov_hi"] == pytest.approx(4 / 6)     # unknown rows counted as hits
    assert row["cov_point"] == pytest.approx(3 / 4)  # uncensored rows only
    assert row["unknown_share"] == pytest.approx(row["cov_hi"] - row["cov_lo"])


def test_coverage_bounds_are_ordered_at_every_tau(scored):
    for row in scored["coverage"]:
        assert row["cov_lo"] <= row["cov_hi"]


def test_with_no_sellout_signal_the_uncensored_figures_become_all_rows_figures():
    res = evaluate.score_quantiles(Q, TAUS, SOLD, np.zeros(6), censoring_known=False)
    assert res["censoring_known"] is False
    assert res["wape_uncensored"] == pytest.approx(res["wape_all_rows"])
    assert res["n_uncensored"] == 6


def test_waste_observed_is_measured_not_modelled(bounded):
    # produced - sold on the four rows that have production AND are day-fresh:
    # 10 + 5 + 5 + 10 = 30 units, at cost 1.00 and retail 4.00
    assert bounded["n_rows_measured"] == 4
    assert bounded["production_coverage"] == pytest.approx(5 / 6)
    assert bounded["waste_observed_units"] == pytest.approx(30.0)
    assert bounded["waste_observed_cost"] == pytest.approx(30.0)
    assert bounded["waste_observed_retail"] == pytest.approx(120.0)


def test_waste_saving_lower_bound(bounded):
    # model waste upper = max(rec - sold, 0) = 5, 0, -, 0, 0 over the measured rows -> 5
    assert bounded["waste_model_upper_units"] == pytest.approx(5.0)
    assert bounded["waste_saving_lower_units"] == pytest.approx(25.0)
    assert bounded["waste_saving_lower_cost"] == pytest.approx(25.0)
    assert bounded["waste_saving_lower_retail"] == pytest.approx(100.0)


def test_lost_units_uses_every_row_because_sold_is_a_lower_bound(bounded):
    # max(sold - rec, 0) = 0, 5, 0, 0, 10, 5 -> 20 units at a 3.00 margin
    assert bounded["lost_units_lower"] == pytest.approx(20.0)
    assert bounded["lost_margin_lower"] == pytest.approx(60.0)


def test_lost_margin_is_not_bounded_from_above_and_says_so(bounded):
    assert bounded["lost_margin_upper"] is None
    assert "upper bound on demand" in bounded["lost_margin_note"]


def test_sellout_days_bracket_the_answer(bounded):
    # rec < sold on rows 1, 4, 5 -> 3/6 certain; row 2 is censored with rec >= sold, so
    # it is the one row nobody can classify
    assert bounded["sellout_days_model_lower"] == pytest.approx(3 / 6)
    assert bounded["sellout_days_model_unknown"] == pytest.approx(1 / 6)
    assert bounded["sellout_days_model_upper"] == pytest.approx(4 / 6)
    assert bounded["sellout_days_sq"] == pytest.approx(2 / 6)


def test_a_multi_day_item_is_excluded_from_the_waste_bound_and_named(bounded):
    assert bounded["excluded_multi_day_items"] == ["multi"]


def test_a_row_with_no_production_record_is_excluded_from_the_measured_side_only(bounded):
    # row 2 has produced=NaN: it contributes nothing to waste, but its shortfall still counts
    assert bounded["n_rows_measured"] == 4
    assert bounded["lost_units_lower"] == pytest.approx(20.0)


def test_the_recorded_waste_column_is_only_ever_a_cross_check(bounded):
    assert bounded["waste_recorded_max_abs_diff"] == pytest.approx(0.0)


def test_with_no_sellout_signal_the_flag_dependent_bounds_go_null():
    res = evaluate.bounds(REC, SOLD, np.zeros(6), PRODUCED, WASTED, COST, PRICE,
                          DAY_FRESH, censoring_known=False, item=ITEM)
    assert res["sellout_days_sq"] is None
    assert res["sellout_days_model_upper"] is None
    assert res["sellout_days_model_lower"] == pytest.approx(3 / 6)   # still certain


# ---- the bounds must be bounds ----

def _censored_world(seed=11, n=600):
    """A world where demand is known, so a bound can be checked against the truth."""
    rng = np.random.default_rng(seed)
    demand = rng.gamma(9.0, 4.0, n)
    produced = np.round(rng.uniform(45, 75, n))     # a padded status-quo par sheet
    sold = np.minimum(demand, produced)
    cens = (sold >= produced - 1e-9).astype(float)
    rec = np.round(rng.uniform(35, 50, n))          # a leaner recommendation
    q = np.column_stack([sold * 0.7 + 4, sold * 1.0 + 2, sold * 1.3 + 6])
    return demand, produced, sold, cens, rec, q


def test_the_waste_saving_bound_never_overstates_the_true_saving():
    demand, produced, sold, cens, rec, _ = _censored_world()
    res = evaluate.bounds(rec, sold, cens, produced, np.full(len(sold), np.nan),
                          np.ones(len(sold)), np.full(len(sold), 4.0),
                          np.ones(len(sold), dtype=bool))
    true_saving = float((np.maximum(produced - demand, 0)
                         - np.maximum(rec - demand, 0)).sum())
    assert res["waste_saving_lower_units"] <= true_saving + 1e-9
    assert res["waste_saving_lower_units"] > 0          # and it is not vacuous


def test_the_model_sellout_rate_is_never_overstated():
    demand, produced, sold, cens, rec, _ = _censored_world()
    res = evaluate.bounds(rec, sold, cens, produced, np.full(len(sold), np.nan),
                          np.ones(len(sold)), np.full(len(sold), 4.0),
                          np.ones(len(sold), dtype=bool))
    true_rate = float((rec < demand).mean())
    assert res["sellout_days_model_lower"] <= true_rate + 1e-9
    assert res["sellout_days_model_upper"] >= true_rate - 1e-9


def test_the_coverage_interval_brackets_the_true_coverage_at_every_tau():
    demand, produced, sold, cens, rec, q = _censored_world()
    res = evaluate.score_quantiles(q, TAUS, sold, cens)
    for j, row in enumerate(res["coverage"]):
        true_cov = float((demand <= q[:, j]).mean())
        assert row["cov_lo"] <= true_cov + 1e-9
        assert true_cov <= row["cov_hi"] + 1e-9


def test_lost_units_lower_never_overstates_unmet_demand():
    demand, produced, sold, cens, rec, _ = _censored_world()
    res = evaluate.bounds(rec, sold, cens, produced, np.full(len(sold), np.nan),
                          np.ones(len(sold)), np.full(len(sold), 4.0),
                          np.ones(len(sold), dtype=bool))
    assert res["lost_units_lower"] <= float(np.maximum(demand - rec, 0).sum()) + 1e-9


# ---- the pieces around the metrics ----

def test_recommend_is_the_shared_newsvendor_definition(items):
    from model import newsvendor
    q = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]])
    taus = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99])
    got = evaluate.recommend(q, np.array(["doughnut"]), items, taus)
    fractile = evaluate.ht_config.critical_fractiles(items)["doughnut"]
    want = newsvendor.quantity(q[0], taus, fractile, items["doughnut"]["batch"], False)
    assert got[0] == pytest.approx(want)
    assert got[0] % items["doughnut"]["batch"] == 0        # a whole number of trays


def test_recommend_refuses_an_item_the_config_has_never_heard_of(items):
    q = np.ones((1, 11))
    taus = np.linspace(0.05, 0.99, 11)
    with pytest.raises(ValueError) as exc:
        evaluate.recommend(q, np.array(["ghost"]), items, taus)
    assert "ghost" in str(exc.value)


def test_by_group_refuses_to_print_a_number_from_too_few_rows():
    pack = evaluate.row_pack(q_units=Q, taus=TAUS, sold=SOLD, cens=CENS, rec_qty=REC,
                             produced=PRODUCED, wasted=WASTED, cost=COST, price=PRICE,
                             day_fresh=DAY_FRESH, item=ITEM)
    table = evaluate.by_group(pack, ITEM, min_n=5)
    assert set(table["group"]) == {"a", "b", "multi"}
    assert any("n/a" in str(v) for v in table.to_numpy().ravel())


def test_skill_is_a_paired_comparison_on_the_same_rows():
    model = evaluate.score_quantiles(Q, TAUS, SOLD, CENS)
    worse = evaluate.score_quantiles(Q * 1.5, TAUS, SOLD, CENS)
    better = evaluate.skill(model, worse)
    assert better["wape_skill"] > 0
    assert better["pinball_skill"] > 0
    assert better["n_rows"] == 6
    assert evaluate.skill(model, model)["wape_skill"] == pytest.approx(0.0)


def test_load_panel_refuses_a_frame_carrying_simulator_truth(tmp_path, synth_raw):
    from ht import schema
    path = str(tmp_path / "panel.csv")
    synth_raw.to_csv(path, index=False)
    panel = evaluate.load_panel(path)           # conform drops them before anything reads them
    schema.assert_no_truth(panel)
    assert not [c for c in schema.SIM_ONLY if c in panel.columns]


def test_the_measured_waste_matches_what_the_simulator_settlement_reported(synth_panel, repo):
    """The observable path and the simulator path must agree on the store's own waste.

    results.json's status_quo block was settled inside the simulator; produced - sold is
    computed here from columns a real store exports. They are the same number because
    sold = min(produced, demand) identically, which is what makes waste_observed a
    measurement rather than an estimate.
    """
    with open(f"{repo}/results/results.json", encoding="utf-8") as fh:
        published = json.load(fh)["summary"]["status_quo"]
    test_year = synth_panel[(synth_panel.date >= "2025-01-01") & (synth_panel.is_closed == 0)]
    waste = (test_year.produced.astype(float) - test_year.sold.astype(float))
    assert float(waste.sum()) == pytest.approx(published["waste_units"], rel=1e-6)
    assert float((waste * test_year.unit_cost.astype(float)).sum()) == pytest.approx(
        published["waste_cost"], rel=1e-5)
    assert float(test_year.stockout.mean()) == pytest.approx(published["sellout_days"],
                                                             rel=1e-9)


def test_a_policy_with_no_quantity_on_a_row_does_not_nan_the_whole_bound():
    """The status-quo policy's quantity IS the store's production record, and a real one has
    holes -- the mock export models 8% of days missing. One NaN in a plain sum turns every
    total into nan, which is how the observed-settlement table came to print `nan` for the
    status-quo row. A row the policy named no quantity for is not evidence about that policy:
    it drops out of the bound and is counted, rather than passing through as a silent zero.
    """
    n = 4
    rec = np.array([10.0, np.nan, 8.0, 6.0])       # no recommendation on row 1
    sold = np.array([12.0, 20.0, 5.0, 9.0])
    produced = np.array([14.0, 22.0, 8.0, 10.0])
    cens = np.zeros(n)
    cost, price = np.ones(n), np.full(n, 4.0)
    fresh = np.ones(n, dtype=bool)

    bnd = evaluate.bounds(rec, sold, cens, produced, np.full(n, np.nan), cost, price, fresh,
                          item=np.array(["a"] * n))

    assert bnd["n_rows_recommended"] == 3
    assert bnd["n_rows_measured"] == 3               # row 1 has production but no quantity
    for key in ("lost_units_lower", "lost_margin_lower", "waste_saving_lower_units",
                "waste_saving_lower_cost", "waste_observed_units",
                "sellout_days_model_lower"):
        assert np.isfinite(bnd[key]), f"{key} is not finite"
    # shortfall counts rows 0 and 3 only: (12-10) + (9-6) = 5. Row 1 is not a zero, it is
    # absent, and row 2 sold under its quantity.
    assert bnd["lost_units_lower"] == pytest.approx(5.0)
    assert bnd["lost_margin_lower"] == pytest.approx(15.0)
    # sellout lower bound over the three rows with a quantity: rows 0 and 3 -> 2/3
    assert bnd["sellout_days_model_lower"] == pytest.approx(2 / 3)


def test_cov_lo_is_a_floor_when_the_sellout_flag_could_not_be_evaluated():
    """A row nobody could read is not a row where demand was observed.

    Built so the truth is known: four days of demand 10, of which two really sold out at 6
    and carry stockout_known=0 (no production record). q_tau = 8 covers `sold` on both of
    those days and covers real demand on none of the four, so a floor built from ~cens
    would read 0.50 against a true coverage of 0.00.
    """
    demand = np.array([10.0, 10.0, 10.0, 10.0])
    sold = np.array([6.0, 6.0, 10.0, 10.0])
    known = np.array([0.0, 0.0, 1.0, 1.0])
    cens = np.zeros(4)                       # stockout * stockout_known is 0 on every row
    q = np.full((4, 1), 8.0)

    res = evaluate.score_quantiles(q, [0.5], sold, cens, censoring_known=False, known=known)
    c = res["coverage"][0]
    true_cov = float((demand <= 8.0).mean())
    assert c["cov_lo"] <= true_cov <= c["cov_hi"]
    assert c["cov_lo"] == 0.5 - 0.5          # the two unknown rows cannot count as certain
    assert c["cov_hi"] == pytest.approx(0.5)
    assert c["n_observed"] == 2


def test_the_reported_sellout_rate_is_measured_over_the_rows_that_could_be_read():
    """Pooling 'nobody could tell' with 'did not sell out' understates the store's own
    service level, and G4 compares the model against exactly that number."""
    known = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    cens = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    res = evaluate.score_quantiles(np.ones((5, 1)), [0.5], np.ones(5), cens, True, known)
    assert res["sellout_rate"] == pytest.approx(0.5)      # 1 of the 2 evaluable rows
    assert res["censored_share"] == pytest.approx(0.2)    # and it is no longer the same number

    bnd = evaluate.bounds(np.ones(5), np.ones(5), cens, np.full(5, 2.0), np.full(5, np.nan),
                          np.ones(5), np.full(5, 4.0), np.ones(5, dtype=bool),
                          item=np.array(["a"] * 5), known=known)
    assert bnd["sellout_days_sq"] == pytest.approx(0.5)


def test_a_group_thin_on_uncensored_rows_does_not_print_a_wape():
    """60 scored days of which 8 are uncensored is an eight-day WAPE labelled n=60."""
    n = 60
    cens = np.zeros(n)
    cens[8:] = 1.0
    pack = evaluate.row_pack(q_units=np.full((n, 1), 10.0), taus=np.array([0.5]),
                             sold=np.full(n, 10.0), cens=cens, rec_qty=np.full(n, 10.0),
                             produced=np.full(n, 10.0), wasted=np.full(n, np.nan),
                             cost=np.ones(n), price=np.full(n, 4.0),
                             day_fresh=np.ones(n, dtype=bool), item=np.array(["a"] * n),
                             known=np.ones(n))
    row = evaluate.by_group(pack, np.array(["Bakery"] * n), min_n=20).iloc[0]
    assert row["n"] == 60 and row["n_uncensored"] == 8
    assert str(row["wape_uncensored"]).startswith("n/a")
