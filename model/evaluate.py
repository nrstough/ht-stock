"""Evaluation with nothing but what a store can actually see.

model/backtest.py settles every policy against the simulator's own latent demand,
which is the one quantity no real store will ever hand us. This module is the
version that survives contact with a real export: every number here comes from the
panel's observable columns, and the simulator-only column names do not appear in
this file at all.

It is built around one awkward fact. On a day the case ran empty, `sold` is a
lower bound on demand, not demand -- so an error against `sold` on that day is not
an error. Every metric below therefore states what it does with those rows and
returns that choice next to the number:

  accuracy    excludes them (wape_uncensored), and prints the all-rows figure
              beside it so the choice is visible rather than argued about later
  calibration brackets them ([cov_lo, cov_point, cov_hi]); the interval width is
              exactly the censoring rate, so it narrows as sellout data improves
  bias        excludes them, plus one figure that counts only the certain half
  economics   MEASURED where the identity is exact (status-quo waste really is
              produced - sold, because sold = min(produced, demand)), and a
              rigorous one-sided BOUND everywhere else

Nothing here integrates the model's own predictive distribution to produce an
expected cost: that is self-referential, and a miscalibrated model would report a
confident saving. The bounds are weaker and they are true. Lost margin is the one
quantity with no honest bound at all -- bounding it needs an upper bound on
demand, which nothing observable provides -- so it is reported as None with a
note rather than left for a reader to assume it is zero.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from ht import config as ht_config
from ht import schema

from . import baselines, features, newsvendor
from .net import DemandNet

FLAG_BIAS_PCT = 0.10       # a slice this far off is called out by name
FLAG_MIN_N = 20            # ... but only once it has enough rows to mean anything

LOST_MARGIN_NOTE = ("not estimable without an upper bound on demand; the lower bound "
                    "counts only the units we can prove were missed")


def load_panel(path):
    """Read a canonical panel and refuse it if it still carries simulator columns."""
    df = schema.read_panel(path)
    schema.assert_no_truth(df)
    return df


# ---- forecasts ----

def _predict_z(b, artifacts_dir, batch_size=1024):
    """The checkpoint's quantile matrix, in z-space, for every row of b."""
    with open(os.path.join(artifacts_dir, "meta.json")) as f:
        meta = json.load(f)
    features.assert_compatible(meta, b)
    model = DemandNet(len(meta["items"]), meta["ctx_dim"], meta["cov_dim"], len(meta["taus"]))
    model.load_state_dict(torch.load(os.path.join(artifacts_dir, "demandnet.pt"),
                                     weights_only=True))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(b["iidx"]), batch_size):
            sl = slice(i, i + batch_size)
            out.append(model(torch.tensor(b["iidx"][sl]), torch.tensor(b["ctx"][sl]),
                             torch.tensor(b["cov"][sl])).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(meta["taus"])), dtype=np.float32)


def to_units(zmat, b):
    """z-scored log1p quantiles -> units, per item, clipped at zero."""
    stats = b["stats"]["items"]
    std = np.array([stats[it]["std"] for it in b["item"]])[:, None]
    mean = np.array([stats[it]["mean"] for it in b["item"]])[:, None]
    return np.expm1(zmat * std + mean).clip(min=0)


def predict(b, artifacts_dir, batch_size=1024):
    """The (n, n_taus) quantile matrix in UNITS. Raises SpecMismatch on a stale checkpoint."""
    return to_units(_predict_z(b, artifacts_dir, batch_size), b)


def model_version(artifacts_dir):
    with open(os.path.join(artifacts_dir, "demandnet.pt"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def recommend(q_units, item, items, taus, tau_by_item=None):
    """Batch-rounded newsvendor quantity per row -- the one definition shadow also uses."""
    taus = np.asarray(taus, dtype=float)
    tau_by_item = ht_config.critical_fractiles(items) if tau_by_item is None else tau_by_item
    missing = sorted({str(k) for k in item} - set(items))
    if missing:
        raise ValueError("items config has no entry for: " + ", ".join(missing))
    out = np.zeros(len(item))
    for i, (key, row) in enumerate(zip(item, q_units)):
        it = items[key]
        out[i] = newsvendor.quantity(row, taus, tau_by_item[key], it["batch"],
                                     bool(it.get("continuous", False)))
    return out


def median_forecast(q_units, taus):
    taus = np.asarray(taus, dtype=float)
    j = np.where(np.isclose(taus, 0.5))[0]
    if len(j):
        return q_units[:, int(j[0])]
    return np.array([float(np.interp(0.5, taus, row)) for row in q_units])


# ---- metrics ----

def _ratio(num, den):
    den = float(den)
    return float(num) / den if den else float("nan")


def score_quantiles(q_units, taus, sold, cens, censoring_known=True, known=None):
    """Accuracy, pinball and calibration for one quantile matrix. Pure numpy.

    q_units (n, n_taus) and sold are in units; cens is 1 on rows where the sellout
    flag both fired and was evaluable. censoring_known=False says the panel carries
    no sellout signal at all, in which case cens is all zeros and every "uncensored"
    figure is really an all-rows figure -- the flag travels with the numbers so a
    reader cannot mistake one for the other.

    `known` is stockout_known: 1 where the sellout rule could actually be evaluated.
    It is NOT the complement of cens, and the calibration bracket needs the difference.
    A row where nobody could tell is a row where `sold <= q_tau` proves nothing, so it
    can only sit in cov_hi; counting it in cov_lo -- which is what happens if the floor
    is built from ~cens -- makes the "lower bound" exceed the truth by up to the hidden
    sellout rate, and under rule "none" that is every row. An absent `known` means the
    panel had no such column, which schema.conform defaults to all-ones.

    Two things this deliberately does NOT claim. wape_all_rows is printed as the
    conventional pessimistic reading, but it is not a bound: measured on the synthetic
    panel it came in 1.4 points BELOW the same WAPE against real demand, because a
    censored `sold` sits below demand and the median forecast often sits below `sold`.
    And bias_pct is computed on uncensored rows, which are a biased subsample -- the
    sellout days it drops are exactly the days the model is short. On the synthetic
    panel that flipped the sign: +2.7% here against -1.5% against real demand. Read it
    as the bias on days that were fully served, and read bias_lower_bound beside it.
    """
    taus = np.asarray(taus, dtype=float)
    sold = np.asarray(sold, dtype=float)
    cens = np.asarray(cens, dtype=float) > 0
    known = np.ones(len(sold), dtype=bool) if known is None \
        else np.asarray(known, dtype=float) > 0
    unc = ~cens
    observed = known & ~cens          # demand exactly observed: the only certain rows
    p50 = median_forecast(q_units, taus)
    err = p50 - sold

    u = sold[:, None] - q_units
    full = np.maximum(taus * u, (taus - 1) * u)
    under = taus * np.clip(u, 0, None)
    pin = np.where(cens[:, None], under, full)

    coverage = []
    for j, tau in enumerate(taus):
        hit = sold <= q_units[:, j]
        coverage.append(dict(
            tau=float(tau),
            cov_lo=float(np.mean(hit & observed)) if len(sold) else float("nan"),
            cov_point=float(np.mean(hit[observed])) if observed.any() else float("nan"),
            cov_hi=float(np.mean(hit)) if len(sold) else float("nan"),
            unknown_share=float(np.mean(hit & ~observed)) if len(sold) else float("nan"),
            n=int(len(sold)), n_uncensored=int(unc.sum()), n_observed=int(observed.sum()),
        ))

    return dict(
        n_rows=int(len(sold)),
        n_uncensored=int(unc.sum()),
        censored_share=float(cens.mean()) if len(sold) else float("nan"),
        censoring_known=bool(censoring_known),
        n_observed=int(observed.sum()),
        # the store's service level, over the rows where it could be measured -- pooling
        # "did not sell out" with "nobody could tell" understates it by the unknown share
        sellout_rate=float(cens[known].mean()) if known.any() else float("nan"),
        wape_uncensored=_ratio(np.abs(err[unc]).sum(), sold[unc].sum()),
        wape_all_rows=_ratio(np.abs(err).sum(), sold.sum()),
        pinball_censored=float(pin.mean()) if len(sold) else float("nan"),
        pinball_per_tau={f"{t:g}": float(pin[:, j].mean()) if len(sold) else float("nan")
                         for j, t in enumerate(taus)},
        bias_pct=_ratio(err[unc].sum(), sold[unc].sum()),
        bias_units=float(err[unc].mean()) if unc.any() else float("nan"),
        # only the half that is certain: an under-prediction is definitely an error
        # because demand is at least sold, an over-prediction on a sellout day may
        # be no error at all. Negative means "definitely short, on average, by".
        bias_lower_bound=float(np.minimum(err, 0.0).mean()) if len(sold) else float("nan"),
        coverage=coverage,
    )


def _maybe(cens, known):
    """Rows where demand may exceed sold: flagged sellouts, plus rows nobody could evaluate."""
    return cens | ~known


def bounds(rec_qty, sold, cens, produced, wasted, cost, price, day_fresh,
           censoring_known=True, item=None, known=None):
    """Measured waste and the one-sided economic bounds, in units and dollars.

    Two inequalities carry the whole block and neither needs an assumption:

      status-quo waste == produced - sold, because sold = min(produced, demand);
      the model's waste <= max(rec_qty - sold, 0), because demand >= sold.

    The second is unconditional. The first carries ONE assumption -- that every unit which
    left the case either scanned or was discarded. Employee meals, samples, a catering tray
    rung to another department, damage and theft all break it the same way, making
    produced - sold an upper bound on real discard rather than an equality. That direction
    is safe for the saving below (both sides shrink together) and it is why the heading says
    "no model" rather than "no assumption".

    So their difference is a genuine LOWER bound on units saved -- the real saving is
    that or better. Both halves are restricted to rows that carry a production record
    AND belong to a day-fresh item (day_fresh), because for anything with a shelf life
    longer than a day the identity is simply false. lost_units_lower uses every row the
    policy actually named a quantity for -- sold - rec_qty bounds unmet demand whether or not
    production was recorded, but a row with no recommendation is not evidence of anything, and
    n_rows_recommended reports how many there were. `item` names the excluded multi-day items.
    """
    rec_qty = np.asarray(rec_qty, dtype=float)
    sold = np.asarray(sold, dtype=float)
    cens = np.asarray(cens, dtype=float) > 0
    produced = np.asarray(produced, dtype=float)
    wasted = np.asarray(wasted, dtype=float)
    cost = np.asarray(cost, dtype=float)
    price = np.asarray(price, dtype=float)
    fresh = np.asarray(day_fresh, dtype=bool)
    known = np.ones(len(sold), dtype=bool) if known is None \
        else np.asarray(known, dtype=float) > 0

    has_prod = np.isfinite(produced)
    # A row where the policy has no quantity says nothing about that policy. It matters for
    # the status-quo policy in particular, whose quantity IS the store's production record --
    # and a real production record always has holes. Counting those rows as a zero shortfall
    # would understate the bound; letting the NaN through turns every total into nan.
    rec_ok = np.isfinite(rec_qty)
    have = has_prod & fresh & rec_ok
    sq_waste = np.maximum(produced - sold, 0.0)
    model_waste_upper = np.maximum(rec_qty - sold, 0.0)
    short = np.maximum(sold - rec_qty, 0.0)

    short = np.where(rec_ok, short, 0.0)
    excluded = sorted({str(k) for k, f in zip(item, fresh) if not f}) if item is not None else []
    recorded = np.isfinite(wasted) & have
    lower_units = float((sq_waste[have] - model_waste_upper[have]).sum())

    return dict(
        n_rows=int(len(sold)),
        production_coverage=float(has_prod.mean()) if len(sold) else float("nan"),
        n_rows_measured=int(have.sum()),
        n_rows_recommended=int(rec_ok.sum()),
        n_produced_below_sold=int((has_prod & (produced < sold - 1e-9)).sum()),
        waste_observed_units=float(sq_waste[have].sum()),
        waste_observed_cost=float((sq_waste * cost)[have].sum()),
        waste_observed_retail=float((sq_waste * price)[have].sum()),
        # cross-check only: does the panel's own waste column agree with produced - sold?
        # It is evidence exactly where those cells came from the store's own record -- and
        # the panel does not carry that provenance, so read it beside ingest's waste_cells
        # counts, which say how many cells the export supplied and how many were derived.
        waste_recorded_units=float(wasted[recorded].sum()) if recorded.any() else None,
        waste_recorded_max_abs_diff=(float(np.abs(wasted[recorded] - sq_waste[recorded]).max())
                                     if recorded.any() else None),
        waste_model_upper_units=float(model_waste_upper[have].sum()),
        waste_saving_lower_units=lower_units,
        waste_saving_lower_cost=float(((sq_waste - model_waste_upper) * cost)[have].sum()),
        waste_saving_lower_retail=float(((sq_waste - model_waste_upper) * price)[have].sum()),
        lost_units_lower=float(short.sum()),
        lost_margin_lower=float((short * (price - cost)).sum()),
        lost_margin_upper=None,
        lost_margin_note=LOST_MARGIN_NOTE,
        # the store's own rate over the rows where the flag could be evaluated, which is
        # what G4 compares the model against; pooling the unevaluable rows in as
        # "did not sell out" would hand the model a baseline the store never had
        sellout_days_sq=(float(cens[known].mean())
                         if censoring_known and known.any() else None),
        # the denominator of the line above, printed beside it: a rate over one row and a
        # rate over a thousand read identically on the page unless the page says so
        n_flag_evaluable=int(known.sum()) if censoring_known else 0,
        sellout_days_model_lower=(float((rec_qty[rec_ok] < sold[rec_ok]).mean())
                                  if rec_ok.any() else float("nan")),
        # unknown: rec_qty covered `sold`, but sold is not demand -- either because the day
        # was flagged a sellout or because nobody could tell whether it was
        sellout_days_model_unknown=(float((_maybe(cens, known)[rec_ok]
                                           & (rec_qty[rec_ok] >= sold[rec_ok])).mean())
                                    if censoring_known and rec_ok.any() else None),
        sellout_days_model_upper=(float(((rec_qty[rec_ok] < sold[rec_ok])
                                         | _maybe(cens, known)[rec_ok]).mean())
                                  if censoring_known and rec_ok.any() else None),
        excluded_multi_day_items=excluded,
    )


def row_pack(*, q_units, taus, sold, cens, rec_qty, produced, wasted, cost, price,
             day_fresh, item, censoring_known=True, known=None):
    """The per-row arrays every metric above is computed from, in one dict.

    by_group slices this; evaluate and shadow build it once and hand it around. It is
    deliberately not part of any report: reports carry numbers, not row arrays.
    """
    return dict(q_units=np.asarray(q_units, dtype=float), taus=np.asarray(taus, dtype=float),
                sold=np.asarray(sold, dtype=float), cens=np.asarray(cens, dtype=float),
                rec_qty=np.asarray(rec_qty, dtype=float),
                produced=np.asarray(produced, dtype=float),
                wasted=np.asarray(wasted, dtype=float), cost=np.asarray(cost, dtype=float),
                price=np.asarray(price, dtype=float),
                day_fresh=np.asarray(day_fresh, dtype=bool), item=np.asarray(item),
                known=(np.ones(len(sold)) if known is None else np.asarray(known, dtype=float)),
                censoring_known=bool(censoring_known))


def _slice(pack, mask):
    out = dict(pack)
    for k in ("q_units", "sold", "cens", "rec_qty", "produced", "wasted", "cost", "price",
              "day_fresh", "item", "known"):
        out[k] = pack[k][mask]
    return out


def by_group(res, key, min_n=FLAG_MIN_N):
    """Per-group accuracy, bias and economics. `res` is a row_pack, not a report dict.

    Groups thinner than min_n print "n/a (n=k)": a WAPE over eleven rows is a number,
    not a measurement, and printing it invites someone to act on it.
    """
    key = np.asarray(key).astype(str)
    rows = []
    for g in sorted(set(key)):
        m = key == g
        n = int(m.sum())
        p = _slice(res, m)
        acc = score_quantiles(p["q_units"], p["taus"], p["sold"], p["cens"],
                              p["censoring_known"], p["known"])
        # the guard counts the rows the WAPE is measured on, not the rows in the group: a
        # high-sellout department can carry 60 rows and eight uncensored ones, and printing
        # "n=60, wape 4%" beside it invites a reader to trust eight days
        if acc["n_uncensored"] < min_n:
            # with no sellout signal there is no uncensored subset to name: every row is in
            # the count, and calling it n_unc beside an all-rows heading invents a distinction
            label = (f"n/a (n_unc={acc['n_uncensored']})" if p["censoring_known"]
                     else f"n/a (n={acc['n_uncensored']})")
            rows.append(dict(group=g, n=n, n_uncensored=acc["n_uncensored"],
                             wape_uncensored=label, bias_pct=label,
                             sellout_days_model_lower=label, waste_saving_lower_cost=label))
            continue
        bnd = bounds(p["rec_qty"], p["sold"], p["cens"], p["produced"], p["wasted"],
                     p["cost"], p["price"], p["day_fresh"], p["censoring_known"], p["item"],
                     p["known"])
        rows.append(dict(group=g, n=n, n_uncensored=acc["n_uncensored"],
                         wape_uncensored=acc["wape_uncensored"],
                         bias_pct=acc["bias_pct"],
                         sellout_days_model_lower=bnd["sellout_days_model_lower"],
                         waste_saving_lower_cost=bnd["waste_saving_lower_cost"]))
    return pd.DataFrame(rows)


def skill(res_model, res_baseline):
    """Paired improvement over a baseline scored on exactly the same rows."""
    return dict(
        wape_skill=1.0 - _ratio(res_model["wape_uncensored"], res_baseline["wape_uncensored"]),
        pinball_skill=1.0 - _ratio(res_model["pinball_censored"],
                                   res_baseline["pinball_censored"]),
        wape_model=res_model["wape_uncensored"],
        wape_baseline=res_baseline["wape_uncensored"],
        n_rows=res_model["n_rows"],
    )


def bias_slices(pack, slices, min_n=FLAG_MIN_N, flag_at=FLAG_BIAS_PCT):
    """Bias by item, dow, month and weather, with the slices worth an argument flagged.

    Aggregate accuracy hides structure. A model 4% off overall but 22% low every Friday
    gets found by a kitchen manager in week one and never trusted again.
    """
    out, flagged = {}, []
    for name, key in slices.items():
        key = np.asarray(key).astype(str)
        rows = []
        for g in sorted(set(key)):
            m = key == g
            p = _slice(pack, m)
            acc = score_quantiles(p["q_units"], p["taus"], p["sold"], p["cens"],
                                  p["censoring_known"], p["known"])
            entry = dict(group=g, n=acc["n_rows"], n_uncensored=acc["n_uncensored"],
                         bias_pct=acc["bias_pct"], bias_units=acc["bias_units"],
                         wape_uncensored=acc["wape_uncensored"])
            rows.append(entry)
            if acc["n_uncensored"] >= min_n and abs(acc["bias_pct"]) > flag_at:
                flagged.append(dict(slice=name, **entry))
        out[name] = rows
    out["flagged"] = flagged
    return out


# ---- the report ----

def _spec_for(meta, df, artifacts_dir):
    """Score with the boundaries the checkpoint was trained on, never a fresh split.

    The frozen artifact predates the spec field, so there the legacy layout is assumed
    rather than read. features.spec_from_meta checks that assumption against the panel
    and refuses when it does not hold, instead of scoring a store's 2026 export on the
    simulator's 2024 boundaries and reporting the result as accuracy.
    """
    return features.spec_from_meta(meta, df, artifacts_dir)


def _coverage_of_scoring(df, b, mask, artifacts_dir, items_path, sellout_source):
    dates = b["date"][mask]
    lo, hi = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    d = pd.to_datetime(df["date"])
    window = df[(d >= lo) & (d <= hi) & df["item"].isin(set(b["items"]))]
    status = window["row_status"].astype(str).value_counts().to_dict()
    excluded = {k: int(v) for k, v in status.items() if k != "ok"}
    excluded["short_history_item"] = int(((d >= lo) & (d <= hi)
                                          & ~df["item"].isin(set(b["items"]))).sum())
    excluded["non_contiguous_window"] = int(sum(b.get("dropped_rows", {}).values()))
    scored = int(mask.sum())
    expected = int((window["is_closed"].astype(int) == 0).sum())
    excluded["context_warmup_or_other"] = max(expected - scored - excluded["non_contiguous_window"],
                                              0)
    return dict(
        n_rows_expected=expected, n_rows_scored=scored,
        n_uncensored=int((b["cens"][mask] == 0).sum()),
        censored_share=float((b["cens"][mask] > 0).mean()) if scored else float("nan"),
        n_rows_excluded=excluded,
        n_items_scored=len(b["items"]),
        n_items_excluded=len(b.get("excluded_items", [])),
        excluded_items=b.get("excluded_items", []),
        date_min=str(lo.date()), date_max=str(hi.date()),
        sellout_source=sellout_source,
        spec_hash=b.get("spec_hash"), model_version=model_version(artifacts_dir),
        items_config_hash=ht_config.config_hash(items_path),
        panel_hash=schema.panel_hash(df)[:12],
    )


def _caveats(res_sellout, items, b, bnd, spec_assumed):
    out = []
    if not res_sellout["censoring_known"]:
        out.append("NO SELLOUT DATA: the model was fitted to sales, not demand, so these "
                   "quantities run low on the busiest days by an amount nothing here "
                   "measures. Every 'uncensored' figure below is really an all-rows figure.")
    elif res_sellout["known_share"] < 1.0:
        out.append(f"The sellout flag could only be evaluated on "
                   f"{res_sellout['known_share']:.0%} of scored rows. The rest are 'nobody "
                   "could tell', not 'did not sell out': they are excluded from cov_lo and "
                   "counted in the calibration bracket's unknown column.")
    if bnd["excluded_multi_day_items"]:
        out.append("Excluded from the waste bound (shelf_life_days > 1, so waste is not "
                   "produced - sold): " + ", ".join(bnd["excluded_multi_day_items"]))
    if bnd["waste_observed_units"] == 0 and bnd["n_rows_measured"] == 0:
        out.append("No production record on any scored row: there is no measured waste "
                   "baseline, so no saving is claimed.")
    imputed = sorted(k for k, it in items.items() if it.get("cost_imputed"))
    if imputed:
        out.append("Cost imputed from a department gross margin for: " + ", ".join(imputed)
                   + " -- every dollar figure involving them rests on that assumption.")
    if b.get("spec", {}).get("stats_scope") == "train_val":
        out.append("Legacy feature spec: normalization statistics were computed over train+val, "
                   "so first and second moments of the validation window leaked into z-scoring.")
    if spec_assumed:
        sp = b.get("spec", {})
        out.append("The checkpoint's meta.json records NO feature spec, so the legacy layout "
                   f"was assumed, not read: train_end {sp.get('train_end')}, val_start "
                   f"{sp.get('val_start')}, test_start {sp.get('test_start')}, holiday "
                   "countdown off. Those are the simulator's own dates; every split figure "
                   "above is cut on them.")
    out.append("The naive benchmark is model/baselines.naive_forecast, which averages the "
               "trailing four same-weekday sales with no is_closed filter, so a closed day "
               "drags it down; it is a slightly weaker benchmark than an honest par sheet.")
    return out


def evaluate(panel_path, artifacts_dir, items_path, *, split="test", date_from=None,
             date_to=None, out=None):
    """Score the checkpoint and both baselines on identical rows, observables only."""
    df = load_panel(panel_path)
    items_all = ht_config.load_items(items_path, include_inactive=True)
    with open(os.path.join(artifacts_dir, "meta.json")) as f:
        meta = json.load(f)

    # the checkpoint's own normalizers and trend origin, never ones refitted from this
    # frame: a re-export that starts on a different date would otherwise silently re-z-score
    # every context window and move quantities nobody changed
    b = features.build(df, spec=_spec_for(meta, df, artifacts_dir), stats=meta.get("stats"))
    mask = np.ones(len(b["y"]), dtype=bool) if split == "all" else (b["split"] == split)
    if date_from is not None:
        mask &= b["date"] >= np.datetime64(pd.Timestamp(date_from))
    if date_to is not None:
        mask &= b["date"] <= np.datetime64(pd.Timestamp(date_to))
    if not mask.any():
        raise ValueError(f"no rows in split {split!r} between {date_from} and {date_to}")

    taus = np.asarray(b["taus"], dtype=float)
    dl_z = _predict_z(b, artifacts_dir)
    dl_units = to_units(dl_z, b)
    naive_point = baselines.naive_forecast(df, b)
    stats = b["stats"]["items"]
    std = np.array([stats[it]["std"] for it in b["item"]])
    mean = np.array([stats[it]["mean"] for it in b["item"]])
    naive_units = to_units(baselines.quantiles_from_point(
        (np.log1p(naive_point) - mean) / std, b, taus), b)
    ridge_units = to_units(baselines.quantiles_from_point(
        baselines.fit_predict_ridge(b), b, taus), b)

    # the panel rows behind b's scored rows, in the same order
    key = pd.MultiIndex.from_arrays([b["item"][mask], b["date"][mask]])
    rows = df.set_index(["item", "date"]).loc[key].reset_index()
    item = rows["item"].astype(str).values
    sold = rows["sold"].values.astype(float)
    cens = b["cens"][mask]
    price = _fill_from_config(rows.get("unit_price"), item, items_all, "price")
    cost = _fill_from_config(rows.get("unit_cost"), item, items_all, "cost")
    produced = (rows["produced"].values.astype(float) if "produced" in rows
                else np.full(len(rows), np.nan))
    wasted = (rows["wasted"].values.astype(float) if "wasted" in rows
              else np.full(len(rows), np.nan))
    fresh = np.array([int(items_all[k].get("shelf_life_days", 1)) == 1 for k in item])
    known = (rows["stockout_known"].values.astype(float) if "stockout_known" in rows
             else np.ones(len(rows)))
    known_share = float(known.mean()) if len(known) else 1.0
    sellout_source = b.get("sellout_source", "unknown")
    censoring_known = bool(b.get("censoring_known", True))

    rec = recommend(dl_units[mask], item, {k: items_all[k] for k in set(item)}, taus)
    pack = row_pack(q_units=dl_units[mask], taus=taus, sold=sold, cens=cens, rec_qty=rec,
                    produced=produced, wasted=wasted, cost=cost, price=price,
                    day_fresh=fresh, item=item, censoring_known=censoring_known, known=known)

    acc = {name: score_quantiles(units[mask], taus, sold, cens, censoring_known, known)
           for name, units in (("dl", dl_units), ("naive", naive_units),
                               ("ridge", ridge_units))}
    # the same loss the training curve reports, so meta.json's best_val is comparable
    acc["dl"]["pinball_censored_z"] = score_quantiles(
        dl_z[mask], taus, b["y"][mask], cens, censoring_known, known)["pinball_censored"]
    bnd = bounds(rec, sold, cens, produced, wasted, cost, price, fresh, censoring_known, item,
                 known)

    sellout = dict(
        sellout_source=sellout_source, censoring_known=censoring_known,
        known_share=known_share,
        sellout_rate=acc["dl"]["sellout_rate"],
        rows_by_source=(rows["sellout_source"].astype(str).value_counts().to_dict()
                        if "sellout_source" in rows else {}),
        sellout_days_sq=bnd["sellout_days_sq"],
        sellout_days_model_lower=bnd["sellout_days_model_lower"],
        sellout_days_model_unknown=bnd["sellout_days_model_unknown"],
        sellout_days_model_upper=bnd["sellout_days_model_upper"],
    )
    dates = pd.DatetimeIndex(rows["date"])
    slices = dict(item=item, dow=rows["dow"].values, month=dates.month.values,
                  weather=rows["weather"].astype(str).values)
    measured_keys = ("production_coverage", "n_rows_measured", "n_produced_below_sold",
                     "waste_observed_units", "waste_observed_cost", "waste_observed_retail",
                     "waste_recorded_units", "waste_recorded_max_abs_diff")

    report = dict(
        coverage_of_scoring=_coverage_of_scoring(df, b, mask, artifacts_dir, items_path,
                                                 sellout_source),
        accuracy={k: {m: v for m, v in r.items() if m != "coverage"} for k, r in acc.items()},
        calibration=acc["dl"]["coverage"],
        bias_slices=bias_slices(pack, slices),
        sellout=sellout,
        measured={k: bnd[k] for k in measured_keys},
        bounds={k: v for k, v in bnd.items() if k not in measured_keys},
        skill=dict(vs_naive=skill(acc["dl"], acc["naive"]),
                   vs_ridge=skill(acc["dl"], acc["ridge"])),
        caveats=_caveats(sellout, items_all, b, bnd, "spec" not in meta),
    )
    report["by_group"] = {
        name: by_group(pack, key).to_dict(orient="records")
        for name, key in (("item", item), ("dept", rows["dept"].astype(str).values),
                          ("dow", rows["dow"].values))}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True, default=_jsonable)
    return report


def _fill_from_config(series, item, items, field):
    """Realized per-unit dollars where the export has them, config price/cost elsewhere."""
    fallback = np.array([float(items[k][field]) for k in item])
    if series is None:
        return fallback
    v = np.asarray(series.values, dtype=float)
    return np.where(np.isfinite(v), v, fallback)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(v).date())
    return str(v)


def _pct(v, nd=1):
    return "n/a" if v is None or not np.isfinite(float(v)) else f"{float(v) * 100:.{nd}f}%"


def _num(v, nd=1):
    return "n/a" if v is None or not np.isfinite(float(v)) else f"{float(v):,.{nd}f}"


def format_report(res, width=100):
    """The text rendering. Coverage first, measured and bounded never in one table."""
    cov, acc, bnd = res["coverage_of_scoring"], res["accuracy"], res["bounds"]
    ms, so = res["measured"], res["sellout"]
    L = ["=" * width, "FRESH FORECAST - OBSERVABLE-ONLY EVALUATION", "=" * width,
         "", "COVERAGE OF SCORING (every number below is conditional on this row set)"]
    L.append(f"  {cov['date_min']} .. {cov['date_max']}   scored {cov['n_rows_scored']:,} of "
             f"{cov['n_rows_expected']:,} expected item-days   "
             f"items {cov['n_items_scored']} scored / {cov['n_items_excluded']} excluded")
    drops = sorted(cov["n_rows_excluded"].items())
    L.append("  excluded: " + ", ".join(f"{k}={v}" for k, v in drops if v))
    L.append(f"  model {cov['model_version']}  spec {cov['spec_hash']}  "
             f"items-config {cov['items_config_hash']}  panel {cov['panel_hash']}")
    L.append(f"  sellout source {so['sellout_source']}  known_share {_pct(so['known_share'])}  "
             f"observed sellout rate {_pct(so['sellout_rate'])}")
    if not so["censoring_known"]:
        L.append("  CENSORING UNKNOWN: 'uncensored' figures below are all-rows figures.")

    L += ["", "MEASURED (no model; assumes every unit that left the case scanned or was "
          "thrown out)"]
    L.append(f"  production record on {_pct(ms['production_coverage'])} of scored rows; "
             f"{ms['n_rows_measured']:,} rows are day-fresh AND have one")
    L.append(f"  status-quo waste  {_num(ms['waste_observed_units'])} units   "
             f"${_num(ms['waste_observed_cost'])} at cost   "
             f"${_num(ms['waste_observed_retail'])} at retail")
    if ms["waste_recorded_units"] is not None:
        L.append(f"  panel waste column {_num(ms['waste_recorded_units'])} units; "
                 f"max per-row disagreement with produced - sold "
                 f"{_num(ms['waste_recorded_max_abs_diff'], 3)}")
        L.append("  that agreement is evidence only for cells the store's own report "
                 "supplied; the panel does not")
        L.append("  record which those are -- ingest's report does (waste_cells)")

    L += ["", "ACCURACY (median forecast vs sold, on rows where demand is exactly observed)",
          f"  {'model':10s} {'wape_unc':>10s} {'wape_all':>10s} {'bias':>9s} "
          f"{'defshort':>9s} {'pinball':>9s} {'n_unc':>7s}"]
    for name in ("dl", "naive", "ridge"):
        r = acc[name]
        L.append(f"  {name:10s} {_pct(r['wape_uncensored']):>10s} {_pct(r['wape_all_rows']):>10s} "
                 f"{_pct(r['bias_pct']):>9s} {_num(r['bias_lower_bound'], 2):>9s} "
                 f"{_num(r['pinball_censored'], 3):>9s} {r['n_uncensored']:>7,d}")
    L.append("  wape_all counts sellout days at face value (the spec's 'pessimistic bound'). It "
             "is NOT a bound on the")
    L.append("  truth in either direction: sold understates demand on those days, so the error "
             "can move either way.")
    L.append(f"  skill vs naive {_pct(res['skill']['vs_naive']['wape_skill'])} on WAPE, "
             f"{_pct(res['skill']['vs_naive']['pinball_skill'])} on pinball (paired rows)")

    L += ["", "CALIBRATION (cov_lo and cov_hi bracket the truth; the gap is the share of rows "
          "this quantile",
          "             covers whose demand is not observed -- flagged sellouts plus rows the "
          "rule could not read)",
          f"  {'tau':>6s} {'cov_lo':>8s} {'cov_hi':>8s} {'unknown':>8s} {'cov_pt':>8s}"]
    for c in res["calibration"]:
        L.append(f"  {c['tau']:>6.3f} {_pct(c['cov_lo']):>8s} {_pct(c['cov_hi']):>8s} "
                 f"{_pct(c['unknown_share']):>8s} {_pct(c['cov_point']):>8s}")
    L.append("  cov_pt is the coverage over observed rows ONLY -- a different denominator, so "
             "it sits outside the bracket.")

    flagged = res["bias_slices"]["flagged"]
    L += ["", f"BIAS SLICES OFF BY MORE THAN {FLAG_BIAS_PCT:.0%} (n >= {FLAG_MIN_N})"]
    if not flagged:
        L.append("  none")
    for f in flagged[:20]:
        L.append(f"  {f['slice']:8s} {str(f['group']):18s} bias {_pct(f['bias_pct'])}  "
                 f"wape {_pct(f['wape_uncensored'])}  n={f['n_uncensored']}")

    L += ["", "BOUNDS (one-sided, never in the model's favour)"]
    L.append(f"  waste saving  >= {_num(bnd['waste_saving_lower_units'])} units, "
             f"${_num(bnd['waste_saving_lower_cost'])} at cost, "
             f"${_num(bnd['waste_saving_lower_retail'])} at retail")
    L.append(f"  units definitely missed >= {_num(bnd['lost_units_lower'])}, "
             f"margin definitely missed >= ${_num(bnd['lost_margin_lower'])}")
    L.append(f"  lost margin upper bound: none. {bnd['lost_margin_note']}")
    L.append(f"  sellout days: store {_pct(bnd['sellout_days_sq'])}   model >= "
             f"{_pct(bnd['sellout_days_model_lower'])}"
             + (f", <= {_pct(bnd['sellout_days_model_upper'])}"
                if bnd["sellout_days_model_upper"] is not None else ", upper unknown"))

    L += ["", "CAVEATS"]
    L += [f"  - {c}" for c in res["caveats"]]
    L.append("=" * width)
    return "\n".join(L)


def _check_args(args):
    """Refuse a mistyped path or date before anything is read. One line, naming the flag."""
    for flag, what in (("panel", "panel csv"), ("items", "items config")):
        path = getattr(args, flag)
        if not os.path.exists(path):
            raise schema.HtError(f"--{flag}: no {what} at {path}")
    if not os.path.exists(os.path.join(args.artifacts, "meta.json")):
        raise schema.HtError(f"--artifacts: {args.artifacts} has no meta.json -- point it at "
                             f"a trained model directory, e.g. model/artifacts")
    for flag, value in (("from", args.date_from), ("to", args.date_to)):
        if value is None:
            continue
        try:
            pd.Timestamp(value)
        except ValueError as exc:
            raise schema.HtError(f"--{flag}: {value!r} is not a date ({exc}); write it as "
                                 f"YYYY-MM-DD")


def main(argv=None):
    ap = argparse.ArgumentParser(description="observable-only evaluation of a checkpoint")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--split", choices=("test", "val", "train", "all"), default="test")
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--by", choices=("item", "dept", "dow"), default=None)
    ap.add_argument("--json", dest="out", default=None)
    ap.add_argument("--width", type=int, default=100)
    args = ap.parse_args(argv)

    # a mistyped path or date is the common failure here and it is a person's mistake, not a
    # bug: name the flag in one line and exit 1, exactly as ht.ingest's main does
    try:
        _check_args(args)
        res = evaluate(args.panel, args.artifacts, args.items, split=args.split,
                       date_from=args.date_from, date_to=args.date_to, out=args.out)
    except schema.HtError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        # features.SpecMismatch and evaluate's own "no rows in split" land here; both already
        # say what to do about it, and neither is helped by forty lines of stack
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(format_report(res, args.width))
    if args.by:
        print()
        print(f"BY {args.by.upper()}")
        print(pd.DataFrame(res["by_group"][args.by]).to_string(index=False))
    if args.out:
        print(f"wrote {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
