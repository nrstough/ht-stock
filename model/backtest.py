"""Shadow-replay backtest over the held-out window.

Run:  python -m model.backtest

For every open test day each policy names a production quantity and the day is
settled. There are two settlements, and which one is available is a property of
the DATA, not a preference:

  --settlement sim       the simulator's true demand (never seen by any model)
                         settles what would have sold, wasted and been missed.
                         Only a generated panel has those columns.
  --settlement observed  a real store's panel. Demand is never observed, so the
                         economics come from model/evaluate.py: waste measured
                         exactly as produced - sold, and one-sided bounds for
                         everything else. No number here is settled against a
                         quantity a store cannot export.

Policies:

  status_quo  what the store actually produced
  naive       trailing same-weekday average + gaussian quantiles + newsvendor
  ridge       linear model + gaussian quantiles + newsvendor
  dl          quantile network + newsvendor
  dl_matched  quantile network pinned to the status quo's service level
  oracle      knows the true demand distribution (ceiling, irreducible noise
              only). SIMULATOR ONLY -- it needs the generative noise widths, so
              under --settlement observed it is skipped with a printed reason
              rather than approximated into something meaningless.

Item economics come from config/items.example.json via ht/config.py, not from
sim/params.py: the nine records there are the same numbers, and severing the
import is what lets this file run against a store's export. The one remaining
simulator dependency is a lazy import inside oracle_q().

A supplied --panel must name a --spec. The split decides which rows the dollar
figures cover, and the legacy boundaries are the simulator's own dates: silently
inheriting them on a store's export would report a saving over a window nobody
chose. `python -m model.backtest` with no arguments is untouched and still means
legacy -- it is the provenance of results/results.json.

Writes results/results.json with the summary, per-item detail, and the chart
series used by the PoC dashboard and the proposal.
"""
import argparse
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
import torch

from ht import config as ht_config
from ht import schema

from . import baselines, evaluate, features, newsvendor
from .net import DemandNet

REPO = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(REPO, "results")
# the frozen sim-settlement replay. Only that exact run may write here -- see
# _guard_frozen_out, which is what stops a flag from quietly replacing the file every dollar
# figure in proposal/ and poc/ is settled against.
DEFAULT_OUT = os.path.abspath(os.path.join(RESULTS, "results.json"))
ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
CONFIG = os.path.join(REPO, "config", "items.example.json")

# Two orders, both frozen by results/results.json and the console table it prints
# beside. POLICIES is the order the summary block records; PRINT_ORDER is the order
# the table reads in, which puts the two dl variants next to each other.
POLICIES = ("status_quo", "naive", "ridge", "dl", "oracle", "dl_matched")
PRINT_ORDER = ("status_quo", "naive", "ridge", "dl", "dl_matched", "oracle")
SIM_ONLY_POLICIES = ("oracle",)

# The service-matching calibration window, stated rather than derived. Like the
# legacy split boundaries it is a property of the frozen run, and no rule
# reproduces it from a panel's date range without being reverse-engineered to.
# --calib-window DAYS replaces it with the N days ending the day before the test
# split, which is what a real panel should use.
CALIB_START = "2024-01-01"
CALIB_END = "2024-12-31"

# The proposal's two illustrated items and the eight weeks it plots them over.
# Also stated: on a real panel both are derived (see _chart_items / _chart_window).
CHART_ITEMS = ("pizza-whole", "hotbar-lb")
CHART_WINDOW = ("2025-01-06", "2025-03-02")

# The simulator's latent columns, named once. SIM_DEMAND doubles as the results.json
# series key under sim settlement, which is why it is a constant rather than a literal.
SIM_DEMAND = "true_demand"
SIM_MEAN = "true_mean"


def sim_truth(df_rows, column=SIM_DEMAND):
    """The one gateway to simulator-only columns. SIMULATOR SETTLEMENT ONLY.

    Every read of true_demand and true_mean in this file goes through here, so the whole
    dependency on the generator is a single function -- and one a real panel cannot
    satisfy, because --settlement observed drops those columns from the frame immediately
    after load and this raises instead of quietly returning a number.
    """
    if column not in df_rows.columns:
        raise KeyError(
            f"{column} is a simulator-only column and this frame carries none; "
            "--settlement observed settles against realized sales instead")
    return df_rows[column].to_numpy(dtype=float)


def predict_dl(b, artifacts_dir=ARTIFACTS):
    with open(os.path.join(artifacts_dir, "meta.json")) as f:
        meta = json.load(f)
    # The frozen meta.json predates the feature spec, so assert_compatible falls back
    # to its dimension-and-roster check and warns about it. That is the expected path
    # for this file, not news; a real disagreement still raises SpecMismatch.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features.assert_compatible(meta, b)
    model = DemandNet(len(meta["items"]), meta["ctx_dim"], meta["cov_dim"],
                      len(meta["taus"]))
    model.load_state_dict(torch.load(os.path.join(artifacts_dir, "demandnet.pt"),
                                     weights_only=True))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(b["iidx"]), 1024):
            sl = slice(i, i + 1024)
            preds.append(model(torch.tensor(b["iidx"][sl]),
                               torch.tensor(b["ctx"][sl]),
                               torch.tensor(b["cov"][sl])).numpy())
    return np.concatenate(preds)


def to_units(zmat, b):
    stds = np.array([b["stats"]["items"][it]["std"] for it in b["item"]])[:, None]
    means = np.array([b["stats"]["items"][it]["mean"] for it in b["item"]])[:, None]
    return np.expm1(zmat * stds + means).clip(min=0)


def score_sim(df_rows, q):
    """Economics of producing q against true demand. SIMULATOR ONLY."""
    d = sim_truth(df_rows)
    sold = np.minimum(q, d)
    waste = q - sold
    lost = d - sold
    price, cost = df_rows.unit_price.values, df_rows.unit_cost.values
    return dict(
        produced=float(q.sum()),
        waste_units=float(waste.sum()),
        waste_retail=float((waste * price).sum()),
        waste_cost=float((waste * cost).sum()),
        lost_units=float(lost.sum()),
        lost_margin=float((lost * (price - cost)).sum()),
        econ_cost=float((waste * cost).sum() + (lost * (price - cost)).sum()),
        sellout_days=float((sold >= q).mean()),
        fill_rate=float(sold.sum() / max(d.sum(), 1e-9)),
        waste_pct_of_production=float((waste * price).sum() / max((q * price).sum(), 1e-9)),
    )


def _censoring(df_rows):
    """(cens, known, censoring_known). cens = stockout AND the flag was evaluable.

    `known` travels separately because "did not sell out" and "nobody could tell" are not
    the same row, and the calibration bracket in evaluate depends on the difference. An
    absent stockout_known column is all-ones, which is what schema.conform defaults it to.
    """
    stockout = df_rows.stockout.values.astype(float)
    known = (df_rows["stockout_known"].values.astype(float)
             if "stockout_known" in df_rows else np.ones(len(df_rows)))
    return stockout * known, known, bool(known.sum() > 0)


def score_observed(df_rows, q, items):
    """Observable-only economics: waste MEASURED, everything else a one-sided bound.

    There is no econ_cost and no fill_rate here, deliberately. Both need demand on
    the days the store ran out, and nothing a store can export supplies it; the
    bounds below are what survives a hostile reading instead.
    """
    sold = df_rows.sold.values.astype(float)
    cens, known, censoring_known = _censoring(df_rows)
    n = len(df_rows)
    produced = (df_rows["produced"].values.astype(float)
                if "produced" in df_rows else np.full(n, np.nan))
    wasted = (df_rows["wasted"].values.astype(float)
              if "wasted" in df_rows else np.full(n, np.nan))
    price = df_rows.unit_price.values.astype(float)
    cost = df_rows.unit_cost.values.astype(float)
    item = df_rows.item.values
    day_fresh = np.array([items[k]["shelf_life_days"] == 1 for k in item])

    bnd = evaluate.bounds(q, sold, cens, produced, wasted, cost, price, day_fresh,
                          censoring_known=censoring_known, item=item, known=known)
    # nansum: the status-quo policy's quantity is the store's own production record, and a
    # real one has holes. Those rows are excluded from every bound above, so the produced
    # total has to be over the same rows rather than nan.
    out = dict(produced=float(np.nansum(q)))
    out.update(bnd)
    # nansum on the denominator too: max(nan, 1e-9) is nan in Python, and one nan here writes
    # a bare NaN token into results JSON, which no parser but Python's will read
    out["waste_pct_of_production"] = float(bnd["waste_observed_retail"]
                                           / max(float(np.nansum(q * price)), 1e-9))
    return out


def _quantile_scores(units_mat, taus, df_rows, mask):
    """score_quantiles for one policy's quantile matrix on the test rows."""
    cens, known, censoring_known = _censoring(df_rows)
    return evaluate.score_quantiles(units_mat[mask], taus, df_rows.sold.values.astype(float),
                                    cens, censoring_known=censoring_known, known=known)


def _fill_economics(df, items):
    """A real export may carry no price or cost column; the config is the fallback."""
    for field, col in (("price", "unit_price"), ("cost", "unit_cost")):
        vals = pd.Series([items[k][field] if k in items else np.nan for k in df.item],
                         index=df.index)
        if col not in df.columns:
            df[col] = vals
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(vals)
    return df


def _chart_items(dfi, items):
    if all(k in set(dfi.item) for k in CHART_ITEMS):
        return CHART_ITEMS
    value = (dfi.sold * dfi.unit_price).groupby(dfi.item).sum().sort_values(ascending=False)
    return tuple(value.index[:2])


def _chart_window(dfi, settlement):
    if settlement == "sim":
        return CHART_WINDOW
    start = dfi.date.min()
    return (start.strftime("%Y-%m-%d"),
            (start + pd.Timedelta(days=55)).strftime("%Y-%m-%d"))


def _demand_structure(df, settlement):
    """dow x weather and monthly-by-dept indices of daily retail value.

    Under sim settlement the basis is true demand, which is what results.json
    records. On a real panel the only observable basis is realized sales, which
    is censored on sellout days -- so the index understates exactly the busiest
    day/weather cells, and the returned dict says which basis was used.
    """
    dfall = df[df.is_closed == 0].copy()
    basis = SIM_DEMAND if settlement == "sim" else "sold"
    demand = sim_truth(dfall) if settlement == "sim" else dfall.sold.to_numpy(dtype=float)
    dfall["demand_retail"] = demand * dfall.unit_price
    day_val = dfall.groupby(["date", "dow", "weather"]).demand_retail.sum().reset_index()
    overall = day_val.demand_retail.mean()
    mat = day_val.groupby(["dow", "weather"]).demand_retail.mean() / overall
    dow_weather = [dict(dow=int(dow), weather=w, index=round(float(v), 3))
                   for (dow, w), v in mat.items()]

    dfall["month"] = dfall.date.dt.month
    dept_month = dfall.groupby(["dept", "month", "date"]).demand_retail.sum() \
        .groupby(["dept", "month"]).mean()
    dept_avg = dept_month.groupby("dept").mean()
    seasonality = [dict(dept=dept, month=int(mo), index=round(float(v / dept_avg[dept]), 3))
                   for (dept, mo), v in dept_month.items()]
    return dow_weather, seasonality, basis


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="replay the held-out window under each policy")
    ap.add_argument("--panel", default=None, help="canonical panel CSV (default: the simulator's)")
    ap.add_argument("--items", default=CONFIG, help="item economics JSON")
    ap.add_argument("--artifacts", default=ARTIFACTS, help="directory holding demandnet.pt")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--spec", choices=("legacy", "auto"), default=None,
                    help="auto: boundaries and vocabularies derived from the panel. "
                         "legacy: the frozen simulator's. Required with --panel")
    ap.add_argument("--settlement", choices=("sim", "observed"), default="sim")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--calib-window", type=int, default=None,
                    help="days of service-matching calibration ending before the test split")
    ap.add_argument("--val-days", type=int, default=None)
    ap.add_argument("--test-days", type=int, default=None)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--allow-short", action="store_true")
    return ap.parse_args(argv)


def _guard_spec(panel, spec):
    """A supplied panel with no --spec must not quietly replay on the simulator's window.

    This is the same refusal model.train makes, and it matters more here: the number this
    command prints is the dollar saving. On a 2024-2026 export the legacy boundaries park
    two years in "test" and forecast them with z-scoring fitted through 2024-12-31, so the
    money would cover a window nobody chose. The no-panel default still means legacy --
    `python -m model.backtest` is the provenance of results/results.json.
    """
    if panel is None or spec is not None:
        return
    raise SystemExit(features.spec_refusal(panel, features.AUTO_NOTE_DERIVE))


def _not_the_frozen_run(args):
    """Which arguments make this something other than the frozen replay. Empty means it is."""
    why = []
    if args.panel:
        why.append("--panel")
    if args.spec == "auto":
        why.append("--spec auto")
    if args.settlement != "sim":
        why.append(f"--settlement {args.settlement}")
    if args.policies != ",".join(POLICIES):
        why.append(f"--policies {args.policies}")
    for flag, value in (("--calib-window", args.calib_window), ("--val-days", args.val_days),
                        ("--test-days", args.test_days)):
        if value is not None:
            why.append(f"{flag} {value}")
    for flag, value in (("--no-test", args.no_test), ("--allow-short", args.allow_short)):
        if value:
            why.append(flag)
    if os.path.abspath(args.artifacts) != os.path.abspath(ARTIFACTS):
        why.append(f"--artifacts {args.artifacts}")
    if os.path.abspath(args.items) != os.path.abspath(CONFIG):
        why.append(f"--items {args.items}")
    return why


def _guard_frozen_bytes(out, payload):
    """The frozen replay writes results/results.json only while it still reproduces it.

    The configuration guard cannot see everything that moves the numbers: retrain
    model/artifacts with --force-frozen and the plain command's own output changes, with no
    flag to notice it by. So the last check is the bytes. Reproducing them is a no-op write;
    not reproducing them means this run is no longer the one the proposal is settled against,
    which is a thing to be told rather than a thing to have happen quietly.
    """
    if os.path.abspath(out) != DEFAULT_OUT or not os.path.exists(out):
        return
    with open(out, "rb") as fh:
        published = fh.read()
    if payload.encode() == published:
        return
    raise SystemExit(
        "the frozen replay no longer reproduces results/results.json byte for byte, so "
        "nothing was written. Something underneath it moved -- the checkpoint in "
        "model/artifacts, data/store_synth.csv, or a library version. Re-run with "
        "--out .rehearsal/results_new.json and diff the two; replacing the published file "
        "replaces the proposal's dollar figures with numbers nothing else has been settled "
        "against.")


def _guard_frozen_out(args):
    """results/results.json is the frozen replay, and only that run may write over it.

    The other two frozen writers refuse outright and take --force-frozen. This one cannot:
    `python -m model.backtest` with no arguments IS the provenance of results.json and has to
    go on reproducing it byte for byte, which is the claim the README makes. So the refusal is
    on the CONFIGURATION instead -- anything that changes the numbers has to name its own
    --out. Without this, `--policies dl,naive` replaced a six-policy file with a two-policy one
    and printed nothing but "wrote".
    """
    if os.path.abspath(args.out) != DEFAULT_OUT:
        return
    why = _not_the_frozen_run(args)
    if not why:
        return
    raise SystemExit(
        "results/results.json is the frozen sim-settlement replay the proposal's dollar "
        f"figures rest on, and this run is not it ({', '.join(why)}). Name your own --out, "
        "e.g. --out .rehearsal/results_real.json. Only `python -m model.backtest` with no "
        "arguments writes results/results.json, and it reproduces the published bytes.")


def main(argv=None):
    args = _parse_args(argv)
    try:
        df = features.load(args.panel) if args.panel else features.load()
    except features.PanelNotFound as exc:
        raise SystemExit(f"--panel: {exc}")

    has_truth = all(c in df.columns for c in schema.SIM_ONLY)
    if args.settlement == "sim" and not has_truth:
        raise SystemExit("model.backtest --settlement sim settles policies against "
                         "simulator truth; for a real panel use `python -m model.evaluate`")
    if args.settlement == "observed":
        # Hard Rule 6 made structural: the truth columns cannot be read further down
        # because they are not on the frame any more.
        df = df.drop(columns=[c for c in schema.SIM_ONLY if c in df.columns])

    # after the settlement check, which is a property of the frame: a panel that cannot be
    # settled at all is not helped by first being asked to choose a split for it
    _guard_spec(args.panel, args.spec)

    if args.no_test:
        raise SystemExit(
            "model.backtest replays every policy over the HELD-OUT test window, and --no-test "
            "says this panel has none: there would be no rows to settle and no dollar figure "
            "to print. Drop --no-test, or pass --test-days N to hold out a window this panel "
            "can afford. (model.train takes --no-test because fitting needs only train and "
            "val; backtesting is the step that needs the window nobody fitted on.)")

    # before any forecasting work: a refusal after four minutes of replay is a worse refusal
    _guard_frozen_out(args)

    wanted = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in POLICIES]
    if unknown:
        raise SystemExit(f"unknown policy {unknown}; choose from {list(POLICIES)}")
    if args.settlement == "observed":
        for p in SIM_ONLY_POLICIES:
            if p in wanted:
                wanted.remove(p)
                print(f"skipping policy '{p}': it needs the simulator's latent mean and both "
                      "generative noise widths, which no store export contains")
    selected = [p for p in POLICIES if p in wanted]

    if args.spec == "auto":
        spec = features.spec_for_panel(df, val_days=args.val_days, test_days=args.test_days,
                                       no_test=args.no_test, allow_short=args.allow_short)
    else:                # "legacy", or None -- which _guard_spec allows only with no panel
        spec = features.legacy_spec()
    b = features.build(df, spec=spec)

    items = {k: v for k, v in ht_config.load_items(args.items).items() if k in b["items"]}
    missing = [k for k in b["items"] if k not in items]
    if missing:
        raise SystemExit(f"{args.items} has no economics for {missing}; every item in the "
                         "panel needs a price, cost and batch before a quantity can be named")
    q_star = ht_config.critical_fractiles(items)

    taus = b["taus"]
    test = b["split"] == "test"

    # ---- forecasts (z-space quantile matrices) ----
    try:
        dl_units = to_units(predict_dl(b, args.artifacts), b)
    except features.SpecMismatch as exc:
        raise SystemExit(f"{exc}\n\nThe checkpoint in {args.artifacts} was fitted to a "
                         "different feature layout than the one just built. Retrain it against "
                         "this panel, or point --artifacts at the matching run.")

    naive_units_point = baselines.naive_forecast(df, b)
    stds = np.array([b["stats"]["items"][it]["std"] for it in b["item"]])
    means = np.array([b["stats"]["items"][it]["mean"] for it in b["item"]])
    naive_z = (np.log1p(naive_units_point) - means) / stds
    naive_units = to_units(baselines.quantiles_from_point(naive_z, b, taus), b)

    ridge_z = baselines.fit_predict_ridge(b)
    ridge_units = to_units(baselines.quantiles_from_point(ridge_z, b, taus), b)

    # ---- align test rows of the raw frame with b's test rows ----
    key = pd.MultiIndex.from_arrays([b["item"][test], b["date"][test]])
    dfi = df.set_index(["item", "date"]).loc[key].reset_index()
    if args.settlement == "observed":
        dfi = _fill_economics(dfi, items)

    def decide(units_mat, tau_by_item):
        out = np.zeros(len(dfi))
        um = units_mat[test]
        for i, (item, row_q) in enumerate(zip(dfi.item.values, um)):
            it = items[item]
            out[i] = newsvendor.quantity(row_q, taus, tau_by_item[item],
                                         it["batch"], it.get("continuous", False))
        return out

    # oracle: true lognormal quantile around the latent mean
    def oracle_q():
        import sim.params as sim_params        # simulator-only policy

        traffic_sig = sim_params.STORE["traffic_noise_sigma"]
        means = sim_truth(dfi, SIM_MEAN)
        out = np.zeros(len(dfi))
        for i, row in enumerate(dfi.itertuples()):
            it = items[row.item]
            sig = np.hypot(sim_params.ITEMS[row.item]["sigma"], traffic_sig)
            z = baselines._norm_ppf(np.array([q_star[row.item]]))[0]
            q = means[i] * np.exp(sig * z - sig ** 2 / 2)
            out[i] = newsvendor.quantity(np.array([q]), np.array([0.5]), 0.5,
                                         it["batch"], it.get("continuous", False))
        return out

    forecasts = {"naive": naive_units, "ridge": ridge_units, "dl": dl_units}
    policies = {}
    for name in selected:
        if name == "status_quo":
            policies[name] = dfi.produced.values.astype(float)
        elif name in forecasts:
            policies[name] = decide(forecasts[name], q_star)
        elif name == "oracle":
            policies[name] = oracle_q()

    if "dl_matched" in selected and args.settlement == "observed" and not b["censoring_known"]:
        # With sellout rule "none" the observed sellout rate is 0 by construction, not
        # because the store never ran out, so matching it pins tau at the top of the
        # grid and the policy quietly becomes "produce the p99". Refuse instead.
        selected.remove("dl_matched")
        print("skipping policy 'dl_matched': this panel carries no sellout signal, so the "
              "service level it would match to is 0% by construction, not by measurement")

    if "dl_matched" in selected:
        policies["dl_matched"] = decide(dl_units, _match_service(
            df, b, dl_units, dfi, items, taus, q_star, args, spec))

    if args.settlement == "sim":
        summary = {name: score_sim(dfi, q) for name, q in policies.items()}
    else:
        summary = {name: score_observed(dfi, q, items) for name, q in policies.items()}

    # per-item detail for the headline policies
    score = score_sim if args.settlement == "sim" else \
        (lambda rows, q: score_observed(rows, q, items))
    per_item = {}
    for item, grp in dfi.groupby("item"):
        m = (dfi.item == item).values
        per_item[item] = dict(
            name=items[item]["name"],
            q_star=round(q_star[item], 3),
            sq=score(grp, policies["status_quo"][m]) if "status_quo" in policies else None,
            dl=score(grp, policies["dl"][m]) if "dl" in policies else None,
        )

    # forecast accuracy of the median forecast on the test rows
    wape = {}
    for name in ("dl", "naive", "ridge"):
        units = forecasts[name]
        if args.settlement == "sim":
            p50 = units[test][:, list(taus).index(0.5)]
            d = sim_truth(dfi)
            wape[name] = float(np.abs(p50 - d).sum() / d.sum())
        else:
            wape[name] = _quantile_scores(units, taus, dfi, test)["wape_uncensored"]

    charts = _charts(dfi, df, policies, items, args.settlement)

    out = dict(
        test_year=int(pd.Timestamp(dfi.date.max()).year),
        n_test_days=int(dfi.date.nunique()),
        taus=list(map(float, taus)),
        q_star={k: round(v, 3) for k, v in q_star.items()},
        wape=wape,
        summary=summary,
        per_item=per_item,
        charts=charts,
    )
    if args.settlement == "observed":
        # extra provenance, appended so the frozen sim-settlement bytes are unchanged
        out["settlement"] = "observed"
        out["wape_basis"] = "uncensored realized sales"
        out["spec_hash"] = b["spec_hash"]
        out["sellout_source"] = b["sellout_source"]
        out["censoring_known"] = bool(b["censoring_known"])
        out["items_config_hash"] = ht_config.config_hash(args.items)

    # allow_nan=False after the sweep: a bare NaN token is not JSON, and this file is read
    # by poc/dashboard.html's JSON.parse, which stops at the first one
    payload = json.dumps(_json_safe(out), indent=1, allow_nan=False)
    _guard_frozen_bytes(args.out, payload)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(payload)

    _report(summary, wape, [p for p in PRINT_ORDER if p in summary], args.settlement)
    print(f"wrote {os.path.abspath(args.out)}")
    return 0


def _match_service(df, b, dl_units, dfi, items, taus, q_star, args, spec):
    """Per item, the tau whose calibration-window service matches the status quo's.

    Under sim settlement the target is the realized FILL RATE, which needs true
    demand. Under observed settlement it is the observed SELLOUT-DAY RATE -- the
    share of days the store ran out -- which is what a store can actually measure,
    and which a quantile is a more direct statement about anyway (the note below
    the original code already said so).
    """
    if args.calib_window:
        end = pd.Timestamp(spec["test_start"]) - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=args.calib_window - 1)
    else:
        start, end = pd.Timestamp(CALIB_START), pd.Timestamp(CALIB_END)
    cal = (b["date"] >= np.datetime64(start)) & (b["date"] <= np.datetime64(end))
    if not cal.any():
        raise SystemExit(f"no rows in the service-matching window {start.date()}..{end.date()}; "
                         "pass --calib-window DAYS to size it from the panel's own range")
    cal_key = pd.MultiIndex.from_arrays([b["item"][cal], b["date"][cal]])
    df_cal = df.set_index(["item", "date"]).loc[cal_key].reset_index()
    dl_cal = dl_units[cal]

    matched = {}
    for item in b["items"]:
        m = (df_cal.item == item).values
        it = items[item]
        rows = df_cal[m]
        qs = np.array([[newsvendor.quantity(np.array([v]), np.array([0.5]), 0.5,
                                            it["batch"], it.get("continuous", False))
                        for v in dl_cal[m][:, j]] for j in range(len(taus))])
        if args.settlement == "sim":
            d_cal = sim_truth(rows)
            target = rows.sold.values.sum() / max(d_cal.sum(), 1e-9)
            # fill rate rises with tau, so the curve is already increasing in tau
            curve = np.array([np.minimum(qj, d_cal).sum() / max(d_cal.sum(), 1e-9)
                              for qj in qs])
        else:
            sold = rows.sold.values.astype(float)
            cens, known, _ = _censoring(rows)
            target = float(cens[known > 0].mean()) if (known > 0).any() else 0.0
            # sellout rate FALLS as tau rises; np.interp needs an increasing x, so
            # both sides are negated rather than the tau grid reversed.
            curve = -np.array([float((qj < sold).mean()) for qj in qs])
            target = -target
        matched[item] = float(np.clip(np.interp(target, curve, taus), taus[0], taus[-1]))
    return matched


def _json_safe(value):
    """Non-finite floats -> null. A policy with no quantity on a row is genuinely nothing."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _charts(dfi, df, policies, items, settlement):
    charts = {}
    lo, hi = _chart_window(dfi, settlement)
    window = (dfi.date >= lo) & (dfi.date <= hi)
    for item in _chart_items(dfi, items):
        m = ((dfi.item == item) & window).values
        series = dict(dates=[d.strftime("%Y-%m-%d") for d in dfi.date[m]])
        if settlement == "sim":
            series[SIM_DEMAND] = list(map(float, sim_truth(dfi)[m]))
        else:
            series["sold"] = list(map(float, dfi.sold.values[m]))
        for name in ("status_quo", "dl"):
            if name in policies:
                series[name] = list(map(float, policies[name][m]))
        series["holidays"] = [h if isinstance(h := dfi.holiday.values[i2], str) else ""
                              for i2 in np.where(m)[0]]
        charts[f"series_{item}"] = series

    if "status_quo" in policies and "dl" in policies:
        charts.update(_savings_chart(dfi, policies, items, settlement))

    dow_weather, seasonality, basis = _demand_structure(df, settlement)
    charts["dow_weather"] = dow_weather
    charts["seasonality"] = seasonality
    if settlement != "sim":
        charts["demand_basis"] = basis
    return charts


def _savings_chart(dfi, policies, items, settlement):
    price = dfi.unit_price.values
    cost = dfi.unit_cost.values
    if settlement == "sim":
        # cumulative economic saving (waste cost + lost margin) of dl over the status quo
        daily = {}
        d_true = sim_truth(dfi)
        for name in ("status_quo", "dl"):
            q = policies[name]
            sold = np.minimum(q, d_true)
            econ = (q - sold) * cost + (d_true - sold) * (price - cost)
            daily[name] = pd.Series(econ).groupby(dfi.date.values).sum()
        sav = (daily["status_quo"] - daily["dl"]).cumsum()
        return {"cumulative_savings": dict(
            dates=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in sav.index],
            dollars=[round(float(v), 2) for v in sav.values],
        )}

    # observed: the only rigorous series is the waste-saving LOWER bound at cost,
    # over day-fresh rows carrying a production record. It is not econ savings and
    # is not named as if it were.
    sold = dfi.sold.values.astype(float)
    produced = dfi.produced.values.astype(float) if "produced" in dfi else \
        np.full(len(dfi), np.nan)
    fresh = np.array([items[k]["shelf_life_days"] == 1 for k in dfi.item.values])
    have = np.isfinite(produced) & fresh
    sq = np.where(have, np.maximum(produced - sold, 0.0), 0.0)
    mdl = np.where(have, np.maximum(policies["dl"] - sold, 0.0), 0.0)
    sav = pd.Series((sq - mdl) * cost).groupby(dfi.date.values).sum().cumsum()
    return {"cumulative_waste_saving_lower": dict(
        dates=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in sav.index],
        dollars=[round(float(v), 2) for v in sav.values],
    )}


def _report(summary, wape, selected, settlement):
    if settlement == "sim":
        print(f"{'policy':12s} {'waste retail$':>13s} {'waste cost$':>12s} "
              f"{'lost margin$':>13s} {'econ cost$':>11s} {'fill':>6s} {'waste%':>7s}")
        for name in selected:
            s = summary[name]
            print(f"{name:12s} {s['waste_retail']:13,.0f} {s['waste_cost']:12,.0f} "
                  f"{s['lost_margin']:13,.0f} {s['econ_cost']:11,.0f} "
                  f"{s['fill_rate']:6.1%} {s['waste_pct_of_production']:7.1%}")
        print(f"\nWAPE (median forecast vs true demand): "
              + ", ".join(f"{k} {v:.1%}" for k, v in wape.items()))
        if "status_quo" in summary and "dl" in summary:
            sq, dl = summary["status_quo"], summary["dl"]
            print(f"\ndl vs status quo: waste retail "
                  f"-{1 - dl['waste_retail']/sq['waste_retail']:.1%}, "
                  f"econ cost -{1 - dl['econ_cost']/sq['econ_cost']:.1%}, "
                  f"fill {sq['fill_rate']:.1%} -> {dl['fill_rate']:.1%}")
        return

    print("OBSERVED SETTLEMENT: only the store's own waste is measured; every policy column "
          "below is a one-sided bound.")
    any_row = summary[selected[0]]
    print(f"MEASURED waste (produced - sold, day-fresh rows with a production record): "
          f"{any_row['waste_observed_units']:,.0f} units, "
          f"${any_row['waste_observed_cost']:,.0f} at cost, "
          f"${any_row['waste_observed_retail']:,.0f} at retail "
          f"over {any_row['n_rows_measured']:,} of {any_row['n_rows']:,} rows")
    print(f"\n{'policy':12s} {'produced':>10s} {'waste<=$':>10s} {'saving>=$':>10s} "
          f"{'lost units>=':>12s} {'sellout>=':>9s}")
    for name in selected:
        s = summary[name]
        # measured status-quo waste minus the saving bound is the policy's own waste
        # upper bound in cost dollars; bounds() reports that side only in units.
        waste_upper = s["waste_observed_cost"] - s["waste_saving_lower_cost"]
        print(f"{name:12s} {s['produced']:10,.0f} {waste_upper:10,.0f} "
              f"{s['waste_saving_lower_cost']:10,.0f} {s['lost_units_lower']:12,.0f} "
              f"{s['sellout_days_model_lower']:9.1%}")
    print("\nWAPE (median forecast vs realized sales, uncensored rows only): "
          + ", ".join(f"{k} {v:.1%}" for k, v in wape.items()))
    print("Lost margin is bounded from below only: an upper bound would need an upper "
          "bound on demand, which nothing observable provides.")


if __name__ == "__main__":
    raise SystemExit(main())
