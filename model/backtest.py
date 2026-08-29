"""Shadow-replay backtest on the held-out year (2025).

Run:  python -m model.backtest

For every open day of 2025 each policy names a production quantity; the
simulator's true demand (never seen by any model) settles what would have
sold, wasted, and been missed. Policies:

  status_quo  what the simulated store actually produced
  naive       trailing same-weekday average + gaussian quantiles + newsvendor
  ridge       linear model + gaussian quantiles + newsvendor
  dl          quantile network + newsvendor
  dl_matched  quantile network pinned to the status quo's availability level
              ("same service, less waste" scenario)
  oracle      knows the true demand distribution (ceiling, irreducible noise only)

Writes results/results.json with the summary, per-item detail, and the chart
series used by the PoC dashboard and the proposal.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

import sim.params as sim_params

from . import baselines, features, newsvendor
from .net import DemandNet

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")


def predict_dl(b):
    with open(os.path.join(ARTIFACTS, "meta.json")) as f:
        meta = json.load(f)
    model = DemandNet(len(meta["items"]), meta["ctx_dim"], meta["cov_dim"],
                      len(meta["taus"]))
    model.load_state_dict(torch.load(os.path.join(ARTIFACTS, "demandnet.pt"),
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


def score(df_rows, q):
    """Economics of producing q against true demand."""
    d = df_rows.true_demand.values
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


def main():
    df = features.load()
    b = features.build(df)

    # lag features for ridge, aligned to b's rows
    lag_full = baselines._lag_features(df, b["stats"])
    pos = {(it, d): i for i, (it, d) in enumerate(zip(df.item.values, df.date.values))}
    b["lags"] = lag_full[[pos[(it, d)] for it, d in zip(b["item"], b["date"])]]

    taus = b["taus"]
    test = b["split"] == "test"

    # ---- forecasts (z-space quantile matrices) ----
    dl_units = to_units(predict_dl(b), b)

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

    items = {k: sim_params.ITEMS[k] for k in b["items"]}
    traffic_sig = sim_params.STORE["traffic_noise_sigma"]

    q_star = {k: newsvendor.critical_fractile(it["price"], it["cost"])
              for k, it in items.items()}

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
        out = np.zeros(len(dfi))
        for i, row in enumerate(dfi.itertuples()):
            it = items[row.item]
            sig = np.hypot(it["sigma"], traffic_sig)
            z = baselines._norm_ppf(np.array([q_star[row.item]]))[0]
            q = row.true_mean * np.exp(sig * z - sig ** 2 / 2)
            out[i] = newsvendor.quantity(np.array([q]), np.array([0.5]), 0.5,
                                         it["batch"], it.get("continuous", False))
        return out

    policies = {
        "status_quo": dfi.produced.values.astype(float),
        "naive": decide(naive_units, q_star),
        "ridge": decide(ridge_units, q_star),
        "dl": decide(dl_units, q_star),
        "oracle": oracle_q(),
    }

    # service-matched dl: per item, find on 2024 the quantile whose realized
    # fill rate matches the status quo's, then apply that quantile to 2025.
    # (a quantile is a no-sellout probability, not a fill rate -- fill runs
    # much higher because sellout days still serve most of their demand)
    cal = (b["date"] >= np.datetime64("2024-01-01")) & \
          (b["date"] <= np.datetime64("2024-12-31"))
    cal_key = pd.MultiIndex.from_arrays([b["item"][cal], b["date"][cal]])
    df_cal = df.set_index(["item", "date"]).loc[cal_key].reset_index()
    dl_cal = dl_units[cal]
    tau_matched = {}
    for item in b["items"]:
        m = (df_cal.item == item).values
        it = items[item]
        d_cal = df_cal.true_demand.values[m]
        sq_fill = df_cal.sold.values[m].sum() / max(d_cal.sum(), 1e-9)
        fills = []
        for j in range(len(taus)):
            qj = np.array([newsvendor.quantity(np.array([v]), np.array([0.5]), 0.5,
                                               it["batch"], it.get("continuous", False))
                           for v in dl_cal[m][:, j]])
            fills.append(np.minimum(qj, d_cal).sum() / max(d_cal.sum(), 1e-9))
        fills = np.array(fills)
        # smallest tau achieving the status-quo fill (interp on the fill curve)
        tau_matched[item] = float(np.clip(np.interp(sq_fill, fills, taus), taus[0], taus[-1]))
    policies["dl_matched"] = decide(dl_units, tau_matched)

    summary = {name: score(dfi, q) for name, q in policies.items()}

    # per-item detail for the headline policies
    per_item = {}
    for item, grp in dfi.groupby("item"):
        m = (dfi.item == item).values
        per_item[item] = dict(
            name=items[item]["name"],
            q_star=round(q_star[item], 3),
            sq=score(grp, policies["status_quo"][m]),
            dl=score(grp, policies["dl"][m]),
        )

    # forecast accuracy (median forecast vs true demand, test rows)
    wape = {}
    for name, units in (("dl", dl_units), ("naive", naive_units), ("ridge", ridge_units)):
        p50 = units[test][:, list(taus).index(0.5)]
        wape[name] = float(np.abs(p50 - dfi.true_demand.values).sum()
                           / dfi.true_demand.sum())

    # ---- chart series ----
    charts = {}
    window = (dfi.date >= "2025-01-06") & (dfi.date <= "2025-03-02")
    for item in ("pizza-whole", "hotbar-lb"):
        m = ((dfi.item == item) & window).values
        charts[f"series_{item}"] = dict(
            dates=[d.strftime("%Y-%m-%d") for d in dfi.date[m]],
            true_demand=list(map(float, dfi.true_demand.values[m])),
            status_quo=list(map(float, policies["status_quo"][m])),
            dl=list(map(float, policies["dl"][m])),
            holidays=[h if isinstance(h := dfi.holiday.values[i2], str) else ""
                      for i2 in np.where(m)[0]],
        )

    # cumulative economic savings (dl vs status quo) across 2025
    daily = {}
    price = dfi.unit_price.values
    cost = dfi.unit_cost.values
    d_true = dfi.true_demand.values
    for name in ("status_quo", "dl"):
        q = policies[name]
        sold = np.minimum(q, d_true)
        econ = (q - sold) * cost + (d_true - sold) * (price - cost)
        daily[name] = pd.Series(econ).groupby(dfi.date.values).sum()
    sav = (daily["status_quo"] - daily["dl"]).cumsum()
    charts["cumulative_savings"] = dict(
        dates=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in sav.index],
        dollars=[round(float(v), 2) for v in sav.values],
    )

    # demand structure: dow x weather index (all 3 years, retail-value demand)
    dfall = df[df.is_closed == 0].copy()
    dfall["demand_retail"] = dfall.true_demand * dfall.unit_price
    day_val = dfall.groupby(["date", "dow", "weather"]).demand_retail.sum().reset_index()
    overall = day_val.demand_retail.mean()
    mat = day_val.groupby(["dow", "weather"]).demand_retail.mean() / overall
    charts["dow_weather"] = [
        dict(dow=int(dow), weather=w, index=round(float(v), 3))
        for (dow, w), v in mat.items()
    ]

    # monthly seasonality by dept (average daily retail demand, indexed)
    dfall["month"] = dfall.date.dt.month
    dept_month = dfall.groupby(["dept", "month", "date"]).demand_retail.sum() \
        .groupby(["dept", "month"]).mean()
    dept_avg = dept_month.groupby("dept").mean()
    charts["seasonality"] = [
        dict(dept=dept, month=int(mo), index=round(float(v / dept_avg[dept]), 3))
        for (dept, mo), v in dept_month.items()
    ]

    out = dict(
        test_year=2025,
        n_test_days=int(dfi.date.nunique()),
        taus=list(map(float, taus)),
        q_star={k: round(v, 3) for k, v in q_star.items()},
        wape=wape,
        summary=summary,
        per_item=per_item,
        charts=charts,
    )
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    # ---- console report ----
    sq, dl = summary["status_quo"], summary["dl"]
    print(f"{'policy':12s} {'waste retail$':>13s} {'waste cost$':>12s} "
          f"{'lost margin$':>13s} {'econ cost$':>11s} {'fill':>6s} {'waste%':>7s}")
    for name in ("status_quo", "naive", "ridge", "dl", "dl_matched", "oracle"):
        s = summary[name]
        print(f"{name:12s} {s['waste_retail']:13,.0f} {s['waste_cost']:12,.0f} "
              f"{s['lost_margin']:13,.0f} {s['econ_cost']:11,.0f} "
              f"{s['fill_rate']:6.1%} {s['waste_pct_of_production']:7.1%}")
    print(f"\nWAPE (median forecast vs true demand): "
          + ", ".join(f"{k} {v:.1%}" for k, v in wape.items()))
    print(f"\ndl vs status quo: waste retail -{1 - dl['waste_retail']/sq['waste_retail']:.1%}, "
          f"econ cost -{1 - dl['econ_cost']/sq['econ_cost']:.1%}, "
          f"fill {sq['fill_rate']:.1%} -> {dl['fill_rate']:.1%}")
    print(f"wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
