"""Generate the synthetic store dataset.

Run:  python -m sim.generate --out .rehearsal/store_synth.csv

Writes one row per item per day. Columns a real store could observe are the
first group; `true_demand`, `lost_sales`, and `true_mean` exist only because
this is a simulation and are used solely for scoring policies, never for
training.

It refuses to overwrite data/store_synth.csv without --force-frozen, for the
same reason model/train.py refuses to write into model/artifacts/: those bytes
are provenance. results/results.json, the checkpoint and every dollar figure in
the proposal are settled against that one file, and a re-run that lands on it --
after a params change, on another numpy, or by simply typing the default -- moves
the ground under all of them silently. The seed is fixed, so a clean re-run into
a scratch path and `cmp` against the frozen file is the check that this
generator still produces it.
"""
import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd

from . import calendar_events, demand, params, weather
from .policy import StatusQuoManager


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


FROZEN = os.path.join(os.path.dirname(__file__), "..", "data", "store_synth.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description="generate the synthetic store dataset")
    ap.add_argument("--out", default=FROZEN, help="where to write the CSV")
    ap.add_argument("--force-frozen", action="store_true",
                    help="permit writing data/store_synth.csv, whose bytes back the proposal")
    args = ap.parse_args(argv)
    if os.path.realpath(args.out) == os.path.realpath(FROZEN) and not args.force_frozen:
        raise SystemExit(
            "refusing to write data/store_synth.csv: those bytes are provenance -- "
            "results/results.json, model/artifacts/ and every dollar figure in the proposal "
            "are settled against them. Write somewhere else, e.g. "
            "--out .rehearsal/store_synth.csv, and cmp the two if you mean to check that this "
            "generator still reproduces the frozen file; or pass --force-frozen if you really "
            "do mean to replace it.")

    rng = np.random.default_rng(params.SEED)
    start = dt.date.fromisoformat(params.START)
    end = dt.date.fromisoformat(params.END)
    dates = list(daterange(start, end))
    n = len(dates)

    tmax, wkind, snow_tomorrow = weather.simulate(dates, rng)
    holidays = calendar_events.holiday_map(start, end)
    event_days = rng.random(n) < params.STORE["event_prob"]
    traffic_noise = np.exp(
        rng.normal(0, params.STORE["traffic_noise_sigma"], n)
        - params.STORE["traffic_noise_sigma"] ** 2 / 2
    )

    managers = {k: StatusQuoManager(k, it) for k, it in params.ITEMS.items()}
    rows = []
    for i, date in enumerate(dates):
        holiday = holidays.get(date)
        closed = holiday and params.HOLIDAYS[holiday] == "closed"
        payday = date.day in (1, 2, 3, 15, 16, 17)
        event_mult = params.STORE["event_mult"] if event_days[i] else 1.0
        is_snow = wkind[i] == "snow"
        is_rain = wkind[i] == "rain"

        for key, item in params.ITEMS.items():
            mean = demand.latent_mean(
                key, item, date, i, wkind[i], tmax[i],
                snow_tomorrow[i], holiday, payday, event_mult,
            )
            true_d = demand.realize(item, mean, traffic_noise[i], rng)

            if closed:
                produced = sold = wasted = lost = 0.0
            else:
                produced = managers[key].decide(date, holiday, snow_tomorrow[i], is_snow, is_rain)
                sold = min(produced, true_d)
                wasted = produced - sold
                lost = true_d - sold
                managers[key].observe(date, produced, sold)

            rows.append(dict(
                # ---- observable by a real store ----
                date=date.isoformat(),
                item=key, item_name=item["name"], dept=item["dept"],
                dow=date.weekday(),
                holiday=holiday or "",
                is_closed=int(bool(closed)),
                tmax_f=tmax[i], weather=wkind[i],
                snow_tomorrow=int(snow_tomorrow[i]),
                payday=int(payday),
                produced=produced, sold=sold, wasted=wasted,
                stockout=int(produced > 0 and sold >= produced),  # observable sellout flag
                unit_price=item["price"], unit_cost=item["cost"],
                # ---- simulation-only (scoring, never training) ----
                true_demand=true_d,
                lost_sales=lost,
                true_mean=round(mean * 1.0, 3),
            ))

    df = pd.DataFrame(rows)
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)

    # quick sanity report
    open_rows = df[df.is_closed == 0]
    waste_retail = (open_rows.wasted * open_rows.unit_price).sum()
    prod_retail = (open_rows.produced * open_rows.unit_price).sum()
    lost_margin = (open_rows.lost_sales * (open_rows.unit_price - open_rows.unit_cost)).sum()
    years = n / 365.0
    print(f"rows: {len(df)}  ({n} days x {len(params.ITEMS)} items)")
    print(f"waste (retail $): {waste_retail:,.0f} total | {waste_retail/years:,.0f}/yr "
          f"| {waste_retail/prod_retail:.1%} of production")
    print(f"lost margin ($): {lost_margin:,.0f} total | {lost_margin/years:,.0f}/yr")
    print(f"stockout day rate: {open_rows.stockout.mean():.1%}")
    print("\nper item (waste% of production, stockout rate):")
    for key, grp in open_rows.groupby("item"):
        wpct = grp.wasted.sum() / max(grp.produced.sum(), 1)
        print(f"  {key:12s} waste {wpct:5.1%}  stockouts {grp.stockout.mean():5.1%}")
    print(f"\nwrote {os.path.abspath(out)}")


if __name__ == "__main__":
    raise SystemExit(main())
