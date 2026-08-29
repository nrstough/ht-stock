# HT Stock — Prepared Food Waste: Measure, Predict, Reduce

Cut grocery prepared-food waste (bakery, pizza, hot foods, fresh cases) by replacing gut-feel
par sheets with demand forecasts. Three pieces, all in this repo:

1. **Phase 1 — measure** (`index.html`): a phone-friendly waste logger that produces a store's
   real baseline in two weeks of use. Nobody funds "reduces waste 15%"; people fund
   "recovers $X/year at this store." This tool produces $X.
2. **Proof of concept — predict** (`sim/`, `model/`): a synthetic 3-year store, a global quantile
   demand network, a newsvendor decision layer, and a shadow-replay backtest of the held-out
   year. Built on **zero real data** so it settles the method without touching data-permission
   or IP questions.
3. **The pitch** (`proposal/`, `poc/`): an executive proposal and an interactive results
   dashboard generated from the backtest.

## Proof-of-concept results (simulated held-out year, one store)

| Policy | Waste (retail) | Waste % of production | Fill rate | Economic cost |
|---|---|---|---|---|
| Status quo (par sheet) | $138,462 | 18.8% | 98.1% | $56,011 |
| Model, availability held | $87,351 (−37%) | 12.8% | 97.6% | $41,078 |
| Model, profit-optimal | $55,883 (−60%) | 8.8% | 95.5% | $39,723 |
| Oracle (perfect knowledge) | $57,435 | 8.9% | 96.6% | $35,788 |

The network's forecast error (13.0% WAPE) beats a trailing-average (15.5%) and a linear model
(15.1%), and captures ~81% of the improvement perfect knowledge would allow. Every number is
reproducible:

```bash
pip install -r requirements.txt
python -m sim.generate     # 3 years of synthetic store data -> data/store_synth.csv
python -m model.train      # train the quantile network (CPU, ~1 min)
python -m model.backtest   # replay 2025 under six policies -> results/results.json
```

Model: GRU encoder over each item's trailing 28 days + item embeddings + calendar/weather
covariates → 11 demand quantiles (pinball loss, censored on sellout days, since sales only
bound demand from below when the case sold out). Decision layer: per-item newsvendor critical
fractile, batch-rounded. See `data/README.md` for what the simulation does and doesn't claim.

---

## Phase 1: the waste logger

## What it does

- **Log** — tap an item, set a quantity, pick a reason, done. Built for one-handed use in the back
  room. Supports backdating and decimal quantities (hot bar by the pound).
- **History** — every entry, grouped by day, with day totals.
- **Stats** — documented waste in units and dollars, weekly run rate, annualized baseline
  (weekly × 52), 10/15/20% recovery scenarios, breakdowns by item, department, and day of week.
- **Pitch** — a copy-ready, print-ready one-page summary generated from your real data, including
  district-level extrapolation. It warns you if you have less than 14 days of coverage.
- **Setup** — manage items and retail prices, export CSV for spreadsheets, and back up / restore
  your data as JSON.

## How to use it

The whole app is a single file with no dependencies, no build step, and no server.

- **On a computer:** open `index.html` in any browser.
- **On your phone (recommended):** enable GitHub Pages for this repo (Settings → Pages → deploy
  from branch), open the URL on your phone, and add it to your home screen. It behaves like an app.
- Everything is stored in the browser's local storage **on that device only**. Nothing is uploaded
  anywhere. Back up regularly from the Setup tab — clearing the browser clears the data.

### The two-week baseline

1. In **Setup**, fix the item list and enter your store's real retail prices. The defaults are
   placeholders.
2. For at least **14 consecutive days**, log every discarded item you see across your pilot items.
3. Open **Pitch**, set your district's store count, and copy or print the summary.

## How the math works

- **Weekly run rate** = total documented waste ÷ calendar days covered × 7. Calendar span is used
  (not just days with entries), so days you logged nothing count as zero-waste days — conservative
  by design.
- **Annualized baseline** = weekly × 52. This ignores seasonality; the app says so wherever the
  number appears.
- **Recovery scenarios** = annualized × 10% / 15% / 20%, the typical first-year range for retail
  perishable waste-reduction pilots. Pitch the middle, not the top.

## Roadmap

- **Phase 1:** hand-logged baseline (this tool). ✅
- **Proof of concept:** demand network + newsvendor on synthetic data, validated in shadow
  replay. ✅ (`sim/`, `model/`)
- **Phase 2:** with written permission, retrain the same model on the store's real sales
  export.
- **Phase 3:** shadow mode — the model prints its morning sheet, nobody follows it, and its
  forecasts are scored against reality for four weeks.
- **Phase 4:** live pilot with a measured before/after against the Phase 1 baseline.

## A note on data and permission

This tool is a digital clipboard for what you personally observe being thrown away — the same
thing you could do on paper. It does not connect to, scrape, or store anything from company
systems. Before Phase 2 (which needs real sales data), get written permission for data access and
clarity on IP ownership first. That ordering protects both the project and you.
