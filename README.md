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
python -m model.backtest   # replay 2025 under six policies -> results/results.json
```

`data/store_synth.csv`, `model/artifacts/` and `results/results.json` are checked in and
frozen: the proposal's dollar figures are settled against them, and `model.backtest` above
reproduces `results.json` byte for byte. Regenerating any of them changes those figures, so
none of the three writers will do it by accident:

```bash
python -m sim.generate --out .rehearsal/synth.csv   # writing the frozen CSV needs --force-frozen
python -m model.train --artifacts .rehearsal/art    # writing model/artifacts/ needs --force-frozen
python -m model.backtest --policies dl,naive        # refused: that run is not the frozen replay
python -m model.backtest --settlement observed --out .rehearsal/real.json   # fine: its own --out
```

The first two refuse the frozen *path* and take `--force-frozen`. The third cannot work that
way, because `python -m model.backtest` with no arguments **is** the provenance of
`results.json` and has to go on reproducing it — so it refuses the *configuration* instead:
anything that would change the numbers (a `--panel`, a different split, a policy subset,
another `--artifacts` or `--items`, observed settlement) has to name its own `--out`. Before
that guard existed, `--policies dl,naive` silently replaced the six-policy file with a
two-policy one. And because a retrained checkpoint moves the numbers with no flag to notice
it by, the plain command checks the bytes as well: it writes `results.json` only while it
still reproduces it, and refuses with the diff to make rather than replacing it quietly.

Because the generator will write anywhere it is pointed, one more claim is checkable:
regenerating into scratch reproduces the frozen CSV byte for byte — `cmp data/store_synth.csv
.rehearsal/synth.csv` is silent — which is what makes "the seed is fixed" a provenance claim
rather than a comment.

Model: GRU encoder over each item's trailing 28 days + item embeddings + calendar/weather
covariates → 11 demand quantiles (pinball loss, censored on sellout days, since sales only
bound demand from below when the case sold out). Decision layer: per-item newsvendor critical
fractile, batch-rounded. See `data/README.md` for what the simulation does and doesn't claim.

## The real-data path

The pipeline no longer reads only the simulator's CSV. `ht/` turns a store's own raw export
into one canonical panel, and everything downstream -- features, training, evaluation, the
morning sheet -- runs off that panel. The simulator is now just one producer of a schema a real
store can also produce.

**It has never seen real data.** What has been run is a dress rehearsal: the synthetic store
exported as a raw, real-shaped store file with the simulation-only columns stripped, the columns
renamed, and deliberate dirt added (CRLF and a declared cp1252 encoding, a report title block and
a `DEPT TOTAL` footer, `MM/DD/YY`, parenthesised refunds, a duplicated export window, a multi-day
outage, random-weight barcodes, an item number that changes mid-history, an unmapped code, a
weather condition no alias table knows, and the same report as the district office runs it —
a `STORE` column and a second store's rows), then the whole real chain run over it:

```bash
bash scripts/rehearse.sh          # 3m04 on 4 CPUs here; --fast for the 6-epoch CI configuration
```

It fails on a one-byte change to the frozen artifacts. It writes under `.rehearsal/` with one
exception: step 9 re-runs the frozen backtest, which rewrites `results/results.json` — the file
is copied first, compared byte for byte, and restored if it ever differs.
The individual steps, against any canonical panel:

```bash
python -m ht.ingest   --mapping MAP.json --items ITEMS.json --out panel.csv --report ingest.json
python -m ht.validate --panel panel.csv --items ITEMS.json --mapping MAP.json \
    --ingest-report ingest.json
python -m model.train --panel panel.csv --items ITEMS.json --spec auto --artifacts artifacts/
python -m model.evaluate --panel panel.csv --artifacts artifacts/ --items ITEMS.json --split test

# the daily loop: print it, the kitchen writes on it, key it back in, score it
python -m model.shadow morning --panel panel.csv --artifacts artifacts/ --items ITEMS.json \
    --date 2026-03-02 --out shadow --format both
python -m model.shadow enter   --items ITEMS.json --date 2026-03-02 --out shadow --by kmurphy
python -m model.shadow score   --panel panel.csv --items ITEMS.json --date 2026-03-02 \
    --out shadow
python -m model.shadow weekly  --panel panel.csv --items ITEMS.json --artifacts artifacts/ \
    --week-ending 2026-03-07 --out shadow
```

`--spec` is required whenever `--panel` is: `model.train` and `model.backtest` refuse to pick a
split for you, because the frozen default boundaries are the simulator's own dates. `enter` reads
`item, made, sold out at[, note]` per line — from a pipe, a file (`--file`), or an item-by-item
prompt that walks the page in the order it printed (departments, then CARRY-OVER, then NO
FORECAST) showing what it said — and writes `shadow/overrides/<date>.csv`. The item may be
typed as the sheet printed it, truncated ITEM column and all. Nothing is
written unless every line parses. On a row keyed in that way, and only there, an **empty** "sold
out at" cell counts as "it did not sell out", which is what lets a store with no sellout column
in its export measure accuracy on fully-served days at all.

`MAP.json` and `ITEMS.json` start as `config/source_mapping.example.json` and
`config/items.example.json`. `docs/DATA_CONTRACT.md` is what a store's category manager or IT
contact reads; `docs/REAL_DATA_READINESS.md` is the day-one checklist, including what is ready,
what still needs the store, and what is still missing.

Evaluation on a real panel uses only what a store can see: accuracy on days where demand was
fully served, calibration as a two-sided bracket rather than a point, and economics reported as
measurements and one-sided bounds instead of point estimates. Nothing in that path reads
`true_demand`, `true_mean` or `lost_sales` -- those exist only inside the simulator.

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
- **Recovery scenarios** = annualized × 10% / 15% / 20%. That band is a commonly quoted
  first-year range for perishable waste pilots; nothing in this repo measures it, so it is an
  expectation, not a result. Pitch the middle, not the top.

## Roadmap

- **Phase 1:** hand-logged baseline (this tool). ✅
- **Proof of concept:** demand network + newsvendor on synthetic data, validated in shadow
  replay. ✅ (`sim/`, `model/`)
- **Phase 2 — plumbing built, rehearsed, still unused.** The code to consume a store's export
  exists: a documented data contract, a mapping-driven ingest, a validation gate, splits derived
  from the panel's own date range, and training that takes about a minute. It has been run end
  to end on a mock export built from the synthetic store with the simulation-only columns
  stripped — `bash scripts/rehearse.sh`. **It has never been run on real data**, and no
  permission has been sought yet. When permission lands, the work is writing the item file and
  the source mapping and then running the chain, not building it. `docs/REAL_DATA_READINESS.md`
  is the day-one checklist and is candid about what still needs the store — chiefly real costs,
  batch sizes, and whether any record of daily production exists at all.
- **Phase 3:** shadow mode — the model prints its morning sheet, nobody follows it, and its
  forecasts are scored against reality for four weeks. The commands exist
  (`python -m model.shadow morning|enter|score|weekly`), the prediction log is append-only, and
  the five go/no-go criteria are fixed in advance and printed on every weekly report from week
  one. `enter` is the return path: what the kitchen wrote on the printed sheet goes back in by
  hand, which for a store whose export carries no sellout column is the only way accuracy on
  fully-served days can be measured at all.
- **Phase 4:** live pilot with a measured before/after against the Phase 1 baseline. The
  baseline is not re-keyed: `python -m ht.ingest --logger-backup <the JSON from Setup > Data >
  Download backup>` folds this app's logged markouts into the panel's `wasted` column. The
  precedence is stated, applies cell by cell, and is what keeps the headline number honest — a
  waste cell the export itself supplied wins, then the logger, then `produced - sold` for
  day-fresh items only. Cell by cell matters: a shrink report covering one department is
  ordinary, and it must not switch the derivation off for every other item in the panel. The
  ingest report says how many cells came from each (`waste_cells`).
  So the logger does replace a `produced - sold` figure, which is arithmetic on a label-printer
  proxy, and it never *adds* to a waste number the export reported (two records of one markout,
  summed, would double exactly the number being claimed); those collisions keep the export's
  figure and are reported with both. Logged item-days outside the export's date range, logger
  items matching no configured item, repeated entry ids and unreadable dates are counted and
  named rather than folded in, and a backup where nothing matched is refused. `scripts/rehearse.sh`
  does not exercise this path — there is no logger backup in the mock export — so it is covered
  by `tests/test_ingest_logger.py` and by hand. Only if all five gates read PASS.

Three things the rehearsal deliberately does **not** establish. The published checkpoint does not
transfer — it is shape-locked to these nine items, so Phase 2 means retraining, not reusing.
On the synthetic panel every candidate sellout rule agrees on every row, because the simulator
defines them as the same event, so a green rehearsal says nothing about which rule a real store
needs. And `tools/make_mock_export.py` and `ht/ingest.py` were written in this repo against each
other: the same person chose the dirt and the handling, so the rehearsal tests an importer
against its own author's imagination, and the first real export will break in a way nobody here
thought of.

## A note on data and permission

This tool is a digital clipboard for what you personally observe being thrown away — the same
thing you could do on paper. It does not connect to, scrape, or store anything from company
systems. Before Phase 2 (which needs real sales data), get written permission for data access and
clarity on IP ownership first. That ordering protects both the project and you.
