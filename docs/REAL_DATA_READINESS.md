# Real-data readiness

The checklist to read on the day permission is granted.

Every line in READY names a command that has actually been run. It was run against a *mock*
store export: the synthetic panel written back out as a raw store file with the simulator-only
columns stripped, every column renamed, and deliberate dirt added. That is a dress rehearsal,
not a pilot. **No part of this repo has ever seen real store data.** Where a claim below has no
command beside it, treat it as a claim rather than a capability.

Run the whole chain yourself:

```bash
bash scripts/rehearse.sh              # ~2 min on 4 CPUs; --fast for the 6-epoch CI configuration
```

It writes only under `.rehearsal/`, guards the four frozen provenance files by md5 before and
after, and exits non-zero on a one-byte change in `results/results.json`.

---

## READY — exercised end to end, no store input required

| Capability | Command |
|---|---|
| Raw export to canonical panel | `python -m ht.ingest --mapping MAP.json --items ITEMS.json --out panel.csv --report ingest.json` |
| Fitness gate over a panel | `python -m ht.validate --panel panel.csv --items ITEMS.json --mapping MAP.json` |
| Holidays / payday / weekday from real dates, any year | `python -c "import ht.calendar as c; print(c.easter(2026), c.super_bowl(2026))"` |
| Weather from a store CSV or a downloaded daily summary | `ht.weather.CsvWeather`; see `tests/test_weather.py` |
| Splits resolved from the panel's own date range | `python -m model.features --panel panel.csv --spec auto` |
| Training on a real panel | `python -m model.train --panel panel.csv --spec auto --items ITEMS.json --artifacts art/` |
| Evaluation with no simulator truth | `python -m model.evaluate --panel panel.csv --artifacts art/ --items ITEMS.json --split test` |
| Morning sheet, prediction log, day scoring, weekly report | `python -m model.shadow morning\|score\|catch-up\|weekly\|status` |
| Policy replay with no simulator truth | `python -m model.backtest --panel panel.csv --settlement observed --spec auto --out OUT.json` |

The rehearsal absorbs, and the validator names, the failure modes a real export actually has:
cp1252 with CRLF, a report title block above the header and a `DEPT TOTAL` footer below it,
`MM/DD/YY` dates, `$` and thousands separators, parenthesised refunds, a duplicated export
window, a multi-day outage, random-weight barcodes, an item number that changes mid-history, an
unmapped code, an item with too little history, and weather conditions no alias table knows.

`ht.weather.CsvWeather` reads either a file with condition text (mapped through the store's
`kind_map` and a built-in alias table) or a downloaded daily summary carrying `TMAX`, `PRCP` and
`SNOW` and no words at all; it uses the words first and falls back to the gauges only where a
word was unrecognised.

**"No sellout signal" is a supported mode, not a broken one.** `sellout.rule = "none"` ingests,
validates and trains end to end; the rehearsal proves it on a separate degraded run. The cost is
stated rather than compensated for: the model then fits the distribution of *sales* rather than
demand, so recommended quantities run roughly 1-8% low on the busiest days, and `ht.validate`,
`model.train`, `model.evaluate` and every morning sheet print that sentence.

---

## Day one, in order

Nothing below is automated, and steps 1 and 3 are the ones that take an afternoon.

**0. Get the day-close cutoff in writing** before anything else. See NEEDS THE STORE item 5.

**1. Write the item file.** Copy `config/items.example.json` and replace all nine records with
the store's real items, prices, costs and batch sizes. The key you choose for each item is the
join key for everything downstream, so pick it once and do not rename it later.

**2. Put the raw export somewhere** — one file, one per year, or one per department, all fine.

**3. Write the source mapping.** Copy `config/source_mapping.example.json`. It is written to
point at the rehearsal's mock files, so its `files[].path` values are `.rehearsal/raw/...` and
must be repointed. Then fix, in this order: `date.format` (an explicit strftime string; `"auto"`
is rejected on purpose, because `3/4/25` is two different dates), the `columns` block, the
`items.map` from raw item numbers to your keys, and `sellout.rule`. The rule key is required and
`"none"` is a valid, expected answer.

**4. Dry-run the ingest until it exits 0.** Each failure names both what is wrong and which
mapping field would authorise the repair:

```bash
python -m ht.ingest --mapping MAP.json --items ITEMS.json --dry-run
```

**5. Write the panel and its report.**

```bash
python -m ht.ingest --mapping MAP.json --items ITEMS.json \
    --out data/panel.csv --report data/ingest_report.json
```

Exit 1 means nothing was written. Exit 2 means the panel *was* written but validation found
errors — look at it rather than re-running.

**6. Read the validation report. Do not just check its exit code.**

```bash
python -m ht.validate --panel data/panel.csv --items ITEMS.json --mapping MAP.json
```

Exit 0 means no error-level findings; warnings can still be present and usually are. This is
also the document to hand back to the store when their export needs fixing. A single
mis-configured mapping field corrupts every downstream number uniformly, and the model will fit
the corruption without complaint, so this report is the only place that failure is visible.

**7. Train.** About a minute for a three-year panel at four threads.

```bash
python -m model.train --panel data/panel.csv --items ITEMS.json \
    --spec auto --artifacts artifacts/
```

`--spec auto` validates the panel first and exits 1 with the printed report on any error, so a
bad panel fails at the data layer with a data message rather than twelve epochs later inside
torch. Training refuses to write into `model/artifacts/` without `--force-frozen`; write
somewhere else.

**8. Evaluate the held-out split.**

```bash
python -m model.evaluate --panel data/panel.csv --artifacts artifacts/ \
    --items ITEMS.json --split test --json eval.json
```

**9. Run shadow mode, every morning, for four weeks.** The sheet is logged before it is
rendered, so a crash while printing still leaves the prediction on the record.

```bash
python -m model.shadow morning --panel data/panel.csv --artifacts artifacts/ \
    --items ITEMS.json --date 2026-03-02 --out shadow --store "Store 0123" --format both
python -m model.shadow score    --panel data/panel.csv --items ITEMS.json --date 2026-03-01 \
    --out shadow
python -m model.shadow catch-up --panel data/panel.csv --items ITEMS.json --out shadow
python -m model.shadow weekly   --panel data/panel.csv --items ITEMS.json --artifacts artifacts/ \
    --week-ending 2026-03-07 --out shadow --format both
python -m model.shadow status   --out shadow
```

`morning` refuses a panel that runs to or past the forecast date, refuses a panel more than
`--max-staleness` days behind, and refuses a past date unless `--backfill` — which stamps
`backfilled=1` and quarantines those rows out of every headline number.

---

## NEEDS THE STORE — the code is waiting, the data is not here

1. **The export itself.** Item movement at store x item x business-day grain, 104 weeks. The
   pilot's own weeks of logging are a validation window, not a training set. See the floors
   below. Details, in a store's own language, are in `docs/DATA_CONTRACT.md`.

2. **An item list with real prices and costs.** Every recommended quantity is a newsvendor
   critical fractile of price and cost; nothing else sets it. Cost appears in neither the
   Phase-1 logger nor a standard movement report, and is often a more sensitive ask than sales.
   Without it, cost is imputed from a department gross margin, `cost_imputed` is set, and every
   report using that item's dollars prints the assumption — but the imputation error goes
   straight into the quantity, so this is a guess wearing a decimal point.

3. **Batch sizes.** One tray, one bake, one pan. A recommendation is rounded to a whole batch
   and never goes below one, so a wrong batch size is a wrong sheet even with a perfect
   forecast. Nobody but the department can supply these.

4. **A production count — the sellout question.** This is the single most valuable column and
   the most important question in the permission conversation, ahead of asking for sales:
   *does any record exist of how much we put out each day?* A photographed clipboard keyed in
   is enough; a label-printer log is an accepted proxy. With it, the sellout flag is correct by
   construction and waste is measured for free. Without it the honest configuration is
   `sellout.rule = "none"`, and the consequence is the 1-8% low bias above plus no measured
   waste baseline at all — which means no denominator for any savings claim, and gate G4 reads
   PENDING forever. Both `produced - sold` and the flag derived from it assume every unit that
   left the case was scanned or discarded; employee meals, samples and department transfers
   break that assumption in the safe direction (they make waste look larger and sellouts look
   rarer). A proxy that undercounts production is tolerated: days where sales exceed it are
   counted and warned about rather than refused, up to `production.max_overrun_share`.

5. **The day-close cutoff, in writing.** `mapping.date.business_day` is asserted by whoever
   writes the mapping and is never verified. A cloud BI export in UTC shifts a US store 4-5
   hours and pushes evening sales into the next day, scrambling day-of-week — the model's
   strongest signal. The panel validates clean and the model trains happily on a Friday that is
   partly Saturday. No data-only test detects this reliably.

6. **Store hours or a closure list.** Otherwise a closed day is indistinguishable from a demand
   collapse and the model learns the collapse. The validator warns on any date where every open
   item sold zero and points at `mapping.closures.dates`, which catches the obvious cases and
   not the subtle ones.

7. **A weather source.** A store CSV, or a daily summary downloaded once and kept beside the
   panel. Nothing in this codebase makes a network call, ever, so no feed is fetched for you.
   With `provider: "none"` the pipeline runs and four covariate dimensions and two encoder
   channels sit dead.

**History floors**, from `model/features.py`, and what happens below them:

| Mode | Panel floor | Per-item floor | Command |
|---|---|---|---|
| train / val / test | 126 days | 84 open days | default |
| no held-out test | 98 days | 84 open days | `--no-test` |
| short history, provisional | 70 days | 56 open days | `--allow-short` |

Below the floor, `features.build` raises `InsufficientHistory` naming the actual span, the
requirement, its decomposition and the remedy. `ht.validate` predicts the same outcome from the
panel alone, and both agree, so a panel too short fails at the data layer.

The two floors count different things, and this bites at the margin: the panel floor counts
calendar days, the per-item floor counts *open* days. On the rehearsal panel `--allow-short` at
exactly 70 days fails with `sushi: 55 of 56 days`, because Thanksgiving sits closed inside the
56-day training window; 72 days works. Expect the effective floor to be the stated one plus the
number of closed days that land in the training window.

---

## KNOWN GAP

### Would block a pilot on the affected items

- **Multi-day shelf life is fenced off, not solved, and it lands on bread and cake.** For an
  item with `shelf_life_days > 1`, demand is served from mixed-age inventory, production on day
  *t* is not availability on day *t*, and `wasted = produced - sold` is simply false — so the
  single-period newsvendor is confidently wrong on two of the largest-dollar lines. Those items
  are excluded from the waste bound by name, marked `(multi-day - shadow only)` on the sheet,
  and never get `wasted` derived. The code is honest about them; it does not forecast them
  usefully. A multi-period age-cohort model is a separate piece of work. Set
  `shelf_life_days` truthfully before the live phase — the shipped example says 1 for bread and
  cake only because the simulator is day-fresh by construction.

- **Promotions, ads and markdowns are not modelled at all.** Ad data lives in a pricing system,
  not the movement report. An ad item can double for a week; the model sees the spike and fits
  it as noise, and the error inflates every quantile for that item. The `unit_price` divergence
  check is a *detector*, not a covariate, and it only fires if the export carries a dollars
  column — silence from it means "not measurable", not "no promotions". On a store with a heavy
  ad cadence this could dominate the residual. Markdowns are a related hole: dollars are net of
  markdown, which is why units are never derived from dollars, and a marked-down unit that sells
  is not waste but is not a full-price sale either. Nothing here distinguishes the two.

### Imperfect, not blocking

- **The frozen checkpoint does not transfer.** `model/artifacts/demandnet.pt` is shape-locked to
  nine items, `ctx_dim` 6 and `cov_dim` 35, with normalizers fitted on synthetic sales. "Point
  it at the export" means point it and *retrain*. Two things routinely change the covariate
  width on a real panel: a panel under 540 days degrades the Fourier terms, and a single
  `unknown` weather day appends a fifth weather kind — the rehearsal panel comes out at
  `cov_dim` 36 for exactly that reason. `features.assert_compatible` refuses the mismatch with a
  named list of differences rather than a torch shape error.

- **Gate G3 can be unresolvable rather than failed.** Its second term is `cov_lo` at tau 0.90,
  and `cov_lo` is a strict lower bound that widens with the sellout rate: on the rehearsal it
  reads 0.663 against a threshold of 0.75, while `cov_hi` reads 0.855, so the true coverage is
  somewhere in a bracket that straddles the gate. Its first term, `cov_point`, is measured only
  on rows where demand was exactly observed — sellout days excluded — which is a biased
  subsample, since those are the high-demand days; on the rehearsal `cov_point` sits above
  `cov_hi` at every tau. Read the bracket, and do not tune the model to hit G3. With no sellout
  signal at all there are no observed rows and G3 correctly reads PENDING.

- **The history floors are judgement, not measurement.** 84 open days per item and 126 panel
  days encode "eight observations of every weekday". They were never tuned by measuring accuracy
  against history length, because the only data available for tuning is synthetic. Re-measure
  them during the pilot and expect them to move.

- **Train/serve skew on weather.** `snow_tomorrow` in the synthetic data is a noiseless one-day
  lookahead, so a model that learns to lean on it will not transfer; a real probabilistic
  forecast is strictly worse. Where no forecast column exists the value is hindcast from the
  next day's observed condition, and the hindcast is 0 on the last day of the feed by
  construction — so tomorrow's sheet sees `snow_tomorrow = 0` unless the store supplies a real
  forecast column. Expect shadow-mode accuracy below the backtest for this reason alone.

- **Item identity is a hand-written table.** Random-weight barcode extraction and the explosion
  ceiling catch the mechanical failures. A product discontinued under one item number and
  re-added under another is only stitched back together if somebody notices and writes the
  alias; otherwise one series becomes two short ones, both fail the per-item floor and drop out,
  visible only to someone who reads the excluded-item table. A product that quietly changes what
  it is under a constant item number is undetectable by anything here.

- **`row_status = "not_carried"` is declared and never produced.** The schema enum lists it and
  no code path assigns it; an item's out-of-span days are simply absent from the grid instead.
  Nothing depends on it today, but the enum promises a distinction the ingest does not make.

- **Soft censoring is not available.** `model/net.py`'s censored pinball is a hard
  `torch.where`, so a probabilistic sellout weight would need that file changed.

- **The Phase-1 logger has no "sold out?" toggle.** Adding one to `index.html` would produce the
  cheapest sellout ground truth this project will ever get, and the code already accepts an
  out-of-stock log through the `flag` rule and through the sheet's write-in columns.

- **Live weather is deliberately not wired.** `ht.weather.LiveWeather.frame` raises
  `NotImplementedError` naming what a real implementation must do. A real one should fetch
  archived *forecasts* for training rather than archived observations.

- **The legacy path normalizes over train+val.** `features.legacy_spec()` computes per-item
  statistics over rows up to `train_end`, which sits after `val_start`, so first and second
  moments leak into the validation window. It is pinned that way because `results/results.json`
  is settled against it. `spec_for_panel()` — the path a real store uses — sets
  `stats_end = train_end` and does not leak.

- **The naive benchmark is weaker than an honest par sheet.** `model/baselines.naive_forecast`
  averages the trailing four same-weekday sales with no `is_closed` filter, so a closed day
  drags it down for four consecutive same-weekday targets. Fixing it would move
  `results.json`, so it stands. `model/shadow.par_quantity` computes its own open-days-only par
  precisely so a printed sheet never inherits the defect — which is why the weekly report's
  "your par" is a harder benchmark than the backtest's "naive".

- **`python -m sim.generate` has no overwrite guard** and rewrites `data/store_synth.csv` the
  moment it runs, which would invalidate the frozen results. Training and observed-settlement
  backtests are guarded; this is not.

- **Operator fatigue is the most likely cause of failure and no software fixes it.** The daily
  loop asks for several commands and one hand-keyed CSV every morning for twenty-eight
  consecutive days, in a job that starts at 5am. G1's 95% completeness threshold is realistic
  but not easy. `catch-up` makes recovery cheap and completeness makes the gap visible; neither
  makes anyone do it.

---

## What a green rehearsal does not prove

**It does not prove the model is good.** In the rehearsal's own 28-day shadow replay the model
loses to the store's trailing par: 15.9% WAPE against 11.9%, and G2, G3 and G4 all read FAIL.
Most of that gap is one week — New Year's, where the model had seen only two prior New Year's
Eves and over-forecast doughnuts 140 against 63. On the held-out test split the same model reads
12.0% against the naive 12.5%. Both numbers are real and they disagree, which is the honest
state of a 40-epoch model on a synthetic panel and is exactly why four shadow weeks exist. The
gates are meant to be able to fail.

**It does not prove which sellout rule is right.** `stockout`, `wasted <= 0` and
`sold >= produced` are the same event in all 9,837 open rows of the synthetic panel, because
`sim/generate.py` defines them that way. The rehearsal exercises the sellout plumbing and says
nothing about which rule a real store needs. Only real data — or the hand labels a "sold out?"
toggle would produce — can settle it.

**It does not prove the ingest reads your store's export.** It proves the ingest reads one
deliberately dirty mock export whose messiness we chose. The first real export will have a
problem nobody here anticipated. The dry-run loop in step 4 is where that gets found.
