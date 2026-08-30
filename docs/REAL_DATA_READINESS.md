# Real-data readiness

The checklist to read on the day permission is granted.

Every line in READY names a command that has actually been run. It was run against a *mock*
store export: the synthetic panel written back out as a raw store file with the simulator-only
columns stripped, every column renamed, and deliberate dirt added. That is a dress rehearsal,
not a pilot. **No part of this repo has ever seen real store data.** Where a claim below has no
command beside it, treat it as a claim rather than a capability.

Run the whole chain yourself:

```bash
bash scripts/rehearse.sh              # 3m04 on 4 CPUs here; --fast for the 6-epoch CI configuration
```

It guards the four frozen provenance files by md5 before and after and exits non-zero on a
one-byte change in `results/results.json`. It writes under `.rehearsal/` with one exception worth
knowing before you run it on a repo you care about: step 9 re-runs the frozen backtest, which
rewrites `results/results.json` in place. The file is copied to `.rehearsal/results.before.json`
first, compared byte for byte afterwards, and restored if it differs.

---

## READY — exercised end to end, no store input required

Every row was run against the mock export while this page was written. Rows marked † are run by
hand and by unit test but are **not** part of `scripts/rehearse.sh`, so a green rehearsal does
not cover them.

| Capability | Command |
|---|---|
| Raw export to canonical panel | `python -m ht.ingest --mapping MAP.json --items ITEMS.json --out panel.csv --report ingest.json` |
| Fitness gate over a panel | `python -m ht.validate --panel panel.csv --items ITEMS.json --mapping MAP.json --ingest-report ingest.json` |
| Holidays / payday / weekday from real dates, any year | `python -c "import ht.calendar as c; print(c.easter(2026), c.super_bowl(2026))"` |
| Weather from a store CSV or a downloaded daily summary | `python -c "import pandas as pd; from ht.weather import CsvWeather; print(CsvWeather('WEATHER.CSV', {'date':'DATE','tmax_f':'TMAX','kind':'CONDITION'}).frame(pd.date_range('2025-12-29','2025-12-31')))"` |
| Splits resolved from the panel's own date range | `python -m model.features --panel panel.csv --spec auto` |
| Training on a real panel | `python -m model.train --panel panel.csv --spec auto --items ITEMS.json --artifacts art/` |
| Evaluation with no simulator truth | `python -m model.evaluate --panel panel.csv --artifacts art/ --items ITEMS.json --split test` |
| Morning sheet, returned-sheet intake, day scoring, weekly report | `python -m model.shadow morning\|enter\|score\|catch-up\|weekly\|status` |
| A district movement report filtered to one store | `python -m ht.ingest --mapping MAP.json --items ITEMS.json --store 0123 --out panel.csv` |
| † The Phase-1 logger's markouts folded into the panel's waste | `python -m ht.ingest --mapping MAP.json --items ITEMS.json --logger-backup backup.json --out panel.csv` |
| Policy replay with no simulator truth | `python -m model.backtest --panel panel.csv --settlement observed --spec auto --out OUT.json` |

The rehearsal absorbs the failure modes a real export actually has: CRLF line endings and a
declared cp1252 encoding, a report title block above the header and a `DEPT TOTAL` footer below
it, `MM/DD/YY` dates, `$` and thousands separators, parenthesised refunds, a duplicated export
window, a multi-day outage, random-weight barcodes, an item number that changes mid-history, an
unmapped code, an item with too little history, and weather conditions no alias table knows.

**Which artifact names which.** This matters, because the two are different documents and only
one of them is the thing you hand back to the store.

| Absorbed | Named by |
|---|---|
| the multi-day outage, the short-history item, the unknown weather condition, thin sellout coverage | the **validation report**, derived from the panel itself |
| the collapsed duplicate lines (96), the unmapped code `777777` (6 lines), the code `items.exclude` names (6), the grid days ingest inserted (57), the declared closures it stamped (42), negatives seen and negatives clipped | the **ingest report** — and repeated on the validation page as `[repair_*]` INFOs **only if you pass `--ingest-report`**. Without it the validator prints `[repair_report_absent]`, which says what it cannot count rather than implying nothing happened |
| the title block, the footer, `MM/DD/YY`, `$` and separators, the parenthesised refunds, the random-weight barcode collapse, the item number that changed | **nothing counts these.** They are absorbed by mapping fields you set, silently and by design. The mapping file is their only record |

One number to read carefully: the rehearsal reports `negatives 0 seen / 0 clipped`. Its two
parenthesised refunds net positive inside their own item-day, so nothing was ever clipped. The
refund *parsing* is exercised; refund *clipping* is not.

The mock export also emits the same movement report as the district office runs it — a `STORE`
column and a second store's rows — and step 4b of the rehearsal checks that all three layers give
the same answer to it: `ht.ingest` refuses the raw file and names both store numbers, `--store
0123` reproduces the single-store panel row for row, and `ht.validate` and `features.build` both
refuse a panel that was merged some other way.

### Two inputs that are named on the command line, not in the mapping

`--store` and `--logger-backup` are flags rather than `mapping.files` roles, and deliberately:
the store number answers a question about *this* export rather than describing the store's
system, and the logger backup is not a file the store's system produced at all. Both are printed
in the ingest summary (`stores kept 0456 of {...} rows dropped 8794`; a `logger` block naming
entries used, item-days written, units, collisions, unmatched item names and logged days outside
the export) and both are kept in the ingest report JSON under `stores` and `logger`. Neither is
repeated on the validation page — its `[repair_*]` block covers the grid fill, the collapsed
duplicates, negatives, closures, dropped lines and filled weather days, and stops there. If you
need a record of which store was kept or what the logger contributed, that is the ingest summary
and the report JSON, not the validation report.

A district export with no `columns.<role>.store` mapped is the one silent case left: without
that column `ht.ingest` stamps `mapping.store` on every row, the district is summed onto one
store's series, and no later layer can see it — the panel it gets carries exactly one store.
Map the column; the refusal is what tells you the file has more than one store in it. Passing
`--store` at least cannot make it worse: with no store column mapped there is nothing to filter
on, so `--store` would only relabel the panel as a store whose sales it does not contain, and
ingest refuses it rather than doing that quietly.

`ht.weather.CsvWeather` reads either a file with condition text (mapped through the store's
`kind_map` and a built-in alias table) or a downloaded daily summary carrying `TMAX`, `PRCP` and
`SNOW` and no words at all; it uses the words first and falls back to the gauges only where a
word was unrecognised.

**"No sellout signal" is a supported mode, not a broken one.** `sellout.rule = "none"` ingests,
validates and trains end to end; the rehearsal proves it on a separate degraded run. The cost is
stated rather than compensated for: the model then fits the distribution of *sales* rather than
demand, so recommended quantities run low on the busiest days, and `ht.validate`, `model.train`,
`model.evaluate` and every morning sheet print that sentence.

**How low is not known**, and those four places now say only that — they used to print
"roughly 1-8% low on the busiest days", a band nothing in this repo measures: no test pins it
and no experiment produces it. One measurement, run while writing this page: train the same 40 epochs twice on the rehearsal panel,
once with the sellout signal and once with `rule = "none"` (identical sales, so the only
difference is the censored loss and the `stockout` context channel), then compare the newsvendor
quantile each model produces on the test split. The no-sellout model came out **0.1% lower over
all 1,728 rows, 0.8% lower on the busiest tenth of days per item, and 2.1% lower on cake**, the
item with the highest sellout rate. That is the right direction and well under the advertised
band. Treat any figure of that kind as an order-of-magnitude expectation carried over from the
design argument, not as a result — and note that one synthetic panel at one sellout rate is thin
evidence in either direction.

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

Exit 1 with one line of explanation means the ingest refused and nothing was written. Exit 2
means the panel *was* written but validation found errors — look at it rather than re-running.
Exit 1 with a *traceback* is the third case and is a bug, not a refusal; none is known today.
The panel is written before the report and before the validation page, so a failure in either
of those leaves a usable panel on disk to re-validate.

**6. Read the validation report. Do not just check its exit code.**

```bash
python -m ht.validate --panel data/panel.csv --items ITEMS.json --mapping MAP.json \
    --ingest-report data/ingest_report.json
```

Pass the report. Without it the validator can see the panel but not what was repaired to make
it, and it says so (`[repair_report_absent]`) instead of leaving a silence that reads as "nothing
was repaired". With it you get seven `[repair_*]` INFOs instead of four, including the collapsed
duplicates and the dropped item codes — the two numbers a store's first export most often turns
on.

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
python -m model.shadow enter    --items ITEMS.json --date 2026-03-01 --out shadow --by kmurphy
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

`enter` is the return path, and it runs **before** `score`: `score_day` freezes a day's verdict
once, so a sheet keyed in afterwards does not change it. It is not lost — re-running `score`
writes the difference to `scores/_revisions.csv` (`2025-12-30, bread, produced, 30.0 -> 44.0`) —
but the row the weekly report reads is still the one scored before the sheet arrived. Key the
sheet in first.

It takes `item, made, sold out at[, note]` per line from a pipe, from `--file PATH`, or from an
item-by-item prompt (`Bread Loaf (sheet said 42) - made:`), which walks the sheet's own order:
the day-fresh departments, then the CARRY-OVER block, then NO FORECAST. The item is the config
key or the printed name — including the name **as the sheet truncated it**, since the ITEM
column is 18 characters wide and a real POS description is often longer; the time reads
`14:30`, `2:30pm`, `1430`, `930am`, `2pm`, or a bare `yes` for a circled item; blank, `no`,
`0`, `none` and `n/a` all mean "it did not sell out", and anything else is refused **by line
number** rather than guessed at, because a guessed sellout time is a fabricated observation. A
file written by Excel is read through its byte-order mark rather than refused over it. Nothing is written unless every line parses. Rows append to
`shadow/overrides/<date>.csv` — nine columns, `date, item, rec_qty, actual_produced, sold_out_at,
note, entered_by, entered_ts, sellout_source` — stamped `sellout_source=sheet`.

That stamp carries the load. On a row that came in through `enter`, an **empty** "sold out at"
cell means "it did not sell out" (`stockout=0, stockout_known=1`); on a hand-authored overrides
file it stays unknown, because a blank there promises nothing. This is what lets a store whose
export has no sellout column measure accuracy on fully-served days at all — the weekly report's
censoring line then reads `sheet`, or `produced_vs_sold+sheet` in a week that mixed the two.

---

## NEEDS THE STORE — the code is waiting, the data is not here

1. **The export itself.** Item movement at store x item x business-day grain, 104 weeks. The
   pilot's own weeks of logging are a validation window, not a training set. See the floors
   below. Details, in a store's own language, are in `docs/DATA_CONTRACT.md`.

2. **An item list with real prices and costs.** Every recommended quantity is a newsvendor
   critical fractile of price and cost; nothing else sets it. Cost appears in neither the
   Phase-1 logger nor a standard movement report, and is often a more sensitive ask than sales.
   Without it, the analyst writes `cost = price x (1 - margin)` into the items file by hand and
   sets `cost_imputed`; `ht.config` requires a cost, so nothing fills one in for you. `ht.ingest`
   re-derives it from `mapping.items.dept_gross_margin` and refuses the run if the two disagree by
   more than a cent, if the department has no margin, or if a margin was typed as `58` rather than
   `0.58`. The ingest summary, `model.evaluate`'s caveats, the morning sheet and the weekly report
   each print the assumption; **`model.backtest` does not**, and its policy table is the one that
   quotes dollars — read it beside the ingest summary. The imputation error goes straight into the
   quantity, so this is a guess wearing a decimal point.

3. **Batch sizes.** One tray, one bake, one pan. A recommendation is rounded to a whole batch
   and never goes below one, so a wrong batch size is a wrong sheet even with a perfect
   forecast. Nobody but the department can supply these.

4. **A production count — the sellout question.** This is the single most valuable column and
   the most important question in the permission conversation, ahead of asking for sales:
   *does any record exist of how much we put out each day?* A photographed clipboard keyed in
   is enough; a label-printer log is an accepted proxy. With it, the sellout flag is correct by
   construction and waste is measured for free. Without it the honest configuration is
   `sellout.rule = "none"`, and the consequence is the low bias described above plus no measured
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
   With `provider: "none"` the pipeline runs and the weather inputs go constant rather than
   absent: measured on the rehearsal panel re-ingested with no weather file, **7 of 36 covariate
   dimensions** (a temperature z-score, a five-wide condition one-hot that is all `unknown`, and
   tomorrow's snow flag) and **3 of the encoder's 6 channels** (`tmax_z`, `rain`, `snow`) carry a
   single value for every row.

**History floors**, from `model/features.py`, and what happens below them:

| Mode | Panel floor | Per-item floor | Command |
|---|---|---|---|
| train / val / test | 126 days | 84 open days | default |
| no held-out test | 98 days | 84 open days | `--no-test` |
| short history, provisional | 70 days | 56 open days | `--allow-short` |

`ht.validate` checks the floor of the mode the run actually asked for, which is what makes the
third row reachable through `model.train --allow-short`; without that it checked 84 open days
per item on every run and refused panels the trainer itself would have accepted.
`model.backtest` takes `--allow-short` too but **not** `--no-test`: a backtest replays policies
over a held-out window, and `--no-test` says there is none, so it refuses up front instead of
failing later on an empty test mask.

`--spec` has no default when a `--panel` is supplied. `model.train` and `model.backtest` both
refuse the run and name both options, because the legacy boundaries (`train_end 2024-12-31`,
`val_start 2024-11-04`, `test_start 2025-01-01`, holiday countdown off, statistics fitted over
train+val) are the *simulator's* dates and are silently wrong for every other store. With no
`--panel` the default is still legacy: that is the provenance of `results/results.json`.
Scoring is the same rule from the other end — `model.evaluate` and `model.shadow` score with the
spec the checkpoint's `meta.json` recorded, and where it recorded none (the frozen checkpoint
predates the field) they check the assumed legacy layout against the panel and refuse rather
than score a 2026 export on 2024 boundaries.

Below the floor, `features.build` raises `InsufficientHistory` naming the actual span, the
requirement, its decomposition and the remedy. `ht.validate` reaches the same verdict from the
panel alone and reaches it first — on the 70-day panel it exits 1 with `[all_items_excluded]` and
a `[short_history]` line per item — so a panel too short fails at the data layer with a data
message. Both measure against the floor of the mode the run asked for: `ht.validate --allow-short`
(and `model.train --allow-short`, which passes the mode through) checks 56 open days per item,
the same number `features.build` uses under that flag, and without the flag both check 84. That
matters more than it sounds -- with the per-item floor pinned at 84 whatever was asked for, the
short-history row of the table above could not be reached through the documented command at all:
every panel under about 99 days exited 1 on `[all_items_excluded]`.

The two floors count different things, and this bites at the margin: the panel floor counts
calendar days, the per-item floor counts *open* days. On the rehearsal panel (the last N days
ending 2025-12-31) `--allow-short` at exactly 70 days is refused with **every** item at
`55 of 56 days`, because Thanksgiving 2025-11-27 sits closed inside the 56-day training window
and a closure is the *store's*, so it costs every item the same day at once — the failure is
never one unlucky item. 71 days is the first span that works (8 items, 320 target rows; `sub` is
excluded for the unrelated reason that the mock export keeps only its last 30 days of it). At 69
days the refusal comes earlier and from a different check, `resolve_splits` on the span alone.
Expect the effective floor to be the stated one plus the number of closed days that land in the
training window.

---

## KNOWN GAP

### Would block a pilot on the affected items

- **Multi-day shelf life is fenced off, not solved, and it lands on bread and cake.** For an
  item with `shelf_life_days > 1`, demand is served from mixed-age inventory, production on day
  *t* is not availability on day *t*, and `wasted = produced - sold` is simply false — so the
  single-period newsvendor is confidently wrong on two of the largest-dollar lines. Those items
  are excluded from the waste bound by name, lifted out of the department MAKE blocks into a
  `CARRY-OVER ITEMS - NOT AN ORDER. CHECK WHAT IS LEFT FIRST.` block at the foot of the sheet
  (each row carrying its own "N-day shelf life: MAKE is one day's demand and does not subtract
  what is already on the shelf"), and never get `wasted` derived. The code is honest about them;
  it does not forecast them usefully. A multi-period age-cohort model is a separate piece of
  work. Set `shelf_life_days` truthfully before the live phase — the shipped example says 1 for
  bread and cake only because the simulator is day-fresh by construction.

- **A production count may not exist in any system, and then neither does the business case.**
  Everything the
  pilot claims in dollars needs a waste denominator, and every route to one — the export's own
  waste report, a label-printer log, a clipboard, the Phase-1 logger — is something a person has
  to produce. If the answer to NEEDS THE STORE item 4 turns out to be "no record exists and
  nobody will start one", the pipeline still runs and G4 reads PENDING forever. That is not a bug
  to fix in this repo; it is the question to ask before the pilot, not during it.

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
  reads 0.649 against a threshold of 0.75, while `cov_hi` reads 0.856, so the true coverage is
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
Eves: on 2025-12-31 its median forecast for doughnuts was 140, the sheet printed MAKE 168, and 63
sold. On the held-out test split the same model reads
12.0% against the naive 12.5%. Both numbers are real and they disagree, which is the honest
state of a 40-epoch model on a synthetic panel and is exactly why four shadow weeks exist. The
gates are meant to be able to fail.

**It does not prove which sellout rule is right.** `stockout`, `wasted <= 0` and
`sold >= produced` are the same event in all 9,837 open rows of the synthetic panel, because
`sim/generate.py` defines them that way. The rehearsal exercises the sellout plumbing and says
nothing about which rule a real store needs. Only real data — or the hand labels a "sold out?"
toggle would produce — can settle it.

**It does not prove the ingest reads your store's export.** `tools/make_mock_export.py` and
`ht/ingest.py` were written in this repo against each other: the same person chose which dirt
to add and which dirt to handle, so the rehearsal is a test of an importer against its own
author's imagination. Every failure mode it absorbs is one somebody thought of. The first real
export will have a problem nobody here anticipated — a column that means something else, a
day-close that is not a day, a department that was outsourced last year — and the dry-run loop
in step 4 is where that gets found, one refusal at a time. Budget for it: the honest estimate
for a first real export is days of back-and-forth with whoever runs the report, not an
afternoon.
