# Production plan: everything done, everything left

The single status page for this project. `README.md` says what the project is,
`docs/DATA_CONTRACT.md` says what the store has to hand over, and
`docs/REAL_DATA_READINESS.md` is the checklist for the day permission lands. This page
sits above all three and answers one question: what is built, what is not, and what is
blocking what.

Every DONE row below was verified against the repository, not remembered. Status means:

| Mark | Meaning |
|---|---|
| **DONE** | Built, tested, and exercised end to end at least once |
| **PARTIAL** | Built but not covered by the rehearsal, or built against invented inputs |
| **TODO** | Not started |
| **BLOCKED** | Cannot proceed without something only the store or a lawyer can supply |

The ordering inside each chunk is the order to do the work in.

---

## Chunk 1 — Measurement (Phase 1)

| Item | Status | Notes |
|---|---|---|
| Waste logger app | DONE | `index.html`, single file, no build step, no server, no dependencies |
| Log / history / stats / pitch / setup tabs | DONE | Backdating, decimal quantities, CSV export, JSON backup and restore |
| Annualised baseline and recovery scenarios | DONE | Weekly run rate times 52; the 10/15/20% band is labelled an expectation, not a result |
| **Two weeks of real logging at store 0298** | **TODO** | The whole point of the tool. Nothing else in this project produces a defensible dollar figure without it |

The logger needs no permission from anyone. It records what a person observes being thrown
away, which is the same thing a paper clipboard would do. Until it runs for fourteen
consecutive days, every dollar figure in the pitch descends from a guess.

---

## Chunk 2 — Simulator and proof of concept

| Item | Status | Notes |
|---|---|---|
| Synthetic store generator | DONE | `sim/`, seven modules: demand, weather, policy, calendar, params, generate |
| Frozen dataset | DONE | `data/store_synth.csv`, 9,864 rows, 9 items, 1,096 days, reproducible byte for byte from a fixed seed |
| Status-quo policy | DONE | `sim/policy.py`, deliberately a competent manager rather than a straw man |
| Quantile demand network | DONE | `model/net.py`, GRU over a 28-day window, item embeddings, 11 quantiles, pinball loss with sellout censoring |
| Decision layer | DONE | `model/newsvendor.py`, per-item critical fractile, batch rounding |
| Benchmark forecasters | DONE | `model/baselines.py`, trailing average and linear model |
| Six-policy backtest | DONE | `model/backtest.py`, frozen to `results/results.json`, refuses configurations that would change the numbers |
| **Dollar magnitudes** | **PARTIAL** | See Chunk 6. The ratios hold; the dollars trace to one hand-tuned constant |

### Known limits of this chunk, already documented

The published checkpoint is shape-locked to these nine items, so Phase 2 means retraining
rather than reusing. On the synthetic panel every candidate sellout rule agrees on every
row, because the simulator defines them as the same event, so a green run says nothing
about which rule a real store needs.

---

## Chunk 3 — Real-data pipeline

| Item | Status | Notes |
|---|---|---|
| Canonical panel schema | DONE | `ht/schema.py`, the one table every downstream module agrees on |
| Mapping-driven ingest | DONE | `ht/ingest.py`, raw export to panel |
| Validation gate | DONE | `ht/validate.py`, says whether a panel is fit to train on and what the store must fix |
| Item economics and source mapping | DONE | `ht/config.py`, the only place a store's own numbers enter |
| Real calendar | DONE | `ht/calendar.py`, holidays and paydays derived for any year |
| Weather intake | DONE | `ht/weather.py`, three fields, one closed vocabulary |
| Feature build with panel-derived splits | DONE | `model/features.py` |
| Training on a real panel | DONE | `model/train.py`, about a minute on four CPU threads |
| Evaluation without simulator truth | DONE | `model/evaluate.py`, reads nothing the store cannot see |
| District export filtered to one store | DONE | `ht/ingest.py --store` |
| Logger markouts folded into the panel | PARTIAL | Built and unit-tested in `tests/test_ingest_logger.py`, but the rehearsal does not exercise it because the mock export carries no logger backup |
| Dress rehearsal on a dirty mock export | DONE | `scripts/rehearse.sh`, absorbs CRLF, cp1252, title blocks, footers, `MM/DD/YY`, parenthesised refunds, duplicate windows, an outage, random-weight barcodes, a changing item number, an unmapped code and an unknown weather condition |
| **Run on a real export** | **BLOCKED** | Needs written data access. Nothing here has ever seen real data |

The honest caveat, which stays in every document: `tools/make_mock_export.py` and
`ht/ingest.py` were written in this repository against each other. The same person chose
the dirt and the handling, so the rehearsal tests an importer against its own author's
imagination, and the first real export will break in a way nobody here anticipated.

---

## Chunk 4 — Daily operating loop (Phase 3)

| Item | Status | Notes |
|---|---|---|
| Morning sheet | DONE | `model.shadow morning`, prints for the kitchen |
| Returned-sheet intake | DONE | `model.shadow enter`, pipe, file or item-by-item prompt; nothing written unless every line parses |
| Day scoring | DONE | `model.shadow score` |
| Weekly report with five fixed go/no-go criteria | DONE | `model.shadow weekly`, criteria printed from week one |
| Catch-up and status | DONE | `model.shadow catch-up`, `model.shadow status` |
| Append-only prediction log | DONE | The record that makes shadow mode auditable |
| **Four weeks of shadow running** | **BLOCKED** | Phase 3 proper. Needs the export and a store willing to print the sheet |

---

## Chunk 5 — Engineering gaps

These are ours to close and need nobody's permission.

| # | Item | Status | Effort |
|---|---|---|---|
| 1 | **Wire up CI.** `ci/github-actions-ci.yml` exists with instructions to copy it to `.github/workflows/ci.yml`. That directory does not exist, so 360 test functions across 21 files have never run automatically | **TODO** | Minutes |
| 2 | Add gradient-boosted trees as a fourth backtest policy | TODO | Hours |
| 3 | Benchmark a pretrained time-series foundation model zero-shot | TODO | Hours |
| 4 | Exercise the logger-to-panel path in the rehearsal | TODO | Hours |
| 5 | Retraining path for a real item list (checkpoint is shape-locked to nine items) | TODO | Covered by Chunk 3 once data lands |

On item 1: the suite is inference-only, the backtest replays a frozen checkpoint, and
`tests/conftest.py` monkeypatches the socket layer so an accidental network fetch fails
rather than passing quietly. It is ready to run. It simply is not running.

On item 2: the evidence favours it. Retail daily demand forecasting with covariates is
what gradient boosting wins at, and the oracle bound says the entire remaining headroom
past the current model is $3,935 per store-year, so a simpler model that lands close is
worth more than a more complex one that lands slightly closer.

---

## Chunk 6 — Grounding the numbers

The most important open work, and the least glamorous. Raised because the dollar figures
in the pitch do not currently rest on anything checkable.

| # | Item | Status | Who can close it |
|---|---|---|---|
| 1 | `BASE_SCALE = 0.72` in `sim/params.py` was hand-tuned so that weekly retail waste lands near a guessed order of magnitude. Every dollar figure in the deck descends from it | **TODO** | Us, once a real waste rate exists |
| 2 | The claim that waste "runs 15–25% of production" is unsourced. Published figures are lower: bakery shrink 8.5% and deli 8.7% against 3.1% store-wide (FMI, 2018 results); prepared foods 8.38% unsold and breads and bakery 8.06% (ReFED / Pacific Coast Food Waste Commitment, 2022) | **TODO** | Us, today |
| 3 | Retail prices in `config/items.example.json` are invented | TODO | Walk the department and read the tags. Public information |
| 4 | Production batch sizes are invented | TODO | You, from the department. It is how the equipment works, not company data |
| 5 | Unit costs are invented | BLOCKED | Needs company data. Drives the newsvendor targets and the entire economic-cost column |
| 6 | Store counts used in the scale table are approximate | TODO | Publicly checkable |

Two cautions worth carrying into any room. Shrink as a percent of sales dollars and waste
as a percent of units produced are different denominators and are not interchangeable.
And the deck and the simulator currently agree with each other because they share an
assumption, not because either was independently validated.

**What survives all of this:** the relative results. That the forecast beats a well-run par
sheet by 37% on waste at held availability, and closes 81% of the gap to perfect knowledge,
are ratios measured against benchmarks inside the same simulation and are far less
sensitive to the scale constant. Lead with the percentages. Let the dollars wait for the
baseline.

---

## Chunk 7 — Legal, ownership and commercial

Not code, and currently the critical path.

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | **Repository is public with no licence file** | **TODO** | Disclosed to anyone who finds it, and nobody can legally build on it either. Make it private if this may become a product |
| 2 | Obtain the signed onboarding packet from store or district HR | TODO | An ordinary request. Read the Conflicts of Interest, Outside Employment, Confidential Information, Company Property and inventions sections |
| 3 | Confirm whether any inventions-assignment agreement was signed | TODO | Hourly retail associates usually sign no such thing. This single fact decides everything downstream |
| 4 | Written IP position settled before any data access is requested | TODO | Currently the proposal defers this to before the live phase, which is too late. Move it ahead of shadow mode |
| 5 | Pricing structure decided | TODO | Per-store licence, gainshare against the shrink line, or hours. Exclusivity priced separately as a minimum annual commitment with a snap-back clause, never a lump sum |
| 6 | Fill the proposal placeholders | TODO | `[name]`, `[role]`, `[store]` in `proposal/executive-proposal.html` |

The asset being protected is that this system has never touched company data, and the
repository can prove it: the generator reproduces the frozen dataset byte for byte from a
fixed seed, and `tests/test_no_sim_import.py` walks the AST of every module under `ht/`
and `model/` to keep the simulator out of the real-data path. That provenance is worth
more than the code and is destroyed the first time an unscoped pilot puts real sales
history into the training pipeline.

---

## Chunk 8 — The pitch

| Item | Status | Notes |
|---|---|---|
| Seventeen-slide leadership deck | DONE | `proposal/Fresh-Forecast-Leadership-Deck.pptx`, native charts, speaker notes on every slide |
| Browser and print deck | DONE | `proposal/deck.html`, arrow-key navigation, one landscape page per slide |
| Executive proposal memo | DONE | `proposal/executive-proposal.html`, prints to four letter pages |
| PDFs of both | DONE | Generated from the HTML by a headless browser |
| Original store-level proposal | DONE | `proposal/proposal.html` |
| Interactive results dashboard | DONE | `poc/dashboard.html` |
| Both decks generated from frozen results | DONE | `tools/build_deck.py`, `tools/build_deck_pptx.js`; no slide can drift from the backtest |
| **Numbers re-anchored on cited benchmarks** | **TODO** | Chunk 6, items 1 and 2 |

---

## Critical path

Everything else is optional until these four are done, in this order:

1. **Read your signed onboarding packet.** Free, takes a day to request. Decides whether
   this is a job conversation or a business.
2. **Make the repository private.** Minutes.
3. **Log waste for fourteen consecutive days.** Needs no permission and replaces the one
   guessed constant that the entire dollar model rests on.
4. **Fix the unsourced industry claim** and re-derive the scale table from the logged
   baseline plus published benchmarks.

Only after those four does the pitch survive a numerate reader, and only after the pitch
survives does the data-access request make sense to send.
