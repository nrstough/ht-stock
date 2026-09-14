# Handoff: Fresh Forecast pilot proposal

Written 2026-09-14 at the end of a working session. Read this first in a new chat.

## Who and what

Nathan Stough, Harris Teeter associate, built this repo on his own time with no company
data. It is a prepared-food waste forecasting tool. He wants to pitch a paid 90-day pilot at
his store. He has **not yet been given the okay to present**, and no company data has been
touched. All work is on branch `claude/vigilant-goldberg-5ur0lo`, pushed.

## What exists in the repo (as of this session)

- `proposal/one_pager.html` — the two-minute manager brief. Current state: ask-first, plain
  English, no architecture talk. Asks: two-plus years of item-level sales history, a daily
  production count, price/cost/batch/shelf life per item, about 80 paid hours at $30/hr
  (~$2,400, with a one-time-payment fallback if payroll cannot run a project rate), a
  formal written agreement on access, ownership and payment (tool stays his, store keeps
  results), and if the pilot
  passes, a completion bonus plus the manager carrying the result to district. Ten
  highest-dollar items. Byline filled in. Placeholders left: store number, `[$ baseline]`,
  `[≈20]` stores.
- `docs/pilot_prep.html` — the working document. Part 1: sixteen things the pilot needs
  before it goes anywhere. Part 2: thirty-six questions for Nathan, unanswered. Part 3:
  plain-language study guide to the pipeline, model, decision rule, gates, simulated store,
  likely Q&A, glossary.
- `proposal/proposal.html`, `poc/dashboard.html` — older, longer, pre-existing pitch and
  dashboard. Still accurate on numbers, but predate the real-data layer.
- Everything else is the code: `sim/` (synthetic store), `model/` (quantile GRU +
  newsvendor + backtest + shadow mode), `ht/` (ingest/validate/config/weather/calendar for
  real exports), `docs/DATA_CONTRACT.md`, `docs/REAL_DATA_READINESS.md`.

Published artifacts (same content as the files):
- Brief: https://claude.ai/code/artifact/ef5312ef-20c4-4b82-80b2-7d2d1cc6341f
- Workbook: https://claude.ai/code/artifact/99bd6b1a-fac9-4566-93ca-71512e66007f

## Decisions Nathan has made so far

- Worth doing: ~80 hours over 90 days is under an hour a day.
- Wants to be paid properly; said $30/hr feels fair. Then said the proposal has too many
  if/then options and a menu invites the cheapest choice. **He wants one firm structure and
  a solid written agreement.** Nothing final chosen yet.
- No lawyer. The agreement is a **formal contract** in strict legal language: recitals,
  defined terms, numbered sections, specific and hard to pick apart, NC governing law.
  Drafted together, clause by clause; Nathan approves every line.
- No architecture or jargon in anything the manager sees. Keep it that way.
- Wants the proposal to be more convincing and straightforward, probably slides or a
  shorter HTML page. Not built yet; waiting on his answers.
- Wants to understand the tool and the simulated data well enough to answer questions
  himself. Part 3 of the workbook is for that; a quiz session was offered.

## Recommendation on the table (not yet accepted)

Flat project fee for the 90 days, in the $2,500 to $4,000 range, half at signing, half at
the end regardless of result. Ownership stays with Nathan; store owns its data and the
results; rollout is a separate conversation. Do not negotiate the rollout in the pilot
meeting.

## Open items, in order

1. Nathan reads his hiring paperwork for an inventions/IP clause. Everything depends on it.
2. Nathan answers the Part 2 questions (any order, short answers fine).
3. Draft the formal agreement together from those answers.
4. Build the short deck or page together for the meeting; the brief becomes the leave-behind.
5. Cut the data contract to one page together for whoever runs the item movement report.
6. Nathan runs the waste logger (`index.html`) for two weeks to replace `[$ baseline]`.
7. Quiz Nathan on Part 3 until he can answer without looking.
8. Produce PDFs of the brief and the agreement.

## Facts to keep straight

- Proof-of-concept numbers (simulated 2025, one store, nine items): status quo waste
  $138,462 (18.8% of production, 98.1% availability); model with availability held −37%;
  model profit-optimal −60%; WAPE 13.0% vs 15.5% trailing average vs 15.1% linear; model
  closes 81% of the gap to the oracle. All simulated, all reproducible from
  `python -m model.backtest`.
- The pilot promises 10 to 15%. Shadow-mode gate G4 requires a provable 15% before going
  live; live success bar is 10% against the measured baseline.
- The frozen checkpoint does not transfer; real data means retraining.
- Known limits: multi-day items (bread, cake) get a reminder not a quantity; ads/markdowns
  not modeled; no production count means no waste baseline and G4 pending forever.
- Nathan's current wage never appears in any document. Only the $30 figure does.

## Working rule

Nothing is Claude's to decide or finish alone. Every document is drafted together and Nathan
approves every line before it leaves his hands.

## Style rules that held up

Full sentences, no fragments, no slogans, no bold-label bullet lists, no em dashes, first
person from Nathan, every ask carries one sentence on why it is reasonable to grant.
