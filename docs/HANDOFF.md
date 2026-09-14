# Handoff: Fresh Forecast pilot proposal

Written 2026-09-14 at the end of a working session, updated the same day when the pitch branch
was merged with `main`. Read this first in a new chat.

## Who and what

Nathan Stough, Harris Teeter associate, built this repo on his own time with no company
data. It is a prepared-food waste forecasting tool. He wants to pitch a paid 90-day pilot at
his store. He has **not yet been given the okay to present**, and no company data has been
touched. All work is on branch `claude/handoff-docs-uhugi8`, pushed. That branch carries
everything from `claude/vigilant-goldberg-5ur0lo` (the brief, the workbook, this note) merged
with `main`, which had meanwhile gained a leadership deck, an executive proposal, print PDFs
and a production plan. The two lines touched only the README in common and the merge kept both
descriptions.

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
- `proposal/deck.html`, `proposal/Fresh-Forecast-Leadership-Deck.pptx` and `.pdf` — a
  seventeen-slide deck for district and division leadership, generated from the frozen backtest
  by `tools/build_deck.py` and `tools/build_deck_pptx.js`. Written for a room above the store
  manager, so it talks architecture and scale; it is not the manager-meeting deck the open items
  below call for, though it is raw material for one. Placeholders for name, role and store.
- `proposal/executive-proposal.html` and `Fresh-Forecast-Executive-Proposal.pdf` — the
  four-page memo that goes with the deck. Same audience, same caveats.
- `proposal/README.md` — which pitch file is for whom and how to regenerate the generated ones.
- `proposal/proposal.html`, `poc/dashboard.html` — older, longer, pre-existing pitch and
  dashboard. Still accurate on numbers, but predate the real-data layer.
- `docs/PRODUCTION_PLAN.md` — one status page over everything: what is built, what is not,
  what blocks what, with a four-item critical path. Read it before promising anything new.
  It records that CI cannot be installed from a Claude session (GitHub refuses workflow files
  from the app), so the 360 tests only run when a human runs them.
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

## Answers Nathan gave on 2026-09-14 (first batch of Part 2)

- Hiring paperwork: does not know what he signed. Still has to find it or get a copy from HR.
  Open item 1 stays open.
- Money: for the pilot he would accept the equivalent of $50 an hour, so about $4,000. He
  would do it for $2,000 to $2,500. A floor was mentioned but not clearly; confirm it.
  The brief still says $30 an hour, about 80 hours, roughly $2,400, with a one-time-payment
  fallback. That wording is now out of date and should become one flat number.
- Production sheet: exists and is reasonably well followed, not perfectly. So the pilot can
  measure production.
- Waste logging: cataloged very well for anything with a barcode. Donuts, pizza and leftover
  sub bread are not accounted for at all; they are thrown away in a lump. So barcoded items
  already have a waste baseline in the store's own records, and the unbarcoded ones need the
  logger or need to be left out of the ten.
- Employment: hourly. Willing to set up a small company so the store can pay an invoice
  instead of a special payroll rate. Payment clause can be written as a vendor invoice.
- Format: slides only if he gets a room, which he thinks unlikely. Expects the manager to read
  on a phone. Wants a document, and is fine with both.
- Approved on the same day, and applied to the brief: a flat $4,000 project fee, half at
  signing and half at the end regardless of result, invoiced through a small business; and
  the removal of the "15 to 25 percent" industry claim in favor of "we do not know exactly
  what that costs this store today." The $30 figure is gone from the brief. No hourly rate
  appears in it now, on purpose, so the fee does not invite an hourly counter.
- Scope: Nathan thinks ten items at one store is too small to be a real proof of concept and
  wants at least ten items per department. Not yet applied anywhere. It changes the hours,
  the fee arithmetic and the logging burden on the department; see the open discussion.
- Production sheet: filled in by the department managers or by Nathan, depending on what the
  managers he talks to prefer.
- Scope, resolved in principle: deli, bakery, sub shop and pizza are all one department at
  his store, so "ten per department" and "ten items" are the same thing. He says it is hard
  to name ten separate barcoded items because the department does not make that many. So
  the pilot can plausibly cover everything the department makes in house, and the scored set
  is whichever of those have barcodes. Not yet worded into the brief.
- Store number is 298. Filled into the brief.
- Scale, from Nathan on 2026-09-14: the department makes about 50 items in house and puts
  out roughly another 100 that it does not make. The pilot should cover the 50 made in
  house; the 100 put-out items are ordering, not production, and stay out of scope. Named
  in-house items so far: fried chicken in several forms, rotisserie chicken in several
  sizes, ribs, pasta salads, chicken salads, cookies and other baked goods, pizza, subs.
- What actually gets thrown away most, from Nathan, all barcoded: grilled bites, popcorn
  chicken, fried four piece, baked eight piece, fried legs, fried bucket, rotisserie
  chicken. More rarely a fresh slice or a chub of meat. Waste is concentrated in hot foods
  and chicken, and the heavy items are barcoded, which means the store's own records can
  supply most of the baseline. The unbarcoded lump (donuts, pizza, leftover sub bread)
  matters less than first assumed.
- Nathan asked Claude to log into his HR portal through his browser. Not possible from this
  session and not appropriate in any case. He pulls the packet himself.
- HR portal: Nathan asked Claude to check it from his computer. Not possible from a cloud
  session, and not something to do on his behalf in any case. He needs to pull his new-hire
  packet himself or ask HR for a copy.
- Target date: in front of the manager roughly two weeks out, so around 2026-09-28. That
  makes the hiring-paperwork check and the baseline question urgent.

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
6a. Done: the "15 to 25 percent" sentence is out of the brief.
6b. Settle the scope question (ten items total, or ten per department) before the agreement
    is drafted, because the fee and the hours follow from it.
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
- The manager brief says industry estimates put prepared-food waste at 15 to 25 percent of
  production. `docs/PRODUCTION_PLAN.md` flags that line as unsourced and cites published
  figures that are lower (bakery shrink 8.5% and deli 8.7% in FMI 2018; prepared foods 8.38%
  unsold in the 2022 ReFED / Pacific Coast Food Waste Commitment report). The brief has not
  been changed at first because Nathan approves every line; he approved the change on
  2026-09-14 and the sentence is gone.
- Every dollar figure in the pitch descends from a hand-tuned scale constant in the simulator
  (`BASE_SCALE` in `sim/params.py`). The percentages hold; the dollars are illustrative until
  the logged baseline replaces them. Say so if asked.

## Working rule

Nothing is Claude's to decide or finish alone. Every document is drafted together and Nathan
approves every line before it leaves his hands.

## Style rules that held up

Full sentences, no fragments, no slogans, no bold-label bullet lists, no em dashes, first
person from Nathan, every ask carries one sentence on why it is reasonable to grant.
