# Run spec — Shadow-mode operator UI

Status: implemented and committed (Phase 4). Frozen from here.
Written: 2026-09-07 · Implemented: 2026-09-09
Branch: `claude/plan-review-frontend-12nyhm`
Plan file: `docs/reports/2026-09-09_shadow-operator-ui-plan.md`
Feature spec: `docs/features/SHADOW_MODE.md`

This spec is the audit artifact for one change. It is frozen once the change is committed.

---

## Problem

Phase 3 shadow mode works and is exposed only as six terminal commands:

```
python -m model.shadow morning|enter|score|catch-up|weekly|status
```

To run the four-week pilot, a store needs someone who can open a shell, has Python with
pandas and torch installed, and types a command with four path flags at 5:30am — then again
for the returned paper sheet, then again to score it, then again on Sunday. Every morning,
without missing days, because a missed morning is a hole in gate G1.

`docs/REAL_DATA_READINESS.md` is candid that day one takes an afternoon. The operational risk
is not the model; it is that nobody in a grocery back room will drive this from a command
line for twenty-eight consecutive days. That is what this change removes.

## Solution

A local, single-machine web application:

- **`ht/serve.py`** — a stdlib-only HTTP service that wraps the existing `model.shadow`
  functions. It adds no arithmetic. Every number it returns is a number the CLI would have
  printed.
- **`web/`** — a React + TypeScript + Vite single-page app served by it: today's sheet
  (printable), the returned-sheet intake walked item by item in the printed order, day
  scores, and the weekly district-manager page with the five go/no-go gates.

Scope is **Stage 1 only**. Absorbing the Phase-1 waste logger and building a public front
page are separate changes with their own run specs.

## Design decisions

**D1 — No `GET` in this API writes.**
`log_predictions` (`model/shadow.py:692`) is append-only and `read_predictions`
(`model/shadow.py:738`) resolves duplicates with `keep="last"`. A browser refresh, a
prefetch, a React StrictMode double-mount or a double click would append a second run for a
date, and `keep="last"` would then make the *later* forecast the live one. If that second run
happened after the store opened, the pilot's central claim — every number traces to a
forecast that provably existed before the day did — becomes quietly false with nothing in the
record saying so. Therefore: `GET /api/morning` returns the already-logged sheet and never
logs; logging happens only on `POST /api/morning`.

**D2 — A second run for a date is refused, not silently appended.**
`POST /api/morning` for a date that already has a logged run returns 409 naming the existing
`run_id`. A deliberate re-forecast requires explicit intent and is stamped so the sheet and
the weekly page disclose it.

**D3 — All writes are serialised behind one lock, and one shadow directory has one writer.**
The CSV writers use `open(..., "a")` + `csv.DictWriter`; two simultaneous callers can tear a
row. A second server process on the same shadow directory is refused via a lock file.

**D4 — The server reproduces the CLI's refusals verbatim.**
Every refusal reachable from the daily loop (the backfill guard, the staleness guard,
`schema.assert_no_truth`, the missing-shadow-directory check, the all-or-nothing
`parse_entries`) surfaces as the same sentence the CLI prints, with the same meaning. The UI
never offers a path around one.

**D5 — No new Python runtime dependency.**
`requirements.txt` documents a deliberate policy: direct dependencies only, because a store's
IT contact installs this. The API is a small number of JSON endpoints; stdlib `http.server`
is sufficient. Handlers are pure functions and the server glue is thin, which is also what
makes them testable — `tests/conftest.py` monkeypatches `socket.socket` to raise in every
test, so an API test that binds a port is impossible here by design.

**D6 — No number is computed in JavaScript.**
Every figure rendered traces to a field the Python layer emitted. PENDING is rendered as
PENDING and never as a pass.

**D7 — Loopback only by default, no authentication.**
It is a back-room terminal. Binding a non-loopback address requires an explicit opt-in flag
that prints what it means.

## What will change

Created:

- `ht/serve.py`
- `tests/test_serve.py`
- `tests/fixtures/entry_vectors.json`, `scripts/make_entry_vectors.py`
- `web/` (Vite + React + TypeScript app, `node_modules` and `dist` git-ignored)

Already existed when this list was first written, and were edited rather than created: this
file, `docs/features/SHADOW_MODE.md`, and the plan.

Edited:

- `README.md` — the three-piece framing, the "single file with no dependencies, no build step,
  and no server" claim (now scoped to the logger), "nothing is uploaded anywhere", the Phase 3
  roadmap bullet's four-of-six subcommand list, and a new **Running the daily loop in a
  browser** section
- `docs/REAL_DATA_READINESS.md` — a READY row for `ht.serve`, day-one step 9, and the
  operator-fatigue bullet, which claimed a cost this change removes half of
- `ci/github-actions-ci.yml` — a separate `web:` job (the python job's steps from "feature
  build" onward are one order-dependent chain ending in the frozen-artifact guard, so nothing
  may be inserted into it) and the header's "pip is the only thing that touches the network"
- `.gitignore` — node build output, and `/shadow/`: the served `--out` default is relative,
  and `tests/conftest.py`'s RULE TWO is that nothing is written inside the repo

`requirements.txt` was **not** edited. Staying stdlib-only keeps its opening claim — that
`ht/` and `model/` import numpy, pandas and torch and nothing else — literally true.

Must not change:

- `results/results.json`, `data/store_synth.csv`, `model/artifacts/demandnet.pt`,
  `model/artifacts/meta.json` — guarded by `git diff --exit-code` in CI
- `requirements.txt`
- `model/shadow.py` behaviour — at most extraction of existing logic into callable form

## Acceptance criteria

Each is pinned by a named test; the full failure-mode inventory and its test mapping live in
the plan file.

1. `GET /api/morning` never writes: two successive GETs leave `predictions.csv` byte-identical.
2. A second `POST /api/morning` for the same date is refused, naming the existing `run_id`.
3. A date with more than one logged run is disclosed as superseded, naming the live `run_id`.
   *(Rewritten. `PREDICTION_COLUMNS` has no field for "this replaced an earlier sheet" and
   `morning_sheet` prints no marker, so the only honest evidence is the one the log already
   keeps. A `superseded` column is a change to the record's schema and gets its own run.)*
4. Concurrent writes produce a CSV that `read_predictions` parses, with the exact row count.
5. Forecasting a date the panel already covers is refused with `forecast()`'s own sentence.
6. `backfilled=1` survives to the payload and is visibly stamped in the UI.
7. One unreadable line in a returned sheet writes nothing.
8. A blank sold-out cell counts as "did not sell out" only via `enter`; a hand-authored
   overrides file's blanks stay unknown.
9. `"0"` in the sold-out cell is a negative, not midnight.
10. A truncated printed item name resolves; an ambiguous truncation is refused.
11. Re-scoring an unchanged day records no revisions; a changed day discloses rather than applies.
12. Weekly excludes backfilled rows by default; `include_backfilled` stamps `reconstructed`.
13. PENDING is never rendered or encoded as a pass, for all five gates.
14. A missing shadow directory is refused rather than reported as an empty pilot.
15. `par_fallback` rows with NaN quantiles are excluded, not summed into `nan`.
16. The simulator-only columns cannot reach a forecast through the API.
    *(Rewritten. `schema.read_panel` returns `conform(df)`, and `conform` drops those columns
    unconditionally, so such a panel is silently stripped rather than refused — the refusal a
    reader would assume is there is not. What is testable, and what matters, is that nothing
    downstream can see them.)*
17. The panel is reloaded when its mtime changes; `items_config_hash` is recomputed per request.
18. `model_version` tracks the artifacts directory.
19. Repeated forecasts of the same inputs are identical.
20. `Settings` defaults match the **`morning` subparser's** defaults for the flags they share.
    *(Narrowed. The CLI has six subparsers whose defaults diverge — `--artifacts` is required
    for `morning` and `None` for `weekly` — so "the CLI's defaults" is not one set.)*
21. Default bind is loopback; a non-loopback host requires explicit opt-in.
    *(Tested at the parsed-arguments layer only. `tests/conftest.py` makes binding a socket
    impossible in this suite, so the bind itself is the one part of D7 that no test covers.)*
22. Path traversal is refused on both the sheet route and static serving.
23. A missing `web/dist` prints the build command rather than a bare 404.
24. Client-side entry pre-validation agrees with `parse_entries` on a shared vector set —
    `tests/fixtures/entry_vectors.json`, generated by `scripts/make_entry_vectors.py` and read
    by *both* suites, so neither language owns the expectations.
25. With the API unreachable, operator routes are disabled rather than silently empty.
26. The four frozen files and `requirements.txt` are byte-unchanged.
27. The existing test suite passes with no regressions.

## Regression definition

Any existing test in `tests/` failing; any byte change in the four frozen files or
`requirements.txt`; any socket opened at test time; any CLI-visible behaviour change in
`model/shadow.py`.

## Verification record

Baseline before any code, recorded so "no regressions" is checkable rather than assumed:
**449 passed, 5 failed, 3 deselected**. All five failures pre-date this change —
`tests/test_integration_guards.py:21` shells out to `REPO/.venv/bin/python` and no `.venv`
exists in this container.

- **Python suite:** 449 → **531 passed**, the same 5 pre-existing failures. `tests/test_serve.py`
  contributes 80.
- **Frontend:** `npm run build` clean (tsc + vite), **65 tests passed** across 5 files.
- **End-to-end, over real HTTP:** `python -m ht.serve` driven with curl through the whole loop
  (morning → 409 on a repeat → printed sheet → entry order → bad line rejected writing nothing
  → good sheet entered → scored → weekly gates), then the built SPA driven in Chromium through
  Today / Enter / Scores / Weekly / Status. Screenshots taken.
- **Frozen-artifact guard:** `git diff --exit-code` silent over all four, and over
  `requirements.txt`. No new Python dependency.
- **Two bugs found by running it rather than by testing it**, both fixed and pinned:
  taking the lock creates the shadow directory, which quietly undid the CLI's rule that only
  `morning` does — so a typo in `--out` would have reported a pilot that lost its record; and
  `Today` rendered `sheet.caveats` without checking the field was present, crashing the page
  instead of degrading.
- **Claude adversarial critique:** run against the plan before implementation — 23 findings,
  4 acceptance criteria that could not pass as written, and the change's worst failure mode
  (a Score button freezing an empty verdict) absent from the plan entirely. All dispositioned
  in the plan file. The post-commit pass is the remaining gate.
- **Codex audit:** **unavailable in this environment** — no `codex` binary, no `.claude/`
  directory and no `review-audit.sh` in this repo or container; the pipeline's path names a
  different repository. The adversarial Claude critique stands in as the gate, per the
  pipeline's quota-failure provision.
