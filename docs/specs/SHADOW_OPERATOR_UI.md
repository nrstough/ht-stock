# Run spec — Shadow-mode operator UI

Status: planned (Phase 3 of the plan-review pipeline)
Date: 2026-09-07
Branch: `claude/plan-review-frontend-12nyhm`
Plan file: `docs/reports/2026-09-07_shadow-operator-ui-plan.md`
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
- `web/` (Vite + React + TypeScript app, `node_modules` and `dist` git-ignored)
- `docs/specs/SHADOW_OPERATOR_UI.md` (this file)
- `docs/features/SHADOW_MODE.md`
- `docs/reports/2026-09-07_shadow-operator-ui-plan.md`

Edited:

- `README.md` — the Phase 3 roadmap bullet and the daily-loop section
- `docs/REAL_DATA_READINESS.md` — the READY table's daily-loop row
- `ci/github-actions-ci.yml` — a node job and the new Python test
- `.gitignore` — node build output

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
3. An explicit re-forecast is stamped and the stamp reaches the rendered sheet.
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
16. A panel carrying `true_demand` is refused at load.
17. The panel is reloaded when its mtime changes; `items_config_hash` is recomputed per request.
18. `model_version` tracks the artifacts directory.
19. Repeated forecasts of the same inputs are identical.
20. Server defaults match the CLI parser's defaults.
21. Default bind is loopback; a non-loopback host requires explicit opt-in.
22. Path traversal is refused on both the sheet route and static serving.
23. A missing `web/dist` prints the build command rather than a bare 404.
24. Client-side entry pre-validation agrees with `parse_entries` on a shared vector set.
25. With the API unreachable, operator routes are disabled rather than silently empty.
26. The four frozen files and `requirements.txt` are byte-unchanged.
27. The existing test suite passes with no regressions.

## Regression definition

Any existing test in `tests/` failing; any byte change in the four frozen files or
`requirements.txt`; any socket opened at test time; any CLI-visible behaviour change in
`model/shadow.py`.

## Verification record

Filled in during Phase 4.

- Unit tests: _pending_
- Frontend tests / build: _pending_
- End-to-end walk: _pending_
- Frozen-artifact guard: _pending_
- Claude adversarial critique: _pending_
- Codex audit: **unavailable in this environment** — no `codex` binary, no `.claude/`
  directory and no `review-audit.sh` in this repo or container. The adversarial Claude
  critique stands in as the gate, per the pipeline's quota-failure provision.
