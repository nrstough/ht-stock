# Implementation plan — Shadow-mode operator UI (Stage 1)

Date: 2026-09-09
Branch: `claude/plan-review-frontend-12nyhm`
Run spec: `docs/specs/SHADOW_OPERATOR_UI.md`
Feature spec: `docs/features/SHADOW_MODE.md`

Grounded in the current state of `model/shadow.py` (1892 lines), `tests/conftest.py` (146 lines),
`ci/github-actions-ci.yml` (69 lines), `.gitignore` (4 lines) and `index.html` (868 lines).

---

## Step 0 — Pre-flight

```bash
git branch --show-current      # must be claude/plan-review-frontend-12nyhm
pwd                            # must be /home/user/ht-stock
git status --short             # must be empty
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

torch is the only missing prerequisite; `numpy` and `pandas` are already installed. Without it
`model/shadow.py` cannot be imported and `tests/test_serve.py` cannot collect. **This install
needs your approval — it was declined earlier in the session.**

---

## Step 1 — `.gitignore` (append after line 4)

```
node_modules/
web/dist/
web/coverage/
shadow/
```

`shadow/` matters independently of the frontend: `ht.serve`'s `--out` default is a relative
path, and `tests/conftest.py:8-10` (RULE TWO) says nothing is written inside the repo.

---

## Step 2 — `ht/serve.py` (new, stdlib only)

### 2a. Module layout

```python
"""The daily loop, served to a back-room browser. Adds no arithmetic."""
import argparse, contextlib, http.server, json, mimetypes, os, posixpath, socketserver
import threading, urllib.parse
import pandas as pd
from ht import config as ht_config, schema
from model import shadow
```

Layering check: `ht/serve.py` importing `model.shadow` inverts the usual direction (`model/`
imports `ht/`). `tests/test_no_sim_import.py` guards `sim` imports, not this. **Verify at
execution** that no test asserts `ht/` never imports `model/`; if one does, the module moves to
`model/serve.py` and every path below changes accordingly.

### 2b. `Settings` — one frozen record of the CLI flags

Fields: `panel_path, items_path, artifacts_dir, out_dir, store, max_staleness`. Defaults must
match `model/shadow.py:1809-1861`'s parser exactly — `out="shadow"`, `store=""`,
`max_staleness=MAX_STALENESS_DAYS` (=2, `shadow.py:54`). Pinned by `test_server_defaults_match_the_cli_parser`.

### 2c. `Api` — the pure handler (no sockets, no HTTP objects)

```python
class Api:
    def __init__(self, settings): ...
    def handle(self, method, path, query=None, body=None) -> (int, dict): ...
```

This shape is forced by `tests/conftest.py:42-52`: `_NoNetwork` subclasses `socket.socket` and
raises in `__init__`, patched autouse for **every** test, plus `socket.create_connection`.
Anything that binds a port or opens a loopback connection raises `RuntimeError` before it
reaches a route. So the entire test surface calls `Api.handle` directly.

Internals:

- `self._write_lock = threading.Lock()` — every mutating route holds it (D3).
- `_panel()` — caches `shadow._panel(settings.panel_path)` (`shadow.py:1665`, which calls
  `schema.assert_no_truth`) keyed on the file's `st_mtime_ns` + size; re-reads on change.
- `_items()` — caches `ht_config.load_items` keyed the same way;
  `ht_config.config_hash(items_path)` recomputed on every request that logs (C2).
- `_require_out()` — mirrors `shadow.py:1805-1808`: every route except `POST /api/morning`
  refuses a missing `--out` directory with that function's own sentence.

### 2d. Routes

| Method + path | Wraps | Notes |
|---|---|---|
| `GET /api/status` | `shadow.status` (`:1627`) | |
| `GET /api/config` | `ht_config.load_items`, `shadow._load_meta` (`:102`), panel date range | |
| `GET /api/morning?date=` | `shadow.read_predictions` (`:738`) windowed to the date | **Never writes** (D1). 404 when no run exists. |
| `POST /api/morning` | `shadow.forecast` (`:259`) → `shadow.log_predictions` (`:692`) → `shadow.sheet_caveats` (`:618`) → `shadow._yesterday` (`:654`) → `shadow.morning_sheet` (`:469`) both formats | 409 naming the existing `run_id` unless body `{"reforecast": true}` (D2); `{"backfill": true}` maps to `allow_backfill` |
| `GET /api/sheet/{date}.{txt,html}` | the file under `out/sheets/` | containment-checked |
| `GET /api/entry-order?date=` | `shadow.entry_order` (`:893`) + what the sheet said | drives the Enter form's order |
| `POST /api/enter` | `shadow.parse_entries` (`:829`) → `shadow.record_actuals` (`:947`) | all-or-nothing: on any error return 400 with the error list and write nothing |
| `POST /api/score` | `shadow.score_day` (`:1079`) | |
| `POST /api/catch-up` | `shadow.catch_up` (`:1150`) | |
| `GET /api/weekly?week_ending=&weeks=&include_backfilled=` | `shadow.weekly_report` (`:1195`), whose result already carries `gates` (`:1408`) | |
| `GET /api/scores`, `GET /api/predictions` | `shadow.read_scores` (`:1130`), `read_predictions` | date-windowed |
| `GET /api/results` | `results/results.json`, read-only | |

Six command routes for the six subcommands, matching `docs/REAL_DATA_READINESS.md:40`.

### 2e. Serialisation

One helper converts DataFrames and numpy scalars to JSON-safe values: `NaN`/`NaT` → `null`,
`numpy.int64`/`float32` → Python scalars, `Timestamp` → `YYYY-MM-DD`. `NaN` must become `null`
and never `0` — a `par_fallback` row's quantiles are NaN by design (`shadow.py:1263`), and
zeroing them would print "make none".

### 2f. Errors

`schema.HtError` and `ValueError` from `forecast()` → 400 `{"error": "<the sentence>"}`,
verbatim, no traceback, no absolute paths (D4). Unknown route → 404 JSON. Body over 1 MiB →
413. Malformed JSON → 400.

### 2g. Server glue (thin)

`ThreadingHTTPServer` + a `BaseHTTPRequestHandler` that reads the body, calls `Api.handle`,
writes the JSON. Static: serves `web/dist` with `posixpath.normpath` containment; a missing
`dist` returns a message naming `npm --prefix web run build`. `log_message` silenced except
errors.

### 2h. `main(argv)`

Flags mirroring the shadow CLI: `--panel --items --artifacts --out --store --max-staleness
--port --host --allow-remote`. `--host` other than `127.0.0.1` requires `--allow-remote` and
prints what it means (D7). Path/date validation reuses the `shadow._check_args` (`:1781`)
pattern: one sentence, exit 1. A `out/.serve.lock` file with the pid refuses a second server on
the same shadow directory (D3).

---

## Step 3 — `tests/test_serve.py` (new)

Imports `from tests.conftest import SYNTH_CSV, ITEMS_JSON, ARTIFACTS` (`conftest.py:25-30`).
Module-scoped fixtures follow `tests/test_shadow.py:19-40` — **not** `make_panel`, whose panels
are not shape-compatible with the frozen nine-item checkpoint. `tmp_path` for every `--out`.

Tests, grouped by the run spec's acceptance criteria (numbers are the criteria they pin):

**No-write GETs and duplicate runs** — 1,2,3,4
`test_get_morning_never_writes`, `test_get_morning_404_before_any_run`,
`test_second_post_for_a_date_is_refused`, `test_refusal_names_the_existing_run_id`,
`test_explicit_reforecast_is_stamped`, `test_concurrent_posts_produce_a_readable_csv`

**CLI parity refusals** — 5,6,14,16
`test_forecast_of_a_covered_date_is_refused` (+ counterfactual note),
`test_backfill_flag_allows_it_and_stamps_it`, `test_stale_panel_is_refused`,
`test_missing_shadow_dir_refuses`, `test_truth_columns_are_refused_at_load`

**The returned sheet** — 7,8,9,10
`test_one_bad_line_writes_nothing`, `test_blank_sellout_via_enter_is_a_negative`,
`test_hand_authored_override_blank_stays_unknown`, `test_zero_is_a_negative_not_midnight`,
`test_truncated_name_resolves`, `test_ambiguous_truncation_is_refused`,
`test_entry_order_matches_the_printed_sheet`

**Scoring and the weekly page** — 11,12,13,15
`test_rescoring_an_unchanged_day_records_nothing`,
`test_changed_day_discloses_rather_than_applies`,
`test_score_before_enter_leaves_the_frozen_row` (readiness:211-215),
`test_weekly_excludes_backfilled_by_default`,
`test_include_backfilled_stamps_reconstructed`,
`test_pending_is_not_pass` (all five gates), `test_no_forecast_rows_are_excluded_not_summed`

**State, identity, determinism** — 17,18,19,20
`test_panel_reload_on_mtime_change`, `test_items_hash_is_recomputed_per_request`,
`test_model_version_tracks_the_artifacts_dir`, `test_forecast_is_deterministic`,
`test_server_defaults_match_the_cli_parser`

**Process boundary** — 21,22,23
`test_default_bind_is_loopback`, `test_non_loopback_requires_explicit_optin`,
`test_sheet_path_traversal_refused`, `test_static_path_traversal_refused`,
`test_missing_build_prints_the_build_command`, `test_second_server_on_the_same_dir_refuses`

**Request hygiene**
`test_bad_json_body_is_one_sentence`, `test_unknown_route_is_json_404`,
`test_oversized_body_refused`, `test_errors_carry_no_traceback`,
`test_date_errors_match_the_cli_wording`, `test_nan_serialises_as_null_not_zero`

**Repo integrity** — 26
`test_requirements_unchanged`, `test_no_socket_is_opened` (implicit via the autouse fixture)

≈40 Python tests. During execution each guard marked counterfactual in the run spec is verified
by temporarily removing it and confirming the named test fails.

---

## Step 4 — `web/` (Vite + React + TypeScript)

```
web/
  package.json  vite.config.ts  tsconfig.json  index.html
  src/
    main.tsx  App.tsx
    api/client.ts        one place that knows the endpoints
    api/types.ts         mirrors PREDICTION_COLUMNS / SCORE_COLUMNS / weekly_report's keys
    lib/format.ts        qty/usd/date — vectors ported from shadow._fmt_qty (:397)
    lib/parseEntries.ts  client mirror of shadow.parse_entries, shared vectors
    styles/tokens.css    the nine custom properties from index.html:8-32, verbatim
    routes/Today.tsx  Enter.tsx  Scores.tsx  Weekly.tsx  Status.tsx
  src/**/*.test.ts       vitest
```

Runtime deps: `react`, `react-dom`, `react-router-dom`. Dev: `vite`, `typescript`, `vitest`,
`@testing-library/react`, `jsdom`. No chart library — inline SVG; load the `dataviz` skill
before writing the first chart.

Behaviour that is not negotiable:

- **Order is enforced, not offered** (readiness:211-215): Score is disabled until the day has
  been entered, with the reason shown. `score_day` freezes a verdict once.
- **Stamps are loud**: `backfilled`, `reconstructed`, `par_fallback`, and the sellout-source
  caveat render as visible banners, not tooltips.
- **PENDING is its own state** — never green, never counted as a pass.
- **No arithmetic in the render path** — verified by grep during the audit.
- Print stylesheet reproduces the sheet's columns; `@media print` hides navigation.
- With `/api/status` unreachable, operator routes are disabled with an explanation.

Client tests: `format.test.ts`, `parseEntries.test.ts` (shared vector set with Python),
`gates.test.tsx`, `stamps.test.tsx`, `apiAbsent.test.tsx`, `entryOrder.test.ts`. ≈15 tests.

---

## Step 5 — CI (`ci/github-actions-ci.yml`)

- **No change needed for the Python side**: line 36-37's `python -m pytest tests/ -q -m "not slow"`
  already collects `tests/test_serve.py`.
- Add a **second job** `web:` after line 69 — `actions/checkout@v4`, `actions/setup-node@v4`
  (node 20, `cache: npm`, `cache-dependency-path: web/package-lock.json`), then `npm ci`,
  `npm run build`, `npm test` with `working-directory: web`.
- Nothing may be inserted into lines 39-69: that is a contiguous, order-dependent chain ending
  in the `git diff --exit-code` frozen-artifact guard at 62-65. `npm ci`, never `npm install`,
  so `package-lock.json` is not mutated.
- Amend the header comment at lines 8-11: "pip is the only thing that touches the network here"
  becomes "pip and npm".

---

## Step 6 — Documentation sweep (before commit)

| File | Edit |
|---|---|
| `README.md:145` | "The whole app is a single file…" — scope to `index.html` |
| `README.md:150` | "Nothing is uploaded anywhere" — scope to the logger |
| `README.md:3-14` | the three-piece framing gains the operator UI |
| `README.md:97-104` | the daily-loop block gains the served equivalent |
| `README.md:186-187` | "The commands exist (…)" — the stale forward reference |
| `README.md:189-190` | "goes back in by hand" / "the only way" |
| `README.md:218-223` | the permission note, re-read against a localhost server |
| `docs/REAL_DATA_READINESS.md:40` | READY row gains `python -m ht.serve` |
| `docs/REAL_DATA_READINESS.md:192-205` | day-one step 9 gains the served loop |
| `docs/REAL_DATA_READINESS.md:439-443` | the operator-fatigue bullet |
| `docs/features/SHADOW_MODE.md` | "(planned)" → shipped |
| `docs/specs/SHADOW_OPERATOR_UI.md` | verification record filled in |

`requirements.txt` is **not** edited — stdlib-only keeps its line 1 claim true.

Then the staleness grep: for every behaviour this change resolves, grep `docs/` and `README.md`
for now-contradicted claims. The readiness doc's own rule (lines 5-9) is that a READY row names
a command that has been run — so the new row must name `python -m ht.serve` and that command
must actually have been run.

---

## Step 7 — Verify

```bash
python -m pytest tests/test_serve.py -v -o "addopts="
python -m pytest tests/ -v -o "addopts="
python -m pytest tests/ -q -m "not slow"
npm --prefix web ci && npm --prefix web run build && npm --prefix web test
bash scripts/rehearse.sh --fast
python -m ht.serve --panel .rehearsal/panel.csv --items .rehearsal/items.json \
    --artifacts .rehearsal/art --out .rehearsal/shadow --port 8765
# Playwright walk against the preinstalled Chromium: print -> enter -> score -> weekly
git diff --exit-code -- results/results.json data/store_synth.csv \
    model/artifacts/demandnet.pt model/artifacts/meta.json
git diff --exit-code -- requirements.txt
```

---

## Step 8 — Commit, critique, audit

`feat: serve the shadow-mode daily loop to a browser`. Then the adversarial Claude critique
loop against `.claude/codex-audit-prompt.md`'s rubric — **which does not exist in this repo**,
so the critique agent is given the seven dimensions explicitly (plan adherence, scope
discipline, test coverage, review compliance, freeze integrity, regression check,
documentation) and told to verify every finding at `file:line`. Loop until Acceptable.

**Codex is unavailable** — no `codex` binary, no `.claude/`, no `review-audit.sh` in this repo
or container, and the pipeline's path points at a different repository. Recorded in the run
spec; the passing Claude critique stands in as the gate.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **A `GET` that writes corrupts the append-only record** — the dominant risk; `read_predictions` (`:738`) is `keep="last"`, so a second run silently substitutes a later forecast | D1/D2; four tests; `POST` is the only writer and refuses a duplicate date |
| **Torn CSV rows under concurrency** — `open(..., "a")` + `csv.DictWriter` in `log_predictions` and `record_actuals` | One write lock + a per-directory lock file; `test_concurrent_posts_produce_a_readable_csv` |
| **Out-of-order operation** — score before enter freezes the wrong verdict (readiness:211-215) | UI enforces order; `test_score_before_enter_leaves_the_frozen_row` |
| **`ht/` importing `model/` breaks a layering test** | Checked at Step 2a; falls back to `model/serve.py` |
| **`make_panel` panels don't fit the frozen checkpoint** | Forecast tests use `SYNTH_CSV`/`ITEMS_JSON`/`ARTIFACTS` per `test_shadow.py:24-40` |
| **A relative `--out` default writes inside the repo** | `.gitignore` entry + every test uses `tmp_path` |
| **NaN quantiles serialised as 0** → "make none" on a `par_fallback` row | `test_nan_serialises_as_null_not_zero` |
| **The frontend needs node on the store's terminal** | `web/dist` is built in CI for Pages; if a store terminal can't have node, a CI-built release zip — **flagged, not solved in this change** |
| **No auth on the API** | Loopback-only default, explicit opt-in flag, documented |
| **torch install declined** → no E2E, no forecast tests | Step 0 asks again; if declined, Stage 1 ships unit-tested only and the run spec records it |
| **Scope creep into Stages 2-3** | The logger and the public shell are out of scope; `index.html` is not touched |

---

## Deferred to later runs

Stage 2 (absorb the Phase-1 logger, preserving `htw.*.v1` localStorage keys and the
`{app,v,items,logs,settings}` backup envelope), Stage 3 (public front page, and importing
`results/results.json` at build time so `poc/dashboard.html:234`'s hand-pasted copy can stop
drifting), and the "sold out?" toggle that `docs/REAL_DATA_READINESS.md:418-420` names as the
cheapest sellout ground truth available.
