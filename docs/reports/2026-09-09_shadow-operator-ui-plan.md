# Implementation plan — Shadow-mode operator UI (Stage 1)

Date: 2026-09-09
Branch: `claude/plan-review-frontend-12nyhm`
Run spec: `docs/specs/SHADOW_OPERATOR_UI.md`
Feature spec: `docs/features/SHADOW_MODE.md`

Grounded in the current state of `model/shadow.py` (1892 lines), `tests/conftest.py` (146 lines),
`ci/github-actions-ci.yml` (69 lines), `.gitignore` (4 lines) and `index.html` (868 lines).

---

## Revision 2 — after adversarial review

A critique pass verified this plan against the code and found four acceptance criteria that
cannot pass as designed, plus the change's worst new failure mode missing entirely. Everything
below is revised; `## Review findings and disposition` at the foot records each one.

## Step 0 — Pre-flight

```bash
git branch --show-current      # must be claude/plan-review-frontend-12nyhm
pwd                            # must be /home/user/ht-stock
git status --short             # only untracked plan/spec docs permitted
pip install torch==2.13.0 pytest==9.1.1     # default PyPI index
python -m pytest tests/ -q -m "not slow"    # ESTABLISH THE BASELINE, record the counts
```

Prerequisites actually missing, verified: **torch**, **pytest**. `numpy` and `pandas` are
installed. `download.pytorch.org` is blocked by this sandbox's proxy (403), so the CPU index in
`requirements.txt:14` cannot be used here; PyPI carries `torch==2.13.0` and
`requirements.txt:17-18` says a CUDA-flavoured torch is fine because the code never asks for a
device.

`.venv/` does not exist, so every subprocess test in `tests/test_integration_guards.py` (which
uses `PY = REPO/.venv/bin/python` at `:21`) is **already red before this change**. The baseline
run above is what makes "no regressions" a checkable claim rather than an assumption; its
counts go into the run spec before any code is written.

---

## Step 1 — `.gitignore` (append after line 4)

```
node_modules/
web/dist/
web/coverage/
/shadow/
```

`/shadow/` with the leading slash: a bare `shadow/` matches a directory of that name at any
depth, so a future `web/src/shadow/` component directory would silently vanish from git.

It matters independently of the frontend: `ht.serve`'s `--out` default is a relative path, and
`tests/conftest.py:8-10` (RULE TWO) says nothing is written inside the repo.

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

**Layering — settled, not a risk.** Nothing forbids `ht/` importing `model/`; the direction is
already inverted in the codebase (`ht/config.py:16` does `from model import newsvendor` at
module level). `tests/test_no_sim_import.py:74-77` guards `sim` only.

**The real landmine, which this file must not step on.**
`tests/test_no_sim_import.py:80-86` is parametrised over *every* `.py` under `ht/` and fails if
the literal string `true_demand`, `true_mean` or `lost_sales` appears **anywhere in the file,
comments and docstrings included** — `schema.py` is the sole exemption. A comment like "refuses
a panel carrying true_demand" above the panel loader fails the suite. `ht/serve.py` must never
write those three strings; where it needs to refer to them it says "the simulator-only
columns" and defers to `schema.SIM_ONLY`.

### 2b. `Settings` — one frozen record of the CLI flags

Fields: `panel_path, items_path, artifacts_dir, out_dir, store, max_staleness, timezone,
entered_by`. The shared-flag defaults mirror the **`morning` subparser** (`main()` at
`model/shadow.py:1813`, its arguments `:1814-1870`): `out="shadow"`, `store=""`,
`max_staleness=MAX_STALENESS_DAYS` (=2, `shadow.py:54`). See 2h for why "the CLI's defaults" is
not a single well-formed set. Pinned by `test_server_defaults_match_the_morning_parser`.

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

- `self._rw_lock` — a readers/writer lock, **not** a plain write lock. `log_predictions:702`
  and `record_actuals:966` are non-atomic `open(...,"a")` appends, and `read_predictions:740` /
  `read_overrides:757` go straight to `pd.read_csv`. Under `ThreadingHTTPServer` a `GET` runs
  concurrently with a `POST`, so a reader can hit a half-written line and get a pandas
  `ParserError` naming no file. Readers take the shared lock; writers take it exclusive.
- `_panel()` — caches `shadow._panel(settings.panel_path)` (`shadow.py:1665`) keyed on the
  file's `st_mtime_ns` + size; re-reads on change. **Note what this does not do:**
  `schema.read_panel` (`ht/schema.py:267-270`) returns `conform(df)`, and `conform`
  (`ht/schema.py:239`) drops the simulator-only columns unconditionally. So a panel CSV
  carrying them is silently stripped, never refused, and the `assert_no_truth` calls at
  `shadow.py:1667` and `:270` are assertions on an already-clean frame that cannot fire on
  anything read from disk. Criterion 16 is rewritten accordingly (see disposition F2).
- `_items()` — caches `ht_config.load_items` keyed the same way;
  `ht_config.config_hash(items_path)` recomputed on every request that logs (C2).
- `_require_out()` — mirrors `shadow.py:1808-1810` (the `if` at 1808, the sentence at
  1809-1810): every route except `POST /api/morning` refuses a missing `--out` directory with
  that function's own words. Note `status()` itself does **not** refuse — `shadow.py:1627-1651`
  returns all-`None` on a missing directory — so the guard has to be `_require_out`, not the
  wrapped function.
- `_write_state(**fields)` — **every route that the CLI's equivalent state-writes must too.**
  `_cmd_morning:1694`, `_cmd_score:1742` and `_cmd_weekly:1771` all call `shadow._write_state`
  (`:1653`), and `status()` (`:1627`) reads `current_artifacts_dir`, `last_ingested_date` and
  `last_gates` **only** from `state.json`. Omitting it leaves `GET /api/status` permanently
  empty and criterion 18 unsatisfiable.

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
| `POST /api/score` | `shadow.score_day` (`:1079`) | **server-side precondition** — see 2d-i |
| `POST /api/catch-up` | `shadow.catch_up` (`:1150`) | the safe route; skips days with no panel data |
| `POST /api/weekly` | `shadow.weekly_report` (`:1195`) + `format_weekly` (`:1473`) + `_write_state(last_gates=…)` | **POST, not GET** — the CLI's `weekly` writes `weekly/<week>.txt\|.json` (`_cmd_weekly:1763-1772`) and stamps `last_gates`. A GET that writes violates D1; dropping the writes loses a documented artifact. So it is a POST, and `GET /api/weekly?week_ending=` returns an already-written report or 404. |
| `GET /api/scores`, `GET /api/predictions` | `shadow.read_scores` (`:1130`), `read_predictions` | date-windowed |

Six command routes for the six subcommands, matching `docs/REAL_DATA_READINESS.md:40`.
`GET /api/results` is **removed** — `results/results.json` is the simulator's backtest, of no
use to an operator running a live pilot, and the plan's own deferral section assigns it to
Stage 3.

### 2d-i. `POST /api/score` — the precondition (the worst new failure mode)

`score_day` (`:1079-1090`) is write-once **by file existence**. On a date whose panel export has
not landed, `_score_rows` (`:1000-1077`) returns every row as `status="missing_data"` (`:1034`),
or an empty frame if there are also no predictions — and `score_day` writes that and freezes it.
`catch_up` cannot repair it: it calls the same `score_day` (`:1165`), which sees the file and
only records revisions. `read_scores` yields nothing from a header-only file, so the date never
enters `weekly_report`'s `settled` set (`:1235`), stays in `unscored` forever (`:1236`), and
counts against **G1 completeness** (`gates:1418-1420`) for the life of the pilot. One click,
silently unrecoverable without hand-deleting a CSV.

The CLI's protection is that a person typed the command. A button in front of a 6am operator is
exactly what turns that footgun into an inevitability — and shipping that button is this
change's whole premise, so the guard is server-side, not React state:

`POST /api/score` refuses with a sentence unless the panel carries rows for that date. The
precondition is **panel data present**, not "the day has been entered" — the earlier draft had
the wrong precondition; `docs/REAL_DATA_READINESS.md:211-215` is about the export landing.
Entry order is a second, separate check. `POST /api/catch-up` is offered as the safe default
because `catch_up:1160-1168` already skips days with no panel data.

### 2d-ii. `POST /api/enter` — two things the CLI does that a naive wrapper drops

- **The wrong-date warning.** `_cmd_enter:1705-1708` prints `warning: no sheet was logged for
  <date>; recording it anyway`. It is the strongest wrong-date signal in the loop. Compounding
  it, `entry_order:900-901` falls back to `sorted(items)` when there are no predictions, so a
  mistyped date silently presents an *alphabetical* form instead of the printed order, and
  `record_actuals:951-952` then writes `rec_qty=""` on every row. The API returns this warning
  in the payload and the UI renders it as a blocking confirmation, not a toast.
- **`entered_by`.** `--by` defaults to `os.environ.get("USER","")` (`:1836`) and is recorded on
  every row. A server has no such notion, so `Settings` carries an operator name and the UI
  asks for it once per session; an empty `entered_by` loses the "who keyed it in" field.

### 2d-iii. What "stamped" can honestly mean for a re-forecast

The original criterion 3 — "an explicit re-forecast is stamped and the stamp reaches the
rendered sheet" — has no mechanism. `PREDICTION_COLUMNS` (`:91-97`) has `run_id`, `made_at` and
`backfilled` and no re-forecast field; `morning_sheet` (`:469`) and `_morning_html` (`:546`)
print no backfill marker at all, and `sheet_caveats` (`:618-642`) adds none. Setting
`backfilled=1` to fake a stamp would be a lie that also quarantines a legitimate morning row out
of every headline number (`weekly_report:1217`). Adding a column or a sheet marker means
changing `model/shadow.py`'s behaviour, which the run spec forbids.

So the honest evidence is the one the log already carries: **a `for_date` with more than one
`run_id`**. `GET /api/morning` returns `runs: [{run_id, made_at}, …]` computed from the raw
predictions file (before `read_predictions`' `keep="last"` collapses them), and the UI renders a
banner naming how many sheets exist for that date and which one is live. Criterion 3 is
rewritten to that. Adding a real `superseded` column to `PREDICTION_COLUMNS` is the better fix
and is deferred to its own run spec, because it is a change to the record's schema.

### 2d-iv. `recs.attrs` does not survive the CSV

`forecast()` returns covariates in `out.attrs` — `conditions`, `day_source`, `warnings`,
`staleness_days` (`:379-383`) — and `_cmd_morning:1688` passes `attrs["conditions"]` into
`morning_sheet` as a required keyword. `.attrs` survives neither `to_dict("records")` nor a CSV
round-trip, so **a sheet cannot be re-rendered from `predictions.csv`**. Two consequences:

- `POST /api/morning` renders and writes both sheet formats in the same call, while `attrs` is
  still in hand. It always writes both (`--format both`), so `GET /api/sheet/{date}.html` can
  never 404 on a file the run chose not to write.
- `GET /api/morning` serves the stored sheet plus the logged rows, and returns the caveats and
  conditions from a small `sheets/<date>.meta.json` the POST writes alongside them. It does not
  attempt to reconstruct them.

`day_source == "carried_forward"` and its caveat (`sheet_caveats:637-639`, *"No row for today in
the panel…"*) reach the UI through that meta file.

### 2d-v. The date default and its timezone

Nothing in `shadow.py` computes "today": `--date` is `required=True` for `morning`, `enter` and
`score` (`:1821`, `:1831`, `:1843`). A served UI must default it. If the server's clock is in a
zone **ahead** of the store, it rolls to tomorrow while the store is still on today — and since
a future date is never `backfilled` (`:283`), **no existing guard refuses it**; a whole day gets
forecast under the wrong `for_date`.

So: `--timezone` is a required server flag (no guess), the default date is computed in it, the
UI shows the date it resolved and requires confirmation, and `test_date_default_uses_the_store_timezone`
pins it. Related but benign: `log_predictions:698` and `record_actuals:961` stamp
`dt.datetime.now().astimezone()`; a daemon inheriting `TZ=UTC` records a 5:30am CST sheet as
`11:30:00+00:00`. Ordering survives because the offset is recorded, but the audit trail reads on
the wrong clock, so the server sets `TZ` from `--timezone` at startup. No guard depends on a
clock — `forecast`'s backfill (`:283`) and staleness (`:294`) checks compare panel dates only —
so no refusal can flip.

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
--port --host --allow-remote --timezone --by`. `--host` other than `127.0.0.1` requires
`--allow-remote` and prints what it means (D7). Path/date validation reuses the
`shadow._check_args` (`:1781`) pattern: one sentence, exit 1.

**Which defaults `Settings` mirrors.** The CLI has six subparsers with divergent defaults
(`:1814-1870`) — `--artifacts` is `required=True` for `morning` (`:1819`) but `default=None` for
`weekly` (`:1857`); `--format` and `--width` have no server equivalent. So criterion 20 is
narrowed: `Settings` mirrors the **`morning` subparser's** defaults for the flags they share
(`--out="shadow"`, `--store=""`, `--max-staleness=2`), and the test asserts exactly that rather
than a whole-parser equivalence that is not well formed.

**The lock file, specified properly.** A pid file alone is a trap: a `SIGKILL`, an OOM kill or a
power cut leaves it behind, and next morning the server refuses to start — citing a pid that no
longer exists, in front of an operator the run spec defines as someone who cannot open a shell.
So `<out>/.serve.lock` holds pid **and** process start time, is checked with `os.kill(pid, 0)`,
is reclaimed automatically when the holder is gone, names `--force` in its refusal when the
holder is alive, and is keyed on `os.path.realpath(out)` so `shadow`, `./shadow` and an absolute
path are one directory. Ordering: the lock lives inside `--out`, which only `morning` creates
(`:1808`), so startup creates the directory when absent and takes the lock there —
`_require_out`'s refusal stays a *route*-level guard, unchanged.

**Lock starvation.** `catch_up` (`:1150-1168`) loops `score_day` over every logged date, each
re-reading the panel. Held exclusive with no timeout, a 5:30am `POST /api/morning` queues behind
it and the browser simply hangs. Writers acquire with a timeout and return **503 with a retry
hint** rather than blocking indefinitely.

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

**The failure modes the review surfaced** — new
`test_score_without_panel_data_is_refused` (the frozen-empty-verdict trap, 2d-i),
`test_catch_up_skips_days_with_no_panel_data`,
`test_reader_never_sees_a_torn_row` (a GET during a POST, 2c),
`test_stale_lock_from_a_dead_pid_is_reclaimed`,
`test_live_lock_refusal_names_force`,
`test_lock_is_keyed_on_realpath`,
`test_writer_times_out_with_503_not_a_hang`,
`test_date_default_uses_the_store_timezone`,
`test_future_date_under_a_wrong_timezone_is_caught`,
`test_enter_without_a_logged_sheet_returns_the_warning`,
`test_entry_order_falls_back_alphabetically_and_says_so`,
`test_morning_writes_both_sheet_formats`,
`test_sheet_meta_carries_conditions_and_caveats` (`.attrs` loss, 2d-iv),
`test_multiple_runs_for_a_date_are_reported` (criterion 3 as rewritten),
`test_weekly_post_writes_the_report_and_last_gates`,
`test_entered_by_is_recorded`,
`test_ht_serve_names_no_simulator_column` (the `test_no_sim_import.py:80-86` landmine, checked
in this suite too so it fails loudly rather than as a parametrised surprise)

**Repo integrity** — 26
`test_requirements_unchanged`, `test_no_socket_is_opened` (implicit via the autouse fixture)

≈55 Python tests. During execution each guard marked counterfactual in the run spec is verified
by temporarily removing it and confirming the named test fails.

**Duplication note.** Criteria 5, 7, 8, 9, 11 and 15 are already covered for the *library* by
`tests/test_shadow.py` and `tests/test_shadow_capture.py`. The tests above exercise them
**through `Api.handle`**, which is the point — they pin that the HTTP layer does not swallow or
soften a refusal. Where a library-level test already exists it is named in a comment rather than
re-implemented.

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
    lib/parseEntries.ts  client mirror of shadow.parse_entries
    styles/tokens.css    BOTH token blocks from index.html:8-19 (light) and :20-32 (dark)
    routes/Today.tsx  Enter.tsx  Scores.tsx  Weekly.tsx  Status.tsx
  src/**/*.test.ts       vitest
```

Runtime deps: `react`, `react-dom`, `react-router-dom`. Dev: `vite`, `typescript`, `vitest`,
`@testing-library/react`, `jsdom`. No chart library — inline SVG; load the `dataviz` skill
before writing the first chart.

Behaviour that is not negotiable:

- **Order is enforced on the server** (2d-i), not in React state. The UI disables Score and
  shows the reason, but the refusal that matters is `POST /api/score`'s — grounding the most
  destructive invariant in the layer D1 explicitly distrusts would contradict the spec.
- **Stamps are loud**: `backfilled`, `reconstructed`, `par_fallback`, and the sellout-source
  caveat render as visible banners, not tooltips.
- **PENDING is its own state** — never green, never counted as a pass.
- **No arithmetic in the render path** — verified by grep during the audit.
- Print stylesheet reproduces the sheet's columns; `@media print` hides navigation.
- With `/api/status` unreachable, operator routes are disabled with an explanation.

**The shared vector set is a file, not a convention.** `tests/fixtures/entry_vectors.json` is
committed and read by **both** `tests/test_serve.py` (asserting `shadow.parse_entries` agrees
with it) and `web/src/lib/parseEntries.test.ts`. Hand-copying expectations into TypeScript is
exactly the drift criterion 24 exists to prevent, so neither side may hard-code them.

Client tests: `format.test.ts`, `parseEntries.test.ts` (against that fixture), `gates.test.tsx`,
`stamps.test.tsx`, `apiAbsent.test.tsx`, `entryOrder.test.ts`. ≈15 tests.

**`package-lock.json` must be generated before CI can work.** Step 5 pins `npm ci`, which
hard-fails without a lockfile; `npm install` runs once here to create it and the lockfile is
committed. This needs npm-registry access — verified reachable from this container, unlike
`download.pytorch.org`.

---

## Step 5 — CI (`ci/github-actions-ci.yml`)

- **No change needed for the Python side**: line 36-37's `python -m pytest tests/ -q -m "not slow"`
  already collects `tests/test_serve.py`.
- Add a **second job** `web:` after line 69 — `actions/checkout@v4`, `actions/setup-node@v4`
  (node 20, `cache: npm`, `cache-dependency-path: web/package-lock.json`), then `npm ci`,
  `npm run build`, `npm test` with `working-directory: web`.
- Nothing may be inserted into lines 39-69: a contiguous, order-dependent chain running feature
  build -> split counts -> rehearsal -> backtest -> the `git diff --exit-code` frozen-artifact
  guard at 62-65 -> the slow test at 68-69, which is the file's last line. `npm ci`, never `npm install`,
  so `package-lock.json` is not mutated.
- Amend the header comment at lines 8-11: "pip is the only thing that touches the network here"
  becomes "pip and npm".

---

## Step 6 — Documentation sweep (before commit)

| File | Edit |
|---|---|
| `README.md:145` | "The whole app is a single file…" — scope to `index.html` |
| `README.md:150-151` | "Nothing is uploaded / anywhere" — straddles two lines; scope to the logger |
| `README.md:3-14` | the three-piece framing gains the operator UI |
| `README.md:97-104` | the daily-loop block gains the served equivalent |
| `README.md:186-187` | "The commands exist (…)" — the stale forward reference |
| `README.md:189-191` | "goes back in by / hand" and "the only way" — both straddle lines |
| `README.md:186-187` | also names only four subcommands; the loop has six |
| `README.md:97-104` | same four-of-six omission (no `catch-up`, no `status`) |
| `README.md:218-223` | the permission note, re-read against a localhost server |
| `docs/REAL_DATA_READINESS.md:40` | READY row gains `python -m ht.serve` |
| `docs/REAL_DATA_READINESS.md:192-205` | day-one step 9 gains the served loop |
| `docs/REAL_DATA_READINESS.md:439-443` | the operator-fatigue bullet |
| `docs/features/SHADOW_MODE.md` | "(planned)" → shipped |
| `docs/specs/SHADOW_OPERATOR_UI.md` | verification record filled in; criteria 3/16/18/20/21/24 rewritten per the disposition below; the plan pointer at `:6` and `:95` corrected from the `2026-09-07` filename to `2026-09-09`; the "Created" list corrected — the three docs already exist |

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

---

## Review findings and disposition

Adversarial critique, 2026-09-09. Verified independently before acting: findings F1, F2, F4 and
F5 were each re-checked against the source. 25 of 30 line citations in revision 1 were exact;
the five wrong ones are corrected above.

### Blockers — acceptance criteria that could not pass

| # | Finding | Disposition |
|---|---|---|
| F1 | **Criterion 16 is false.** `schema.read_panel` (`ht/schema.py:267-270`) returns `conform(df)`, and `conform` (`:239`) drops the simulator-only columns unconditionally. A panel carrying them is stripped, never refused, so `assert_no_truth` cannot fire on anything read from disk. **Verified.** | Criterion 16 rewritten: "the simulator-only columns cannot reach the model, whether by refusal or by `conform` stripping them" — tested through `Api.handle` on a panel file that carries them, asserting they are absent downstream. Changing `_panel` to read unconformed and assert first was considered and rejected: it changes `model/shadow.py` behaviour, which the run spec forbids. |
| F2 | **Criterion 3 has no mechanism.** `PREDICTION_COLUMNS` (`:91-97`) has no re-forecast field; `morning_sheet` (`:469`) and `_morning_html` (`:546`) print no backfill marker; faking it via `backfilled=1` would quarantine a legitimate row out of every headline number. | Criterion 3 rewritten to the evidence the log already carries — more than one `run_id` for a `for_date`, surfaced by `GET /api/morning` and rendered as a banner (2d-iii). A real `superseded` column is deferred to its own run spec, being a change to the record's schema. |
| F3 | **Criterion 18 cannot pass** — the plan never called `_write_state`, which `_cmd_morning:1694`, `_cmd_score:1742` and `_cmd_weekly:1771` all do and which `status()` (`:1627`) reads exclusively. `weekly` as a GET also silently dropped `weekly/<week>.txt\|.json`. | `_write_state` added to every writing route (2c). `weekly` becomes a **POST** so it can write without violating D1 (2d). |
| F4 | **Criterion 27 had no baseline, and the baseline is already red** — `tests/test_integration_guards.py:21` uses `REPO/.venv/bin/python`, and `.venv` does not exist. pytest was also missing. **Verified.** | Step 0 now installs pytest and records the pre-change counts before any code is written. |

### Major — failure modes the plan did not know about

| # | Finding | Disposition |
|---|---|---|
| F5 | **A Score button permanently freezes an empty verdict.** `score_day` (`:1079-1090`) is write-once by file existence; on a date whose export has not landed it writes all-`missing_data` rows and freezes them. `catch_up` cannot repair it. The date then counts against G1 completeness for the life of the pilot. **Verified.** The earlier draft's precondition ("the day has been entered") was the wrong one. | New section 2d-i: a **server-side** precondition on `POST /api/score` requiring panel rows for that date, `catch-up` offered as the safe default, two new tests. This is the change's worst new failure mode and it was absent from the risk table, the test list and the run spec alike. |
| F6 | Reads were not lock-protected — only writers were serialised, so a GET during a POST can hit a torn row. | Readers/writer lock (2c) + `test_reader_never_sees_a_torn_row`. |
| F7 | The lock file had no staleness handling, no realpath keying, and contradicted `_require_out`'s ordering. | Specified in 2h: pid + start time, `os.kill(pid, 0)`, auto-reclaim, realpath keying, startup creates the directory. **Superseded by Addendum 2 of the run spec**: the first implementation's `O_EXCL`-then-write left the race open, and `--force` was removed entirely rather than fixed. |
| F8 | `catch_up` under an untimed lock hangs the 5:30am browser. | Writers acquire with a timeout and return 503 with a retry hint. |
| F9 | **No date default and no timezone.** Nothing in `shadow.py` computes "today", and a server clock ahead of the store rolls to tomorrow with no guard refusing it, since a future date is never `backfilled`. | `--timezone` is a required flag; the resolved date is shown and confirmed; two new tests (2d-v). The `made_at` stamp is a lesser, related issue — handled by setting `TZ` at startup. No guard depends on a clock. |
| F10 | The CLI's `warning: no sheet was logged for <date>` (`_cmd_enter:1705-1708`) was dropped, and `entry_order:900-901` silently falls back to alphabetical order on a mistyped date. | Both surfaced in the payload; the UI blocks on confirmation (2d-ii). |
| F11 | `recs.attrs` (`conditions`, `day_source`, `warnings`) survives neither `to_dict("records")` nor CSV, so a sheet cannot be re-rendered from `predictions.csv` — and `morning_sheet` requires `conditions`. | `POST` renders both formats while `attrs` is in hand and writes `sheets/<date>.meta.json`; `GET` serves those (2d-iv). |
| F12 | `entered_by` had no server equivalent, losing the "who keyed it in" field on every row. | `Settings.entered_by`, asked once per session. |

### Specification and hygiene

| # | Finding | Disposition |
|---|---|---|
| F13 | The `ht/` landmine: `tests/test_no_sim_import.py:80-86` is parametrised over every `.py` under `ht/` and fails on the literal simulator column names **in a comment**. Criterion 16's own wording is the comment a developer would write. **Verified.** | Called out in 2a; a test in this suite too, so it fails loudly rather than as a parametrised surprise. |
| F14 | The layering hedge was unnecessary — `ht/config.py:16` already imports from `model/`. | Removed; stated as settled. |
| F15 | `npm ci` cannot run without a committed `package-lock.json`, which the plan never generated. | Step 4 generates and commits it; npm-registry reachability verified. |
| F16 | Criterion 20 was ill-formed — six subparsers with divergent defaults (`--artifacts` required for `morning`, `default=None` for `weekly`). | Narrowed to the `morning` subparser's shared-flag defaults (2h). |
| F17 | Criterion 21 cannot test the bind itself under the no-socket rule; it can only assert on parsed args. | Stated plainly rather than implied. The bind is the one part of D7 with a security consequence that stays untested here. |
| F18 | Criterion 24's "shared vectors" were not actually shared. | `tests/fixtures/entry_vectors.json`, read by both suites. |
| F19 | `GET /api/results` was Stage-3 scope contradicting the plan's own deferral section. | Route removed. |
| F20 | `index.html:8-32` is **two** token blocks; "the nine properties" would drop dark mode. | Both blocks (Step 4). |
| F21 | `.gitignore` `shadow/` matches at any depth. | `/shadow/`. |
| F22 | The run spec's plan pointer (`:6`, `:95`) names a `2026-09-07` file that does not exist, and lists three already-existing files as "Created". | Corrected in the Step 6 sweep. |
| F23 | `README.md:186-187` and `:97-104` name four subcommands; the loop has six. Two more citations straddle line breaks. | Corrected in the Step 6 sweep. |

### Not accepted

- **Criterion 4's "exact row count" under D2.** The critique calls it ambiguous, since two POSTs
  for the same date are one write and one 409. Correct, and that *is* the assertion: one row set
  written, one refusal, file parseable. Concurrency across *different* dates is the separate case
  and is tested as such.
- **§2.7's `_overrides_columns` migration.** The critique confirms it is correct and atomic; no
  change needed, but 2d-ii now names it so the next reader knows why.

### Still true after revision

Codex is unavailable in this environment (no binary, no `.claude/`, and the pipeline's script
path names another repository), so the adversarial Claude critique is the gate. This pass is the
plan-review half of it; the post-commit half runs at Step 8.
