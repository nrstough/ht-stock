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


---

## Addendum, 2026-09-09 — post-commit audit

The spec above is frozen; this section records what the adversarial post-commit audit found
against it and what changed in response. Nothing above was rewritten.

**Overall on the first pass: Fail.** Fourteen findings. Freeze integrity, regression check and
scope discipline passed; test coverage failed; plan adherence, review compliance and
documentation needed work.

The three that mattered:

- **Arbitrary `.json` disclosure.** `GET /api/weekly?week=` put the query string straight into
  `os.path.join` with no shape check and no containment check — and `os.path.join` discards the
  prefix on an absolute second argument, so `week=/etc/…` escaped entirely. Reachable over HTTP
  because `parse_qs` unquotes `..%2f` before the handler sees it. The sheet route and the static
  server were both defended; the third file-reading route was simply missed. Criterion 22 named
  only two routes, so the spec never asked for it — but D7's threat model did.
- **The change's own headline guard was half-built.** `POST /api/score` refused only when
  *every* row was unsettled. A partly-landed export — eight items in, one not — passed the
  guard and froze the missing one permanently, which is the exact mechanism D-i was written to
  prevent. Now any waiting row refuses, and the message names the items.
- **`GET /api/status` was not under the read lock**, and it is the route the page polls. Under
  a concurrent write it surfaced raw pandas parse errors to the browser: 1080 failures in a
  contention probe, against 0 for the read-locked `/api/predictions`.

Also fixed: the lock file was `exists()`-then-`open()` rather than `O_EXCL` (two simultaneous
starts could both take it, verified with a barrier); `--force` could steal a *live* holder's
lock, leaving two servers on one append-only record — the flag is removed entirely, since
reclaiming a dead lock needs no flag and overriding a live one should not be available;
`json_safe` passed unknown types through to `json.dumps` at the socket; readers could starve a
waiting writer; and `weeks` / `from` / `to` leaked raw pandas and `int()` messages, breaking D4.

Three tests were found to prove nothing and were rewritten: the defaults test asserted a
hard-coded copy of the parser's defaults instead of reading them (so the drift criterion 20
exists to catch would have passed); the static-traversal test was satisfied by its own `or`
escape hatch; and the shadow-directory test checked 6 of 12 routes while being named for all
of them. A one-in-three-hundred flake was root-caused rather than dismissed: `uuid4().hex[:12]`
is all digits often enough that pandas reads `run_id` back as an integer.

Added coverage: criterion 4's "exact row count" now has a test that actually writes
concurrently across dates; `web/src/routes/Today.test.tsx` delivers the `stamps` tests the plan
promised, and `web/src/routes/Enter.test.tsx` the entry-order ones; and the `Today` crash the
browser walk found is now pinned rather than merely fixed.

**How criteria 18 and 20 are read.** Neither was rewritten in the frozen list above; both are
read as the implementation reads them, and the difference is recorded here instead. 18 is
satisfied only with `_write_state` on every route whose CLI equivalent writes it, because
`status()` reads `current_artifacts_dir` from `state.json` and nowhere else. 20 is read against
the `morning` subparser, per F16 in the plan.

Two limits stand, recorded rather than closed:

- Criterion 21 still tests the bind at the parsed-arguments layer only. `tests/conftest.py`
  makes binding a socket impossible in this suite by design.
- `_require_out` cannot fire under `python -m ht.serve`, because taking the lock creates the
  directory. The compensation is `new_shadow_dir` in the status payload and a startup warning,
  both tested; the guard remains meaningful for direct `Api` construction.


---

## Addendum 2, 2026-09-09 — second audit

The first addendum was written from a fix commit that had not been audited. The re-audit graded
it **Fail** again and was right on three counts, one of which was a claim in the addendum
itself.

- **The lock race was not closed, and the test added to prove it failed about one run in
  six.** `O_EXCL` creates an *empty* file and writes the payload as a second step, so a racing
  starter could read nothing out of it, conclude the owner was dead, and delete a live lock —
  the precise outcome the lock exists to prevent. The file is now built in a temp file and
  `link()`ed into place, so it never exists in a state that says nothing, and an unreadable
  lock is assumed **live** rather than dead. Hammered 30× with warnings as errors. Fixing it
  also exposed a bug in the fix: two racers in one process built the same temp filename, so
  one deleted the other's.

- **The freeze hazard was one route wide.** `POST /api/score` refused a partly-landed day
  while `POST /api/catch-up` froze it — through the button the refusal recommended by name.
  Both now share one precondition, and catch-up reports what it held back.

- **That precondition then introduced a permanent lockout.** An item discontinued mid-pilot,
  or added to the items file before the export carries it, has no panel row and never will,
  while `forecast()` keeps logging a prediction for it — so "any missing row refuses" refused
  such a day on every attempt, forever, with no exit. Missing rows are now separated by
  whether the item sold within `RECENT_DAYS`: a gap the export can still fill blocks, an item
  that has stopped is recorded and frozen with the rest.

Corrections to Addendum 1, which stated three things that were not true when written: the
entry-order frontend test did not exist (it does now); the lock fix was described as "verified
with a barrier" when the barrier test was failing intermittently; and "Nothing above was
rewritten" was false — that commit had edited criteria 18 and 24 inside the frozen list. Those
two edits are reverted and the reading is recorded above instead.

Also closed: an unreadable or unreclaimable lock now produces a sentence rather than a
`ValueError` traceback or an unbounded spin, on the acquire path.


---

## Addendum 3, 2026-09-09 — third audit

Fail again. The three headline fixes from Addendum 2 held up under re-examination, but two
round-2 findings were not actually closed, both new mechanisms had a defect of their own, and
Addendum 2 repeated the failure it was written to apologise for.

**Two findings I reported as fixed and had not fixed.**

- The `assert status in (200, 409)` hatch was still there. A second, deterministic test had
  been added for the same scenario — with a docstring saying "Deterministic, not `assert
  status in (200, 409)`" — and the original was left standing beside it. The old test is now
  deleted.
- The corrupt-lock fix covered the acquire path only. Releasing a lock whose file had been
  corrupted or replaced while held still raised `ValueError` or `TypeError` out of the context
  manager, at shutdown, against this module's own "never a traceback" contract.

**Three defects introduced by the previous round's fixes.**

- `os.link` was a portability regression against the `O_EXCL` it replaced: it fails with
  EPERM, EXDEV or ENOTSUP on FAT, several CIFS/NFS mounts and some container layers. A store
  keeping the pilot record on a USB stick could not start the server at all, and got a
  traceback rather than a sentence. `link` is now attempted first for its atomicity, with
  `O_EXCL` as the fallback and a reader that tolerates the brief empty window that leaves.
- A lock released between our failed `link()` and our read of the file — the ordinary
  stop-old-start-new restart window — was reported as "unreadable … delete that file", naming
  a file that no longer existed. A disappearing lock now retries.
- `_unsettled` computed which items were beyond saving and then discarded that list at both
  call sites, so an item idle longer than the window was frozen as `missing_data` with nothing
  naming it. Freezing a row is permanent and only the operator can say whether the item has
  really stopped or its export is behind, so both routes now name it. `RECENT_DAYS` also went
  from 7 to 14: at 7 a weekly item survived exactly one cycle with no margin, and one skipped
  week — a holiday, a stockout, a closure — would have reclassified it.

**Corrections to Addendum 2**, which stated two things that were not true: the missing-sheet
test was not deterministic (see above), and only one of the plan file's two `--force`
references had been marked. Both are now right.

**On the numbers.** Addendum 2 said "553 → 560 passed". Measured properly this time, in a
separate worktree at the base commit rather than from memory: **449 at 250b28d, 560 at the
previous commit, 570 now**, with the same 5 pre-existing `test_integration_guards.py` failures
throughout. Frontend: 87 tests. `tests/test_serve.py` passes under `-W error`, which the
previous round claimed of the lock subset only.

Also closed this round: `catch_up` no longer writes `last_scored_date` (the CLI's does not, and
writing it could move the stored value backwards when an earlier day is filled after a later
one) and its date selection is now pinned by test to `shadow.catch_up`'s; what catch-up held
back and what scoring froze reach the operator's screen rather than the JSON alone; a moved
panel file no longer answers a browser with an `OSError` and an absolute path; the weekly
listing no longer advertises reports the single-week route would refuse; stale lock temp files
are cleared; and `@testing-library/user-event` is pinned exactly like every other dependency.

**Still open, recorded rather than closed.** `serve.catch_up` remains a reimplementation rather
than a delegation, which the module header says this file must never become. The justification
is specific: `shadow.catch_up` freezes a partly-landed day, which is the defect this whole
thread of findings is about. It is held to the CLI's behaviour by test on everything except
that precondition. If `model/shadow.py` ever gains the precondition itself, this should
collapse back to a call.

---

## Addendum 4, 2026-09-09 — fourth audit

Thirteen of sixteen third-round findings genuinely fixed, none regressed — and **one blocking
new defect**, in the mechanism the previous three rounds had each rewritten.

**The lock was the wrong primitive, three times over.** Every version tried to work out
whether the previous holder was still alive by reading a pid out of a file. That question
cannot be answered without a race, and the audit measured the consequence: two starters racing
a *stale* lock both read the corpse's pid, both removed the file, and both claimed — **57
times in 300**, across genuine separate processes, in exactly the scenario the reclaim existed
for. A power cut leaves a lock behind, the box reboots, cron and the operator both start the
server, and two of them append to one append-only record. The suite never saw it because its
race test started from an *empty* directory, so the reclaim branch was never executed.

`fcntl.flock` does not ask the question. The lock lives on an open file descriptor; the kernel
releases it when the holder exits for any reason, including a kill or a power loss. There is
nothing to reclaim, no staleness to detect, no pid to trust, no temp file, no empty window and
no retry budget — and with it go four findings that were all symptoms of the old scheme: the
`os.link` portability regression, the "unreadable — delete that file" advice that would have
had an operator delete a live server's lock on a slow filesystem, the startup sweep that could
break a concurrent claim, and the bounded-retry corner. The pid inside the file is now only a
note for a person reading it.

Verified with two real servers: the second is refused and told that deleting the file will not
help, because the lock is held on an open descriptor rather than by the file existing; and a
lock left behind by `kill -9` does not block the next start. The replacement race test uses
real processes against a stale lock — 84 races, all clean.

Also closed this round: the frozen-items note disagreed with itself on number ("cake, sushi …
that item is still selling") and said "has not sold" where the code measures "has no row in
the panel"; the weekly listing filter gained the test it lacked; and two comments that had
gone stale in the opposite direction are corrected.

**Recorded, not closed.** `score()` still writes `last_scored_date`, which moves backwards
when an earlier day is scored after a later one. It stays because `_cmd_score` does exactly
the same thing and parity with the CLI is this module's contract, and because nothing reads
the stored value — `status()` derives that date from the score files. `catch_up` does *not*
write it, because the CLI's catch-up does not either. The previous commit message implied that
asymmetry was a fix rather than a deliberate difference; it is the latter, and it is now
commented where it happens.
