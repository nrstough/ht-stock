# Feature — Shadow mode

The living truth for Phase 3: the model prints a morning sheet, nobody follows it, and its
forecasts are scored against what the store actually did. Four weeks of that produce one page
a district manager reads, carrying five go/no-go criteria fixed in advance.

This document describes the feature across every surface it has. Today that is the CLI. The
served operator UI is being added under `docs/specs/SHADOW_OPERATOR_UI.md` and is marked
**(planned)** below until it ships.

---

## Why the record has to be append-only

Four shadow weeks are worth nothing unless every number traces to a forecast that provably
existed before the day it is for. Four properties carry that claim, and every surface must
preserve all four:

1. **The prediction log is append-only.** `log_predictions` (`model/shadow.py:692`) writes to
   `shadow/predictions.csv` *before* the sheet is rendered.
2. **A day's score is written once.** A later change is disclosed in
   `shadow/scores/_revisions.csv` rather than applied (`_record_revisions`).
3. **A backfilled forecast is stamped.** `backfilled=1`, and `weekly_report` quarantines those
   rows unless `include_backfilled=True`, which stamps the page `reconstructed`.
4. **The forecaster refuses to see the date it is forecasting.** `forecast()` raises when the
   panel carries actuals on or after `for_date`.

`tests/test_shadow.py` and `tests/test_shadow_capture.py` are the guard on all four.

One consequence worth stating plainly, because it is the trap any new surface falls into:
`read_predictions` (`model/shadow.py:738`) resolves duplicates with `keep="last"`. A second
run for the same `for_date` **replaces** the live forecast. Under a human typing one command
at 5:30am that is safe. Under anything that can be triggered twice — a browser refresh, a
retry, a scheduler — it silently substitutes a later forecast for the morning one and nothing
in the record says so.

## The daily loop

| Step | When | CLI | UI (planned) |
|---|---|---|---|
| Print the sheet | before open | `model.shadow morning` | Today → Print |
| Kitchen writes on the paper | during the day | — | — |
| Key the sheet back in | after close | `model.shadow enter` | Enter |
| Freeze the day's verdict | after the export lands | `model.shadow score` | Enter → Score |
| Score every day that now has data | any time | `model.shadow catch-up` | Status → Catch up |
| The district-manager page | weekly | `model.shadow weekly` | Weekly |
| What is behind | any time | `model.shadow status` | Status |

### `morning`

Forecasts one date for every active item, logs the predictions, then writes
`shadow/sheets/<date>.txt` and `.html`. Refuses a date the panel already covers unless
`--backfill` is passed; refuses a panel more than `MAX_STALENESS_DAYS` behind the day before
the forecast date. An item the checkpoint never saw, or one without a full context window,
gets its trailing par with `source="par_fallback"` rather than a second unvalidated forecast
printed with the model's authority.

### `enter`

Reads `item, made, sold out at[, note]` per line — from a pipe, a file, or an item-by-item
prompt that walks the page in the order it printed. Nothing is written unless every line
parses; a half-keyed day that looks entered is worse than one that obviously is not.

Three details that are load-bearing:

- A blank **sold out at** cell counts as "it did not sell out" **only** on a row that came in
  through `enter`, which stamps `sellout_source="sheet"`. A hand-authored overrides file
  carries no such promise and its blanks stay unknown (`_sheet_sellout`). For a store whose
  export has no sellout column, this is the only way accuracy on fully-served days can be
  measured at all.
- `"0"` in that cell is a written negative, not a sellout at midnight (`parse_time`).
- The item may be typed as the sheet printed it, truncated ITEM column and all — but only
  where the truncation is unambiguous, because two names that shorten to the same
  `SHEET_ITEM_WIDTH` characters would otherwise file one item's production under the other
  (`_item_index`).

### `score`

Freezes one day's verdict into `shadow/scores/<date>.csv`. Re-scoring an unchanged day must
record nothing — `_cell` exists so that a `missing_sheet` row read back from CSV as NaN does
not read as a revision.

### `weekly`

Accuracy from the log joined to the frozen scores, never a re-forecast. Rows whose quantiles
are NaN (`par_fallback`) go to the exclusion ledger rather than into the model's accuracy,
because one NaN turns every headline sum into `nan` and because scoring model and par on the
same row set is what makes the comparison paired.

## The five gates

`gates()` (`model/shadow.py:1408`). **PENDING is not PASS** — an unmeasured criterion is
reported as unmeasured, on every surface.

| Gate | What it asks |
|---|---|
| G1 | Completeness — ≥95% of expected rows scored, no gap longer than one day. A day the export never explained counts against it exactly like a morning nobody printed a sheet on. |
| G2 | Accuracy — model WAPE ≤ 90% of par's, and ahead in at least 3 of 4 weeks. |
| G3 | Calibration — the median quantile within 10 points of 50%, and the 0.90 quantile's lower coverage bound ≥ 75%. |
| G4 | Economics — waste saving lower bound ≥ 15% of observed waste retail, without raising sellout days more than 3 points. |
| G5 | Coverage — at most two of the top items dropped for short history. |

## On-disk layout

```
shadow/
  predictions.csv          append-only; PREDICTION_COLUMNS
  overrides/<date>.csv     append-only; OVERRIDE_COLUMNS; last row for (date,item) wins
  scores/<date>.csv        written once
  scores/_revisions.csv    disclosures, never overwrites
  sheets/<date>.txt|.html  what was printed
  weekly/<week>.txt|.json  the district-manager page
  state.json               last ingested / sheet / scored date, current gates
```

## Surfaces

### CLI — shipped

```bash
python -m model.shadow morning --panel PANEL.csv --artifacts ART/ --items ITEMS.json \
    --date YYYY-MM-DD --out shadow --format both
python -m model.shadow enter  --items ITEMS.json --date YYYY-MM-DD --out shadow --by NAME
python -m model.shadow score  --panel PANEL.csv --items ITEMS.json --date YYYY-MM-DD --out shadow
python -m model.shadow weekly --panel PANEL.csv --items ITEMS.json --week-ending YYYY-MM-DD --out shadow
python -m model.shadow status --out shadow
```

Only `morning` creates the shadow directory; every other command refuses a directory that is
not there, because a typo in `--out` would otherwise read an empty log and report a pilot that
lost its record.

### Served operator UI — planned

`ht/serve.py` plus `web/`. See `docs/specs/SHADOW_OPERATOR_UI.md`. It wraps these same
functions and adds no arithmetic. The constraints it inherits from this document — no `GET`
that writes, a second run for a date refused rather than appended, PENDING never rendered as
a pass, and every CLI refusal reproduced verbatim — are not UI preferences. They are what
keeps the four properties above true.
