#!/usr/bin/env bash
# The dress rehearsal: the whole real-data chain, one command.
#
# Everything so far runs on the simulator's own CSV, which proves nothing about a store's
# export. This takes that CSV, writes it back out as a raw real-shaped store file with the
# simulation-only columns stripped and every column renamed, and then runs the REAL path over
# it -- ingest, validate, train, evaluate, shadow, weekly -- plus a degraded run with no
# sellout signal at all, which is the case a 90-day pilot is most likely to be in. If this
# chain is green, "point it at the export" is a claim with a command behind it.
#
# It writes only inside .rehearsal/ (gitignored) and guards the four frozen provenance files
# by md5 before and after: results/results.json and model/artifacts/ back the dollar figures
# in the proposal and a rehearsal must never move them.
#
# usage: scripts/rehearse.sh [--fast] [--dirt none|light|full]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"     # never bare python3: it has none of the dependencies
REH="$ROOT/.rehearsal"
ITEMS="config/items.example.json"

DIRT="full"
EPOCHS="${EPOCHS:-40}"                 # 40 is ~50s here; the model beats the naive par well
REPLAY_DAYS=28                         # four weeks of morning sheets, the pilot's shadow phase
while [ $# -gt 0 ]; do
  case "$1" in
    --fast) EPOCHS=6; REPLAY_DAYS=7 ;;                       # CI: prove the plumbing, not the fit
    --dirt) DIRT="$2"; shift ;;
    --dirt=*) DIRT="${1#*=}" ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

FROZEN=(results/results.json model/artifacts/demandnet.pt model/artifacts/meta.json
        data/store_synth.csv)
# The published bytes. Hard-coded rather than snapshotted at startup so a repo that is ALREADY
# corrupted fails here instead of quietly rehearsing against the wrong provenance.
FROZEN_MD5=(3a70ea8e958e6b9d55158148269a79da
            7a07a8c8e7c65aefa9bd23b9ebd6c7e8
            4b6624375353058f2da43837bd80e2c9
            6aad4bbfc0be9fa61297295878f2809c)

step() { printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }
fail() { printf '\n*** REHEARSAL FAILED: %s\n' "$*" >&2; exit 1; }

check_frozen() {
  local i=0 got
  for f in "${FROZEN[@]}"; do
    [ -f "$f" ] || fail "$f is missing"
    got="$(md5sum "$f" | cut -d' ' -f1)"
    [ "$got" = "${FROZEN_MD5[$i]}" ] || fail "$1: $f is ${got}, expected ${FROZEN_MD5[$i]}"
    i=$((i + 1))
  done
  echo "frozen artifacts intact ($1)"
}

step "0  frozen artifact guard"
check_frozen "before"

step "1  clean scratch"
rm -rf "$REH"
mkdir -p "$REH"

# The frozen md5 above says data/store_synth.csv has not changed. This says it is still the
# file sim/generate.py produces -- the seed is fixed, so a re-run into scratch must land on
# the same bytes. Without it a params.py edit would leave the CSV frozen and the generator
# quietly disagreeing with it.
"$PY" -m sim.generate --out "$REH/store_synth_regen.csv" > "$REH/generate.txt"
cmp "$REH/store_synth_regen.csv" data/store_synth.csv \
  || fail "sim.generate no longer reproduces data/store_synth.csv"
echo "sim.generate still reproduces data/store_synth.csv byte for byte"

step "2  synthetic store -> raw real-shaped export  (dirt=$DIRT)"
"$PY" tools/make_mock_export.py --src data/store_synth.csv --out .rehearsal/raw --dirt "$DIRT"

step "3  raw export -> canonical panel"
# exit 2 means the panel was written but validation found errors; both are failures here.
"$PY" -m ht.ingest --mapping .rehearsal/raw/mapping.json --items "$ITEMS" \
  --out .rehearsal/panel.csv --report .rehearsal/ingest_report.json

step "4  validate the panel"
set +e
# --ingest-report is what makes the repair block real: a collapsed duplicate, a dropped item
# code and a grid-filled zero leave no trace in the panel, so without the report the validator
# can only say it cannot count them.
"$PY" -m ht.validate --panel .rehearsal/panel.csv --items "$ITEMS" \
  --mapping .rehearsal/raw/mapping.json --ingest-report .rehearsal/ingest_report.json \
  --json .rehearsal/validation.json > .rehearsal/validation.txt
VALIDATE_RC=$?
set -e
tail -n 40 .rehearsal/validation.txt
[ "$VALIDATE_RC" -eq 0 ] || fail "ht.validate exited $VALIDATE_RC; see .rehearsal/validation.txt"
grep -q "^ERRORS (0)" .rehearsal/validation.txt || fail "the panel has error-level findings"
# A rehearsal where the validator finds NOTHING has proved nothing: the export is dirty on
# purpose, so the findings it is supposed to raise are asserted, not hoped for.
if [ "$DIRT" = "full" ]; then
  for check in short_history sellout_coverage weather_unknown unexplained_outage; do
    grep -q "\[$check\]" .rehearsal/validation.txt \
      || fail "expected warning [$check] is missing -- the dirt did not reach the validator"
  done
  echo "expected warnings present: short_history, sellout_coverage, weather_unknown,"
  echo "                           unexplained_outage"
fi
for check in repair_grid_filled repair_duplicates_collapsed repair_rows_dropped; do
  grep -q "\[$check\]" .rehearsal/validation.txt \
    || fail "expected INFO [$check] is missing -- the ingest report did not reach the validator"
done
echo "repairs counted on the validation page: grid_filled, duplicates_collapsed, rows_dropped"

step "4b  the district export: ingest, validate and features must agree"
# The same movement report as the district office runs it. Three layers guard this one
# condition and here is where they have to give the same answer, because only ingest can fix
# it: ht.ingest refuses the raw file, ht.validate errors on a merged panel, features.build
# refuses one that reached it without either.
set +e
"$PY" -m ht.ingest --mapping .rehearsal/raw/mapping_district.json --items "$ITEMS" \
  --dry-run --quiet > .rehearsal/district_refusal.txt 2>&1
DISTRICT_RC=$?
set -e
cat .rehearsal/district_refusal.txt
[ "$DISTRICT_RC" -eq 1 ] || fail "ht.ingest accepted a two-store export (rc=$DISTRICT_RC)"
grep -q "0456" .rehearsal/district_refusal.txt \
  || fail "the refusal does not name the store numbers the file holds"

"$PY" -m ht.ingest --mapping .rehearsal/raw/mapping_district.json --items "$ITEMS" \
  --store 0123 --out .rehearsal/panel_district.csv --quiet | tail -n 3
"$PY" - <<'DISTRICTEOF'
import sys
import pandas as pd
a = pd.read_csv(".rehearsal/panel.csv")
b = pd.read_csv(".rehearsal/panel_district.csv")
join = a.merge(b, on=["item", "date"], suffixes=("_s", "_d"))
same = len(join) == len(a) == len(b) and all(
    (join[c + "_s"].fillna(-1) == join[c + "_d"].fillna(-1)).all()
    for c in ("sold", "produced", "wasted", "stockout", "is_closed"))
print(f"  --store 0123 on the district export reproduces the single-store panel: {same} "
      f"({len(a)} rows)")
sys.exit(0 if same else 1)
DISTRICTEOF

# Ingest will not write a merged panel, so the rehearsal builds one by hand and checks that
# the other two layers still refuse it.
"$PY" - <<'MERGEEOF'
import pandas as pd
a = pd.read_csv(".rehearsal/panel.csv")
b = a.copy()
b["store"] = 456
pd.concat([a, b], ignore_index=True).to_csv(".rehearsal/panel_two_stores.csv", index=False)
MERGEEOF
set +e
"$PY" -m ht.validate --panel .rehearsal/panel_two_stores.csv --items "$ITEMS" \
  > .rehearsal/validation_two_stores.txt 2>&1
TWO_RC=$?
set -e
grep -m1 "\[multi_store\]" .rehearsal/validation_two_stores.txt \
  || fail "ht.validate did not raise [multi_store] on a two-store panel"
[ "$TWO_RC" -ne 0 ] || fail "ht.validate passed a two-store panel"
"$PY" - <<'FEATEOF'
from model import features
try:
    features.build(features.load(".rehearsal/panel_two_stores.csv"), spec=features.legacy_spec())
except features.MultiStorePanel as exc:
    print(f"  features.build refuses it too: {str(exc).split('.')[0]}.")
else:
    raise SystemExit("features.build accepted a two-store panel")
FEATEOF

step "5  train on the panel  (spec auto, ${EPOCHS} epochs, artifacts into .rehearsal/)"
"$PY" -m model.train --panel .rehearsal/panel.csv --items "$ITEMS" \
  --artifacts .rehearsal/artifacts --spec auto --max-epochs "$EPOCHS" | tail -n 20

step "6  observable-only evaluation on the held-out split"
"$PY" -m model.evaluate --panel .rehearsal/panel.csv --artifacts .rehearsal/artifacts \
  --items "$ITEMS" --split test --json .rehearsal/evaluate.json | tee .rehearsal/evaluate.txt

step "7  shadow replay: ${REPLAY_DAYS} morning sheets, then score and report"
mapfile -t DATES < <("$PY" - "$REPLAY_DAYS" <<'PYEOF'
import sys
import pandas as pd
n = int(sys.argv[1])
d = pd.read_csv(".rehearsal/panel.csv", usecols=["date"], parse_dates=["date"])["date"]
for day in pd.date_range(d.max() - pd.Timedelta(days=n - 1), d.max(), freq="D"):
    print(day.date())
PYEOF
)
for day in "${DATES[@]}"; do
  # --backfill because these dates are already in the panel: the log stamps backfilled=1 and
  # the weekly report quarantines them out of every headline number unless asked otherwise.
  "$PY" -m model.shadow morning --panel .rehearsal/panel.csv \
    --artifacts .rehearsal/artifacts --items "$ITEMS" --date "$day" \
    --out .rehearsal/shadow --store "Rehearsal Store" --format both --backfill > /dev/null
  printf '  sheet %s\n' "$day"
done

# The kitchen's half of the loop, and it has to run HERE: score_day freezes a day's verdict
# once, so a sheet keyed in after catch-up would never reach it. These returned sheets are
# SYNTHETIC -- the made and sold-out cells are read straight off the panel, not off paper --
# so this proves the intake, the parser and the sellout stamp, and proves nothing about
# handwriting. Without it the rehearsal never exercises `enter` at all, and for a store whose
# export carries no sellout rule the weekly gates G3 and G4 stay permanently unmeasurable.
for day in "${DATES[@]}"; do
  "$PY" - "$day" "$ITEMS" <<'SHEETEOF' > .rehearsal/sheet_back.csv
import json
import sys

import pandas as pd

day, items_path = sys.argv[1], sys.argv[2]
items = json.load(open(items_path))["items"]
panel = pd.read_csv(".rehearsal/panel.csv")
rows = panel[(panel["date"] == day) & (panel["item"].isin(items))]
for _, r in rows.iterrows():
    # An empty SOLD OUT AT cell on a keyed-in row MEANS "it did not sell out". The panel
    # only knows that where stockout_known is 1, so a row it cannot answer for is left off
    # the sheet entirely rather than answered with a blank -- writing one would have this
    # script inventing the very observation the return path exists to collect.
    if float(r["stockout_known"] or 0) != 1:
        continue
    made = "" if pd.isna(r["produced"]) else f"{float(r['produced']):g}"
    out = "13:45" if float(r["stockout"] or 0) == 1 else ""
    print(f"{r['item']},{made},{out}")
SHEETEOF
  "$PY" -m model.shadow enter --items "$ITEMS" --date "$day" --out .rehearsal/shadow \
    --file .rehearsal/sheet_back.csv --by rehearsal | sed 's/^/  /'
done

"$PY" -m model.shadow catch-up --panel .rehearsal/panel.csv --items "$ITEMS" \
  --out .rehearsal/shadow
LAST="${DATES[${#DATES[@]}-1]}"
WEEKS=$(( REPLAY_DAYS / 7 ))
"$PY" -m model.shadow weekly --panel .rehearsal/panel.csv --items "$ITEMS" \
  --artifacts .rehearsal/artifacts --week-ending "$LAST" --weeks "$WEEKS" \
  --out .rehearsal/shadow --format both --include-backfilled | tee .rehearsal/weekly.txt
# a mixed week prints every source it was scored with, e.g. "produced_vs_sold+sheet"
grep -qE "censoring [a-z_+]*sheet" .rehearsal/weekly.txt \
  || fail "the keyed-in sheets did not reach the weekly report's censoring source"
"$PY" -m model.shadow status --out .rehearsal/shadow

step "8  the degraded run: no sellout signal at all"
# The likely real case. It must ingest, validate and train end to end, only worse.
"$PY" -m ht.ingest --mapping .rehearsal/raw/mapping_nosellout.json --items "$ITEMS" \
  --out .rehearsal/panel_nosellout.csv --report .rehearsal/ingest_report_nosellout.json \
  | grep -E "sellout|rows in|wrote|RESULT" || true
"$PY" -m model.train --panel .rehearsal/panel_nosellout.csv --items "$ITEMS" \
  --artifacts .rehearsal/artifacts_nosellout --spec auto --max-epochs 3 | tail -n 8

step "9  provenance: the frozen backtest must still reproduce results/results.json"
cp results/results.json "$REH/results.before.json"

# The claim the README's provenance paragraph rests on: the plain command reproduces this
# file, and any OTHER configuration has to name its own --out. A one-flag run that silently
# replaced six policies with two is what this guard exists for.
set +e
"$PY" -m model.backtest --policies dl,naive > "$REH/backtest_guard.txt" 2>&1
GUARD_RC=$?
set -e
[ "$GUARD_RC" -eq 1 ] \
  || fail "model.backtest --policies did not refuse results/results.json (rc=$GUARD_RC)"
grep -q "is not it" "$REH/backtest_guard.txt" \
  || fail "the refusal does not say which flags make this run a different one"
cmp -s results/results.json "$REH/results.before.json" \
  || fail "the refused run still touched results/results.json"
echo "a non-frozen configuration is refused before it can write results/results.json"

"$PY" -m model.backtest > "$REH/backtest.txt" || {
  cp "$REH/results.before.json" results/results.json
  fail "model.backtest exited non-zero; results/results.json restored"
}
tail -n 12 "$REH/backtest.txt"
if ! diff -q results/results.json "$REH/results.before.json" > /dev/null; then
  cp "$REH/results.before.json" results/results.json
  fail "model.backtest changed results/results.json; the original has been restored"
fi
echo "results/results.json is byte-identical after a full backtest"

step "10  frozen artifact guard"
check_frozen "after"

step "11  summary"
"$PY" - <<'PYEOF'
import json

rep = json.load(open(".rehearsal/ingest_report.json"))
val = json.load(open(".rehearsal/validation.json"))
meta = json.load(open(".rehearsal/artifacts/meta.json"))
ev = json.load(open(".rehearsal/evaluate.json"))
sell = rep["sellout"]

print(f"  export        {rep['date_range'][0]} .. {rep['date_range'][1]}  "
      f"{rep['n_days']} days, {len(rep['items_kept'])} items")
print(f"  sellout rule  {sell['rule']}  rate {sell['rate']}  "
      f"known_share {sell['known_share']:.3f}  latency {sell['latency_days']}d")
print(f"  repairs       duplicates {rep['duplicates_collapsed']}  "
      f"negatives {rep['negatives_seen']} seen / {rep['negatives_clipped']} clipped  "
      f"grid rows {sum(rep['grid_rows_inserted'].values())}")
print(f"  findings      {sum(f['level'] == 'error' for f in val['findings'])} errors, "
      f"{sum(f['level'] == 'warning' for f in val['findings'])} warnings")
print(f"  splits        train_end {meta['train_end']}  val_start {meta['val_start']}  "
      f"test_start {meta['test_start']}  cov_dim {meta['cov_dim']}")
print(f"  excluded      {[e['item'] for e in meta.get('excluded_items', [])] or 'none'}")
acc = ev["accuracy"]
print(f"  test wape     dl {acc['dl']['wape_uncensored']:.3f}  "
      f"naive {acc['naive']['wape_uncensored']:.3f}  ridge {acc['ridge']['wape_uncensored']:.3f}")
try:
    gates = json.load(open(".rehearsal/shadow/state.json")).get("last_gates") or {}
    print(f"  gates         {gates or 'not evaluated'}")
except FileNotFoundError:
    pass
PYEOF

cat <<'EOF'

The synthetic panel cannot discriminate between sellout rules: stockout, wasted<=0 and
sold>=produced are the same event in all 9837 open rows, because sim/generate.py defines them
that way. A green rehearsal says nothing about which sellout rule is right for a real store.

EOF
echo "REHEARSAL PASSED"
