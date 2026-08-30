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

step "2  synthetic store -> raw real-shaped export  (dirt=$DIRT)"
"$PY" tools/make_mock_export.py --src data/store_synth.csv --out .rehearsal/raw --dirt "$DIRT"

step "3  raw export -> canonical panel"
# exit 2 means the panel was written but validation found errors; both are failures here.
"$PY" -m ht.ingest --mapping .rehearsal/raw/mapping.json --items "$ITEMS" \
  --out .rehearsal/panel.csv --report .rehearsal/ingest_report.json

step "4  validate the panel"
set +e
"$PY" -m ht.validate --panel .rehearsal/panel.csv --items "$ITEMS" \
  --mapping .rehearsal/raw/mapping.json --json .rehearsal/validation.json \
  > .rehearsal/validation.txt
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
"$PY" -m model.shadow catch-up --panel .rehearsal/panel.csv --items "$ITEMS" \
  --out .rehearsal/shadow
LAST="${DATES[${#DATES[@]}-1]}"
WEEKS=$(( REPLAY_DAYS / 7 ))
"$PY" -m model.shadow weekly --panel .rehearsal/panel.csv --items "$ITEMS" \
  --artifacts .rehearsal/artifacts --week-ending "$LAST" --weeks "$WEEKS" \
  --out .rehearsal/shadow --format both --include-backfilled | tee .rehearsal/weekly.txt
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
      f"negatives {rep['negatives_clipped']}  grid rows {sum(rep['grid_rows_inserted'].values())}")
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
