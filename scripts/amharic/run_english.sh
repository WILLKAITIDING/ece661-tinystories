#!/usr/bin/env bash
# English baseline matched to the Amharic runs: same number of stories, tokenizer size,
# model sizes and training steps.
#
# Usage:  DATA=/path/to/english RUNS=/path/to/runs_en bash run_english.sh
#   DATA   folder for the English data (created by this script)
#   RUNS   output folder for the English models
#   STEPS  training steps to match (default 7110, the Amharic full-corpus runs)
#   VOCAB  tokenizer vocabulary size (default 4000)
#
# Finished steps are skipped, so the script can be re-run after an interruption.
set -u
: "${DATA:?set DATA to the folder for the English data}"; : "${RUNS:?set RUNS to the output folder}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=${STEPS:-7110}
VOCAB=${VOCAB:-4000}
mkdir -p "$RUNS"

[ -f "$DATA/clean/train.txt" ] || python "$HERE/prepare_english.py" --out "$DATA" || exit 1
[ -f "$DATA/tokenizers/en_bpe_${VOCAB}.model" ] || python "$HERE/train_tokenizer.py" --train "$DATA/clean/train.txt" \
    --valid "$DATA/clean/valid.txt" --out "$DATA/tokenizers" --sizes $VOCAB --prefix en || exit 1

# English stories tokenize to a different length, so the epoch count is set to give the same number of steps.
EPOCHS=$(python -c "
import json,sys
t=json.load(open('$DATA/tokenizers/tokenizer_report.json'))['bpe_${VOCAB}']['train_tokens']
print(round($STEPS/max(1,t//(64*512)),3))")
echo "epochs for $STEPS steps: $EPOCHS"

COMMON="--train $DATA/clean/train.txt --valid $DATA/clean/valid.txt --tokenizer $DATA/tokenizers/en_bpe_${VOCAB}.model --epochs $EPOCHS"
run () {            # run <name> <extra args...>
  local name=$1; shift
  if [ -f "$RUNS/$name/metrics.json" ]; then echo "== $name: already done, skipping"; return; fi
  echo "== $name"; python "$HERE/train_lm.py" $COMMON --out "$RUNS/$name" "$@" 2>&1 | tee "$RUNS/$name.log" | grep -E "scratch:|transfer:|train:|best checkpoint|synthetic_valid|natural  |early stop|Error|error"
}
run en_1M_scratch --size 1M
for S in 3M 8M; do
  run en_${S}_scratch    --size $S
  run en_${S}_scratch_s1 --size $S --seed 1
  run en_${S}_scratch_s2 --size $S --seed 2
done
python "$HERE/summarize_runs.py" "$RUNS"

# The same quality measures as for the Amharic models.
python "$HERE/eval_quality.py" --runs "$RUNS" --valid "$DATA/clean/valid.jsonl" --train "$DATA/clean/train.jsonl" \
    --names "$DATA/en_names.csv" --out "$RUNS/quality"
python "$HERE/copy_check.py" --runs "$RUNS" --train "$DATA/clean/train.jsonl"
