#!/usr/bin/env bash
# Train and summarise every Amharic model.
#
# Usage:  DATA=/path/to/data RUNS=/path/to/runs bash run_all.sh
#   DATA  folder holding clean/train.txt, clean/valid.txt, extracted/ (human-written stories)
#         and tokenizers/am_bpe_4000.model (see clean_corpus.py and train_tokenizer.py)
#   RUNS  output folder; each run writes RUNS/<name>/ and RUNS/<name>.log
#
# A run that already has metrics.json is skipped, so the script can be re-run after an interruption.
set -u
: "${DATA:?set DATA to the data folder}"; : "${RUNS:?set RUNS to the output folder}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="--train $DATA/clean/train.txt --valid $DATA/clean/valid.txt --natural $DATA/extracted --tokenizer $DATA/tokenizers/am_bpe_4000.model"

run () {            # run <name> <extra args...>
  local name=$1; shift
  if [ -f "$RUNS/$name/metrics.json" ]; then echo "== $name: already done, skipping"; return; fi
  echo "== $name"; python "$HERE/train_lm.py" $COMMON --out "$RUNS/$name" "$@" 2>&1 | tee "$RUNS/$name.log" | grep -E "scratch:|transfer:|train:|best checkpoint|synthetic_valid|natural  |early stop|Error|error"
}
mkdir -p "$RUNS"

# 1. Model size: from scratch vs initialised from the English checkpoint, on the full corpus.
for S in 1M 3M 8M; do
  run am_${S}_scratch  --size $S
  run am_${S}_transfer --init roneneldan/TinyStories-$S
done

# 2. Data scaling at 3M with a fixed number of epochs: 6k / 12.5k / 25k stories
#    (the full corpus is covered by step 1).
for N in 6000 12500 25000; do
  run am_3M_scratch_n$N  --size 3M --subset $N
  run am_3M_transfer_n$N --init roneneldan/TinyStories-3M --subset $N
done

# 3. Data scaling at 3M with equal training steps: the epoch count is raised so every subset gets
#    about the same number of updates (~7,100) as a full-corpus run. Early stopping ends a run
#    that starts to overfit.
for pair in "6000 245" "12500 117" "25000 58"; do
  set -- $pair; N=$1; E=$2
  run am_3M_scratch_n${N}_long  --size 3M --subset $N --epochs $E
  run am_3M_transfer_n${N}_long --init roneneldan/TinyStories-3M --subset $N --epochs $E
done

# 4. Seed variation: two additional seeds for the 3M and 8M full-corpus pairs.
for S in 3M 8M; do
  for SEED in 1 2; do
    run am_${S}_scratch_s$SEED  --size $S --seed $SEED
    run am_${S}_transfer_s$SEED --init roneneldan/TinyStories-$S --seed $SEED
  done
done

python "$HERE/summarize_runs.py" "$RUNS"
