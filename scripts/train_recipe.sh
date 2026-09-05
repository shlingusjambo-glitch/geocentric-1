#!/usr/bin/env bash
# Geocentric standard recipe: a ~120M conversational model on one consumer GPU.
#
# Sizing rationale (measured on an RTX 2060, 6 GB, sm_75):
#   throughput 11,700 tok/s at 120M params, ctx 1024, fp16
#   Chinchilla-optimal size for a ~3 day budget on this card is ~112M
#   250M does not fit at all: AdamW state alone is ~4 GB of ~5 GB free
#
# Stages are separate so any of them can be rerun. Pretraining resumes from its
# last checkpoint if interrupted, so Ctrl+C is safe.
set -euo pipefail

PY="${PY:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-runs/geocentric-120m}"
DATA_DIR="${DATA_DIR:-data}"
TOKENS="${TOKENS:-3e9}"
PRESET="${PRESET:-120m}"
CTX="${CTX:-1024}"
VOCAB="${VOCAB:-32000}"
# ~250k tokens per optimizer step. Small models converge better with more, smaller
# steps than with the 0.5M-token batches used for larger ones.
ACCUM="${ACCUM:-122}"

echo "=== 1/5  download ==="
$PY scripts/download_recipe.py --out_dir "$DATA_DIR" --target_tokens "$TOKENS"

echo "=== 2/5  tokenizer (${VOCAB} vocab) ==="
if [ ! -f "$RUN_DIR/tokenizer.json" ]; then
  mkdir -p "$RUN_DIR"
  # Trained on a sample rather than the full corpus: BPE merges converge long
  # before the last gigabyte and the full pass costs hours for no gain.
  $PY scripts/sample_corpus.py --src "$DATA_DIR/pretrain" --out "$DATA_DIR/tokenizer_sample.txt" --gb 2
  $PY -m geocentric.cli train-tokenizer \
      --data_path "$DATA_DIR/tokenizer_sample.txt" \
      --output "$RUN_DIR/tokenizer.json" \
      --vocab_size "$VOCAB" --doc_sep $'\n\n\n'
else
  echo "tokenizer exists, skipping"
fi

echo "=== 3/5  tokenize corpus to binary shards ==="
$PY -m geocentric.cli prepare \
    --data_path "$DATA_DIR/pretrain" \
    --output_dir "$RUN_DIR" \
    --tokenizer "$RUN_DIR/tokenizer.json" \
    --doc_sep $'\n\n\n' \
    --val_fraction 0.002

echo "=== 4/5  pretrain ==="
$PY -m geocentric.cli pretrain \
    --data_path "$DATA_DIR/pretrain" \
    --output_dir "$RUN_DIR" \
    --preset "$PRESET" --block_size "$CTX" --vocab_size "$VOCAB" \
    --gradient_accumulation_steps "$ACCUM" \
    --learning_rate 6e-4 --warmup_ratio 0.01 \
    --doc_sep $'\n\n\n' \
    --eval_every 250 --save_every 100 \
    --modelver "Geocentric"

echo "=== 5/5  instruction tuning ==="
$PY -m geocentric.cli sft \
    --model_dir "$RUN_DIR" \
    --sft_data_path "$DATA_DIR/sft/smoltalk.jsonl" \
    --epochs 2 --learning_rate 1e-4 \
    --modelver "Geocentric"

echo
echo "Done. Chat with it:"
echo "  $PY -m geocentric.cli chat --model_dir $RUN_DIR"
