#!/usr/bin/env bash
# KESTREL: a ~250M reasoning/technical model on one RTX 2060.
#
# Named for Ptolemy's compendium -- the book that held the whole of the
# geocentric world's knowledge. This model is the knowledge-and-reasoning
# counterpart to the 120M conversational one.
#
# Sizing rationale (measured, not estimated -- see docs/KESTREL.md):
#   249,396,480 params, 12L x 1280d (20 heads, 5 kv), ctx 1024, vocab 32k
#   FP32 AdamW needs 3.72 GB for weights+grads+state, which is why the 120M
#   recipe says 250M "does not fit at all" on a 6 GB card. EPICYCLE's
#   partitioned+factored optimizer brings that to 2.09 GB, leaving ~3.2 GB for
#   activations. That single flag is what makes this size possible here.
#
# Stages are separate so any can be rerun. Every stage resumes after Ctrl+C.
set -euo pipefail

PY="${PY:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-runs/kestrel-250m}"
DATA_DIR="${DATA_DIR:-data/kestrel}"
TOKENS="${TOKENS:-5e9}"
PRESET="${PRESET:-250m}"
CTX="${CTX:-1024}"
VOCAB="${VOCAB:-32000}"
MODELVER="${MODELVER:-Kestrel}"
# Every gear except MNEME. `full` also runs all four, but keeps plain momentum
# rings; `maximal` adds the factored+partitioned optimizer, which is 0.47 GB less
# state on this model -- the headroom that makes 250M fit a 6 GB card at all.
EPICYCLE="${EPICYCLE:-maximal}"
# ~500k tokens per optimizer step at the batch size the trainer measures (~2).
# Larger models tolerate -- and prefer -- larger batches than the 120M recipe's.
ACCUM="${ACCUM:-244}"
# 6e-4 suits 120M; a 2x model wants a gentler rate.
LR="${LR:-4e-4}"
# MNEME rebalances learning toward rare targets. It won +20pp fact recall on a
# synthetic probe but cost grammar NLL, and has never run on this card. Measure
# before committing nine days to it:  MNEME=--mneme ./scripts/train_kestrel.sh
MNEME="${MNEME:-}"
# LAN status page, so a nine-day run can be checked on from a phone. It reads
# only training_metrics.json, never imports torch, and runs at nice 19 -- it
# cannot take VRAM or a core from the trainer.
MONITOR_PORT="${MONITOR_PORT:-8787}"

# Publish the current stage so the watchers can show the hours before training
# starts -- a download and a corpus tokenization otherwise render as "waiting",
# which looks exactly like a run that died.
mkdir -p "$RUN_DIR"
stage() {
  printf '{"stage": %s, "of": 5, "name": "%s", "started": "%s"}\n' \
    "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/pipeline.json"
}

stage 1 download
echo "=== 1/5  download (~20 GB text, ~9 GB shards) ==="
$PY scripts/download_kestrel.py --out_dir "$DATA_DIR" --target_tokens "$TOKENS"

stage 2 tokenizer
echo "=== 2/5  tokenizer (${VOCAB} vocab) ==="
# Trained on THIS mix, not the 120M's. A tokenizer fitted to prose spends three
# or four tokens on an identifier like `getaddrinfo` and splits indentation into
# single spaces; on a corpus that is a quarter code, that is a large fraction of
# the context window burned on nothing.
if [ ! -f "$RUN_DIR/tokenizer.json" ]; then
  mkdir -p "$RUN_DIR"
  $PY scripts/sample_corpus.py --src "$DATA_DIR/pretrain" --out "$DATA_DIR/tokenizer_sample.txt" --gb 3
  $PY -m geocentric.cli train-tokenizer \
      --data_path "$DATA_DIR/tokenizer_sample.txt" \
      --output "$RUN_DIR/tokenizer.json" \
      --vocab_size "$VOCAB" --doc_sep $'\n\n\n'
else
  echo "tokenizer exists, skipping"
fi

stage 3 "tokenize corpus"
echo "=== 3/5  tokenize corpus to binary shards ==="
$PY -m geocentric.cli prepare \
    --data_path "$DATA_DIR/pretrain" \
    --output_dir "$RUN_DIR" \
    --tokenizer "$RUN_DIR/tokenizer.json" \
    --doc_sep $'\n\n\n' \
    --val_fraction 0.002

# Started here rather than at stage 1: it reports on training_metrics.json,
# which does not exist until the trainer writes it. --wait keeps it patient
# while the first stage warms up; it shuts itself down when training ends.
$PY scripts/watch_web.py "$RUN_DIR" --data_dir "$DATA_DIR" --port "$MONITOR_PORT" --wait 86400 --linger 900 &
MONITOR_PID=$!
# Whatever ends this script -- finish, Ctrl+C, failure -- takes the page with it.
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT INT TERM
sleep 1

stage 4 pretrain
echo "=== 4/5  pretrain (~9 days at 6,000 tok/s) ==="
$PY -m geocentric.cli pretrain \
    --data_path "$DATA_DIR/pretrain" \
    --output_dir "$RUN_DIR" \
    --preset "$PRESET" --block_size "$CTX" --vocab_size "$VOCAB" \
    --epicycle "$EPICYCLE" $MNEME \
    --gradient_accumulation_steps "$ACCUM" \
    --learning_rate "$LR" --warmup_ratio 0.01 \
    --doc_sep $'\n\n\n' \
    --eval_every 250 --save_every 500 \
    --modelver "$MODELVER"

stage 5 "instruction tuning"
echo "=== 5/5  instruction tuning (code / math / reasoning / chat) ==="
# No refusal data: the download step filters canned refusals out of the public
# sets. This produces the unguarded build. The guarded build starts from this
# same checkpoint -- see stage 6 in docs/KESTREL.md.
$PY -m geocentric.cli sft \
    --model_dir "$RUN_DIR" \
    --sft_data_path "$DATA_DIR/sft/kestrel_sft.jsonl" \
    --epochs 2 --learning_rate 1e-4 \
    --save_every 500 \
    --modelver "$MODELVER"

echo
echo "Monitor: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$MONITOR_PORT"
echo "Done. Score it, then talk to it:"
echo "  $PY -m geocentric.cli bench --model_dir $RUN_DIR"
echo "  $PY -m geocentric.cli try   --model_dir $RUN_DIR"
