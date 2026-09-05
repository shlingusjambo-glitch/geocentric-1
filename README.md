# Geocentric

Train a causal language model from scratch — tokenizer, architecture, data diet, and
all — on a single NVIDIA GPU. No agents, no web search, no desktop app. Just the
training lab.

```bash
pip install -e .
geocentric plan --preset 250m          # what will this cost me?
geocentric pipeline \
  --data_path data/wikitext103 \
  --sft_data_path data/alpaca_data.json \
  --preset 250m
geocentric chat --model_dir runs/geocentric
```

---

## What the model is

A decoder-only transformer built the way current open models are built:

| Component | Choice |
|---|---|
| Normalization | RMSNorm, pre-norm residual blocks |
| Position | Rotary embeddings (RoPE) |
| Attention | Grouped-query attention with a KV cache |
| Feed-forward | SwiGLU, hidden width 8/3·d rounded to 128 |
| Embeddings | Tied input/output |
| Init | 0.02, scaled to 0.02/√(2·n_layer) on residual projections |
| Precision | bf16 autocast over float32 master weights |

Context length defaults to 1024 tokens (2048 at 1B+), and the vocabulary to 32,000.

## Commands

| Command | Purpose |
|---|---|
| `plan` | Show architecture and token budget for a parameter target |
| `train-tokenizer` | Train a byte-level BPE tokenizer |
| `prepare` | Tokenize a corpus into binary shards |
| `pretrain` | Train from random initialization |
| `sft` | Instruction fine-tune a pretrained checkpoint |
| `pipeline` | `pretrain` then `sft` in one command |
| `chat` | Interactive chat with streaming output |
| `generate` | One-shot completion |
| `list-models` | Show local checkpoints |

### Sizing a run

`plan` tells you what you are signing up for before you spend a week on it:

```
$ geocentric plan --preset 250m
Parameters:        249,396,480
Context length:    1,024 tokens
Token budget:      4,987,929,600 (4.99B) for compute-optimal training
Wikipedia (en) is roughly 4B tokens, so this needs about 1.2x English Wikipedia.
```

The token budget is the number that decides whether your model is any good. A 250M
model wants roughly 5B training tokens. Training it on 200M tokens produces exactly
what you would expect: correct grammar, correct punctuation, and no idea what it is
talking about.

### Pretraining

```bash
geocentric pretrain \
  --data_path data/wikitext103 \
  --output_dir runs/geocentric \
  --preset 250m \
  --epochs 1
```

Batch size and gradient accumulation are chosen from your VRAM to land near 500k
tokens per optimizer step; override either with `--batch_size` /
`--gradient_accumulation_steps`. The corpus is tokenized once into
`runs/geocentric/corpus/*.bin` and reused on every later run. Interrupting with
Ctrl+C saves model *and* optimizer state; rerunning the same command resumes at the
step it stopped on.

Hit OOM? Add `--gradient_checkpointing` — roughly 30% slower, much smaller.

### Fine-tuning

```bash
geocentric sft --model_dir runs/geocentric --sft_data_path data/alpaca_data.json
```

Accepts `{"instruction", "input", "output"}`, `{"messages": [...]}`, and ShareGPT
`{"conversations": [...]}`. Multi-turn conversations are supported and only the
assistant's turns contribute to the loss.

## Data formats

Plain `.txt`/`.md` files are treated as **one continuous document**. Pass
`--doc_sep` when your file holds many documents with a real separator:

```bash
geocentric pretrain --data_path corpus.txt --doc_sep $'\n\n\n'
```

`.jsonl`, `.json`, and `.csv` records are one document each. See
[DATA_FORMAT.md](DATA_FORMAT.md).

## Chat template

```
<|system|>
You are Geocentric, a helpful assistant.<|eot|>
<|user|>
What is the capital of France?<|eot|>
<|assistant|>
The capital of France is Paris.<|eot|>
```

`<|eot|>` is supervised during SFT, which is what teaches the model to stop.

## Upgrading from Geocentric 2.1

**Checkpoints do not carry over.** 2.1 used learned absolute position embeddings,
LayerNorm and a GELU MLP; loading one now raises an explicit error. Retrain from
scratch — and given the data bugs described below, you want to anyway.

The agent CLI, provider integrations, tool runtime, web search, macOS app,
FastAPI server, licensing, and Supabase integration were all removed in 3.0.

## Why 2.1 models came out fluent but off-topic

Five defects, each independently damaging, all fixed:

1. **Documents were shredded into lines.** The loader yielded plain text one line at
   a time and the dataset appended `<eos>` after every one, so a corpus became
   millions of ~15-token fragments each marked end-of-sequence. The model was
   explicitly taught that context resets every sentence.
2. **256-token context**, hardcoded for every model under 1B.
3. **8,192-token vocabulary**, so capacity went into respelling words.
4. **No scaled residual init**, so deep models spent their early steps recovering
   from residual-stream blowup.
5. **SFT at 2e-5**, a fine-tune rate for a trained 7B model. On a small
   from-scratch model it barely moved the weights, so instruction-following never
   took hold and the model echoed prompts back.

Plus: the corpus was held in memory as a Python list of ints (~100 bytes/token,
capping training at a few tens of millions of tokens); prompt and response were
tokenized separately, shifting BPE boundaries at the join; overlong SFT examples
were truncated, stripping their stop token and teaching the model never to finish;
and validation windows were drawn randomly from training documents, so eval loss
was optimistic.

## Proving the changes helped

The rewrite is justified by diagnosed bugs and established results, but neither is
evidence about *your* model. `scripts/ab_compare.py` measures it directly: it checks
the 2.1 pipeline out of git history into a worktree, trains both versions on the same
corpus slice, and scores them on the same held-out text.

```bash
geocentric download-wiki
python scripts/ab_compare.py --data data/wikitext103 --corpus-mb 200 --preset 50m
```

Output lands in `runs/ab_compare/REPORT.md` with a metric table and sample
completions from each arm.

**The headline number is bits per byte, not loss.** The two arms use different
vocabularies (8,192 vs 32,000), and cross-entropy per token is not comparable across
tokenizers — a larger vocabulary spreads the same text over fewer, individually
harder tokens, so it looks worse on per-token loss while being strictly better at
modeling the text. Normalizing total negative log likelihood by the *bytes* of source
text removes that dependence. Every scored token is given the same amount of context
in both arms.

Notes on running it:

- Use `--corpus-mb` to control the workload, not `--max_steps`. The 2.1 trainer has
  no step cap, so capping steps would train the arms on different amounts of data;
  the script refuses that combination.
- The legacy arm needs `psutil`, which the current dependency list drops.
- The legacy arm's `OneCycleLR` divides by zero on runs of roughly 50–99 optimizer
  steps. That is a pre-existing 2.1 bug — use a corpus large enough to clear it.
- Both arms train well under a compute-optimal budget, so neither produces a good
  model. This measures which pipeline learns more from identical data.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The suite pins the behaviors above: documents are not split into lines, `<eos>` is
rare, SFT masks prompts and supervises the stop token, KV-cached decoding matches a
full forward pass, and attention is causal.
