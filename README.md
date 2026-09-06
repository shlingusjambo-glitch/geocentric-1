# Geocentric

Train a causal language model from scratch — tokenizer, architecture, data diet, and
all — on a single GPU. Then watermark it, benchmark it, teach it to see, and ship it.
No agents, no web search, no desktop app. Just the training lab.

```bash
pip install -e .
geocentric plan --preset 250m          # what will this cost me?
geocentric pipeline \
  --data_path data/wikitext103 \
  --sft_data_path data/alpaca_data.json \
  --preset 250m --epicycle speed
geocentric bench --model_dir runs/geocentric      # scored report, with reasons
geocentric release --model_dir runs/geocentric --output_dir dist/mymodel
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
| Precision | bf16 where supported; FP16 + scaling on Turing; FP32 master weights |

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
| `train-vision` | Attach a vision tower and train on image/text pairs |
| `try` / `chat` | LAN web chat / terminal chat (`--image` in terminal) |
| `generate` | One-shot completion |
| `bench` (alias `parallax`) | Run the PARALLAX suite, write a scored Markdown report |
| `release` | Package a checkpoint for publication, with a model card |
| `detect` | Test whether text carries a watermark, and whose |
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

Hit OOM? Add `--gradient_checkpointing` — roughly 30% slower, much smaller. Or
`--epicycle memory`, which shrinks optimizer state instead of recomputing activations.

### When the loss goes back up

It will. A from-scratch run descends, wobbles, and occasionally jumps a whole nat when
a batch of mojibake or a truncated table goes through. Left alone, one bad batch at
hour thirty undoes hour twenty-nine.

The loss guard is on by default. It keeps a running mean and variance of the loss and
reacts to *abnormality*, not to direction:

| what it sees | what it does |
|---|---|
| one window far above the running distribution | drops that update — the gradient never lands |
| several in a row | restores known-good weights, clears moments, re-warms LR over 100 steps |
| a sustained rise over hundreds of steps | rolls back, or exits with `--stop_on_divergence` |
| no improvement over a long window | records a plateau; never acts on it |

It does **not** roll back whenever the loss ticks up, and that restraint is the point.
The reported loss is an average over one accumulation window, so it is a noisy
estimate of a noisy quantity; a cosine schedule raises it legitimately at transitions.
A guard that reverts on every uptick turns training into a walk that never advances.

Rollback restores from a CPU snapshot taken every `--snapshot_every` healthy steps
(200 by default, 4 bytes per parameter of host RAM), so a recovery costs a couple of
hundred steps rather than the whole save interval. Set `--snapshot_every 0` to roll
back to the last saved checkpoint instead — free, but it loses more.

Everything it did lands in `training_metrics.json` and is explained in the PARALLAX
report. `--no_loss_guard` turns the whole thing off.

### Resuming from the best checkpoint

Rerunning the same command resumes. `--resume_from auto` (the default) continues from
the *best* checkpoint rather than the latest, but only when the recorded loss actually
says it is better — compared on held-out loss where both have one, training loss
otherwise. `--resume_from best` forces it; `--resume_from last` restores the old
behavior.

A run started before this existed has no `checkpoints.json` to compare, so `auto`
falls back to the latest checkpoint and resumes exactly as it always did.

### Fine-tuning

```bash
geocentric sft --model_dir runs/geocentric --sft_data_path data/alpaca_data.json
```

Accepts `{"instruction", "input", "output"}`, `{"messages": [...]}`, and ShareGPT
`{"conversations": [...]}`. Multi-turn conversations are supported and only the
assistant's turns contribute to the loss.

## EPICYCLE — training-time acceleration

EPICYCLE combines progressive depth, sequence-length warmup, selective token loss,
and memory-efficient optimization for local pretraining. It builds on established
training research; each gear has a measurable cost as well as a potential benefit.

| gear | shortage | mechanism |
|---|---|---|
| **DEFERENT** | speed | elastic depth — start shallow, grow to full depth, function-preserving |
| **HORIZON** | speed, tokens | fold every window into short segments, then grow context |
| **EQUANT** | intelligence per token | backprop only a trimmed band of the hardest tokens |
| **ARMILLARY** | parameters per GB | rotating momentum; optional factored second moments |

```bash
geocentric pretrain --data_path corpus.txt --preset 250m --epicycle speed
```

`off` (default) · `speed` · `quality` · `memory` · `full` · `capacity` (experimental). None of them changes the
architecture: an EPICYCLE checkpoint is an ordinary checkpoint that loads in code that
has never heard of it.

The current systems ablation on an M4 retained the same token count: folding was
about 10% faster; chunked loss reduced sampled forward allocations by 29% but ran
slower. The combined capacity path halved sampled allocations. These are small-run
measurements, not claims about intelligence or RTX 2060 performance. Raw results and
method: [training efficiency report](research/benchmarks/REPORT.md).

```bash
# Speed path: preserves all batch tokens, uses dense loss by default
geocentric pretrain --data_path corpus.txt --preset 120m --epicycle speed
# Experimental capacity path: factored second moments plus rotating momentum
geocentric pretrain --data_path corpus.txt --preset 120m --epicycle capacity
# Bound vocabulary-head memory in any training stage
geocentric sft --model_dir runs/geocentric --sft_data_path sft.json --loss_chunk_size 256
python scripts/training_systems_bench.py
```

`--loss_chunk_size 0` uses dense loss. The default is 256 with the `memory`, `full`,
and `capacity` pretraining presets; otherwise dense. Smaller chunks save vocabulary
activation memory at a recomputation cost. No sampled softmax or vocabulary pruning:
the loss and gradients match dense cross entropy within floating-point tolerance.

For your main training machine: [Ubuntu 26.04 / RTX 2060 guide](CUDA.md).

Full design, prior art, and what each gear costs: **[EPICYCLE.md](EPICYCLE.md)**.

## Watermarking

Text a model generates can be marked so it is attributable later. The scheme is the
green-list construction of Kirchenbauer et al.: the previous token seeds a split of
the vocabulary, tokens on the favoured side get a small logit bias, and a one-sided
z-test recovers the bias from the output. A reader sees nothing unusual.

The split is keyed on an **identity string**, which is what makes this attribution
rather than a bare "AI or not" flag — the right identity gives a large z, every other
identity gives noise.

`pretrain`, `sft`, `train-vision` and `release` all ask, once, in a terminal:

```
  Watermarking — this model is being trained from scratch, so now is the moment to
  decide how its output should be attributed
  Watermark this model's output? [Y/n]: y
  Identify the model as [Geocentric]: Acme Writer v2
  Strength — light / normal / strong [normal]:
```

Scripted runs never block: pass `--watermark_identity "Acme Writer v2"`,
`--no_watermark`, or `-y`, and a non-tty is treated as `-y`.

The identity travels inside the checkpoint config, so `generate_text` and
`stream_text` mark their output without being asked — a watermark you have to
remember to pass is a watermark that gets forgotten.

```bash
geocentric detect --model_dir dist/mymodel --text "$(cat sample.txt)"
geocentric detect --identity "Acme Writer v2" "Someone Else" --file sample.txt
```

```
  identity                                 z          p          green
  --------------------------------------------------------------------
  Acme Writer v2                       11.42   1.62e-30    141/247   <-- watermarked
  Someone Else                          0.83      0.203     68/247
```

Two limits, because a watermark that is oversold is worse than none. Below about 40
scored tokens the z-test has no power, and `detect` says *undecidable* rather than
*clean*. And heavy paraphrasing removes it — this survives light editing, not
rewriting.

## Computer vision — making the model multimodal

```bash
geocentric train-vision \
  --model_dir runs/geocentric \
  --vision_data_path data/captions.jsonl \
  --freeze_lm                     # stage 1: align the projector only
geocentric chat --model_dir runs/geocentric --image photo.jpg
```

A ViT tower encodes the image into a grid of patch states, a two-layer projector maps
that grid into the language model's embedding space, and the result is *substituted*
for a run of `<|image|>` placeholder tokens in the prompt. The decoder is not modified
at all — RoPE, GQA, the KV cache and the chat template all keep working unchanged.

Attention in the tower is bidirectional. A picture has no arrow of time, and causal
masking would leave the top-left patch unable to see anything else in the image.

At 224px with 16px patches the grid is 14x14 — 196 tokens, a fifth of a 1024-token
context for one picture. `--vision_pool 2` average-pools it to 49, which is the
difference between a caption fitting alongside the image and not.

Data is `.json`/`.jsonl`, one record per image:

```json
{"image": "cat.jpg", "caption": "A cat asleep on a sofa."}
{"image": "chart.png", "instruction": "What does this show?", "output": "Quarterly revenue."}
{"images": ["a.jpg", "b.jpg"], "messages": [{"role": "user", "content": "Compare these."}, ...]}
```

Paths resolve against the data file's own directory first, which is how caption sets
are actually laid out.

The tower is trained from scratch rather than loaded from CLIP, because that is what
this project is for. It is a real quality ceiling and the PRISM probe will tell you
whether you have hit it. Needs Pillow: `pip install 'geocentric[vision]'`.

## PARALLAX — benchmarking, with reasons

```bash
geocentric bench --model_dir runs/geocentric
```

```
  PARALLAX 1.0 — Geocentric (sft)
  --------------------------------------------------------------
  ZENITH      58.2  ███████████░░░░░░░░░  1.043 bits/byte
  MERIDIAN    71.4  ██████████████░░░░░░  +0.250 nats gained from context
  SEXTANT     18.3  ████░░░░░░░░░░░░░░░░  15/40 correct (38%)
  ASTROLABE   50.0  ██████████░░░░░░░░░░  6/12 instructions followed
  NADIR       84.1  █████████████████░░░  0.87 distinct-3, 0/6 looped
  ORBIT           —                       412 tok/s decode
  --------------------------------------------------------------
  Parallax Index: 51.7 / 100
```

Parallax is how you measure the distance to a star: observe from two positions and
read the angle. Every probe works the same way — it puts the model in two situations
that should differ in a predictable direction, and measures whether they do.

| probe | question | how |
|---|---|---|
| **ZENITH** | how well does it model text? | bits per byte across five genres |
| **MERIDIAN** | does more context help? | loss bucketed by position in the window |
| **SEXTANT** | does it know things? | ranks a true completion above three false ones |
| **ASTROLABE** | does it do what it was told? | 12 instructions, checked by rule, never by a judge model |
| **NADIR** | how badly does it degenerate? | distinct-n and loop detection, repetition penalty off |
| **ORBIT** | how fast is it? | prefill/decode throughput, peak memory |
| **PRISM** | is it actually looking at the image? | caption loss with the right image vs the wrong one |

SEXTANT and MERIDIAN work on base models — they need no chat template, only
log-probabilities — so a pretrained-only checkpoint gets a real score instead of a
zero. ASTROLABE is skipped for base models and its weight is redistributed, because a
probe that did not run must never count as a zero.

`PARALLAX.md` lands next to the checkpoint with the score table, strengths,
weaknesses, and a **diagnosis section that reads `training_metrics.json` and names the
likely cause**:

> ### 🔴 Trained far below the compute-optimal token budget
> - **Evidence** — saw 204,000,000 tokens against a Chinchilla-style budget of
>   2,400,000,000 for 120,000,000 parameters — 11.8x short
> - **What it means** — A model that has not read enough text is fluent before it is
>   knowledgeable. This depresses SEXTANT first, then ZENITH, and leaves generations
>   that are locally well-formed and factually empty. It is not an architecture
>   problem and no hyperparameter will fix it.
> - **What to do** — add roughly 2,196,000,000 more tokens of data, or drop to a
>   smaller `--preset` so the budget matches the corpus you have

Rules also fire on unconverged runs, overfitting, dead context, short vocabularies,
loop-prone SFT, dropped overlong conversations, ungrounded vision towers, loss spikes,
plateaus and low hardware utilization.

The bundled probes are a few kilobytes written for this suite rather than sampled from
a public corpus, so a model trained on Wikipedia has not already read them. That keeps
them honest and keeps them small — treat the index as a calibrated smoke test, not a
leaderboard. `--eval_text yourdata.txt` runs the same probes on your own held-out data
at whatever size you like, and is strictly better whenever you have it.

## Releasing a model

```bash
geocentric release --model_dir runs/geocentric --output_dir dist/mymodel \
  --license apache-2.0 --description "A 250M model trained on public-domain fiction."
```

Asks the watermark question one last time — this is the last point at which
attribution can be added, since once the weights are out they cannot be marked — then
writes a directory containing the weights (optimizer state stripped), the tokenizer,
the config, `MODEL_CARD.md`, the full `PARALLAX.md` report backing every number the
card claims, and a `manifest.json` with a SHA-256 for each file.

## Testing a checkpoint

```bash
geocentric try --model_dir runs/geocentric-120m
```

`try` starts the bundled Geocentric web app on port 8000, prints localhost and LAN/Wi-Fi URLs, and opens your browser. Connect from another device on the same network using the printed LAN URL. No Node installation or cloud service is needed. The app includes streaming, stop, regenerate, editable turns, searchable browser-local history, code copying, export, settings, and your Geocentric artwork.

```bash
geocentric try --model_dir runs/geocentric-120m --port 8000 --no_browser
geocentric try --model_dir runs/geocentric-120m --host 127.0.0.1
geocentric try --model_dir runs/geocentric-120m --terminal
```

The default bind is `0.0.0.0`: reachable LAN users can use the model without a login. Use `--host 127.0.0.1` for access only on this computer. The server handles one generation at a time and returns a busy response for additional requests. Browser history is separate on each device. `--checkpoint FILE.pt` selects a checkpoint; `--dtype auto` chooses the device precision. Web input is currently text; use `chat --image` for vision.

The mode follows the checkpoint, because the two kinds of model want different
input:

- **pretrained only** — raw continuation. No system prompt, no roles. You type the
  start of a passage and it continues. A base model has never seen a chat template,
  so feeding it one produces role tags it cannot close, which looks like a broken
  model rather than the wrong question.
- **instruction tuned** — chat turns with a system prompt, stopping at `<|eot|>`.

Force either in web settings or with `--mode base` / `--mode chat`. Terminal in-session commands: `/reset`,
`/system <text>`, `/exit`.

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
full forward pass, and attention is causal. It also pins the newer ones — that growing
a DEFERENT layer does not change the model's output, that a watermark detects under
its own identity and reads as noise under any other, that a skipped PARALLAX probe is
excluded from the index rather than scored zero, that the loss guard never fires on a
normal noisy descent, and that a checkpoint written in the pre-3.1 format still loads
and produces identical logits.

## Compatibility

Checkpoints and runs from 3.0 keep working. `GPTConfig` gained two optional fields
that default to `None`, the forward pass gained keyword arguments that default to the
old behavior, and `<|image|>` is reserved in newly trained tokenizers only — an older
tokenizer without it has the token added and the embedding matrix grown by one row
when a vision tower is attached, so no retrain is needed. `tests/test_backward_compat.py`
builds checkpoints in the old shape by hand and asserts all of it.

## Balanced optimizer and inference diagnostics

`--epicycle balanced` is an experimental, opt-in extension of `capacity`. It partitions each tensor across momentum rings, so a dominant embedding no longer determines the largest ring. Factored second moments remain unchanged. This reduces optimizer state; it does not reduce model weights, gradients, or guarantee the same convergence. Existing presets and checkpoint parameter shapes are unchanged. Continue existing training with its original optimizer configuration; switching to `balanced` is a new optimizer experiment, not a transparent resume.

Inference now reuses bounded KV storage and applies top-p/min-p sampling in the top-k candidate space. Exact-k selection can differ at tied cutoff logits. The web client reports a repetition stop after six consecutive copies of a 1–4 token cycle. This serving guard does not establish that a training run diverged. Base models are continuers, and pretraining percentage alone does not diagnose repeated words.

Run a read-only probe on the affected checkpoint:

```bash
python scripts/repetition_diagnostic.py --model_dir runs/geocentric-120m --device cuda
python scripts/inference_bench.py --device cuda
```

See [this pass’s results](research/benchmarks/INFERENCE_AND_BALANCED.md) for measurements and limitations.
