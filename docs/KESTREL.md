# KESTREL — a 250M reasoning and technical-knowledge model

A kestrel is the small falcon that hunts by hovering — holding a fixed point in
moving air, watching one patch of ground with far better resolution than its size
suggests. It is not the largest raptor and does not try to be. That is the right
promise for this model: a quarter of a billion parameters, run locally, that sees
a narrow technical field sharply rather than everything vaguely.

This is the knowledge-and-reasoning counterpart to the 120M conversational model:
same trainer, same card, a different diet and twice the parameters.

```bash
./scripts/train_kestrel.sh          # all five stages, resumable
```

## Architecture

| | |
|---|---|
| Parameters | 249,396,480 |
| Shape | 12 layers × 1,280 wide, 20 heads / 5 KV heads, FFN 3,456 |
| Context | 1,024 tokens |
| Vocabulary | 32,000, **retrained on this corpus** |
| Compute-optimal budget | 4.99B tokens |

The tokenizer is not inherited from the 120M run. A BPE fitted to web prose
spends three or four tokens on `getaddrinfo` and splits leading indentation into
single spaces; on a corpus that is a quarter source code, that is a large
fraction of every context window spent on nothing. Refitting it is the cheapest
quality win available here and it costs one hour.

## Why the optimizer preset is not optional

The 120M recipe states that 250M "does not fit at all: AdamW state alone is ~4 GB
of ~5 GB free." That is correct for the default optimizer. Measured on this exact
architecture:

| optimizer path | optimizer state | weights + grads + state |
|---|---:|---:|
| FP32 AdamW | 1.86 GB | **3.72 GB** |
| `--epicycle memory` | 0.70 GB | 2.56 GB |
| `--epicycle full` (all gears, plain rings) | 0.70 GB | 2.56 GB |
| `--epicycle capacity` | 0.23 GB | 2.09 GB |
| `--epicycle maximal` (all gears, compact) | 0.23 GB | **2.09 GB** |

The RTX 2060 offers ~5.6 GB, less ~0.3 GB of CUDA context, and activations must
fit in what remains. FP32 AdamW leaves ~1.6 GB for activations and does not
survive; `maximal` leaves ~3.2 GB and does.

`maximal` is added by this work: `full` already runs all four gears, but keeps
plain momentum rings, costing 0.47 GB more state than the factored+partitioned
optimizer for no gear you gain. `maximal` is `full` plus that optimizer.

Partitioned over plain rings: the tied embedding is 40,960,000 parameters, **16.4%
of the whole model in a single tensor**. ARMILLARY does not split a tensor across
rings, so unpartitioned ring assignment lets that one tensor set the size of the
largest ring. Partitioning is precisely the case `balanced` exists for. Total
state is the same; the peak is not.

Reload note: an EPICYCLE optimizer must be resumed with the flags it was trained
under. Changing families mid-run raises rather than silently reinitializing.

## The gears, and what each costs here

Every EPICYCLE gear except MNEME is on (`--epicycle maximal`). What each one
actually does to *this* model:

| gear | shortage it addresses | what it does here | the cost |
|---|---|---|---|
| **DEFERENT** | speed | Starts at 6 of 12 layers, all live by 60% of steps. Sleeping blocks are exact identities, so this is function-preserving, not a smaller model. | Late layers receive fewer total updates than early ones. |
| **HORIZON** | speed | Context grows 256 → 1,024 by the 50% mark, folding each window into short segments rather than cropping, so every token and label survives. | Attention sees short segments for the first half. Folding preserves tokens, **not full-context conditioning**. |
| **EQUANT** | intelligence per token | After 15% of training, backprop only the hardest 65% of token losses, discarding the top 2% as probable corruption. | Saves no dense transformer FLOPs. Quality benefit is a hypothesis, not a measured result. |
| **ARMILLARY** | parameters per GB | Rotating first moments across 4 rings, factored second moments, tensors partitioned across rings. | Approximate, not identical to AdamW convergence. This is the gear that makes 250M possible at all. |

Two of these point against this model's own goals, and it is worth knowing which
before nine days elapse rather than after:

**HORIZON vs MERIDIAN.** MERIDIAN — does the model use its context — was the 120M's
weakest probe at **14.1/100**. HORIZON spends the first half of training on folded
256-to-512-token segments, which is the one thing that most directly limits
learning long-range conditioning. Every token is still trained on, so this is not
lost data; it is deferred long-range structure. If MERIDIAN is the number you care
about, `EPICYCLE=capacity` drops EQUANT and keeps everything else, and
`--epicycle balanced` plus a hand-set `horizon_start` closer to 512 softens it
further.

**EQUANT vs deep knowledge.** EQUANT trims the top 2% of losses as corrupt. The
EPICYCLE notes are explicit that "high loss does not prove corruption: rare facts
and difficult reasoning can also have high loss," and that selection "may harm
calibration or rare-token learning." Rare tokens are where facts live, so on a
model built for technical recall this gear is the one most likely to work against
the objective. It is on because you asked for every gear; `capacity` is the same
configuration without it, and is a one-word change.

Neither concern is a reason not to run it. They are the two places to look first
if SEXTANT or MERIDIAN come back lower than the projection.

## Pretraining mix — 5B tokens

| source | share | why it is here |
|---|---:|---|
| `fineweb-edu` | 17% | Language backbone. Without ordinary prose the model reads like a stack trace. |
| `cosmopedia-v2` | 18% | Synthetic textbooks. Teaches the *explaining* register the model is asked for. |
| `finemath` (3+) | 15% | The cheapest dense source of multi-step reasoning structure. |
| `the-stack` (8 langs) | 25% | Code: python, c, cpp, javascript, shell, rust, go, sql. |
| `peS2o` | 8% | Open-access papers — the only expert-written source in the mix. |
| `wikipedia` (en) | 7% | Factual grounding and the connective tissue between fields. |
| **systems corpus** | **10%** | Kernel docs, man pages, RFCs, CWE, ATT&CK, CVEs. |

The systems slice was raised from 5% to 10% — the extra 5% came off `fineweb`,
the most redundant source — because you asked specifically for depth in Linux,
operating systems, cybersecurity and hacking. It is assembled from primary
sources rather than a dataset because no general web crawl carries kernel,
protocol, syscall and vulnerability text at usable density; that is exactly the
material CommonCrawl represents worst. Its six sub-sources, each filling its own
slice and degrading independently if an endpoint is down:

| sub-source | share of slice | what it teaches | licence |
|---|---:|---|---|
| Linux kernel `Documentation/` | 22% | Memory management, scheduling, filesystems, the driver model, locking, namespaces, seccomp — the OS internals, from the kernel's own tree | GPLv2 |
| man pages (local) | 18% | Syscalls (§2), libc (§3), admin tools (§8) — the OS as its own manual | mixed, redistributable |
| RFCs | 25% | How the wires work: TCP/IP, TLS, HTTP, DNS, BGP, from the RFC Editor | public domain |
| MITRE CWE | 5% | Weakness *classes* — overflow, use-after-free, injection, races — described structurally | free w/ attribution |
| MITRE ATT&CK | 5% | Adversary tradecraft: how intrusions proceed, tactic by tactic | free w/ attribution |
| NVD CVE descriptions | 25% | Concrete vulnerabilities, described by the people who catalogue them | public domain |

Only the kernel-docs path is heavy: it streams one 6.6 release tarball (~140 MB)
and reads `Documentation/*.rst|.txt` straight out of the decompression stream,
never writing the source tree to disk. Security content is descriptions and
classifications only — CWE/ATT&CK/CVE catalogue *what* goes wrong and *how*
intrusions are structured; **no exploit code is fetched**. That is the right
shape for the uncensored build's goal anyway: a model that understands security
deeply, taught from the standard reference catalogues rather than from payloads.

Restricting code to eight languages is deliberate. All 300 languages of The Stack
would spend a small model's capacity on syntax it will never be asked to produce.

## SFT mix — 600k conversations

| source | share | covers |
|---|---:|---|
| `smol-smoltalk` | 30% | General chat, already filtered for small models |
| `OpenHermes-2.5` | 20% | Broad instruction following and explanation |
| `Magicoder-OSS-Instruct` | 15% | Code from real repository seeds |
| `Magicoder-Evol-Instruct` | 10% | Harder, evolved coding problems |
| `MetaMathQA` | 15% | Math with worked steps |
| `OpenThoughts-114k` | 10% | Short-form reasoning traces |

Conversations over ~6,000 characters are dropped at download rather than at
training time, so the shares stay honest — the trainer would otherwise discard
them silently and the realised mix would not be the one in this table.

Long chain-of-thought is deliberately excluded. At 250M the model cannot hold a
2,000-token trace, and training on them teaches it to imitate the *shape* of
reasoning it cannot perform — the same failure the 120M recipe avoided by
choosing `smol-smoltalk` over full-size chat sets.

## The two builds

Both share the base model and the whole SFT stage. They differ by one stage at
the end, not by two datasets — which means any measured difference between them
is attributable to that stage alone.

```
pretrain ──► SFT ──┬──► kestrel-250m            (unguarded; internal + partners)
                   └──► kestrel-250m-guarded    (+ align-safety)
```

**Unguarded** is what `train_kestrel.sh` produces. No alignment or refusal data
enters it. Public instruction sets carry refusals emitted by whatever model
generated them, so the download step filters them out — left in, they would teach
refusal behaviour to *both* builds and there would be no controlled difference to
measure. `--keep_refusals` disables that filter.

**Guarded** adds a reviewed refusal/benign/uncertainty set on top:

```bash
cp -r runs/kestrel-250m runs/kestrel-250m-guarded
.venv/bin/python -m geocentric.cli align-safety \
    --model_dir runs/kestrel-250m \
    --output_dir runs/kestrel-250m-guarded \
    --data_path data/kestrel/alignment.jsonl --yes
```

`align-safety` requires all three categories present (`refusal`, `benign`,
`uncertainty`); a set of refusals alone is rejected, because a model trained only
to refuse learns to refuse everything. Format is in
`examples/alignment/format-example.jsonl`. That dataset is not built here — you
said you would supply it.

Then compare them on held-out behaviour rather than by impression:

```bash
.venv/bin/python -m geocentric.cli check-behavior \
    --model_dir runs/kestrel-250m \
    --compare_dir runs/kestrel-250m-guarded \
    --data_path data/kestrel/heldout.jsonl --output runs/guardrail-delta.md
```

## Watching it from somewhere else

Nine days is too long to sit at the desk, so the run broadcasts itself.

```bash
.venv/bin/python scripts/watch_training.py runs/kestrel-250m
```

That is the terminal view. It now also prints a LAN URL and starts the web
monitor if nothing is already serving it, so the same numbers are readable from
a phone on the same network. `train_kestrel.sh` starts the monitor itself, so
the page is up from the moment pretraining begins.

**The page cannot slow the run down, by construction:**

- stdlib only — torch is never imported, so the process never creates a CUDA
  context and never reserves a byte of VRAM. Verified: zero `torch`/`libcuda`
  entries in its memory maps, 27 MB resident.
- `os.nice(19)`, the lowest priority the scheduler offers. On a contended box the
  trainer gets the core first, every time.
- Its only input is `training_metrics.json`, which the trainer already writes.
  It never opens a checkpoint, the corpus, or the GPU.
- Reads are cached per `--poll` window, so twenty phone tabs refreshing cost one
  small file read between them rather than twenty.

It stops when training stops. A terminal status alone is not treated as the end —
the trainer writes `stopped` before its final checkpoint save returns, and a
resumed run rewrites the file moments later — so the monitor requires both a
terminal status *and* no live trainer process, then lingers 15 minutes so the
final numbers are still there to look at. `train_kestrel.sh` also traps EXIT/INT/
TERM, so Ctrl+C on the run takes the page down with it.

Ports and hosts: `MONITOR_PORT=9000 ./scripts/train_kestrel.sh` moves it;
`--host 127.0.0.1` restricts it to this machine. The default binds the LAN, and
anyone who can reach the port can read training status — no checkpoints, no data,
no control, but be aware it is unauthenticated.

## What this will actually cost, and actually score

**Time.** 5B tokens at roughly 6,000 tok/s — half the 120M's measured 12,900,
since the model is twice the size — is about **9–10 days of continuous training**,
plus ~1 hour for the tokenizer, several hours to tokenize the corpus, and ~1 day
for two SFT epochs.

**Disk.** ~20 GB of raw text, ~9 GB of token shards, ~3 GB per checkpoint. You
have 122 GB free; this fits with room, but not with a second corpus beside it.

**Benchmarks.** They will not be flawless, and it is worth being exact about why
rather than finding out in nine days. The 120M scored **39.1/100** on PARALLAX
after full training. A 250M on 5B well-chosen tokens should land somewhere around
**50–60**, with the largest gains in SEXTANT (knowledge) and ZENITH (bits/byte),
because those are the probes that respond to parameters and data quality. What
will *not* reach a high score:

- **MERIDIAN** (does more context help) was 14.1/100 at 120M. Long-range
  conditioning is the last thing to appear as models scale, and 1,024 tokens of
  context is not much room to demonstrate it.
- **ASTROLABE** (instruction following) was 41.7. Doubling parameters moves this,
  but reliable multi-constraint instruction following is a several-billion-
  parameter behaviour.
- Code that compiles and runs, and correct multi-step arithmetic, are both
  substantially above this weight class. Expect plausible-looking code with real
  bugs. That is not a data problem or a hyperparameter problem, and no flag in
  this repo fixes it — it is what 250M parameters buys.

This model will be genuinely useful as a fast local technical autocomplete and
explainer, and it will be the best model this stack has produced. It will not be
competitive with anything you can call over an API. Both of those are true at
once, and the second one is not a reason to skip the first.

## The one knob worth testing first

MNEME (`--mneme`) reweights learning toward rare targets. On a synthetic
fact-recall probe it raised recall from 68.4% to 88.4% — directly the "deep
knowledge" objective — but it worsened grammar NLL in the same control
(0.00214 → 0.00439), was measured on tiny CPU models rather than a real
checkpoint, and has never run on this card.

That is a promising result on an unproven path, and nine days is too long to bet
on one. Test it over a few hundred steps before committing:

```bash
.venv/bin/python scripts/epicycle_bench.py \
    --data data/kestrel/pretrain/finemath.txt --steps 400 \
    --arms baseline capacity --output runs/kestrel-armcheck
MNEME=--mneme ./scripts/train_kestrel.sh    # if it holds up
```
