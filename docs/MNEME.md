# MNEME: knowledge balance, grounded learning and explicit abstention

MNEME is an experimental EPICYCLE extension. PARALLAX's existing **ASTROLABE**
benchmark is unchanged. This system has three separate parts: a pretraining loss,
an optional evidence/abstention SFT curriculum, and a conservative evidence lookup.
None is a proof of general intelligence or hallucination-free language generation.

## 1. Knowledge-balanced pretraining

Common punctuation, templates and frequently repeated targets can dominate a
small model's learning. MNEME measures target-token frequencies within each
microbatch, then gives less frequent targets a bounded increase in learning weight.
Rarity is a proxy for neglected information, **not a test of factual correctness**.
Rare corrupt text can receive additional weight too.

For each valid target token i, let c_i be its token-type count and c_max the largest
count in the microbatch. Ignored targets (-100) do not participate.

```
w_i = min(4, sqrt(c_max / c_i))
balanced = sum(w_i * CE_i) / sum(w_i)
MNEME = 0.5 * ordinary_mean_CE + 0.5 * balanced
```

This bounds the raw rarity-weight ratio at 4:1 and retains half of ordinary CE.
Uniform counts reduce exactly to ordinary CE. All-masked batches produce a
differentiable zero. There is no running teacher, replay corpus, additional model
parameter, hidden optimizer or added inference computation. Frequency storage is
one FP32 vector per vocabulary plus token-sized work buffers: about 125 KiB for a
32,000-token vocabulary, excluding temporary token arrays and the existing CE head.

The watcher reports ordinary token CE, not the reweighted objective. Checkpoint
weight shapes and tokenizer IDs are unchanged. The option is saved with EPICYCLE
configuration; keep the same training flags when resuming. Existing models can
continue training, but enabling a different objective intentionally changes future
updates. The named preset selects chunked vocabulary loss by default to bound
intermediate memory; actual RTX 2060 memory and runtime have not been measured here.

Use `--epicycle knowledge` for MNEME without depth/context/selective scheduling.
To add it to an existing EPICYCLE command, append `--mneme`. It remains opt-in.

## 2. Cooperation with EQUANT

For example, keep the rest of your existing pretraining command and use:

```text
--epicycle selective --mneme
```

Before EQUANT activates, MNEME uses ordinary CE for its first half. Afterwards:

- Rank tokens by **unweighted raw loss**, using one shared sort.
- Exclude EQUANT's highest-loss outlier trim from **both** loss terms.
- Use EQUANT's retained hard band for the first half.
- Use rarity-weighted CE over all remaining valid tokens for the second half.

This preserves the outlier trim while giving easier, underrepresented targets
outside the selected band a learning signal. The gradients outside that band are
intentional. Consequently, EQUANT's sparse-backward replay shortcut is disabled
when MNEME is active, with an explicit startup message. Ordinary chunked backward
remains available. Do not expect the full sparse-replay speed benefit at the same
time as gradients for almost every token. EQUANT alone retains its original loss
and selection behavior.

## Controlled knowledge results

The primary synthetic experiment uses 64 arbitrary entity→value facts, with eight
entities sampled 20 times as often as the other 56. Each arm has identical initial
weights, architecture, sampled data, optimizer settings and update budget. Five
confirmatory seeds (101, 202, 303, 404, 505) are distinct from the three pilot seeds.
The combination study reuses the confirmatory seeds; it is not another independent
replication. These are small CPU models, not the user's trained 120M checkpoint.

At 400 updates, averaged over the five seeds:

| Objective | All-fact recall | Rare-fact recall | Recall with new prefix |
|---|---:|---:|---:|
| Ordinary CE | 27.50% | 18.21% | 28.12% |
| MNEME | 48.75% | 41.79% | 49.06% |
| EQUANT | 35.31% | 26.07% | 35.31% |
| EQUANT + MNEME | 49.06% | 41.79% | 49.06% |

MNEME's all-fact gain is **21.25 percentage points** over ordinary CE. The combination
gains **13.75 points** over EQUANT. Each comparison regressed on one of five seeds.
These are learned facts presented in familiar or recombined contexts, not unseen
world knowledge, free-form reasoning, or a standard language benchmark.

After 160 further updates seeing only common facts, recall falls to 19.69% for
ordinary CE versus 29.06% for MNEME, and 24.06% for EQUANT versus 27.19% for the
combination. **Forgetting still occurs**; MNEME is not a complete retention solution.
The joint retention result is mixed across seeds.

An equally sampled control reaches 68.44% recall for CE, 88.44% for MNEME, 68.75%
for EQUANT and 83.44% for the combination. MNEME alone worsens the small synthetic
grammar NLL in that control (0.00214 → 0.00439). This is a tradeoff to track on a
real held-out corpus, not evidence that language quality is unchanged.

The final four-arm CPU experiment spends approximately 0.925 seconds per baseline
arm versus 0.945 for MNEME across 400 tiny updates (~2.1% more time); EQUANT spends
0.937 versus 0.960 for the combination (~2.5% more time). Such short CPU timings
are noisy and exclude real data loading/checkpoint saving. They do not predict
RTX 2060 speed, especially with a large vocabulary and sparse replay disabled.

Raw outcomes, including pilot results and regressions, are in
`research/benchmarks/mneme-{pilot,confirmatory,equant,balanced-control}.json`.

```bash
PYTHONPATH=. .venv/bin/python scripts/mneme_bench.py --steps 400 --retention_steps 160 --seeds 101 202 303 404 505 --compare_equant --output runs/mneme-study.json
```

## 3. Grounded SFT: evidence must support the answer

Token likelihood is not a truth detector. This separate curriculum teaches the
model to answer from evidence or explicitly abstain. It does not rely on the
pretraining rarity loss to decide whether a claim is true.

Input is reviewed JSON/JSONL. Supported examples require the answer to appear
verbatim in the context. This checks annotation consistency, not source truth or
whether the passage actually answers the question. Explicit unsupported examples
can contain irrelevant, incomplete or contradictory context:

```json
{"question":"What is the capital of France?","context":"Paris is the capital of France.","answer":"Paris"}
{"question":"When was the bridge built?","context":"The bridge crosses the river.","answerable":false}
{"question":"What is the recorded value?","context":"One entry gives 12. A conflicting entry gives 19.","answerable":false}
```

For each supported record, the builder emits a supported answer and a matched
question with its evidence withheld. Explicit `answerable:false` records emit one
abstention example and must omit `answer`. System instructions require abstention
for missing, irrelevant, insufficient or contradictory evidence. Output publication
is atomic and never overwrites an existing dataset. Long examples remain subject
to the normal SFT context limit and overlong-example handling.

```bash
.venv/bin/python -m geocentric.cli mneme-ground --data data/reviewed-grounding.jsonl --output data/mneme-sft.jsonl
.venv/bin/python -m geocentric.cli sft --model_dir runs/geocentric-120m --output_dir runs/geocentric-grounded --sft_data_path data/mneme-sft.jsonl
.venv/bin/python -m geocentric.cli check-grounding --model_dir runs/geocentric-grounded --data data/heldout-grounding.jsonl --output runs/grounding-evaluation.json
```

Use a **new output directory** to retain the original model and avoid changing the
dataset geometry of a resumed SFT run. The three rows above illustrate the schema;
they are not a sufficient training dataset. Use diverse reviewed examples and
separate held-out entities, documents, questions and contradiction types. The
evaluator does not automatically prove absence of train/test overlap.

Grounded training and evaluation use an explicit evidence/question format and
system instruction. This does not automatically change ordinary web-chat behavior.
Review the raw responses as well as the exact-match scores: valid paraphrases can
fail exact match. Measure supported accuracy, unsupported abstention and over-refusal
together. A model that refuses everything has not solved factual reliability.

## Hallucination experiments and a rejected curriculum

In the simple five-seed tokenized evidence-present/missing experiment, both arms
answer supported probes correctly. The evidence-withholding arm abstains on every
missing-evidence probe, while the answer-only arm invents an answer token on every
one. Each condition uses 256 probes on held-out question entities with a shared
answer vocabulary. This validates the **simple curriculum principle**, not the
natural-language builder or unrestricted model behavior.

The harder experiment adds irrelevant and contradictory facts. A 75%-negative
curriculum with a 2-layer, width-64 model **fails**: on unseen entities it refuses
98.55% of supported cases, with only 0.90% supported accuracy. Its high unsupported
abstention is not a useful reliability improvement. The answer-only baseline also
struggles at 49.02% supported accuracy. The failed run is retained in
`research/benchmarks/mneme-hard-grounding.json`; do not cite its refusal rate as a
hallucination cure or adopt that data mixture as a recommended default.

A follow-up uses 50% supported cases, a 4-layer width-128 model, 1,500 updates and
three seeds. On unseen entities its supported accuracy reaches only **20.25%**,
with **61.39% over-refusal**, while the answer-only baseline reaches 49.09%
supported accuracy. Abstention is 100% for missing evidence, 63.87% for irrelevant
evidence and 62.83% for conflicts. This is **also not a successful deployment
result**. Capacity, budget and mixture changed together relative to the first
experiment, so their individual effects cannot be separated. All outcomes are in
`research/benchmarks/mneme-hard-balanced.json`.

These failures are a release boundary: the grounding curriculum is experimental,
not enabled automatically or recommended as a proven cure. The evidence lookup
below offers a narrow deterministic alternative while learned reliability remains
unresolved. Do not promote the perfect simple-test scores without these harder
results beside them.

The builder can emit many negative examples when explicitly unanswerable records
are added; inspect the resulting balance. There is no universally safe ratio.

```bash
PYTHONPATH=. .venv/bin/python scripts/mneme_grounding_bench.py --output runs/mneme-simple.json
PYTHONPATH=. .venv/bin/python scripts/mneme_hard_grounding_bench.py --output runs/mneme-hard.json
PYTHONPATH=. .venv/bin/python scripts/mneme_hard_grounding_bench.py --balanced --steps 1500 --seeds 101 202 303 --output runs/mneme-hard-balanced.json
```

## 4. Strict evidence lookup for bounded use cases

```bash
.venv/bin/python -m geocentric.cli mneme-answer --data data/reviewed-grounding.jsonl --question "What is the capital of France?"
```

This explicitly **does not generate with the model**. It matches a normalized exact
question against a vetted registry, returns the annotated answer with its context,
or abstains when there is no match, an explicitly unanswerable entry or conflicting
answers. It will not improvise a paraphrase, fill missing information or append
claims. The registry is limited to 16 MiB. This is useful for narrow audited facts,
but it is not general semantic retrieval: reworded questions may abstain, erroneous
source annotations remain erroneous, and duplicate questions with different scopes
can cause conservative abstention. It is separate from the web chat and is not a
claimed improvement in the model's internal intelligence.

## Prior work and boundaries

MNEME is a new integration in this repository, not a claim to have invented
frequency weighting or learned abstention. Class reweighting has established prior
work, including [Cui et al., Class-Balanced Loss (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html).
MNEME uses a different bounded microbatch-frequency mixture, not that paper's
effective-number formula. Unanswerable-context training also predates this project;
see [Rajpurkar et al., SQuAD 2.0 (ACL 2018)](https://aclanthology.org/P18-2124/).
No results from those papers are attributed to Geocentric.

No existing user checkpoint was retrained or modified. CPU/MPS tests cover loss
and gradient behavior, EQUANT trim/selection, chunked tied-head execution, actual
pretraining resume, curriculum-to-SFT integration and evaluation. CUDA tests are
included but skipped when unavailable. Before adopting broadly, evaluate real
held-out language, factual recall, contradictory evidence, abstention calibration,
over-refusal and throughput on the intended hardware. Unrestricted hallucination
elimination and a large general-intelligence gain remain unproven.
