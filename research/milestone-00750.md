# Milestone 750 — Saturday 05 September 2026, 01:02

Step **780 of 10,588** (7.4%). 194,887,680 of 2,645,628,484 tokens seen (7.4% of one pass).

Wall clock since this log began: 0:00:43.

## Training state

| Metric | Value | Change since last milestone |
|---|---|---|
| Training loss | 3.7300 | |
| Training perplexity | 41.7 | |
| Eval loss | 4.0434 | |
| Learning rate | 5.95e-04 | |
| Throughput | 16,141 tok/s | |
| MFU | 45.0% | |
| Peak VRAM | 3.15 GB | |
| GPU | 99% util, 3,800 MiB, 85°C, 113 W | |

## Held-out evaluation

Measured on text the model has never trained on, on CPU, without interrupting training.

| Metric | Value | Change |
|---|---|---|
| Nats per token | 3.7270 | |
| Perplexity | 41.6 | |
| Bits per byte | 1.0663 | |

**Code vs prose:** code 2.467 nats, prose 3.237 nats — code is easier by 0.769. Code is lower-entropy text, so the model models it more cheaply; watching this gap narrow or widen shows whether prose modelling is catching up.

## Generations

Same ten prompts, same seed (1234), temperature 0.7 at every milestone, so any difference is the model rather than sampling noise.

### `factual`

**Prompt:** `The capital of France is`

```
also not a government.
- "Military" means the military to be held in a state of war with the British and American colonies, but it is also considered a political entity and has an official language. The term "same" means "to have a political or political party."
- "Italian" means "to have a military" or "to be a military". It
```

### `science`

**Prompt:** `Photosynthesis is the process by which`

```
an atom moves from one another to another, in order to produce a new form of energy.
It is also known as the "internal" process because it involves combining the energy and other elements within an atom into a single atom.
In addition, it is not surprising that the molecule can be absorbed by a material (or any substance) at its ends, even though it is constantly being
```

### `definition`

**Prompt:** `A large language model is`

```
an interactive platform for the project to be used in the development of a new kind.
The programme will be open and open to all users, from the user, from the location of the school library at the local library. The training will then be conducted with the help of the instructor/teacher staff, who will create a project plan for this workshop. This will also allow you to make your own
```

### `narrative`

**Prompt:** `The old lighthouse keeper walked down to the shore and`

```
moved down into the ground.
In the first part of the nineteenth century, the first man had been captured by the British in 1789, but it is said that the French ship was made in a land called the Lighthouse. It is also believed that it was not known from the Dutch East, but from the time of the 16th century, there were no settlers here.
In the
```

### `instructional`

**Prompt:** `To bake bread, first`

```
, we need to turn the ball into a clean jar.
We must keep in mind that the air is an area of warm weather and may have an impact on the temperature. If you are concerned about the temperature, the snow should be wet. If you have a cold winter winter season, you can also use the snow as an indicator for the temperature. The snow should be dry, so it
```

### `list`

**Prompt:** `Three reasons to exercise regularly:`

```
- Increased concentration.
- Reduced caffeine levels
- More severe physical activity
- Impaired muscle tone, and increasing body fat intake
- Higher risk of cardiovascular disease
- Decreased blood pressure
- Increased risk for heart diseases
In addition, it is important that you do not have to take a course before exercising. If you have a high-intensity workout and are not physically active
```

### `code`

**Prompt:** ````python
def fibonacci(n):`

```
return 1.0 * 1.0 * 0.1 * 2.0 / 1.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 / 1.0 * 1.0 * 1.0 * 1.
```

### `dialogue`

**Prompt:** `"Where are you going?" she asked. He`

```
explained that there's a special place where the author will share information and ideas, rather than just a fancy name like "The Book of Stakes" or "A Story of Stakes."
There's also one important part of the story of Stake, also known as "The Book of Stakes," a short-lived person who lived in New England in 1790 and later was a descend
```

### `arithmetic`

**Prompt:** `If a train leaves at 3pm and travels for two hours, it arrives at`

```
the same time.
What is the difference between speed and power?
A speed point is a constant distance to be reached by using speed and force. It is also used in different ways (e.g., through speed and velocity) to establish an altitude.
What are the difference between speed and force?
The difference between speed and power is that the rate of change is always determined by
```

### `conversational`

**Prompt:** `Hello! What can you help me with?`

```
- 1. What do you think is a good way to communicate with each other and to communicate with others?
- 2. How do you feel when you are in this situation?
There are many ways we can help you to communicate with each other. Let’s explore some ways you can use these strategies to make your own communication easier.
- 3. Write a few sentences.
```

## Notes

- Benchmarks run on CPU from the latest checkpoint. Training was not paused, signalled or slowed on the GPU at any point.
- Model: 120,787,072 parameters, context 1024.
- This is a base model: it continues text, it does not answer questions. Instruction following arrives with fine-tuning, after pretraining completes.