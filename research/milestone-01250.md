# Milestone 1,250 — Saturday 05 September 2026, 02:53

Step **1,260 of 10,588** (11.9%). 314,818,560 of 2,645,628,484 tokens seen (11.9% of one pass).

Wall clock since this log began: 1:52:09.

## Training state

| Metric | Value | Change since last milestone |
|---|---|---|
| Training loss | 3.4753 |  (-0.072877, improved) |
| Training perplexity | 32.3 | |
| Eval loss | 3.8192 |  (-0.092299, improved) |
| Learning rate | 5.84e-04 | |
| Throughput | 17,685 tok/s | |
| MFU | 49.3% | |
| Peak VRAM | 3.15 GB | |
| GPU | 99% util, 3,802 MiB, 86°C, 123 W | |

## Held-out evaluation

Measured on text the model has never trained on, on CPU, without interrupting training.

| Metric | Value | Change |
|---|---|---|
| Nats per token | 3.4995 |  (-0.101383, improved) |
| Perplexity | 33.1 |  (-3.5, improved) |
| Bits per byte | 1.0013 |  (-0.029007, improved) |

**Code vs prose:** code 2.201 nats, prose 2.974 nats — code is easier by 0.773. Code is lower-entropy text, so the model models it more cheaply; watching this gap narrow or widen shows whether prose modelling is catching up.

## Generations

Same ten prompts, same seed (1234), temperature 0.7 at every milestone, so any difference is the model rather than sampling noise.

### `factual`

**Prompt:** `The capital of France is`

```
the North Sea, with the sea itself, a region known as South-East Asia (also referred to as South-East Asia), and the North Atlantic is home to more than 2 million people.
Bouillean has been an important part of the country’s economy since the early 20th century; it was a significant contribution to the population’s growth and development. It is also
```

<details><summary>previous milestone</summary>

```
the first major city in the country.
What is the history of the Roman Empire?
The Roman Empire was a cultural settlement, and it was an important place for many cultures to come to understand and appreciate. The Romans built their schools, churches, and libraries, while the Romans used their own languages, languages, and ways of life to communicate, communicate, and share with one another. They
```
</details>

### `science`

**Prompt:** `Photosynthesis is the process by which`

```
small particles of material can be dissolved. This method uses a chemical reaction to form a liquid, and then, in some cases, using a single solvent. The solution is used as an electrolyte (a substance that can be easily converted to a gas) and is also sometimes used as a catalyst for building new products.
The simplest method of building materials is the metal, and it has been demonstrated to
```

<details><summary>previous milestone</summary>

```
carbon atoms are produced.
- Acidic acids
Chemical compounds are chemical compounds that have a number of advantages and disadvantages as they arise from natural chemistry. They are used in various forms, including liquids, oils, plastics and other non-organic materials such as rubber, wood, plastic, etc. These substances include metals like lead, silica, glass, metal, and many other non-chem
```
</details>

### `definition`

**Prompt:** `A large language model is`

```
an interactive map of the world.
It is a free online encyclopedia that allows users to search and browse through the web, including images and videos.
This website is used for research and learning purposes.
The site also provides information about the world's culture, history, religion, politics, culture, history, religions, geography, history and history. This site also includes a vast collection of resources
```

<details><summary>previous milestone</summary>

```
quite similar to the one we want.
What is a language model?
Language model is a powerful tool used in education to help students learn more effectively and easily. The language model has been developed by psychologist Sweeny Duchamp from India, where it is now known as the "language model," based on a study conducted by Cengage Medical School, who studied a variety of learning
```
</details>

### `narrative`

**Prompt:** `The old lighthouse keeper walked down to the shore and`

```
moved down to the ground and sat down for a while until he finally reached his destination. He had been told to walk, and then climbed up to the hill, looking for the elusive fuss of a nearby human being.
He was greeted by an angry crowd of people who looked around him before he could see what happened. The man walked up at his eyes, and his eyes were dark.
```

<details><summary>previous milestone</summary>

```
moved down to the ground and began to find a hidden gem on the beach.

One day, while searching for a hidden gem on the beach, Max noticed something strange happening nearby. He saw a group of people walking through his cave during a long trip. They didn't know what to do, but they needed to be able to understand it better than before!

From then on,
```
</details>

### `instructional`

**Prompt:** `To bake bread, first`

```
lift your hands and turn them into a ball.
Make sure to place the dough in a proper shape, like a table or table. Place each piece on top of the dough so that it is at least one inch thick.
Place the dough inside the oven. Place the pieces on top of the bread with the following ingredients: 1 tablespoon baking powder; 2 tablespoons baking powder; 3 tablespoons baking
```

<details><summary>previous milestone</summary>

```
of all, we can enjoy the delectable dishes.

### Ingredients:

* 1 large onion, thinly sliced, diced, minced garlic (optional)
* 2 large onions, minced
* 1 teaspoonful ginger, minced garlic, minced garlic, minced garlic, minced garlic, and olive oil
* 3 large onion, chopped and stir-fry, chopped parsley, minced garlic,
```
</details>

### `list`

**Prompt:** `Three reasons to exercise regularly:`

```
- Keep a cool and comfortable temperature for at least 30 minutes before going out.
- Avoid placing your bed in an airtight container such as a bathroom or fridge.
- Limit your outdoor activities as much as possible so you can enjoy them anywhere.
- Never leave your room in an open, closed space, or outside. Make sure there is enough space for you to sit on.
```

<details><summary>previous milestone</summary>

```
- Keep the heart and lungs healthy for a healthy heart
- Reduce the amount of time you spend in bed
- Lower your energy levels
- Boost your brain
- Improve your mood
- Avoid high cholesterol
- Get enough sleep
- Drink plenty of water – drink lots of water, drink a warm bath (don’t have to spend too much on that), and get enough
```
</details>

### `code`

**Prompt:** ````python
def fibonacci(n):`

```
n = 10.5

def fibonacci(n):
    n, n = 100, n = 10000, n = 100000, n = 10000, n = 30000, n = 20000, n = 5000000, n = 10000000, n = 5000, n = 25000, n = 10000, n = 100000, n =
```

<details><summary>previous milestone</summary>

```
return 1, len = 1.0.sum()
```

### Linear Regression and Normalization

Linear regression is a complex plot that is used to train the model. In this unit, we will learn how to visualize the results of our linear regression models using Python. We will also use a simple linear regression model to understand how to predict the outcomes of our linear regression models.
```
</details>

### `dialogue`

**Prompt:** `"Where are you going?" she asked. He`

```
chuckled softly and said, "So we want to go on a trip to the United States to travel around the world!"

Maria smiled warmly at her friends, asking, "But why do you take so many miles when traveling?" Her voice replied, "Well, sometimes people want to see beautiful places without even being able to travel anywhere." Her gaze lingered on her ears and replied, "I
```

<details><summary>previous milestone</summary>

```
chuckled, so he responded, "Well, if I can tell you something, maybe we could come up with an answer to our questions."

Just then, the young lady thought, "That's true! You see, my dear friends. Let me share a secret – no one gets lost in their quest for knowledge and guidance whenever they want us."

With excitement, they started learning
```
</details>

### `arithmetic`

**Prompt:** `If a train leaves at 3pm and travels for two hours, it arrives at`

```
4:45 p.m. on the first day of the week. After that, a train will travel to the main train station at 1:15 p.m. on the second day of the week. The trains will arrive at 3pm, with the third day being at 11:30 p.m. on the second day of the week. The next day will arrive at 9:
```

<details><summary>previous milestone</summary>

```
120-108 p.m., which is a very important day in the life of your vehicle and you’ll be able to receive all the necessary care and entertainment out of sight, and your car will also know what kind of training you need.
The most interesting thing about the train is that it’s better to be able to ride at night than riding on a train. It’s
```
</details>

### `conversational`

**Prompt:** `Hello! What can you help me with?`

```
- 1. Educate yourself
- 2. Learn new skills
- 3. Take a break
- 4. Write down
- 5. Practice regularly
- 6. Seek help from an expert
- 7. Read
- 8. Share your thoughts
- 9. Get to know your own feelings
- 11. Get back at home
- 13. Be patient, respectful,
```

<details><summary>previous milestone</summary>

```
- 1. What do you think the author will use?
- 2. How do you think the author thinks about the content of his writing?
- 3. Why do you think this article is important for the reader to know what they are saying?
- 5. What does the writer ask him about this topic?
- 6. How would you feel about this page?
```
</details>

## Notes

- Benchmarks run on CPU from the latest checkpoint. Training was not paused, signalled or slowed on the GPU at any point.
- Model: 120,787,072 parameters, context 1024.
- This is a base model: it continues text, it does not answer questions. Instruction following arrives with fine-tuning, after pretraining completes.