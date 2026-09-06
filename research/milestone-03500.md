# Milestone 3,500 — Saturday 05 September 2026, 13:06

Step **3,500 of 10,588** (33.1%). 874,496,000 of 2,645,628,484 tokens seen (33.1% of one pass).

Wall clock since this log began: 1:24:18.

## Training state

| Metric | Value | Change since last milestone |
|---|---|---|
| Training loss | 3.1768 |  (-0.068521, improved) |
| Training perplexity | 24.0 | |
| Eval loss | 3.4850 |  (-0.040515, improved) |
| Learning rate | 4.72e-04 | |
| Throughput | 18,672 tok/s | |
| MFU | 52.0% | |
| Peak VRAM | 3.15 GB | |
| GPU | 99% util, 3,896 MiB, 85°C, 139 W | |

## Held-out evaluation

Measured on text the model has never trained on, on CPU, without interrupting training.

| Metric | Value | Change |
|---|---|---|
| Nats per token | 3.2955 |  (+0.037952, worse) |
| Perplexity | 27.0 |  (+1.0, worse) |
| Bits per byte | 0.9458 |  (+0.013762, worse) |

**Code vs prose:** code 1.829 nats, prose 2.570 nats — code is easier by 0.740. Code is lower-entropy text, so the model models it more cheaply; watching this gap narrow or widen shows whether prose modelling is catching up.

## Factual probes

Probability the model assigns to the correct next token, which is far more sensitive than sampling: a fact can be largely learned yet appear only occasionally in generated text.

| Prompt | Answer | P(answer) | Rank | Top-1 | Best distractor |
|---|---|---|---|---|---|
| `The capital of France is` | ` Paris` | 1.31% | 7 | ` the` | ` London` 0.39% |
| `The capital of Japan is` | ` Tokyo` | 8.64% | 2 | ` the` | ` Se` 0.32% |
| `The largest planet in the solar system is` | ` Jupiter` | 0.72% | 17 | ` the` | ` Earth` 2.53% |
| `Water is made of hydrogen and` | ` oxygen` | 45.66% | 1 | ` oxygen` | ` carbon` 5.58% |
| `The chemical symbol for gold is` | ` Au` | 0.75% | 19 | ` gold` | ` Ag` 0.07% |
| `Shakespeare wrote a play called Romeo and` | ` Juliet` | 98.43% | 1 | ` Juliet` | ` Caesar` 0.00% |

## Generations

Same ten prompts, same seed (1234), temperature 0.7 at every milestone, so any difference is the model rather than sampling noise.

### `factual`

**Prompt:** `The capital of France is`

```
the town of St. Denis.
It is a narrow and open, with a round tower (called an "moor") on the top. It is surrounded by a river and a forest. The city has a small city, the Sardinia, the largest city in the world. A popular amusement park, the famous "Rosetta" was once used as a shopping center.
```

<details><summary>previous milestone</summary>

```
usually not so good.
So we need a little bit more money to keep up with the increasing number of people in France and, if you have lots of money, you can do it at least once a year. But just because you are rich doesn’t mean that you don’t need to spend most of your time doing everything you want, like doing laundry, eating, etc. And
```
</details>

### `science`

**Prompt:** `Photosynthesis is the process by which`

```
plants produce new tissue. These cells then undergo a process known as growth, which involves several stages and can be divided into two main types:
- Growth is determined by the division of cells (a cell that produces new cells) by the production of new cells. This process can occur due to many different factors such as injury or disease, ageing, infection, or even exposure to toxic substances.
```

<details><summary>previous milestone</summary>

```
plants produce flowers. There are different types of photosynthesis, some chemical plants use sunlight to create food and others use carbohydrates for energy production. The different types of photosynthesis are:
Cultivation (cultivation) happens when plants have grown into a group of organisms. Most plants require the right amount of sunlight to survive, but some plants require it at certain times of day and again in the morning
```
</details>

### `definition`

**Prompt:** `A large language model is`

```
essential for effective communication.
- A language model will provide information about the child's environment, and will help them understand their behavior. The model will also help them recognize patterns and develop language skills.
- Language models are designed to capture the child's needs and help them understand their emotions and experiences. These models will be used in a variety of contexts, including:
- Language learning
-
```

<details><summary>previous milestone</summary>

```
essential for effective communication.
- Consider how the child will learn to use the language in a variety of ways to express his or her needs and concerns. This will help you decide whether it is appropriate to seek out additional resources to address your child’s needs.
- If you do not have any ongoing training, consider seeking professional guidance from a speech-language pathologist at a local pediatrician
```
</details>

### `narrative`

**Prompt:** `The old lighthouse keeper walked down to the shore and`

```
found a hole in the roof.
It was not until two years later that the keeper's family moved to the Lighthouse, which had been waiting for them for more than a year. They were on one of the many other vessels from the area, and the keeper said he could not find it. The keeper was surprised and asked if there was anything else that could be seen.
```

<details><summary>previous milestone</summary>

```
saw a large boulders.
He had been looking for an old lighthouse keeper and had been searching for a night skywalk on his boat. He went in and found a small fire at the bottom of the tower. It was a big, white fire that seemed to be burning across the tower from the tower. This fire was burning at the bottom of the tower. He saw a
```
</details>

### `instructional`

**Prompt:** `To bake bread, first`

```
remove the pastry and roll it into a ball.
2. Preheat oven to 375°F (190°C). Turn on the stove to 400°F (175°C) for medium-rare results. This step ensures even cooking.
3. Transfer the dough onto a baking sheet lined with parchment paper. Make sure there's enough space between each piece so that air
```

<details><summary>previous milestone</summary>

```
sauté your chosen vegetables. Add the chopped potatoes and pour them into a bowl of water until well combined. Remove the pan from heat and set aside.
2. **Fill in the pan**: Dip one cup of butter or non-dairy milk in the same vegetable mixture. Add the remaining half of the chopped ham until fully coated. Allow it to soak for about 5 minutes.
```
</details>

### `list`

**Prompt:** `Three reasons to exercise regularly:`

```
- Helps improve flexibility and mobility.
- May reduce stress and anxiety.
- May help you cope with pain, fatigue, and other health problems.
- May assist in weight loss.
- May aid in healthy bowel movements.
- May improve the immune system.
- May reduce the risk of colon cancer.
- May reduce the risk of prostate cancer.
- May
```

<details><summary>previous milestone</summary>

```
- The first time you should be exercising.
- You will have a good chance of completing the task.
- It is difficult to keep your mind occupied during the actual exercise.
There are also other benefits to exercise:
- When you are exercising, you are not only improving your mental health but it is becoming more and more easier every day.
- You can improve your mood
```
</details>

### `code`

**Prompt:** ````python
def fibonacci(n):`

```
return 1 / len(self.__init__))
print(np.allclose(0, self.__init__))
```
This will print the results of our simulation: `x_pred`, `y_pred`, `d_pred`, `x_tol`. These values are then used to compute the number of features in the input data.

###
```

<details><summary>previous milestone</summary>

```
return [n.get_cuff() for n in zip(n=n)].split('/n-1').split('/n-1')
```
### Plotting Fibonacci Numbers with the Fibonacci Sequence

To plot a Fibonacci number, we will use the following steps:

1. We will define a function `to_c
```
</details>

### `dialogue`

**Prompt:** `"Where are you going?" she asked. He`

```
paused for a moment before answering, "It's because we all know that the best way to travel is through an airplane."

Next, they talked about how the airlines have evolved over time. The Airbus has undergone significant upgrades during its long-term flight, allowing it to run faster and more efficiently than ever before. However, managing such changes can be challenging and expensive, so engineers
```

<details><summary>previous milestone</summary>

```
said, "I'm not going to be here because I don't have any kids in my community."
The next day, she was walking around the campus with her young son, and he began to see the students coming to his house. When he got home, he explained that he had seen them pass by a large group of students who were coming to him as soon as they were about to
```
</details>

### `arithmetic`

**Prompt:** `If a train leaves at 3pm and travels for two hours, it arrives at`

```
4:00 p.m. If you're not sure how to get there, ask a travel counselor or call 911 to help.
"There is no need for an emergency room," said Dale Sweeney, president of the National Safety Council. "It's not safe."The Governing Body of the U.S. House of Representatives, on July 16, 2005,
```

<details><summary>previous milestone</summary>

```
4:55 p.m., the distance is approximately 30 miles per hour (about 7 miles) (the distance is about 1,000 feet).
A train is used to travel 5,000 feet on one day. If you are travelling over 10,000 miles of track, the length is usually around 20,000 miles. If you have been traveling 10,000 miles, the distance is
```
</details>

### `conversational`

**Prompt:** `Hello! What can you help me with?`

```
- 1. What do you think would be the best way to get a good education in English? (I did not know where to go)
- 2. The history of language learning:
- 3. The importance of language learning in everyday life.
- 4. I’m interested in languages and how they are used for our purposes.
- 5. How is it used
```

<details><summary>previous milestone</summary>

```
- Is it difficult to understand the concept of "the social world"?
- How do I know what to expect from my little one?
- Is there anything you can do to get them to do the job safely and comfortably?
- Does this person feel comfortable in their own home, or is they sitting up too much?
- Does there be any way you could move your child
```
</details>

## Notes

- Benchmarks run on CPU from the latest checkpoint. Training was not paused, signalled or slowed on the GPU at any point.
- Model: 120,787,072 parameters, context 1024.
- This is a base model: it continues text, it does not answer questions. Instruction following arrives with fine-tuning, after pretraining completes.