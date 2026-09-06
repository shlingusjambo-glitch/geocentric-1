# Milestone 4,000 — Saturday 05 September 2026, 15:04

Step **4,000 of 10,588** (37.8%). 999,424,000 of 2,645,628,484 tokens seen (37.8% of one pass).

Wall clock since this log began: 3:22:27.

## Training state

| Metric | Value | Change since last milestone |
|---|---|---|
| Training loss | 3.0614 |  (-0.115419, improved) |
| Training perplexity | 21.4 | |
| Eval loss | 3.4497 |  (-0.035291, improved) |
| Learning rate | 4.36e-04 | |
| Throughput | 17,676 tok/s | |
| MFU | 49.3% | |
| Peak VRAM | 3.15 GB | |
| GPU | 99% util, 3,872 MiB, 86°C, 120 W | |

## Held-out evaluation

Measured on text the model has never trained on, on CPU, without interrupting training.

| Metric | Value | Change |
|---|---|---|
| Nats per token | 3.2628 |  (-0.032689, improved) |
| Perplexity | 26.1 |  (-0.9, improved) |
| Bits per byte | 0.9364 |  (-0.009382, improved) |

**Code vs prose:** code 1.757 nats, prose 2.507 nats — code is easier by 0.750. Code is lower-entropy text, so the model models it more cheaply; watching this gap narrow or widen shows whether prose modelling is catching up.

## Factual probes

Probability the model assigns to the correct next token, which is far more sensitive than sampling: a fact can be largely learned yet appear only occasionally in generated text.

| Prompt | Answer | P(answer) | Rank | Top-1 | Best distractor |
|---|---|---|---|---|---|
| `The capital of France is` | ` Paris` | 6.96%  (+5.65 pts) | 2 | ` the` | ` London` 0.62% |
| `The capital of Japan is` | ` Tokyo` | 12.62%  (+3.97 pts) | 1 | ` Tokyo` | ` Beijing` 0.73% |
| `The largest planet in the solar system is` | ` Jupiter` | 11.38%  (+10.65 pts) | 2 | ` the` | ` Mars` 5.30% |
| `Water is made of hydrogen and` | ` oxygen` | 34.75%  (-10.92 pts) | 1 | ` oxygen` | ` carbon` 5.43% |
| `The chemical symbol for gold is` | ` Au` | 1.21%  (+0.47 pts) | 13 | ` gold` | ` Ag` 0.11% |
| `Shakespeare wrote a play called Romeo and` | ` Juliet` | 98.71%  (+0.28 pts) | 1 | ` Juliet` | ` Hamlet` 0.00% |

## Generations

Same ten prompts, same seed (1234), temperature 0.7 at every milestone, so any difference is the model rather than sampling noise.

### `factual`

**Prompt:** `The capital of France is`

```
Paris.
The city and the city are in the middle of an active military dictatorship, with a number of political parties, including the National Revolutionary Party (NPD) and the Communist Party (CPD). It has a rich history dating back to the 19th century, particularly from the 1920s through the 1990s. In addition to this it is the home for several museums, such as the
```

<details><summary>previous milestone</summary>

```
the town of St. Denis.
It is a narrow and open, with a round tower (called an "moor") on the top. It is surrounded by a river and a forest. The city has a small city, the Sardinia, the largest city in the world. A popular amusement park, the famous "Rosetta" was once used as a shopping center.
```
</details>

### `science`

**Prompt:** `Photosynthesis is the process by which`

```
plants produce energy from sunlight and convert it into energy when exposed to light.
The following article examines the main steps in the production of energy for photosynthesis. It provides an overview of the basic steps involved in the process, including:
- The step of plant growth
- The role of water in photosynthesis
- The key steps involved in this process, including:
- Water-soluble vitamins
```

<details><summary>previous milestone</summary>

```
plants produce new tissue. These cells then undergo a process known as growth, which involves several stages and can be divided into two main types:
- Growth is determined by the division of cells (a cell that produces new cells) by the production of new cells. This process can occur due to many different factors such as injury or disease, ageing, infection, or even exposure to toxic substances.
```
</details>

### `definition`

**Prompt:** `A large language model is`

```
essential for effective communication.

In the context of language instruction, mastery of foreign languages is crucial for successful transfer and interpretation of texts. Specifically, developing proficiency in the target language can help facilitate comprehension and retention. For instance, mastering a Spanish sentence could help improve pronunciation, whereas German sentences could benefit from knowing a Spanish word. By employing this approach, learners can enhance their fluency and strengthen their native
```

<details><summary>previous milestone</summary>

```
essential for effective communication.
- A language model will provide information about the child's environment, and will help them understand their behavior. The model will also help them recognize patterns and develop language skills.
- Language models are designed to capture the child's needs and help them understand their emotions and experiences. These models will be used in a variety of contexts, including:
- Language learning
-
```
</details>

### `narrative`

**Prompt:** `The old lighthouse keeper walked down to the shore and`

```
saw a large stone statue and silver candelabras.
The new museum was built in 1873, and is open to visitors from all around the world. It serves as a museum of colonial history with exhibits relating to the city, the period between 1700 and 1713. The museum’s exhibit includes a replica of a medieval castle and a bronze statue of a woman and a portrait of a
```

<details><summary>previous milestone</summary>

```
found a hole in the roof.
It was not until two years later that the keeper's family moved to the Lighthouse, which had been waiting for them for more than a year. They were on one of the many other vessels from the area, and the keeper said he could not find it. The keeper was surprised and asked if there was anything else that could be seen.
```
</details>

### `instructional`

**Prompt:** `To bake bread, first`

```
heat your chosen oven. Once cooled, remove the oven and set aside to cool completely.
2. Slice the bread into thin strips and place them on a baking sheet lined with parchment paper. Bake at 350°F for approximately 15-25 minutes, turning occasionally, until golden brown and crispy. Remove from oven and allow it to rest for at least 10 minutes before slicing.
3.
```

<details><summary>previous milestone</summary>

```
remove the pastry and roll it into a ball.
2. Preheat oven to 375°F (190°C). Turn on the stove to 400°F (175°C) for medium-rare results. This step ensures even cooking.
3. Transfer the dough onto a baking sheet lined with parchment paper. Make sure there's enough space between each piece so that air
```
</details>

### `list`

**Prompt:** `Three reasons to exercise regularly:`

```
- Helps improve focus and memory.
- Supports social skills.
- Helps to manage stress.
- Strengthens the immune system.
- Increases muscle tone and strength.
- Improves body composition.
- Improves coordination.
- Helps improve blood flow.
- Helps reduce inflammation.
- Can help reduce the risk of age related chronic diseases.
```

<details><summary>previous milestone</summary>

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
</details>

### `code`

**Prompt:** ````python
def fibonacci(n):`

```
return 1, n[0]
    if n <= 2:
        return 1 / n[1]
```

### The `Pull` Function

The `Pull` function is used to pull a string from a string, which is returned by `string_to_string`. This is a common way for strings to be pulled, but it also has some disadvantages
```

<details><summary>previous milestone</summary>

```
return 1 / len(self.__init__))
print(np.allclose(0, self.__init__))
```
This will print the results of our simulation: `x_pred`, `y_pred`, `d_pred`, `x_tol`. These values are then used to compute the number of features in the input data.

###
```
</details>

### `dialogue`

**Prompt:** `"Where are you going?" she asked. He`

```
didn't know what to do, so he decided to leave the house and go back to his original home. As they moved into their new home, they noticed that some of the furniture was being worn out or damaged. They wondered why some of them had broken down, while others looked like they were missing from a great adventure.

As they walked through the empty room, they saw many other
```

<details><summary>previous milestone</summary>

```
paused for a moment before answering, "It's because we all know that the best way to travel is through an airplane."

Next, they talked about how the airlines have evolved over time. The Airbus has undergone significant upgrades during its long-term flight, allowing it to run faster and more efficiently than ever before. However, managing such changes can be challenging and expensive, so engineers
```
</details>

### `arithmetic`

**Prompt:** `If a train leaves at 3pm and travels for two hours, it arrives at`

```
4:55 p.m. If the train is on or out, its passengers are advised to stay in an area where they can be rested.
A short distance from the train station is called an "autonomous flight path."
A long distance from the train station is called an "directional flight path". In this situation, it is very important that there be enough airspace around
```

<details><summary>previous milestone</summary>

```
4:00 p.m. If you're not sure how to get there, ask a travel counselor or call 911 to help.
"There is no need for an emergency room," said Dale Sweeney, president of the National Safety Council. "It's not safe."The Governing Body of the U.S. House of Representatives, on July 16, 2005,
```
</details>

### `conversational`

**Prompt:** `Hello! What can you help me with?`

```
I'm happy to tell you that many people around the world are starting to feel more and more comfortable with being part of this wonderful community. So let's dive into how we can become more mindful citizens and contribute positively to our schools, friends, and communities.

Firstly, let's talk about why it's important for us to have a positive role model for ourselves. Just like learning
```

<details><summary>previous milestone</summary>

```
- 1. What do you think would be the best way to get a good education in English? (I did not know where to go)
- 2. The history of language learning:
- 3. The importance of language learning in everyday life.
- 4. I’m interested in languages and how they are used for our purposes.
- 5. How is it used
```
</details>

## Notes

- Benchmarks run on CPU from the latest checkpoint. Training was not paused, signalled or slowed on the GPU at any point.
- Model: 120,787,072 parameters, context 1024.
- This is a base model: it continues text, it does not answer questions. Instruction following arrives with fine-tuning, after pretraining completes.