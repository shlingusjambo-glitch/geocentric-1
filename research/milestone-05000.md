# Milestone 5,000 — Saturday 05 September 2026, 20:49

Step **5,000 of 10,588** (47.2%). 1,249,280,000 of 2,645,628,484 tokens seen (47.2% of one pass).

Wall clock since this log began: 9:07:37.

## Training state

| Metric | Value | Change since last milestone |
|---|---|---|
| Training loss | 3.2493 |  (+0.187859, worse) |
| Training perplexity | 25.8 | |
| Eval loss | 3.6242 |  (+0.174495, worse) |
| Learning rate | 3.58e-04 | |
| Throughput | 20,181 tok/s | |
| MFU | 56.3% | |
| Peak VRAM | 2.84 GB | |
| GPU | 99% util, 3,574 MiB, 87°C, 130 W | |

## Held-out evaluation

Measured on text the model has never trained on, on CPU, without interrupting training.

| Metric | Value | Change |
|---|---|---|
| Nats per token | 3.4234 |  (+0.160589, worse) |
| Perplexity | 30.7 |  (+4.6, worse) |
| Bits per byte | 0.9825 |  (+0.046088, worse) |

**Code vs prose:** code 1.976 nats, prose 2.714 nats — code is easier by 0.738. Code is lower-entropy text, so the model models it more cheaply; watching this gap narrow or widen shows whether prose modelling is catching up.

## Factual probes

Probability the model assigns to the correct next token, which is far more sensitive than sampling: a fact can be largely learned yet appear only occasionally in generated text.

| Prompt | Answer | P(answer) | Rank | Top-1 | Best distractor |
|---|---|---|---|---|---|
| `The capital of France is` | ` Paris` | 0.39%  (-6.57 pts) | 39 | ` the` | ` London` 0.08% |
| `The capital of Japan is` | ` Tokyo` | 0.45%  (-12.17 pts) | 33 | ` the` | ` Beijing` 0.14% |
| `The largest planet in the solar system is` | ` Jupiter` | 3.12%  (-8.26 pts) | 4 | ` the` | ` Mars` 3.23% |
| `Water is made of hydrogen and` | ` oxygen` | 36.70%  (+1.95 pts) | 1 | ` oxygen` | ` nitrogen` 3.62% |
| `The chemical symbol for gold is` | ` Au` | 1.10%  (-0.12 pts) | 16 | ` the` | ` Ag` 0.09% |
| `Shakespeare wrote a play called Romeo and` | ` Juliet` | 99.02%  (+0.32 pts) | 1 | ` Juliet` | ` Caesar` 0.00% |

## Generations

Same ten prompts, same seed (1234), temperature 0.7 at every milestone, so any difference is the model rather than sampling noise.

### `factual`

**Prompt:** `The capital of France is`

```
the town of Soudes-Leh.
The capital is the capital in Luxemburg (now Luxemburg).
|1||-|||||
|2||-|||||
|3||-||–|||
|4||–||–|||||
|5||–||–|||
|6||–||–|||
```

<details><summary>previous milestone</summary>

```
Paris.
The city and the city are in the middle of an active military dictatorship, with a number of political parties, including the National Revolutionary Party (NPD) and the Communist Party (CPD). It has a rich history dating back to the 19th century, particularly from the 1920s through the 1990s. In addition to this it is the home for several museums, such as the
```
</details>

### `science`

**Prompt:** `Photosynthesis is the process by which`

```
photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic photosynthetic
```

<details><summary>previous milestone</summary>

```
plants produce energy from sunlight and convert it into energy when exposed to light.
The following article examines the main steps in the production of energy for photosynthesis. It provides an overview of the basic steps involved in the process, including:
- The step of plant growth
- The role of water in photosynthesis
- The key steps involved in this process, including:
- Water-soluble vitamins
```
</details>

### `definition`

**Prompt:** `A large language model is`

```
an interactive platform that can be used by the students to provide instruction and assessment.
- Language Learning (L2) is a core instructional approach that enables students with learning differences from different language skills to take on the same content areas. It provides a variety of scaffolds for differentiated instruction based on the needs of a learner, based on the strengths and weaknesses of each student and with which they are already
```

<details><summary>previous milestone</summary>

```
essential for effective communication.

In the context of language instruction, mastery of foreign languages is crucial for successful transfer and interpretation of texts. Specifically, developing proficiency in the target language can help facilitate comprehension and retention. For instance, mastering a Spanish sentence could help improve pronunciation, whereas German sentences could benefit from knowing a Spanish word. By employing this approach, learners can enhance their fluency and strengthen their native
```
</details>

### `narrative`

**Prompt:** `The old lighthouse keeper walked down to the shore and`

```
saw a baby crow.
- "Mama Bird" (1868). Retrieved: April 18, 2014.
- "The Staircase of the Queen's Cloak". National Union Star. Archived from the original on March 3, 2016. Retrieved March 4, 2016.
- "A Penny, a Queen's Cage at the Rye
- The Queen's Cl
```

<details><summary>previous milestone</summary>

```
saw a large stone statue and silver candelabras.
The new museum was built in 1873, and is open to visitors from all around the world. It serves as a museum of colonial history with exhibits relating to the city, the period between 1700 and 1713. The museum’s exhibit includes a replica of a medieval castle and a bronze statue of a woman and a portrait of a
```
</details>

### `instructional`

**Prompt:** `To bake bread, first`

```
cut it into slices.
- Bake for about 25 minutes or until they are golden brown and slightly golden brown.
- Once they are golden brown, slice them out of the flour and drizzle a tangy tomato sauce over them. Let the mixture sit on top of the bread for around 10 seconds or so to coat them completely before covering them with a fork or frying pan. Yum!
-
```

<details><summary>previous milestone</summary>

```
heat your chosen oven. Once cooled, remove the oven and set aside to cool completely.
2. Slice the bread into thin strips and place them on a baking sheet lined with parchment paper. Bake at 350°F for approximately 15-25 minutes, turning occasionally, until golden brown and crispy. Remove from oven and allow it to rest for at least 10 minutes before slicing.
3.
```
</details>

### `list`

**Prompt:** `Three reasons to exercise regularly:`

```
- Reduced blood pressure and high cholesterol, which can reduce or block the risk for heart attack and stroke.
- Lowering systolic blood pressure is important because it increases your blood pressure.
Exercise may help prevent hypertension
- Regular physical activity helps improve the blood pressure and blood pressure while lowering the risk of high blood pressure and high cholesterol.
- Regular physical activity also improves your blood
```

<details><summary>previous milestone</summary>

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
</details>

### `code`

**Prompt:** ````python
def fibonacci(n):`

```
for n in len(self.num_k):
        if (np.multiply(self.num_k) < 1e-5):
            for n in self.num_k:
                if (np.multiply(self.num_k) < -1e-5):
                        self.num_k = self.num_k.shape
```

<details><summary>previous milestone</summary>

```
return 1, n[0]
    if n <= 2:
        return 1 / n[1]
```

### The `Pull` Function

The `Pull` function is used to pull a string from a string, which is returned by `string_to_string`. This is a common way for strings to be pulled, but it also has some disadvantages
```
</details>

### `dialogue`

**Prompt:** `"Where are you going?" she asked. He`

```
said, "I'm not sure. I feel very embarrassed."
The child laughed and said, "I'm going to be here." The teacher had him sit down and said, "Please tell me what you've done so far. I'll say, 'Well now I want to walk in.'"
The child was saying, "I didn't know, I can do this. It
```

<details><summary>previous milestone</summary>

```
didn't know what to do, so he decided to leave the house and go back to his original home. As they moved into their new home, they noticed that some of the furniture was being worn out or damaged. They wondered why some of them had broken down, while others looked like they were missing from a great adventure.

As they walked through the empty room, they saw many other
```
</details>

### `arithmetic`

**Prompt:** `If a train leaves at 3pm and travels for two hours, it arrives at`

```
4:55 p.m. on the way to the train station.
- New trains arrive in the morning and depart from 5:00 p.m. until 7:30 p.m. on the way. If you stop after 9:00 p.m., you will see another train arrives from 8:00 - 11:00 p.m. on that train, so stay
```

<details><summary>previous milestone</summary>

```
4:55 p.m. If the train is on or out, its passengers are advised to stay in an area where they can be rested.
A short distance from the train station is called an "autonomous flight path."
A long distance from the train station is called an "directional flight path". In this situation, it is very important that there be enough airspace around
```
</details>

### `conversational`

**Prompt:** `Hello! What can you help me with?`

```
- 1. I would like to thank my teacher for giving me some wonderful writing and a bit of my blogging skills to write about.
- 2. Thank you, I love your advice and we know what you are going to say. Thanks you.
- 3. I’m so glad you got this blog post. Thank you, I will be here and happy.
-
```

<details><summary>previous milestone</summary>

```
I'm happy to tell you that many people around the world are starting to feel more and more comfortable with being part of this wonderful community. So let's dive into how we can become more mindful citizens and contribute positively to our schools, friends, and communities.

Firstly, let's talk about why it's important for us to have a positive role model for ourselves. Just like learning
```
</details>

## Notes

- Benchmarks run on CPU from the latest checkpoint. Training was not paused, signalled or slowed on the GPU at any point.
- Model: 120,787,072 parameters, context 1024.
- This is a base model: it continues text, it does not answer questions. Instruction following arrives with fine-tuning, after pretraining completes.