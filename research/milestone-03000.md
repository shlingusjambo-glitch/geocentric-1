# Milestone 3,000 — Saturday 05 September 2026, 09:43

Step **3,000 of 10,588** (28.3%). 749,568,000 of 2,645,628,484 tokens seen (28.3% of one pass).

Wall clock since this log began: 8:41:58.

## Training state

| Metric | Value | Change since last milestone |
|---|---|---|
| Training loss | 3.2454 |  (-0.025496, improved) |
| Training perplexity | 25.7 | |
| Eval loss | 3.5255 |  (-0.057889, improved) |
| Learning rate | 5.05e-04 | |
| Throughput | 17,841 tok/s | |
| MFU | 49.7% | |
| Peak VRAM | 3.15 GB | |
| GPU | 99% util, 3,801 MiB, 85°C, 112 W | |

## Held-out evaluation

Measured on text the model has never trained on, on CPU, without interrupting training.

| Metric | Value | Change |
|---|---|---|
| Nats per token | 3.2576 |  (-0.032915, improved) |
| Perplexity | 26.0 |  (-0.9, improved) |
| Bits per byte | 0.9320 |  (-0.009418, improved) |

**Code vs prose:** code 1.871 nats, prose 2.627 nats — code is easier by 0.756. Code is lower-entropy text, so the model models it more cheaply; watching this gap narrow or widen shows whether prose modelling is catching up.

## Generations

Same ten prompts, same seed (1234), temperature 0.7 at every milestone, so any difference is the model rather than sampling noise.

### `factual`

**Prompt:** `The capital of France is`

```
usually not so good.
So we need a little bit more money to keep up with the increasing number of people in France and, if you have lots of money, you can do it at least once a year. But just because you are rich doesn’t mean that you don’t need to spend most of your time doing everything you want, like doing laundry, eating, etc. And
```

<details><summary>previous milestone</summary>

```
located on the River Deir-Leh.
According to the report, over two billion people live in an area of about 540 square kilometers (about 1,000 square miles). About 30% of the population are poor. This means that fewer than half of all the city’s residents live in a poverty-stricken area. That’s also why many Americans are illiterate
```
</details>

### `science`

**Prompt:** `Photosynthesis is the process by which`

```
plants produce flowers. There are different types of photosynthesis, some chemical plants use sunlight to create food and others use carbohydrates for energy production. The different types of photosynthesis are:
Cultivation (cultivation) happens when plants have grown into a group of organisms. Most plants require the right amount of sunlight to survive, but some plants require it at certain times of day and again in the morning
```

<details><summary>previous milestone</summary>

```
cells produce a chemical reaction.
In the first part of this article, I will explain how to use the DESCRIPT function to create an enzyme that is used in cell culture to create a fluorescent dye. The DESCRIPT function is also known as the DESCRIPT function. In this article, we will discuss some basic steps for creating and using the DES
```
</details>

### `definition`

**Prompt:** `A large language model is`

```
essential for effective communication.
- Consider how the child will learn to use the language in a variety of ways to express his or her needs and concerns. This will help you decide whether it is appropriate to seek out additional resources to address your child’s needs.
- If you do not have any ongoing training, consider seeking professional guidance from a speech-language pathologist at a local pediatrician
```

<details><summary>previous milestone</summary>

```
essential for effective communication.
- Consider how the person is interacting with you and their environment to create a positive relationship with you.
- Choose a time to feel relaxed and calm, and take an active role in managing your emotions. This could include volunteering, attending workshops or sports events, or practicing mindfulness meditation.
- Avoid using negative words or imagery that may negatively affect you. Instead, use
```
</details>

### `narrative`

**Prompt:** `The old lighthouse keeper walked down to the shore and`

```
saw a large boulders.
He had been looking for an old lighthouse keeper and had been searching for a night skywalk on his boat. He went in and found a small fire at the bottom of the tower. It was a big, white fire that seemed to be burning across the tower from the tower. This fire was burning at the bottom of the tower. He saw a
```

<details><summary>previous milestone</summary>

```
saw a large pond.
One of the first visitors to the town was Mr. Mumford who told them all about the history of the town. As he went on, he found a small stone that belonged to the tower. It had a white marble slab. The walls were decorated with white marble, but the tower was destroyed by fire. He then went on to find a boat. The
```
</details>

### `instructional`

**Prompt:** `To bake bread, first`

```
sauté your chosen vegetables. Add the chopped potatoes and pour them into a bowl of water until well combined. Remove the pan from heat and set aside.
2. **Fill in the pan**: Dip one cup of butter or non-dairy milk in the same vegetable mixture. Add the remaining half of the chopped ham until fully coated. Allow it to soak for about 5 minutes.
```

<details><summary>previous milestone</summary>

```
heat the egg until it forms a dense layer.
2. Spread the flour over the eggs and cover them with aluminum foil or plastic wrap to create an elastic band around the edges of the dough.
3. Pour the softened butter from the melted butter, cover the eggs with aluminum foil, and layer the thin layer on top.
4. Allow the mixture to thicken before adding it to
```
</details>

### `list`

**Prompt:** `Three reasons to exercise regularly:`

```
- The first time you should be exercising.
- You will have a good chance of completing the task.
- It is difficult to keep your mind occupied during the actual exercise.
There are also other benefits to exercise:
- When you are exercising, you are not only improving your mental health but it is becoming more and more easier every day.
- You can improve your mood
```

<details><summary>previous milestone</summary>

```
- Keep your weight at the same height.
- Get enough physical activity.
- Exercise regularly.
It is important that you do not overexert yourself, and exercise frequently can lead to health problems. If you are overweight or obese, you may have an increased risk of developing diabetes, heart disease, high blood pressure and diabetes.
According to The American Diabetes Association, about 40
```
</details>

### `code`

**Prompt:** ````python
def fibonacci(n):`

```
return [n.get_cuff() for n in zip(n=n)].split('/n-1').split('/n-1')
```
### Plotting Fibonacci Numbers with the Fibonacci Sequence

To plot a Fibonacci number, we will use the following steps:

1. We will define a function `to_c
```

<details><summary>previous milestone</summary>

```
return 1 / len(n)
    if not None:
        return 1 / len(n)
    print("No!")
```
In this example, we are plotting the quadratic distribution of the polynomial with the standard deviation. The `polyfit` method returns a 2D array with the mean squared error of the input data and the mean square error of the output.

###
```
</details>

### `dialogue`

**Prompt:** `"Where are you going?" she asked. He`

```
said, "I'm not going to be here because I don't have any kids in my community."
The next day, she was walking around the campus with her young son, and he began to see the students coming to his house. When he got home, he explained that he had seen them pass by a large group of students who were coming to him as soon as they were about to
```

<details><summary>previous milestone</summary>

```
said, "I'm not sure. I guess I've got to get back up and take the leap."

She smiled, "That's right! It's a great exercise to build muscle mass. And what about stretching?" She asked, "Yes, sometimes stretching can help us feel better!"

They went on to explain that stretching is just like stretching for body tissue, but instead
```
</details>

### `arithmetic`

**Prompt:** `If a train leaves at 3pm and travels for two hours, it arrives at`

```
4:55 p.m., the distance is approximately 30 miles per hour (about 7 miles) (the distance is about 1,000 feet).
A train is used to travel 5,000 feet on one day. If you are travelling over 10,000 miles of track, the length is usually around 20,000 miles. If you have been traveling 10,000 miles, the distance is
```

<details><summary>previous milestone</summary>

```
midnight.
The car is powered by the steam engine, which is similar to a diesel engine in that it produces electricity by charging its battery pack. However, since the car's engine has no engine at all, it can't operate from the battery pack, which is not enough for the electric motor to run. The car will drive itself back to its original condition, but with a new power source
```
</details>

### `conversational`

**Prompt:** `Hello! What can you help me with?`

```
- Is it difficult to understand the concept of "the social world"?
- How do I know what to expect from my little one?
- Is there anything you can do to get them to do the job safely and comfortably?
- Does this person feel comfortable in their own home, or is they sitting up too much?
- Does there be any way you could move your child
```

<details><summary>previous milestone</summary>

```
- 1. Ask for help.
- 2. Get a new job.
- 3. Click on the email link.
- 4. Click here to see how I will use it.
- 5. If you are not comfortable, you may want to check out my website.
- 6. In case of any problems, please contact me as soon as possible.
-
```
</details>

## Notes

- Benchmarks run on CPU from the latest checkpoint. Training was not paused, signalled or slowed on the GPU at any point.
- Model: 120,787,072 parameters, context 1024.
- This is a base model: it continues text, it does not answer questions. Instruction following arrives with fine-tuning, after pretraining completes.