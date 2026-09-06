# Milestone 2,000 — Saturday 05 September 2026, 05:47

Step **2,000 of 10,588** (18.9%). 499,712,000 of 2,645,628,484 tokens seen (18.9% of one pass).

Wall clock since this log began: 4:46:35.

## Training state

| Metric | Value | Change since last milestone |
|---|---|---|
| Training loss | 3.2741 |  (-0.141765, improved) |
| Training perplexity | 26.4 | |
| Eval loss | 3.6475 |  (-0.098971, improved) |
| Learning rate | 5.58e-04 | |
| Throughput | 17,724 tok/s | |
| MFU | 49.4% | |
| Peak VRAM | 3.15 GB | |
| GPU | 99% util, 3,802 MiB, 85°C, 128 W | |

## Held-out evaluation

Measured on text the model has never trained on, on CPU, without interrupting training.

| Metric | Value | Change |
|---|---|---|
| Nats per token | 3.3540 |  (-0.082735, improved) |
| Perplexity | 28.6 |  (-2.5, improved) |
| Bits per byte | 0.9596 |  (-0.023672, improved) |

**Code vs prose:** code 1.978 nats, prose 2.778 nats — code is easier by 0.800. Code is lower-entropy text, so the model models it more cheaply; watching this gap narrow or widen shows whether prose modelling is catching up.

## Generations

Same ten prompts, same seed (1234), temperature 0.7 at every milestone, so any difference is the model rather than sampling noise.

### `factual`

**Prompt:** `The capital of France is`

```
located in Paris, France.
In the north-west corner of the country, and along the border with France, the city has become a bustling metropolis. It is home to various religious groups, including Christians, Hindus, Buddhists, etc. The City is also known for its beautiful architecture and beautiful gardens, making it an ideal place for religious festivals, fairs, concerts, and festivals.
However
```

<details><summary>previous milestone</summary>

```
located in the city of Dardan, which is part of the larger town. The capital has been called "Bagala," and it is known as Grenier's Castle. In the city of Sardan there are several cities such as Lünster, Paris, Tavern, Hamburg, and Lünster. The city itself includes a total area of
```
</details>

### `science`

**Prompt:** `Photosynthesis is the process by which`

```
plants produce food, such as fruits, vegetables, and whole grains. However, it is important to note that plant growth is not only limited to photosynthesis; it also involves more complex processes than other plant species. Therefore, when considering the nutritional requirements of plants, it is essential to consider the composition and functions of these plants before introducing them into the diet of your plant.
3.1.2
```

<details><summary>previous milestone</summary>

```
cells produce energy. There are two main types of molecules:
- CAMP (methylphenidate) and CAMP (methylphenidate).
- TSH (Lanine), one of the primary components of the cell, is also known as HLA.
The study of these molecules was conducted in collaboration with the National Institutes of Health and the National Institute of Neurolog
```
</details>

### `definition`

**Prompt:** `A large language model is`

```
well-developed, with the ability to be written by a single person.
There are a variety of models to be used in this application, each with its own set of examples, and the advantages and disadvantages can be tested with other models. However, there are many models that have different features, and one type is a standard model for this application. This version, called a standard, is generally
```

<details><summary>previous milestone</summary>

```
essential for effective communication.

Here are some tips to help you build rapport with your partner:

* Avoid sharing sensitive information or asking questions about them.
* Be curious and open-minded as you can learn from others.
* Share personal experiences, such as therapy sessions, mentorship, or even conversations.
* Use common phrases, phrases, or phrases that make you feel comfortable
```
</details>

### `narrative`

**Prompt:** `The old lighthouse keeper walked down to the shore and`

```
climbed up a tree. He was so afraid he would not tell anyone about his life he had come home, but there was something that he had seen so much.
He looked up and began to ask if he could see what he thought of his life, and the man who knew him well knew how to live. In a speech to the director at the time, he told a friend about what
```

<details><summary>previous milestone</summary>

```
climbed down.
"We were really surprised if we could tell how long it would take to travel around the world!" he said. "It was amazing how many people here are traveling there, but also that one person is so special."
In addition, he noted that the city of Tuscany is a great place for tourists to explore. He explained that, although many of the cities have
```
</details>

### `instructional`

**Prompt:** `To bake bread, first`

```
cut your hands into bite-sized pieces. Then chop them into bite-sized pieces and store them in a cool place where they can be cooked. Top with chopped onions and carrots, then spread them on the plate. Place each piece onto a baking sheet, cover it with a lid, and let them cook for about two hours before serving.

**Step 5: Bake Your Delicious bread**
```

<details><summary>previous milestone</summary>

```
into a loaf of beef or beef, then pour the meat out with the butter and cook until golden brown.
* Stir in all ingredients (such as chicken, vegetables, beans, nuts, seeds, and potatoes). If you prefer to cook, add it to your plate and season with additional ingredients like potatoes, carrots, potatoes, and fish. This will help you enjoy the delicious taste of
```
</details>

### `list`

**Prompt:** `Three reasons to exercise regularly:`

```
- Increased cardiovascular endurance and better heart health.
- More blood flow, leading to a lower risk of stroke and stroke.
- Lower rates of heart attacks and stroke (heart attack).
- Better cardiovascular endurance and stronger muscles.
- Greater cardiovascular fitness.
- Better overall health.
- Improved cardiovascular fitness.
- Less stress, anxiety, and depression.
If you are
```

<details><summary>previous milestone</summary>

```
- The first thing to do is to take a warm bath before going out.
- The second reason to be active is to keep your body hydrated and alert for you from the sun’s rays.
- The third reason to start exercising regularly is to keep your body hydrated.
- You’ll need to make sure that your muscles are running properly, even during a long walk.
```
</details>

### `code`

**Prompt:** ````python
def fibonacci(n):`

```
n = 10.5 * n
    if n < 5:
        return n, n_steps = 1
```

### The `n_steps` Function

The `n_steps` function is a Python function that takes in the rotation of an object and the length of its path to the desired rotation. It returns three values: `N`, `n`, and `
```

<details><summary>previous milestone</summary>

```
for i in range(len(self.n_samples)):
        for j in range(num_samples, num_samples)
        # concatenate the results of all possible outcomes from each point of the dataset
for i in range(num_samples)):
    for j in range(num_samples):
        for j in range(num_samples)
        # conc
```
</details>

### `dialogue`

**Prompt:** `"Where are you going?" she asked. He`

```
wondered if anyone who could identify the type of light that is most likely to be affected by a particular light, like a black star, would do so because it's too small for them to see.

As they walked through the forest, Sarah couldn't help but notice how much darker and darker each one was than before. She saw this object floating outside her window, and noticed that it seemed
```

<details><summary>previous milestone</summary>

```
said, "I'm not going to be in the middle of a new school year, but I will have to be working on my own."
- "I've been teaching math and science together."
- "We're going to make a big difference in the lives of children around the world!"
- "What are we doing?"
- "How do we get involved with it?
```
</details>

### `arithmetic`

**Prompt:** `If a train leaves at 3pm and travels for two hours, it arrives at`

```
4pm.
It is not unusual for a train to run out of fuel. The trains are still in use and the power goes up to 100% to supply the needed fuel. But the engine can also be used by running out of fuel. A train can produce enough fuel for one day while carrying it from the generator. This is called “the fuel-boosting service”.
A train
```

<details><summary>previous milestone</summary>

```
the nearest 1am.
The original 3D printed circuit (available from the GIF) has been used to cut out all of the parts of the building. It’s been found in an open room at the front of the building. The circuit is also a good fit for the walkway, which will make up a complete 3D printed circuit. The other way around, the clock can
```
</details>

### `conversational`

**Prompt:** `Hello! What can you help me with?`

```
- 1. Educate yourself about the risks associated with using chemical pesticides and pesticide residue (to control chemicals, to prevent chemical residue from spreading)
- 2. Use a pesticide applicator (a chemical pesticide applicator or pesticide applicator) to protect your plants
- 3. Consult local toxicologists for advice about potential exposure levels of chemicals
- 4. Provide information on safe use of chemicals
```

<details><summary>previous milestone</summary>

```
- 1. Ask yourself what is the first thing I could do to help. How would you like to make a difference?
- 2. Research your place and ask questions about it.
- 3. Write down what you think you would want in order.
- 4. Take action on how you will be able to handle this situation.
- 5. Be a positive person.
```
</details>

## Notes

- Benchmarks run on CPU from the latest checkpoint. Training was not paused, signalled or slowed on the GPU at any point.
- Model: 120,787,072 parameters, context 1024.
- This is a base model: it continues text, it does not answer questions. Instruction following arrives with fine-tuning, after pretraining completes.