# Milestone 2,500 — Saturday 05 September 2026, 07:45

Step **2,500 of 10,588** (23.6%). 624,640,000 of 2,645,628,484 tokens seen (23.6% of one pass).

Wall clock since this log began: 6:44:17.

## Training state

| Metric | Value | Change since last milestone |
|---|---|---|
| Training loss | 3.2709 |  (-0.003240, improved) |
| Training perplexity | 26.3 | |
| Eval loss | 3.5834 |  (-0.064151, improved) |
| Learning rate | 5.33e-04 | |
| Throughput | 17,797 tok/s | |
| MFU | 49.6% | |
| Peak VRAM | 3.15 GB | |
| GPU | 99% util, 3,801 MiB, 86°C, 135 W | |

## Held-out evaluation

Measured on text the model has never trained on, on CPU, without interrupting training.

| Metric | Value | Change |
|---|---|---|
| Nats per token | 3.2905 |  (-0.063556, improved) |
| Perplexity | 26.9 |  (-1.8, improved) |
| Bits per byte | 0.9415 |  (-0.018184, improved) |

**Code vs prose:** code 1.911 nats, prose 2.693 nats — code is easier by 0.782. Code is lower-entropy text, so the model models it more cheaply; watching this gap narrow or widen shows whether prose modelling is catching up.

## Generations

Same ten prompts, same seed (1234), temperature 0.7 at every milestone, so any difference is the model rather than sampling noise.

### `factual`

**Prompt:** `The capital of France is`

```
located on the River Deir-Leh.
According to the report, over two billion people live in an area of about 540 square kilometers (about 1,000 square miles). About 30% of the population are poor. This means that fewer than half of all the city’s residents live in a poverty-stricken area. That’s also why many Americans are illiterate
```

<details><summary>previous milestone</summary>

```
located in Paris, France.
In the north-west corner of the country, and along the border with France, the city has become a bustling metropolis. It is home to various religious groups, including Christians, Hindus, Buddhists, etc. The City is also known for its beautiful architecture and beautiful gardens, making it an ideal place for religious festivals, fairs, concerts, and festivals.
However
```
</details>

### `science`

**Prompt:** `Photosynthesis is the process by which`

```
cells produce a chemical reaction.
In the first part of this article, I will explain how to use the DESCRIPT function to create an enzyme that is used in cell culture to create a fluorescent dye. The DESCRIPT function is also known as the DESCRIPT function. In this article, we will discuss some basic steps for creating and using the DES
```

<details><summary>previous milestone</summary>

```
plants produce food, such as fruits, vegetables, and whole grains. However, it is important to note that plant growth is not only limited to photosynthesis; it also involves more complex processes than other plant species. Therefore, when considering the nutritional requirements of plants, it is essential to consider the composition and functions of these plants before introducing them into the diet of your plant.
3.1.2
```
</details>

### `definition`

**Prompt:** `A large language model is`

```
essential for effective communication.
- Consider how the person is interacting with you and their environment to create a positive relationship with you.
- Choose a time to feel relaxed and calm, and take an active role in managing your emotions. This could include volunteering, attending workshops or sports events, or practicing mindfulness meditation.
- Avoid using negative words or imagery that may negatively affect you. Instead, use
```

<details><summary>previous milestone</summary>

```
well-developed, with the ability to be written by a single person.
There are a variety of models to be used in this application, each with its own set of examples, and the advantages and disadvantages can be tested with other models. However, there are many models that have different features, and one type is a standard model for this application. This version, called a standard, is generally
```
</details>

### `narrative`

**Prompt:** `The old lighthouse keeper walked down to the shore and`

```
saw a large pond.
One of the first visitors to the town was Mr. Mumford who told them all about the history of the town. As he went on, he found a small stone that belonged to the tower. It had a white marble slab. The walls were decorated with white marble, but the tower was destroyed by fire. He then went on to find a boat. The
```

<details><summary>previous milestone</summary>

```
climbed up a tree. He was so afraid he would not tell anyone about his life he had come home, but there was something that he had seen so much.
He looked up and began to ask if he could see what he thought of his life, and the man who knew him well knew how to live. In a speech to the director at the time, he told a friend about what
```
</details>

### `instructional`

**Prompt:** `To bake bread, first`

```
heat the egg until it forms a dense layer.
2. Spread the flour over the eggs and cover them with aluminum foil or plastic wrap to create an elastic band around the edges of the dough.
3. Pour the softened butter from the melted butter, cover the eggs with aluminum foil, and layer the thin layer on top.
4. Allow the mixture to thicken before adding it to
```

<details><summary>previous milestone</summary>

```
cut your hands into bite-sized pieces. Then chop them into bite-sized pieces and store them in a cool place where they can be cooked. Top with chopped onions and carrots, then spread them on the plate. Place each piece onto a baking sheet, cover it with a lid, and let them cook for about two hours before serving.

**Step 5: Bake Your Delicious bread**
```
</details>

### `list`

**Prompt:** `Three reasons to exercise regularly:`

```
- Keep your weight at the same height.
- Get enough physical activity.
- Exercise regularly.
It is important that you do not overexert yourself, and exercise frequently can lead to health problems. If you are overweight or obese, you may have an increased risk of developing diabetes, heart disease, high blood pressure and diabetes.
According to The American Diabetes Association, about 40
```

<details><summary>previous milestone</summary>

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
</details>

### `code`

**Prompt:** ````python
def fibonacci(n):`

```
return 1 / len(n)
    if not None:
        return 1 / len(n)
    print("No!")
```
In this example, we are plotting the quadratic distribution of the polynomial with the standard deviation. The `polyfit` method returns a 2D array with the mean squared error of the input data and the mean square error of the output.

###
```

<details><summary>previous milestone</summary>

```
n = 10.5 * n
    if n < 5:
        return n, n_steps = 1
```

### The `n_steps` Function

The `n_steps` function is a Python function that takes in the rotation of an object and the length of its path to the desired rotation. It returns three values: `N`, `n`, and `
```
</details>

### `dialogue`

**Prompt:** `"Where are you going?" she asked. He`

```
said, "I'm not sure. I guess I've got to get back up and take the leap."

She smiled, "That's right! It's a great exercise to build muscle mass. And what about stretching?" She asked, "Yes, sometimes stretching can help us feel better!"

They went on to explain that stretching is just like stretching for body tissue, but instead
```

<details><summary>previous milestone</summary>

```
wondered if anyone who could identify the type of light that is most likely to be affected by a particular light, like a black star, would do so because it's too small for them to see.

As they walked through the forest, Sarah couldn't help but notice how much darker and darker each one was than before. She saw this object floating outside her window, and noticed that it seemed
```
</details>

### `arithmetic`

**Prompt:** `If a train leaves at 3pm and travels for two hours, it arrives at`

```
midnight.
The car is powered by the steam engine, which is similar to a diesel engine in that it produces electricity by charging its battery pack. However, since the car's engine has no engine at all, it can't operate from the battery pack, which is not enough for the electric motor to run. The car will drive itself back to its original condition, but with a new power source
```

<details><summary>previous milestone</summary>

```
4pm.
It is not unusual for a train to run out of fuel. The trains are still in use and the power goes up to 100% to supply the needed fuel. But the engine can also be used by running out of fuel. A train can produce enough fuel for one day while carrying it from the generator. This is called “the fuel-boosting service”.
A train
```
</details>

### `conversational`

**Prompt:** `Hello! What can you help me with?`

```
- 1. Ask for help.
- 2. Get a new job.
- 3. Click on the email link.
- 4. Click here to see how I will use it.
- 5. If you are not comfortable, you may want to check out my website.
- 6. In case of any problems, please contact me as soon as possible.
-
```

<details><summary>previous milestone</summary>

```
- 1. Educate yourself about the risks associated with using chemical pesticides and pesticide residue (to control chemicals, to prevent chemical residue from spreading)
- 2. Use a pesticide applicator (a chemical pesticide applicator or pesticide applicator) to protect your plants
- 3. Consult local toxicologists for advice about potential exposure levels of chemicals
- 4. Provide information on safe use of chemicals
```
</details>

## Notes

- Benchmarks run on CPU from the latest checkpoint. Training was not paused, signalled or slowed on the GPU at any point.
- Model: 120,787,072 parameters, context 1024.
- This is a base model: it continues text, it does not answer questions. Instruction following arrives with fine-tuning, after pretraining completes.