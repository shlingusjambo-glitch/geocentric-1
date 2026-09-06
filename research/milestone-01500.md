# Milestone 1,500 — Saturday 05 September 2026, 03:50

Step **1,500 of 10,588** (14.2%). 374,784,000 of 2,645,628,484 tokens seen (14.2% of one pass).

Wall clock since this log began: 2:48:50.

## Training state

| Metric | Value | Change since last milestone |
|---|---|---|
| Training loss | 3.4159 |  (-0.059395, improved) |
| Training perplexity | 30.4 | |
| Eval loss | 3.7465 |  (-0.072676, improved) |
| Learning rate | 5.77e-04 | |
| Throughput | 17,670 tok/s | |
| MFU | 49.3% | |
| Peak VRAM | 3.15 GB | |
| GPU | 99% util, 3,802 MiB, 85°C, 112 W | |

## Held-out evaluation

Measured on text the model has never trained on, on CPU, without interrupting training.

| Metric | Value | Change |
|---|---|---|
| Nats per token | 3.4368 |  (-0.062709, improved) |
| Perplexity | 31.1 |  (-2.0, improved) |
| Bits per byte | 0.9833 |  (-0.017942, improved) |

**Code vs prose:** code 2.124 nats, prose 2.876 nats — code is easier by 0.751. Code is lower-entropy text, so the model models it more cheaply; watching this gap narrow or widen shows whether prose modelling is catching up.

## Generations

Same ten prompts, same seed (1234), temperature 0.7 at every milestone, so any difference is the model rather than sampling noise.

### `factual`

**Prompt:** `The capital of France is`

```
located in the city of Dardan, which is part of the larger town. The capital has been called "Bagala," and it is known as Grenier's Castle. In the city of Sardan there are several cities such as Lünster, Paris, Tavern, Hamburg, and Lünster. The city itself includes a total area of
```

<details><summary>previous milestone</summary>

```
the North Sea, with the sea itself, a region known as South-East Asia (also referred to as South-East Asia), and the North Atlantic is home to more than 2 million people.
Bouillean has been an important part of the country’s economy since the early 20th century; it was a significant contribution to the population’s growth and development. It is also
```
</details>

### `science`

**Prompt:** `Photosynthesis is the process by which`

```
cells produce energy. There are two main types of molecules:
- CAMP (methylphenidate) and CAMP (methylphenidate).
- TSH (Lanine), one of the primary components of the cell, is also known as HLA.
The study of these molecules was conducted in collaboration with the National Institutes of Health and the National Institute of Neurolog
```

<details><summary>previous milestone</summary>

```
small particles of material can be dissolved. This method uses a chemical reaction to form a liquid, and then, in some cases, using a single solvent. The solution is used as an electrolyte (a substance that can be easily converted to a gas) and is also sometimes used as a catalyst for building new products.
The simplest method of building materials is the metal, and it has been demonstrated to
```
</details>

### `definition`

**Prompt:** `A large language model is`

```
essential for effective communication.

Here are some tips to help you build rapport with your partner:

* Avoid sharing sensitive information or asking questions about them.
* Be curious and open-minded as you can learn from others.
* Share personal experiences, such as therapy sessions, mentorship, or even conversations.
* Use common phrases, phrases, or phrases that make you feel comfortable
```

<details><summary>previous milestone</summary>

```
an interactive map of the world.
It is a free online encyclopedia that allows users to search and browse through the web, including images and videos.
This website is used for research and learning purposes.
The site also provides information about the world's culture, history, religion, politics, culture, history, religions, geography, history and history. This site also includes a vast collection of resources
```
</details>

### `narrative`

**Prompt:** `The old lighthouse keeper walked down to the shore and`

```
climbed down.
"We were really surprised if we could tell how long it would take to travel around the world!" he said. "It was amazing how many people here are traveling there, but also that one person is so special."
In addition, he noted that the city of Tuscany is a great place for tourists to explore. He explained that, although many of the cities have
```

<details><summary>previous milestone</summary>

```
moved down to the ground and sat down for a while until he finally reached his destination. He had been told to walk, and then climbed up to the hill, looking for the elusive fuss of a nearby human being.
He was greeted by an angry crowd of people who looked around him before he could see what happened. The man walked up at his eyes, and his eyes were dark.
```
</details>

### `instructional`

**Prompt:** `To bake bread, first`

```
into a loaf of beef or beef, then pour the meat out with the butter and cook until golden brown.
* Stir in all ingredients (such as chicken, vegetables, beans, nuts, seeds, and potatoes). If you prefer to cook, add it to your plate and season with additional ingredients like potatoes, carrots, potatoes, and fish. This will help you enjoy the delicious taste of
```

<details><summary>previous milestone</summary>

```
lift your hands and turn them into a ball.
Make sure to place the dough in a proper shape, like a table or table. Place each piece on top of the dough so that it is at least one inch thick.
Place the dough inside the oven. Place the pieces on top of the bread with the following ingredients: 1 tablespoon baking powder; 2 tablespoons baking powder; 3 tablespoons baking
```
</details>

### `list`

**Prompt:** `Three reasons to exercise regularly:`

```
- The first thing to do is to take a warm bath before going out.
- The second reason to be active is to keep your body hydrated and alert for you from the sun’s rays.
- The third reason to start exercising regularly is to keep your body hydrated.
- You’ll need to make sure that your muscles are running properly, even during a long walk.
```

<details><summary>previous milestone</summary>

```
- Keep a cool and comfortable temperature for at least 30 minutes before going out.
- Avoid placing your bed in an airtight container such as a bathroom or fridge.
- Limit your outdoor activities as much as possible so you can enjoy them anywhere.
- Never leave your room in an open, closed space, or outside. Make sure there is enough space for you to sit on.
```
</details>

### `code`

**Prompt:** ````python
def fibonacci(n):`

```
for i in range(len(self.n_samples)):
        for j in range(num_samples, num_samples)
        # concatenate the results of all possible outcomes from each point of the dataset
for i in range(num_samples)):
    for j in range(num_samples):
        for j in range(num_samples)
        # conc
```

<details><summary>previous milestone</summary>

```
n = 10.5

def fibonacci(n):
    n, n = 100, n = 10000, n = 100000, n = 10000, n = 30000, n = 20000, n = 5000000, n = 10000000, n = 5000, n = 25000, n = 10000, n = 100000, n =
```
</details>

### `dialogue`

**Prompt:** `"Where are you going?" she asked. He`

```
said, "I'm not going to be in the middle of a new school year, but I will have to be working on my own."
- "I've been teaching math and science together."
- "We're going to make a big difference in the lives of children around the world!"
- "What are we doing?"
- "How do we get involved with it?
```

<details><summary>previous milestone</summary>

```
chuckled softly and said, "So we want to go on a trip to the United States to travel around the world!"

Maria smiled warmly at her friends, asking, "But why do you take so many miles when traveling?" Her voice replied, "Well, sometimes people want to see beautiful places without even being able to travel anywhere." Her gaze lingered on her ears and replied, "I
```
</details>

### `arithmetic`

**Prompt:** `If a train leaves at 3pm and travels for two hours, it arrives at`

```
the nearest 1am.
The original 3D printed circuit (available from the GIF) has been used to cut out all of the parts of the building. It’s been found in an open room at the front of the building. The circuit is also a good fit for the walkway, which will make up a complete 3D printed circuit. The other way around, the clock can
```

<details><summary>previous milestone</summary>

```
4:45 p.m. on the first day of the week. After that, a train will travel to the main train station at 1:15 p.m. on the second day of the week. The trains will arrive at 3pm, with the third day being at 11:30 p.m. on the second day of the week. The next day will arrive at 9:
```
</details>

### `conversational`

**Prompt:** `Hello! What can you help me with?`

```
- 1. Ask yourself what is the first thing I could do to help. How would you like to make a difference?
- 2. Research your place and ask questions about it.
- 3. Write down what you think you would want in order.
- 4. Take action on how you will be able to handle this situation.
- 5. Be a positive person.
```

<details><summary>previous milestone</summary>

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
</details>

## Notes

- Benchmarks run on CPU from the latest checkpoint. Training was not paused, signalled or slowed on the GPU at any point.
- Model: 120,787,072 parameters, context 1024.
- This is a base model: it continues text, it does not answer questions. Instruction following arrives with fine-tuning, after pretraining completes.