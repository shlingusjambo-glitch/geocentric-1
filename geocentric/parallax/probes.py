"""Built-in evaluation material for PARALLAX.

All of it is written for this file rather than sampled from a public corpus, for two
reasons that both matter more than convenience. A model trained on Wikipedia has
already seen anything drawn from Wikipedia, so a held-out slice of it measures
memorization as much as modeling. And a benchmark that has to download something is
a benchmark that stops working the week the URL rots.

The cost of that choice is sample size: this is a few kilobytes, not a few hundred
megabytes. It is a calibrated smoke test, not a leaderboard. Pass `--eval_text` with
your own held-out data whenever you have it — the same probes then run against text
the model has genuinely never seen, in your actual domain, at whatever size you like.
"""
from __future__ import annotations

# --- ZENITH: bits-per-byte across genres -----------------------------------
# Held deliberately varied: a model that reads only encyclopedia prose scores well
# on the first slice and falls apart on the rest, and that gap is the finding.

ZENITH_SLICES = {
    "prose": """The lighthouse keeper kept two logs. The first was the official one,
ruled and dated, in which he recorded the weather, the state of the lamp, and the
passage of any vessel close enough to name. The second he kept in a school exercise
book, and in it he wrote whatever he had thought about that day. He had started it
in his first winter, when he discovered that a man can go a long time without
speaking and not notice until he tries. The two logs were never confused. The
official one was a record of the sea. The other was a record of what the sea had
done to him. When the relief boat came in the spring the inspector read the first
one, initialled the bottom of each page, and remarked that the lamp had performed
well. He never asked about the exercise book, which sat on the shelf above the
stove in plain view, and which by then had run to a hundred and forty pages.""",

    "dialogue": """"You're late," she said, without looking up.
"The bridge was out. I had to go round by the mill."
"The bridge has been out for three weeks."
"Then I've been late for three weeks and you've only just mentioned it."
She put the pen down. "I've mentioned it every day. You've been apologising every
day. I assumed we both knew what the conversation was."
"That's fair."
"It isn't fair, it's just accurate. Sit down. There's tea, and it's still hot,
which is more than either of us deserves." He sat. Outside, the rain had settled
into the steady kind that means business, and somewhere below the window a gutter
had given up entirely.
"How bad is it?" he asked.
"Bad enough that I made tea." """,

    "code": """def merge_intervals(intervals):
    \"\"\"Collapse overlapping [start, end] pairs into the smallest covering set.\"\"\"
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda pair: pair[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(pair) for pair in merged]


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._store: dict = {}

    def get(self, key):
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return value

    def put(self, key, value) -> None:
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self.capacity:
            oldest = next(iter(self._store))
            self._store.pop(oldest)
        self._store[key] = value""",

    "reasoning": """A train leaves the station at nine and travels at sixty
kilometres an hour. A second train leaves the same station at ten and travels at
ninety. The question is when the second catches the first. By ten o'clock the first
train has covered sixty kilometres, so the second begins its journey sixty
kilometres behind. It closes that gap at thirty kilometres an hour, the difference
between the two speeds, and sixty divided by thirty is two. The second train draws
level at twelve o'clock, one hundred and eighty kilometres from the station. The
answer does not depend on the direction of travel, only on the fact that both trains
share a route, and it would be unchanged if both speeds were doubled and the hour
halved. What matters is the ratio of the head start to the rate at which it shrinks.""",

    "structured": """{"station": "Tromso", "latitude": 69.6492, "longitude": 18.9553,
"readings": [{"timestamp": "2024-01-14T03:00:00Z", "temperature_c": -8.4,
"pressure_hpa": 998.2, "wind_kt": 22}, {"timestamp": "2024-01-14T06:00:00Z",
"temperature_c": -9.1, "pressure_hpa": 996.8, "wind_kt": 27}], "flags": ["ice",
"gale_warning"], "operator": null, "revision": 3}

| port      | arrivals | departures | tonnage   |
|-----------|----------|------------|-----------|
| Bergen    |      412 |        408 | 1,204,880 |
| Stavanger |      331 |        329 |   980,115 |
| Alesund   |      198 |        201 |   440,302 |""",
}


# --- MERIDIAN: one long document, for loss as a function of position -------
# It has to be a single continuous argument, long enough to fill a context window,
# and it must not repeat itself. Repetition would let a model score well by copying
# rather than by understanding, which is precisely the failure this probe exists to
# catch.

MERIDIAN_DOCUMENT = """The problem of finding one's longitude at sea was, for
roughly two hundred years, the most expensive unsolved problem in Europe. Latitude
had never been difficult. The angle of the sun at noon, or of the pole star at
night, gives it directly, and a sailor with a cross-staff and a table could fix his
distance north or south of the equator to within a few miles. Longitude offered no
such gift. The earth turns, and every point on a line of longitude sees the same
sky at a different moment, so the only way to know how far east or west you have
travelled is to know what time it is somewhere else.

That reframing is the whole of the problem. Fifteen degrees of longitude is one
hour of rotation. If a navigator could compare his local noon, which he can observe
directly by watching the sun reach its highest point, against the simultaneous time
at a reference meridian, the difference in hours would give him his distance from
that meridian in degrees. At the equator a single degree is sixty nautical miles.
An error of four minutes in the comparison puts a ship a degree out of position,
and near the end of a long crossing that is the difference between an anchorage and
a reef.

Two families of solution were pursued, and the rivalry between them shaped the
sciences of the period. The astronomical approach treated the sky itself as the
clock. The moon moves against the fixed stars fast enough to be useful, completing
a circuit in a month, and if its position could be predicted precisely enough in
advance, a navigator could measure the angle between the moon and a chosen star,
look up the time at which that angle was due to occur at the reference meridian,
and subtract. The method was called lunar distance. It demanded a great deal: a
theory of the moon's motion accurate to arcseconds, tables computed years ahead,
an instrument capable of measuring angles from a pitching deck, and a navigator
willing to spend four hours on a calculation. Every one of those demands drove real
progress. The sextant was refined for it. National observatories were founded to
supply the tables. The three-body problem was attacked, unsuccessfully in closed
form and then successfully by approximation, because the moon would not hold still.

The mechanical approach was simpler to state and far harder to build. Carry a clock
set to the reference meridian. Read it. Subtract. A clock that keeps time to within
three seconds a day will, after a six-week crossing, have drifted about two minutes,
which is half a degree of longitude, which is close enough. No clock in existence
could do it. Pendulums are useless at sea, since a pendulum's period depends on
gravity and on being held still, and a ship provides neither. Temperature changes
lengthened and shortened every metal component, altering the rate. Humidity and
salt attacked the lubricants. The rolling of a hull put loads on pivots that a
domestic clock never experienced. Each of these was a separate engineering problem,
and each had to be solved simultaneously, because a clock that handles temperature
but not motion is no more use than one that handles neither.

The solutions, when they came, were unglamorous and specific. Bimetallic strips
made from two metals with different expansion coefficients could be arranged so
that the change in one cancelled the change in the other, keeping the effective
length of a balance spring constant as the temperature moved. Jewelled bearings cut
friction at the pivots and, more importantly, kept it constant, since a bearing
whose friction varies is a bearing whose rate varies. A remontoire rewinds the
escapement from a small secondary spring at short intervals, so the escapement
receives a nearly constant force regardless of how far the mainspring has run down.
None of these ideas is deep. All of them are the result of somebody identifying one
specific way in which a clock loses time and building a mechanism that opposes it.

What is worth noticing is that the two approaches did not compete so much as
converge. The astronomical method won first, in the sense that it was usable at
scale decades before a marine timekeeper could be manufactured in quantity, and the
tables it required were produced and distributed by the same institutions that
would later certify chronometers. The mechanical method won eventually, because
once a chronometer could be built at all it could be built repeatedly, and reading
a dial takes a minute where a lunar takes four hours. For most of the nineteenth
century a well-found ship carried both, and used the sky to check the clock. The
redundancy was not indecision. A chronometer that has failed gives a confident
wrong answer, and there is no way to detect the failure from the dial alone; a
lunar observation is laborious and imprecise, but it is independent, and it is the
independence that makes it valuable.

The pattern recurs whenever a measurement matters enough. A single instrument, no
matter how good, cannot report its own failure. Two instruments that fail in
unrelated ways can, because their disagreement is itself information. Modern
navigation repeats the arrangement almost exactly: a satellite fix is fast,
accurate and occasionally and silently wrong, and an inertial unit drifts steadily
and predictably, and the value of carrying both is not that either is better but
that their errors are uncorrelated. The longitude problem was solved twice, and the
second solution did not make the first obsolete. It made it a check."""


# --- SEXTANT: cloze items with distractors ---------------------------------
# Scored by comparing the total log-probability of each candidate continuation, so
# the probe works on a base model that has never seen an instruction. Items are
# short, unambiguous, and drawn from knowledge any general corpus contains.

SEXTANT_ITEMS = [
    # world knowledge
    ("The capital city of France is", " Paris", [" Berlin", " Madrid", " Rome"]),
    ("The largest ocean on Earth is the", " Pacific", [" Atlantic", " Indian", " Arctic"]),
    ("Water freezes at zero degrees", " Celsius", [" Fahrenheit", " Kelvin", " Rankine"]),
    ("The chemical symbol for gold is", " Au", [" Ag", " Gd", " Go"]),
    ("A triangle has three", " sides", [" corners of equal length", " faces", " diameters"]),
    ("The planet closest to the sun is", " Mercury", [" Venus", " Mars", " Earth"]),
    ("Photosynthesis uses sunlight to convert carbon dioxide and water into", " sugar",
     [" nitrogen", " protein", " salt"]),
    ("The human heart has four", " chambers", [" lungs", " kidneys", " stomachs"]),
    ("Shakespeare wrote plays in the", " English", [" Spanish", " Russian", " Latin"]),
    ("The speed of light is fastest in a", " vacuum", [" solid", " liquid", " gas"]),
    ("Mount Everest is part of the", " Himalayas", [" Alps", " Andes", " Rockies"]),
    ("A decade is ten", " years", [" months", " days", " centuries"]),
    # commonsense causation
    ("She forgot her umbrella, so when it rained she got", " wet", [" dry", " warm", " taller"]),
    ("The glass fell on the stone floor and", " shattered", [" grew", " sang", " floated"]),
    ("He had not eaten all day and was extremely", " hungry", [" asleep", " tall", " purple"]),
    ("Because the road was covered in ice, the driver slowed", " down", [" up", " sideways", " forever"]),
    ("The battery was empty, so the phone would not", " turn on", [" get heavier", " melt", " expand"]),
    ("She studied every night and passed the exam with", " ease", [" a bicycle", " November", " gravity"]),
    ("The soup was far too hot, so he waited for it to", " cool", [" boil", " freeze", " vanish"]),
    ("Nobody watered the plant for a month and it", " died", [" flowered", " doubled", " sang"]),
    # grammar and agreement
    ("The children who live next door", " are", [" is", " am", " was"]),
    ("Yesterday I", " walked", [" walk", " will walk", " walking"]),
    ("Neither of the answers", " was", [" were", " are", " am"]),
    ("She is taller than", " he is", [" him is", " his is", " he am"]),
    ("The book, along with its notes,", " was", [" were", " have", " are"]),
    # arithmetic and quantity
    ("Two plus three equals", " five", [" four", " six", " eight"]),
    ("Half of twenty is", " ten", [" five", " forty", " two"]),
    ("Twelve times twelve is", " 144", [" 124", " 148", " 122"]),
    ("If a shirt costs 20 and is reduced by half, it now costs", " 10", [" 20", " 5", " 40"]),
    ("There are sixty seconds in a", " minute", [" hour", " day", " week"]),
    # sequence and ordering
    ("The days of the week in order begin Monday, Tuesday,", " Wednesday",
     [" Friday", " Sunday", " January"]),
    ("After spring comes", " summer", [" winter", " autumn", " Tuesday"]),
    ("The letters of the alphabet begin A, B,", " C", [" D", " Z", " Q"]),
    ("Counting down from three: three, two,", " one", [" four", " zero point five", " seven"]),
    # domain vocabulary
    ("In programming, a variable that never changes is called a", " constant",
     [" function", " loop", " pointer"]),
    ("A doctor who specialises in the heart is a", " cardiologist",
     [" dermatologist", " neurologist", " podiatrist"]),
    ("The study of earthquakes is called", " seismology",
     [" ecology", " geology of stars", " astronomy"]),
    ("Bread is made by baking", " dough", [" stone", " glass", " metal"]),
    ("A group of wolves is called a", " pack", [" school", " flock", " swarm"]),
    ("The opposite of ancient is", " modern", [" enormous", " quiet", " circular"]),
]


# --- ASTROLABE: instruction following, checked programmatically ------------
# Every item is scored by a rule, never by a judge model. A benchmark whose score
# depends on another model's opinion cannot be reproduced, and at this model scale
# the judge would be doing most of the work.

ASTROLABE_ITEMS = [
    {"id": "echo_word", "prompt": "Reply with exactly one word: BLUE",
     "check": "contains_word", "arg": "blue", "max_tokens": 24},
    {"id": "count_words", "prompt": "Write a sentence about the sea using exactly five words.",
     "check": "word_count", "arg": 5, "max_tokens": 40},
    {"id": "list_three", "prompt": "List three colours, one per line.",
     "check": "min_lines", "arg": 3, "max_tokens": 48},
    {"id": "yes_no", "prompt": "Is the sky blue on a clear day? Answer yes or no.",
     "check": "starts_with_any", "arg": ["yes", "no"], "max_tokens": 16},
    {"id": "arith", "prompt": "What is 7 plus 8? Give only the number.",
     "check": "contains_word", "arg": "15", "max_tokens": 16},
    {"id": "capital", "prompt": "What is the capital of Japan?",
     "check": "contains_word", "arg": "tokyo", "max_tokens": 32},
    {"id": "uppercase", "prompt": "Write the word 'hello' in capital letters.",
     "check": "contains_word", "arg": "HELLO", "max_tokens": 24, "case_sensitive": True},
    {"id": "refuse_length", "prompt": "Say hello.",
     "check": "shorter_than", "arg": 60, "max_tokens": 64},
    {"id": "json_shape", "prompt": 'Reply with JSON only: {"ok": true}',
     "check": "contains_word", "arg": "ok", "max_tokens": 32},
    {"id": "no_echo", "prompt": "What colour is grass?",
     "check": "not_echo", "arg": None, "max_tokens": 32},
    {"id": "stop", "prompt": "Name one fruit.",
     "check": "stops_cleanly", "arg": None, "max_tokens": 64},
    {"id": "follow_format", "prompt": "Complete the pattern: 2, 4, 6, 8,",
     "check": "contains_word", "arg": "10", "max_tokens": 24},
]


# --- NADIR: degeneracy probes ----------------------------------------------
# Open-ended prompts with no correct answer. What is being measured is whether the
# model can produce a hundred tokens without falling into a loop.

NADIR_PROMPTS = [
    "Describe a morning in a small harbour town.",
    "Explain why the sky changes colour at sunset.",
    "Tell me about a machine that measures time.",
    "What happens when you leave bread out for a week?",
    "Write a short paragraph about walking in the rain.",
    "Summarise how a lighthouse works.",
]
