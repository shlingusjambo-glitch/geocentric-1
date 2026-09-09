"""Serving rate limits, and the load state the interface tells users about.

Every user is paced to the same ceiling so one person cannot take the whole
machine. Under pressure the ceiling drops and the interface says so, because a
slow answer with no explanation reads as a broken product.

Two signals raise the state to "high":

  thermal   macOS reports a CPU speed limit below 100% under `pmset -g therm`.
            That is the only thermal reading available without elevated
            privileges, so it is what we use; where it is unavailable the
            signal is simply absent rather than guessed at.
  pressure  Requests arriving while the model is already answering. The engine
            serves one at a time, so a burst of these is the honest definition
            of "more demand than we can serve".

Hysteresis is deliberate: the state will not leave "high" for a minute, so the
interface does not flap two modals at someone while a queue drains.
"""
from __future__ import annotations

import subprocess
import threading
import time

NORMAL_TOKENS_PER_SECOND = 25.0
HIGH_LOAD_TOKENS_PER_SECOND = 10.0

# A burst this size within the window means people are waiting on each other.
PRESSURE_EVENTS = 3
PRESSURE_WINDOW = 60.0
MIN_HIGH_SECONDS = 60.0
THERMAL_POLL_SECONDS = 15.0


def thermal_limited(runner=subprocess.run):
    """True when macOS reports the CPU held below full speed. False if unknown."""
    try:
        result = runner(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    for line in (result.stdout or "").splitlines():
        if "CPU_Speed_Limit" in line:
            _, _, value = line.partition("=")
            try:
                return int(value.strip()) < 100
            except ValueError:
                return False
    return False


class LoadMonitor:
    def __init__(self, normal=NORMAL_TOKENS_PER_SECOND, high=HIGH_LOAD_TOKENS_PER_SECOND,
                 clock=time.monotonic, thermal=thermal_limited):
        self.normal, self.high = float(normal), float(high)
        self._clock, self._thermal = clock, thermal
        self._lock = threading.Lock()
        self._events = []           # timestamps of contended requests
        self._high_since = None     # when the state last became high
        self._last_high = None      # when pressure was last actually present
        self._thermal_checked = 0.0
        self._thermal_state = False
        self._generation = 0        # bumped on every transition

    def note_contention(self):
        """Called when a request arrives while the model is already busy."""
        with self._lock:
            self._events.append(self._clock())

    def _thermal_now(self):
        now = self._clock()
        if now - self._thermal_checked >= THERMAL_POLL_SECONDS:
            self._thermal_checked = now
            self._thermal_state = bool(self._thermal())
        return self._thermal_state

    def state(self):
        """Return the current state dict, applying hysteresis."""
        now = self._clock()
        thermal = self._thermal_now()
        with self._lock:
            self._events = [t for t in self._events if now - t <= PRESSURE_WINDOW]
            pressure = len(self._events) >= PRESSURE_EVENTS
            wants_high = thermal or pressure

            if wants_high:
                self._last_high = now
                if self._high_since is None:
                    self._high_since = now
                    self._generation += 1
            elif self._high_since is not None and now - self._last_high >= MIN_HIGH_SECONDS:
                # Counted from when pressure last existed, not from when the
                # state began, so a long busy spell does not end the moment the
                # oldest event ages out of the window.
                self._high_since = None
                self._generation += 1
                self._events.clear()

            high = self._high_since is not None
            return {
                "load": "high" if high else "normal",
                "tokens_per_second": self.high if high else self.normal,
                "reason": ("thermal" if thermal else "demand") if high else None,
                # Changes whenever the state flips, so a client that was away
                # can tell "still high" from "high again" with one integer.
                "generation": self._generation,
            }

    def tokens_per_second(self):
        return self.state()["tokens_per_second"]


def paced(chunks, tokens_per_second, sleep=time.sleep, clock=time.monotonic):
    """Yield chunks no faster than the given rate.

    Paces against elapsed time rather than sleeping a fixed amount per token, so
    a slow model is never delayed further and the cap only bites when generation
    would otherwise outrun it.
    """
    if not tokens_per_second or tokens_per_second <= 0:
        yield from chunks
        return
    interval = 1.0 / tokens_per_second
    start = clock()
    for index, chunk in enumerate(chunks, 1):
        yield chunk
        behind = start + index * interval - clock()
        if behind > 0:
            sleep(behind)


def demo():
    """Self-check: hysteresis holds, and pacing actually paces."""
    now = [0.0]
    clock = lambda: now[0]
    monitor = LoadMonitor(clock=clock, thermal=lambda: False)

    assert monitor.state()["load"] == "normal"
    assert monitor.state()["tokens_per_second"] == NORMAL_TOKENS_PER_SECOND

    for _ in range(PRESSURE_EVENTS):
        monitor.note_contention()
    high = monitor.state()
    assert high["load"] == "high" and high["reason"] == "demand"
    assert high["tokens_per_second"] == HIGH_LOAD_TOKENS_PER_SECOND

    # Still high a moment later, and the generation has not moved.
    now[0] = 5
    assert monitor.state()["generation"] == high["generation"]

    # Pressure is still present just before the window closes.
    now[0] = PRESSURE_WINDOW - 1
    assert monitor.state()["load"] == "high"

    # Now the events age out. The cool-down keeps the state steady anyway.
    now[0] = PRESSURE_WINDOW + 1
    assert monitor.state()["load"] == "high", "must not flap the moment events age out"

    now[0] = PRESSURE_WINDOW + MIN_HIGH_SECONDS + 1
    back = monitor.state()
    assert back["load"] == "normal"
    assert back["generation"] == high["generation"] + 1

    # Thermal alone is enough.
    hot = LoadMonitor(clock=clock, thermal=lambda: True)
    assert hot.state()["reason"] == "thermal"

    # Pacing: a model that generates instantly is held to the cap. Ten chunks
    # at 10/s costs about a second of sleep in total.
    fake = [0.0]
    slept = []

    def fake_sleep(seconds):
        slept.append(seconds)
        fake[0] += seconds

    chunks = list(paced(iter("abcdefghij"), 10, sleep=fake_sleep, clock=lambda: fake[0]))
    assert chunks == list("abcdefghij")
    assert abs(sum(slept) - 1.0) < 0.01, sum(slept)

    # An unpaced rate passes straight through.
    assert list(paced(iter("abc"), 0, sleep=fake_sleep)) == ["a", "b", "c"]

    # A model already slower than the cap is never delayed further.
    ticks = iter([0.0, 5.0, 10.0, 15.0])
    def must_not_sleep(seconds):
        raise AssertionError("a slow model must not be paced further")
    assert list(paced(iter("ab"), 10, sleep=must_not_sleep,
                      clock=lambda: next(ticks))) == ["a", "b"]

    print("load: ok")


if __name__ == "__main__":
    demo()
