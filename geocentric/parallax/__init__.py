"""PARALLAX — the built-in benchmark suite.

Parallax is how you measure the distance to a star: observe it from two positions
and read the angle between them. Every probe here works the same way — it puts the
model in two situations that should differ in a predictable direction and measures
whether they do.

    ZENITH     how well the model models held-out text, in bits per byte
    MERIDIAN   whether more context makes its predictions better
    SEXTANT    whether it knows things, by ranking a right answer above wrong ones
    ASTROLABE  whether it does what it was asked (instruction-tuned models only)
    NADIR      how badly it degenerates when left to run
    ORBIT      how fast it runs and what it costs to run
    PRISM      whether a multimodal model is actually looking at the image
"""
from geocentric.parallax.suite import (
    PARALLAX_VERSION,
    ProbeResult,
    SuiteResult,
    run_parallax,
)
from geocentric.parallax.report import write_report

__all__ = ["PARALLAX_VERSION", "ProbeResult", "SuiteResult", "run_parallax", "write_report"]
