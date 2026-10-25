"""
One place that decides what a run is called.

Schedules used to be tagged by product mix alone, which was fine while every
run sat at 80% utilisation and became a silent overwrite hazard the moment a
second utilisation was added. Tags now carry both axes, and every module that
reads or writes a schedule goes through here.

    a050_u080   ->  alpha 0.5 (50/50 mix), 80% bottleneck utilisation
"""
from __future__ import annotations
import re

_TAG = re.compile(r"^a(\d{3})_u(\d{3})$")


def tag(alpha: float, util: float) -> str:
    return f"a{int(round(alpha * 100)):03d}_u{int(round(util * 100)):03d}"


def parse(tag_str: str) -> tuple[float, float]:
    m = _TAG.match(tag_str)
    if not m:
        raise ValueError(f"not a run tag: {tag_str!r}")
    return int(m.group(1)) / 100.0, int(m.group(2)) / 100.0


def sched_name(alpha: float, util: float) -> str:
    return f"schedules_{tag(alpha, util)}.npz"


def dash_name(alpha: float, util: float, method: str, web: bool = False) -> str:
    return f"dash_{tag(alpha, util)}_{method}{'_web' if web else ''}.png"


def mix_label(alpha: float) -> str:
    pct = int(round(alpha * 100))
    return f"{pct} / {100 - pct}"


def run_label(alpha: float, util: float) -> str:
    return f"{mix_label(alpha)} mix at {util:.0%} utilisation"
