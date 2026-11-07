"""
Mini-Fab configuration — the single source of truth for the reference simulator,
and later for the tensor port and the CP-SAT model.

Values are transcribed from Kempf (1994) as reproduced in Tsakalis's ASU slides
(docs/Tsakalis main.pdf, slides 3-5) and from Tsakalis, Flores-Godoy & Rodriguez,
ETFA '97 (docs/Tsakalis 97.pdf). The transcription and its checks are in the
vault note `minifab_spec`.

Times are written in MINUTES exactly as the spec gives them, and converted to
integer ticks once, here. If a tick length does not divide a time, building the
config raises instead of rounding, so no silent discretisation error can enter.

Only Tier 1 is built. Tier 1 is the simplification [T97] simulated, and the one
its published results refer to:
  - one product (P_a); no test wafers
  - setups on E for a STEP change only (10 min); no product setups
  - preventive maintenance of 30 min per 12-hour shift on every machine
    (switchable, and OFF by default for now: pass pm=True to reproduce [T97])
  - no emergency maintenance, operators, maintenance tech, transport, or
    buffer limits
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, List, Optional

SHIFT_MINUTES = 720                      # 1 shift = 12 hours

# ---------------------------------------------------------------------------
# The specification, in minutes, exactly as written
# ---------------------------------------------------------------------------
# step: (station, eligible machines, load, process, unload)
STEP_SPEC = {
    1: ("diffusion", ("A", "B"), 20, 225, 40),
    2: ("implant",   ("C", "D"), 15,  30, 15),
    3: ("litho",     ("E",),     10,  55, 10),
    4: ("implant",   ("C", "D"), 15,  50, 15),
    5: ("diffusion", ("A", "B"), 20, 255, 40),
    6: ("litho",     ("E",),     10,  10, 10),
}
STEPS = tuple(sorted(STEP_SPEC))

# machine: (station, most lots in one run)
MACHINE_SPEC = {
    "A": ("diffusion", 3),
    "B": ("diffusion", 3),
    "C": ("implant", 1),
    "D": ("implant", 1),
    "E": ("litho", 1),
}
# The order machines decide in within a tick. Twins share buffers, so the
# second twin sees what the first one left. Fixed so the tensor port can copy it.
MACHINE_ORDER = ("A", "B", "C", "D", "E")

# E setups: Kempf gives 10 / 5 / 12 min for step / product / step-and-product
# changes. Tier 1 has one product, so only the step change can occur.
E_SETUP_STEP_MINUTES = 10
E_SETUP_PRODUCT_MINUTES = 5          # Tier 2
E_SETUP_BOTH_MINUTES = 12            # Tier 2 (not a multiple of 5)

PM_MINUTES_PER_SHIFT_TIER1 = 30      # [T97]: 30 min every 12-hour shift, all machines
PM_MIN_GAP_MINUTES = 6 * 60          # Kempf: window opens at the later of shift
                                     # start and last PM + 6 h

# Anchors published in [T97], used by the verification script.
MIN_TPT_MINUTES = 845
BOTTLENECK_LOTS_PER_SHIFT = 6.38

STARTS_PER_WEEK = {"P_a": 51, "P_b": 30, "TW": 3}   # Kempf; Tier 2


def _ticks(minutes: int, tick_minutes: int, what: str) -> int:
    if minutes % tick_minutes:
        raise ValueError(f"{what} = {minutes} min is not a whole number of "
                         f"{tick_minutes}-min ticks; refusing to round")
    return minutes // tick_minutes


def constant_rate_schedule(lots_per_shift: float, horizon: int,
                           shift_ticks: int) -> List[int]:
    """Release `lots_per_shift` lots per shift as evenly as integers allow.

    Cumulative releases by the end of tick t are floor((t+1) * r / shift_ticks),
    computed in exact rational arithmetic so 6.18 lots/shift over 43,200 ticks
    releases exactly 1,854 lots and never drifts. [T97]'s control-theoretic
    policies use constant release rates like this.
    """
    r = Fraction(str(lots_per_shift))
    out, prev = [], 0
    for t in range(horizon):
        cum = math.floor(Fraction(t + 1) * r / shift_ticks)
        out.append(cum - prev)
        prev = cum
    return out


def single_lot_schedule(horizon: int, at_tick: int = 0) -> List[int]:
    sched = [0] * horizon
    sched[at_tick] = 1
    return sched


def make_minifab_config(
    tier: int = 1,
    tick_minutes: int = 5,
    horizon_shifts: int = 300,
    release_rate_per_shift: Optional[float] = None,
    release_schedule: Optional[List[int]] = None,
    setups: bool = True,
    cold_start_setup: bool = True,
    pm: bool = False,
    revenue_per_lot: float = 100.0,
    wip_holding_per_lot_tick: float = 0.0,
) -> Dict[str, Any]:
    """
    Build the Mini-Fab config.

    tick_minutes       5, as in the ASU simulator [T97]. Every Tier 1 time divides.
    horizon_shifts     300 shifts (150 days), [T97]'s run length.
    release_*          lots released into b1: a constant rate, or an explicit
                       per-tick schedule. Neither given means no releases.
    setups             False removes E's setups entirely (used to check the
                       845-minute flow time, which excludes them).
    cold_start_setup   True treats E's very first run as a changeover, matching
                       the existing two-product config's convention for a tool
                       that has never run. It affects at most one run per horizon.
    pm                 False (the default for now) removes preventive maintenance
                       entirely: no PM jobs, windows or deadlines, and every
                       machine is available every tick. Nothing else changes.
                       True adds [T97]'s 30 min per shift on every machine, and is
                       needed to reproduce its 6.38 lots/shift and pull results.
                       Without PM, E's ceiling is 700 / 105 = 6.67 lots/shift.
    revenue_per_lot,
    wip_holding_per_lot_tick
                       Our addition, not part of the Mini-Fab spec: profit =
                       revenue on outs minus holding on every lot in the system.
                       By Little's law, sweeping the holding price traces the
                       output-vs-throughput-time trade-off [T97] plots.
    """
    if tier != 1:
        raise NotImplementedError("only Tier 1 is built; see the reference "
                                  "simulator note for the Tier 2 plan")
    if SHIFT_MINUTES % tick_minutes:
        raise ValueError("tick must divide a 12-hour shift")

    tk = tick_minutes
    shift_ticks = SHIFT_MINUTES // tk
    H = int(horizon_shifts) * shift_ticks

    steps: Dict[int, Dict[str, Any]] = {}
    for s, (station, machines, load, proc, unload) in STEP_SPEC.items():
        steps[s] = {
            "station": station,
            "machines": machines,
            "load": _ticks(load, tk, f"S{s} load"),
            "process": _ticks(proc, tk, f"S{s} process"),
            "unload": _ticks(unload, tk, f"S{s} unload"),
        }
        # In Tier 1 nothing else (no operators) can stretch a run, so the
        # machine is busy for exactly load + process + unload.
        steps[s]["occupancy"] = steps[s]["load"] + steps[s]["process"] + steps[s]["unload"]

    machines: Dict[str, Dict[str, Any]] = {}
    for m in MACHINE_ORDER:
        station, max_batch = MACHINE_SPEC[m]
        machines[m] = {
            "station": station,
            "max_batch": max_batch,
            "steps": tuple(s for s in STEPS if m in STEP_SPEC[s][1]),
        }

    # setup_times[machine][previous completed step][next step], in ticks.
    # Same shape as the existing two-product config (`None` = never run).
    e_step = _ticks(E_SETUP_STEP_MINUTES, tk, "E step-change setup") if setups else 0
    setup_times: Dict[str, Dict[Any, Dict[int, int]]] = {}
    for m, spec in machines.items():
        own = spec["steps"]
        table: Dict[Any, Dict[int, int]] = {}
        for prev in (None,) + own:
            row = {}
            for nxt in own:
                if m != "E":
                    row[nxt] = 0
                elif prev is None:
                    row[nxt] = e_step if cold_start_setup else 0
                else:
                    row[nxt] = 0 if prev == nxt else e_step
            table[prev] = row
        setup_times[m] = table

    if release_schedule is not None:
        sched = [int(x) for x in release_schedule]
        if len(sched) != H:
            raise ValueError(f"release_schedule has {len(sched)} ticks, horizon is {H}")
    elif release_rate_per_shift is not None:
        sched = constant_rate_schedule(release_rate_per_shift, H, shift_ticks)
    else:
        sched = [0] * H

    return {
        "tier": tier,
        "tick_minutes": tk,
        "shift_ticks": shift_ticks,
        "horizon": H,
        "products": [0],                       # P_a only in Tier 1
        "steps": steps,
        "machines": machines,
        "machine_order": MACHINE_ORDER,
        "setup_times": setup_times,
        "pm": ({"duration": _ticks(PM_MINUTES_PER_SHIFT_TIER1, tk, "PM"),
                "min_gap": _ticks(PM_MIN_GAP_MINUTES, tk, "PM gap")} if pm else None),
        "release_schedule": sched,
        "release_rate_per_shift": release_rate_per_shift,
        "econ": {"revenue_per_lot": float(revenue_per_lot),
                 "wip_holding_per_lot_tick": float(wip_holding_per_lot_tick)},
        "anchors": {"min_tpt_minutes": MIN_TPT_MINUTES,
                    "bottleneck_lots_per_shift": BOTTLENECK_LOTS_PER_SHIFT},
    }
