"""
MiniFabSpec — static description of the Mini-Fab for the tensor simulator.

Derived from `minifab.config.make_minifab_config`, so the parameters have one
source of truth, exactly as `twoprod/spec.py` derives from the SimpleFab config.

What is genuinely new relative to the serial line:

  * Work waits per STEP, not per machine: buffers b1..b6. A machine serves two
    steps, so the machine-to-step relation is a table, `step_of[m, k]`: the step
    served by machine m in its slot k (k = 0, 1, in ascending step order).
    Re-entrancy lives entirely in this table; routing is always step s -> s+1.
  * The register is tagged by slot, so a finished batch on A knows whether it was
    step 1 or step 5. `occ[m, k]` is the busy time of that step on that machine.
  * Batch size is part of the action. Each machine's options are enumerated in
    exactly the order `minifab.sim.action_space` uses: idle, then (step, lots)
    for each eligible step in ascending order and lots = 1..max_batch. Machines
    with fewer than NOPT options are padded with options that request nothing.
  * `setup[m, prev, k]` is sized by the machine's LAST COMPLETED slot, with
    middle index 0 = never run (the cold start), as in the two-product spec.

Register length must cover setup + occupancy: a job started at tick t completes
at t + setup + occ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MiniFabSpec:
    # ---- structure ----
    H: int
    shift_ticks: int
    machines: Tuple[str, ...]    # decision order, A..E
    n_steps: int                 # 6
    n_products: int              # 1 in Tier 1
    n_slots: int                 # eligible steps per machine (2)
    n_opts: int                  # options per machine after padding (7)
    reg_len: int                 # max(setup + occ) + 1

    # ---- per machine ----
    step_of: np.ndarray          # [M, K] step index 0..S-1 served in slot k
    occ: np.ndarray              # [M, K] busy ticks of that step
    max_batch: np.ndarray        # [M]
    setup: np.ndarray            # [M, K+1, K] setup ticks; middle 0 = never run
    opt_slot: np.ndarray         # [M, NOPT] slot of option o, -1 = requests nothing
    opt_lots: np.ndarray         # [M, NOPT] lots requested by option o
    n_valid_opts: np.ndarray     # [M] real options (7 for A/B, 3 for C/D/E)

    # ---- preventive maintenance ----
    pm_on: bool
    pm_duration: int             # ticks
    pm_min_gap: int              # ticks between PM starts (window rule)

    # ---- releases and economics ----
    releases: np.ndarray         # [H] lots into b1 per tick (the config's schedule)
    revenue_per_lot: float
    wip_holding: float           # per lot per tick, on every lot in the system

    @property
    def M(self) -> int:
        return len(self.machines)


def spec_from_config(cfg: Dict[str, Any]) -> MiniFabSpec:
    if cfg.get("tier", 1) != 1:
        raise NotImplementedError("only Tier 1 is tensorised")
    machines = tuple(cfg["machine_order"])
    steps_sorted = tuple(sorted(cfg["steps"]))
    S = len(steps_sorted)
    if steps_sorted != tuple(range(1, S + 1)):
        raise ValueError(f"steps must be 1..S, got {steps_sorted}")
    P = len(cfg["products"])

    K = {len(cfg["machines"][m]["steps"]) for m in machines}
    if len(K) != 1:
        raise ValueError("every machine must serve the same number of steps")
    K = K.pop()
    max_batch = np.array([cfg["machines"][m]["max_batch"] for m in machines], dtype=np.int32)
    nopt = 1 + K * int(max_batch.max())

    M = len(machines)
    step_of = np.zeros((M, K), dtype=np.int32)
    occ = np.zeros((M, K), dtype=np.int32)
    setup = np.zeros((M, K + 1, K), dtype=np.int32)
    opt_slot = np.full((M, nopt), -1, dtype=np.int32)
    opt_lots = np.zeros((M, nopt), dtype=np.int32)
    n_valid = np.zeros(M, dtype=np.int32)
    for i, m in enumerate(machines):
        steps = tuple(cfg["machines"][m]["steps"])
        if list(steps) != sorted(steps):
            raise ValueError(f"{m}: steps must be ascending")
        table = cfg["setup_times"][m]
        for k, s in enumerate(steps):
            step_of[i, k] = s - 1
            occ[i, k] = cfg["steps"][s]["occupancy"]
            setup[i, 0, k] = table[None][s]
            for j, prev in enumerate(steps):
                setup[i, j + 1, k] = table[prev][s]
        o = 1
        for k in range(K):
            for lots in range(1, int(max_batch[i]) + 1):
                opt_slot[i, o], opt_lots[i, o] = k, lots
                o += 1
        n_valid[i] = o

    reg_len = int((setup.max(axis=1) + occ).max()) + 1
    pm = cfg["pm"]
    return MiniFabSpec(
        H=int(cfg["horizon"]),
        shift_ticks=int(cfg["shift_ticks"]),
        machines=machines,
        n_steps=S,
        n_products=P,
        n_slots=K,
        n_opts=nopt,
        reg_len=reg_len,
        step_of=step_of,
        occ=occ,
        max_batch=max_batch,
        setup=setup,
        opt_slot=opt_slot,
        opt_lots=opt_lots,
        n_valid_opts=n_valid,
        pm_on=pm is not None,
        pm_duration=int(pm["duration"]) if pm is not None else 0,
        pm_min_gap=int(pm["min_gap"]) if pm is not None else 0,
        releases=np.asarray(cfg["release_schedule"], dtype=np.int32),
        revenue_per_lot=float(cfg["econ"]["revenue_per_lot"]),
        wip_holding=float(cfg["econ"]["wip_holding_per_lot_tick"]),
    )


def option_index(spec: MiniFabSpec, m: int, action: Optional[Tuple[int, ...]]) -> int:
    """Reference action (None or (step, lots)) on machine m -> option index."""
    if action is None:
        return 0
    step, lots = int(action[0]), int(action[1])
    k = int(np.flatnonzero(spec.step_of[m] == step - 1)[0])
    return 1 + k * int(spec.max_batch[m]) + lots - 1


def option_action(spec: MiniFabSpec, m: int, o: int) -> Optional[Tuple[int, int]]:
    """Option index on machine m -> reference action (None or (step, lots))."""
    k = int(spec.opt_slot[m, o])
    if k < 0:
        return None
    return int(spec.step_of[m, k]) + 1, int(spec.opt_lots[m, o])
