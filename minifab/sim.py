"""
Reference discrete-time simulator for the Intel Mini-Fab, Tier 1.

Built the same way as `simplefab/sim.py`: integer ticks, one `run_step` per
tick, machine objects that hold their own job and event log, buffers that keep
the identity of every lot as FIFO lists, an `actions_override` that replaces the
built-in policy, invalid requests counted rather than raised, and a cost log.

This file is the yardstick, not the fast path. It favours being obviously right
over being quick: lots are real objects with full histories, so every claim the
tensor port makes can later be checked against it lot by lot.

Tick contract, in order (see the vault note for the reasoning behind each):

  1. Release      new lots enter buffer b1
  2. Completions  every machine whose job has finished hands its lots to the
                  next step's buffer (or to outs after step 6)
  3. Maintenance  if PM is on, an idle machine whose preventive-maintenance
                  window is open starts its PM; PM takes priority over a new run
  4. Decide       the policy sees the state after 1-3 and picks, per machine,
                  idle or (step, number of lots)
  5. Start        machines start in the fixed order A, B, C, D, E; each one sees
                  the buffers as the previous one left them, which is how twin
                  machines share a buffer without double-claiming lots
  6. Accrue       work in process, revenue on outs, holding cost

A run started at tick t and lasting d ticks occupies the machine on ticks
t .. t+d-1 and completes at the start of tick t+d. Because completions come
before starts, the machine can begin its next run at tick t+d.

Departure from the existing simulator, on purpose: there, the built-in
Commander decides on the state *before* completions, so a rule cannot reuse a
machine that frees up this tick. Here every policy decides after completions,
which is the order the tensor tick uses. Open-loop action tapes behave the same
under either order, because their requests are validated after completions in
both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass
class Lot:
    lot_id: int
    product: int
    release_tick: int
    # step -> (tick, sequence number) at which the lot joined that step's buffer
    entries: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    # step -> (machine, start tick, setup ticks, finish tick)
    runs: Dict[int, Tuple[str, int, int, int]] = field(default_factory=dict)
    out_tick: Optional[int] = None


@dataclass(frozen=True)
class Job:
    """One occupation of a machine: a run of lots, or a preventive maintenance."""
    kind: str                     # "RUN" or "PM"
    machine: str
    step: Optional[int]           # None for PM
    product: Optional[int]
    lots: Tuple[int, ...]         # empty for PM
    start: int                    # tick the machine becomes occupied (setup starts here)
    setup: int                    # setup ticks at the front of the job
    finish: int                   # completion tick (machine free again at this tick)


# ---------------------------------------------------------------------------
# machines
# ---------------------------------------------------------------------------
class Tool:
    """
    One Mini-Fab machine. Batch tools (A, B) and single-lot tools (C, D, E) are the
    same class; `max_batch` is the only difference, which keeps twin handling uniform.

    `last_step` is the step of the last COMPLETED run, mirroring
    `Machine.last_finished_product` in the existing simulator, and it is what
    sizes E's setup.

    Event log: every Job in start order (runs and PMs).
    """

    def __init__(self, name: str, steps: Tuple[int, ...], max_batch: int,
                 occupancy: Dict[int, int], setup_times: Dict[Any, Dict[int, int]]):
        self.name = name
        self.steps = tuple(steps)
        self.max_batch = int(max_batch)
        self.occupancy = dict(occupancy)
        self.setup_times = setup_times
        self.busy: bool = False
        self.job: Optional[Job] = None
        self.finish_tick: int = 0
        self.last_step: Optional[int] = None
        self.event_log: List[Job] = []
        # preventive maintenance bookkeeping
        self.last_pm_start: Optional[int] = None
        self.pm_shift_done: int = -1

    def setup_for(self, step: int) -> int:
        return int(self.setup_times[self.last_step][step])

    def process(self, step: int, product: int, lots: List[int], current_time: int) -> Job:
        setup = self.setup_for(step)
        finish = int(current_time) + setup + int(self.occupancy[step])
        job = Job("RUN", self.name, int(step), int(product), tuple(lots),
                  int(current_time), setup, finish)
        self._occupy(job)
        return job

    def maintain(self, duration: int, current_time: int, shift: int) -> Job:
        job = Job("PM", self.name, None, None, (), int(current_time), 0,
                  int(current_time) + int(duration))
        self._occupy(job)
        self.last_pm_start = int(current_time)
        self.pm_shift_done = int(shift)
        return job

    def _occupy(self, job: Job) -> None:
        self.busy = True
        self.job = job
        self.finish_tick = job.finish
        self.event_log.append(job)

    def update(self, current_time: int) -> Optional[Job]:
        if self.busy and current_time >= self.finish_tick:
            job = self.job
            self.busy = False
            self.job = None
            if job.kind == "RUN":
                self.last_step = job.step
            return job
        return None


# ---------------------------------------------------------------------------
# the fab
# ---------------------------------------------------------------------------
def action_space(cfg: Dict[str, Any]) -> Dict[str, List[Optional[Tuple[int, int]]]]:
    """Every option each machine has at a tick: idle, or (step, number of lots).

    A, B: idle + {S1, S5} x {1, 2, 3}  -> 7 options
    C, D: idle + {S2, S4}              -> 3 options
    E:    idle + {S3, S6}              -> 3 options

    This is also the categorical action the tensor port will relax.
    """
    space: Dict[str, List[Optional[Tuple[int, int]]]] = {}
    for m in cfg["machine_order"]:
        spec = cfg["machines"][m]
        opts: List[Optional[Tuple[int, int]]] = [None]
        for s in spec["steps"]:
            for k in range(1, spec["max_batch"] + 1):
                opts.append((s, k))
        space[m] = opts
    return space


class MiniFab:
    """
    The five-machine, six-step re-entrant Mini-Fab, Tier 1.

    buffers[s][p]  FIFO list of lot ids waiting for step s, product p (one
                   buffer per step, b1..b6, as in both views of the spec)
    lots           every lot ever released, with its full history
    outs           lot ids that finished step 6, in completion order

    Actions: {"A": (step, k) | None, ..., "E": ...}. A third element may give the
    product; Tier 1 has only product 0.
    """

    def __init__(self, cfg: Dict[str, Any], policy: Any = None):
        self.cfg = cfg
        self.H = int(cfg["horizon"])
        self.shift_ticks = int(cfg["shift_ticks"])
        self.steps = tuple(sorted(cfg["steps"]))
        self.products = list(cfg["products"])
        self.order = tuple(cfg["machine_order"])
        self.release_schedule = cfg["release_schedule"]
        self.pm = cfg["pm"]
        self.econ = cfg["econ"]

        self.tools: Dict[str, Tool] = {}
        for m in self.order:
            spec = cfg["machines"][m]
            occ = {s: cfg["steps"][s]["occupancy"] for s in spec["steps"]}
            self.tools[m] = Tool(m, spec["steps"], spec["max_batch"], occ,
                                 cfg["setup_times"][m])

        self.buffers: Dict[int, List[List[int]]] = {
            s: [[] for _ in self.products] for s in self.steps}
        self.lots: Dict[int, Lot] = {}
        self.outs: List[int] = []
        self._next_lot = 0
        self._seq = 0

        self.costs = {"revenue": 0.0, "wip": 0.0}
        self.cost_log: List[Dict[str, Any]] = []
        # per tick, end of tick: lots waiting in b1..b6, and lots inside machines
        self.buffer_log: List[Tuple[int, ...]] = []
        self.inproc_log: List[int] = []
        self.invalid_total = 0
        self.invalid_log: List[Tuple[int, str, Any]] = []
        self.pm_violations: List[Tuple[str, int, str]] = []

        if policy is None:
            from minifab.policies import Commander
            policy = Commander(cfg)
        self.policy = policy
        self.logging_enabled = False
        self.event_log: List[Dict[str, Any]] = []
        self._t = 0

    # ------------------------------------------------------------------
    # lot movement
    # ------------------------------------------------------------------
    def _enter(self, lot_id: int, step: int, t: int) -> None:
        lot = self.lots[lot_id]
        self.buffers[step][lot.product].append(lot_id)
        lot.entries[step] = (int(t), self._seq)
        self._seq += 1

    def _release(self, t: int, product: int = 0) -> None:
        lot = Lot(self._next_lot, int(product), int(t))
        self.lots[lot.lot_id] = lot
        self._next_lot += 1
        self._enter(lot.lot_id, self.steps[0], t)

    def _complete(self, job: Job, t: int) -> None:
        last = self.steps[-1]
        for lid in job.lots:                       # batch order preserves FIFO
            if job.step == last:
                self.lots[lid].out_tick = int(t)
                self.outs.append(lid)
                self.costs["revenue"] += self.econ["revenue_per_lot"]
            else:
                self._enter(lid, job.step + 1, t)

    # ------------------------------------------------------------------
    # preventive maintenance
    # ------------------------------------------------------------------
    def _maybe_start_pm(self, tool: Tool, t: int) -> None:
        """Start PM on an idle machine whose window is open and whose PM for this
        shift is not yet done. Window: the later of shift start and last PM + 6 h."""
        if self.pm is None or tool.busy:
            return
        shift = t // self.shift_ticks
        if tool.pm_shift_done == shift:
            return
        opens = shift * self.shift_ticks
        if tool.last_pm_start is not None:
            opens = max(opens, tool.last_pm_start + self.pm["min_gap"])
        if t < opens:
            return
        job = tool.maintain(self.pm["duration"], t, shift)
        if job.finish > (shift + 1) * self.shift_ticks:
            self.pm_violations.append((tool.name, shift, "PM finishes after end of shift"))

    def _check_pm_deadline(self, t: int) -> None:
        if self.pm is None or (t + 1) % self.shift_ticks:
            return
        shift = t // self.shift_ticks
        for tool in self.tools.values():
            if tool.pm_shift_done != shift:
                self.pm_violations.append((tool.name, shift, "no PM this shift"))

    # ------------------------------------------------------------------
    # starts
    # ------------------------------------------------------------------
    @staticmethod
    def _parse(action: Any) -> Tuple[int, int, int]:
        if len(action) == 2:
            return int(action[0]), int(action[1]), 0
        return int(action[0]), int(action[1]), int(action[2])

    def _try_start(self, m: str, action: Any, t: int) -> bool:
        tool = self.tools[m]
        try:
            step, k, prod = self._parse(action)
        except (TypeError, ValueError):
            return False
        if tool.busy:
            return False
        if step not in tool.steps:
            return False
        if not 1 <= k <= tool.max_batch:
            return False
        if prod not in self.products:
            return False
        buf = self.buffers[step][prod]
        if len(buf) < k:
            return False
        lots = [buf.pop(0) for _ in range(k)]
        job = tool.process(step, prod, lots, t)
        for lid in lots:
            self.lots[lid].runs[step] = (m, job.start, job.setup, job.finish)
        return True

    # ------------------------------------------------------------------
    # state / masks
    # ------------------------------------------------------------------
    def _state_dict(self, t: int) -> Dict[str, Any]:
        """What a policy sees: buffer contents with the entry tick of each head lot,
        and for each machine whether it is free, its last step and time remaining."""
        buffers = {s: [len(q) for q in self.buffers[s]] for s in self.steps}
        head_entry = {s: [self.lots[q[0]].entries[s][0] if q else None for q in self.buffers[s]]
                      for s in self.steps}
        tools = {m: {"free": not tool.busy,
                     "steps": tool.steps,
                     "max_batch": tool.max_batch,
                     "last_step": tool.last_step,
                     "remaining": max(tool.finish_tick - t, 0) if tool.busy else 0,
                     "in_pm": bool(tool.busy and tool.job.kind == "PM")}
                 for m, tool in self.tools.items()}
        return {"Time": t, "shift_ticks": self.shift_ticks, "order": self.order,
                "buffers": buffers, "head_entry": head_entry, "tools": tools}

    def compute_action_mask(self, t: int) -> Dict[str, np.ndarray]:
        """Per machine, which options are feasible on the current state, judged
        machine by machine. It does not know what an earlier twin will claim this
        tick, so a feasible option for B can become invalid after A starts."""
        space = action_space(self.cfg)
        mask: Dict[str, np.ndarray] = {}
        for m, opts in space.items():
            tool = self.tools[m]
            v = np.zeros(len(opts), dtype=np.int8)
            v[0] = 1
            if not tool.busy:
                for i, o in enumerate(opts[1:], start=1):
                    s, k = o
                    if len(self.buffers[s][0]) >= k:
                        v[i] = 1
            mask[m] = v
        return mask

    # ------------------------------------------------------------------
    # one tick
    # ------------------------------------------------------------------
    def profit_total(self) -> float:
        return self.costs["revenue"] - self.costs["wip"]

    def run_step(self, current_time: int,
                 actions_override: Optional[Dict[str, Any]] = None,
                 invalid_penalty: float = 0.0) -> float:
        t = int(current_time)

        # 1. release
        if t < self.H:
            for _ in range(int(self.release_schedule[t])):
                self._release(t, product=0)

        # 2. completions, in machine order
        for m in self.order:
            job = self.tools[m].update(t)
            if job is not None and job.kind == "RUN":
                self._complete(job, t)

        # 3. preventive maintenance takes priority over new runs
        for m in self.order:
            self._maybe_start_pm(self.tools[m], t)

        # 4. decide
        state = self._state_dict(t)
        actions = self.policy.decide(state) if actions_override is None else actions_override

        prev_profit = self.profit_total()
        invalids = 0

        # 5. start, machine by machine
        for m in self.order:
            a = actions.get(m) if actions else None
            if a is None:
                continue
            if not self._try_start(m, a, t):
                invalids += 1
                self.invalid_log.append((t, m, a))
        self.invalid_total += invalids

        # 6. accrue at the end of the tick
        waiting = tuple(sum(len(q) for q in self.buffers[s]) for s in self.steps)
        inside = sum(len(tool.job.lots) for tool in self.tools.values()
                     if tool.busy and tool.job.kind == "RUN")
        self.buffer_log.append(waiting)
        self.inproc_log.append(inside)
        self.costs["wip"] += self.econ["wip_holding_per_lot_tick"] * (sum(waiting) + inside)

        self._check_pm_deadline(t)

        profit_now = self.profit_total()
        step_reward = (profit_now - prev_profit) + float(invalid_penalty) * invalids

        if self.logging_enabled:
            snap = self._state_dict(t)
            snap["actions"] = dict(actions) if actions else {}
            self.event_log.append(snap)
        self.cost_log.append({"Time": t, "revenue": self.costs["revenue"],
                              "wip": self.costs["wip"], "profit": profit_now,
                              "step_reward": step_reward, "invalid_actions": invalids,
                              "outs": len(self.outs)})
        self._t = t + 1
        return float(step_reward)

    def run(self, horizon: Optional[int] = None,
            actions: Optional[List[Dict[str, Any]]] = None) -> "MiniFab":
        """Run from the current tick to `horizon`. `actions`, if given, is an
        open-loop tape: one action dict per tick, replacing the policy."""
        H = self.H if horizon is None else int(horizon)
        for t in range(self._t, H):
            self.run_step(t, None if actions is None else actions[t])
        return self

    # ------------------------------------------------------------------
    # metrics, in the benchmark's own terms
    # ------------------------------------------------------------------
    def wip_total_log(self) -> List[int]:
        return [sum(w) + i for w, i in zip(self.buffer_log, self.inproc_log)]

    def metrics(self) -> Dict[str, Any]:
        tk = self.cfg["tick_minutes"]
        T = len(self.buffer_log)                     # ticks actually simulated
        shifts = T / self.shift_ticks if T else 0.0
        tpts = [(self.lots[i].out_tick - self.lots[i].release_tick) for i in self.outs]
        tpt_min = float(np.mean(tpts)) * tk if tpts else float("nan")
        anchor = self.cfg["anchors"]["min_tpt_minutes"]
        wip = self.wip_total_log()

        tool_time: Dict[str, Dict[str, int]] = {}
        for m, tool in self.tools.items():
            run = setup = pm = 0
            n_setups = n_runs = n_lots = 0
            for j in tool.event_log:
                lo, hi = j.start, min(j.finish, T)
                if hi <= lo:
                    continue
                if j.kind == "PM":
                    pm += hi - lo
                else:
                    s_hi = min(j.start + j.setup, T)
                    setup += max(s_hi - lo, 0)
                    run += hi - max(s_hi, lo)
                    n_runs += 1
                    n_lots += len(j.lots)
                    n_setups += 1 if j.setup > 0 else 0
            tool_time[m] = {"run": run, "setup": setup, "pm": pm,
                            "idle": T - run - setup - pm, "runs": n_runs,
                            "lots": n_lots, "setups": n_setups}

        return {
            "ticks": T,
            "shifts": shifts,
            "released": len(self.lots),
            "outs": len(self.outs),
            "outs_per_shift": len(self.outs) / shifts if shifts else float("nan"),
            "avg_tpt_minutes": tpt_min,
            "normalised_tpt": tpt_min / anchor - 1.0 if tpts else float("nan"),
            "avg_wip_lots": float(np.mean(wip)) if wip else 0.0,
            "final_buffers": self.buffer_log[-1] if self.buffer_log else None,
            "tool_time": tool_time,
            "invalid_total": self.invalid_total,
            "pm_violations": len(self.pm_violations),
            "profit": self.profit_total(),
        }
