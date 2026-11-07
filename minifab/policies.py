"""
Dispatching rules for the Mini-Fab reference simulator.

A policy turns the state the fab exposes into one action per machine:
None (idle) or (step, number of lots). Machines are considered in the fab's
fixed order, and the policy keeps its own count of lots it has already handed to
an earlier machine this tick. Without that, twins A and B would both claim the
same three lots from b1 and B's request would be rejected as invalid.

Rules (all deterministic):

  priority   which eligible buffer a free machine serves
    "fifo"   the buffer whose head lot has waited longest
    "lbfs"   last buffer first served: the later step first (S5 over S1, S4 over
             S2, S6 over S3). Pulls work towards the end of the line.
    "fbfs"   first buffer first served: the earlier step first

  min_batch  A and B start only when at least this many lots are waiting
             (1 = start with whatever is there; 3 = full batches only).
             They always take as many as they can, up to 3.

  e_campaign E stays on its current step for up to this many consecutive runs
             while that step has work, then reconsiders by `priority`. None means
             E reconsiders every run. Campaigns trade setups against waiting.

[T97] does not define its "simple pull policy" precisely; LBFS is the standard
pull rule for re-entrant lines, and is the default here for that reason.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class DispatchPolicy:
    def __init__(self, cfg: Dict[str, Any], priority: str = "lbfs",
                 min_batch: int = 1, e_campaign: Optional[int] = None):
        if priority not in ("fifo", "lbfs", "fbfs"):
            raise ValueError(f"unknown priority {priority!r}")
        self.cfg = cfg
        self.priority = priority
        self.min_batch = int(min_batch)
        self.e_campaign = e_campaign
        self._e_run_step: Optional[int] = None    # step of E's current campaign
        self._e_run_count = 0

    def _rank(self, steps, avail, head_entry):
        cands = [s for s in steps if avail[s] > 0]
        if self.priority == "lbfs":
            return sorted(cands, reverse=True)
        if self.priority == "fbfs":
            return sorted(cands)
        # fifo: oldest head lot first; ties go to the later step
        return sorted(cands, key=lambda s: (head_entry[s][0], -s))

    def decide(self, state: Dict[str, Any]) -> Dict[str, Optional[Tuple[int, int]]]:
        avail = {s: n[0] for s, n in state["buffers"].items()}
        head = state["head_entry"]
        actions: Dict[str, Optional[Tuple[int, int]]] = {}
        for m in state["order"]:
            tool = state["tools"][m]
            actions[m] = None
            if not tool["free"]:
                continue
            ranked = self._rank(tool["steps"], avail, head)

            if m == "E" and self.e_campaign is not None and self._e_run_step is not None:
                s = self._e_run_step
                if avail.get(s, 0) > 0 and self._e_run_count < self.e_campaign:
                    ranked = [s] + [x for x in ranked if x != s]

            for s in ranked:
                need = self.min_batch if tool["max_batch"] > 1 else 1
                if avail[s] < need:
                    continue
                k = min(tool["max_batch"], avail[s])
                actions[m] = (s, k)
                avail[s] -= k
                if m == "E":
                    if s == self._e_run_step:
                        self._e_run_count += 1
                    else:
                        self._e_run_step, self._e_run_count = s, 1
                break
        return actions


class Commander(DispatchPolicy):
    """The built-in default, mirroring the role of `Commander` in simplefab/sim.py:
    LBFS pull with full batches on A and B, and no E campaigns.

    Full batches, because starting A/B with whatever is waiting is unstable even at
    5.5 lots/shift: LBFS then runs step 5 with 1.5 lots on average, and A/B overtake
    E as the bottleneck at about 4.5 lots/shift (their time budget at those batch
    sizes is 4.52; the simulator measures 4.54). With full batches this rule
    reproduces [T97]'s pull experiment: stable at 6.18 lots/shift, unstable at 6.36
    with buffer 3 growing, because roughly four setups a shift cut E below 6.36.
    Those are [T97]'s rates, with PM on. With PM off the same happens one capacity
    step up: stable at 6.36, unstable at 6.64."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg, priority="lbfs", min_batch=3, e_campaign=None)


def make_policy(cfg: Dict[str, Any], name: str) -> DispatchPolicy:
    """Names like 'lbfs', 'fifo-full', 'lbfs-camp6', 'fifo-full-camp4'."""
    parts = name.split("-")
    priority = parts[0]
    min_batch = 3 if "full" in parts else 1
    camp = next((int(p[4:]) for p in parts if p.startswith("camp")), None)
    return DispatchPolicy(cfg, priority=priority, min_batch=min_batch, e_campaign=camp)
