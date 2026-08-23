from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np


class BatchProcessingMachine:
    """
    Batch machine (single product):
      - requires `batch_size` items to start
      - processes a batch as one job
      - on completion, emits `batch_size` units
    Event log tuple format:
      (start_time, finish_time, batch_size)
    """
    def __init__(self, processing_time: int, batch_size: int):
        self.processing_time = int(processing_time)
        self.batch_size = int(batch_size)
        self.busy: bool = False
        self.finish_time: int = 0
        self.event_log: List[Tuple[int, int, int]] = []

    def process(self, current_time: int) -> int:
        self.busy = True
        self.finish_time = int(current_time) + self.processing_time
        self.event_log.append((int(current_time), int(self.finish_time), self.batch_size))
        return self.finish_time

    def update(self, current_time: int) -> bool:
        """Returns True if a batch just completed."""
        if self.busy and current_time >= self.finish_time:
            self.busy = False
            return True
        return False


class Machine:
    """
    Single-unit machine (single product, no setup times).
    Event log tuple format:
      (start_time, finish_time)
    """
    def __init__(self, processing_time: int):
        self.processing_time = int(processing_time)
        self.batch_size = 1          # so batch and single tools share one interface
        self.busy: bool = False
        self.finish_time: int = 0
        self.event_log: List[Tuple[int, int]] = []

    def process(self, current_time: int) -> int:
        self.busy = True
        self.finish_time = int(current_time) + self.processing_time
        self.event_log.append((int(current_time), int(self.finish_time)))
        return self.finish_time

    def update(self, current_time: int) -> bool:
        """Returns True if a unit just completed."""
        if self.busy and current_time >= self.finish_time:
            self.busy = False
            return True
        return False


def commander_decide_general(machines: List[Any], queues: Dict[str, List[int]]) -> Dict[str, bool]:
    """Greedy: run any tool that is idle and has enough input. Depth-agnostic."""
    return {
        f"machine{m}": (not mach.busy) and len(queues[f"queue{m}"]) >= mach.batch_size
        for m, mach in enumerate(machines)
    }


def commander_decide(
    machine0: BatchProcessingMachine,
    machine1: Machine,
    machine2: BatchProcessingMachine,
    machine3: Machine,
    queues: Dict[str, List[int]],
) -> Dict[str, bool]:
    """
    Simple heuristic for single-product: run if the machine is idle and
    the input queue has enough items.

    Returns dict like {"machine0": True/False, ...} where True = run.
    """
    actions: Dict[str, bool] = {
        "machine0": False,
        "machine1": False,
        "machine2": False,
        "machine3": False,
    }

    if (not machine0.busy) and len(queues["queue0"]) >= machine0.batch_size:
        actions["machine0"] = True

    if (not machine1.busy) and len(queues["queue1"]) >= 1:
        actions["machine1"] = True

    if (not machine2.busy) and len(queues["queue2"]) >= machine2.batch_size:
        actions["machine2"] = True

    if (not machine3.busy) and len(queues["queue3"]) >= 1:
        actions["machine3"] = True

    return actions


class ProductionLine:
    """
    Discrete-time simulation of a 4-machine, single-product line:

      queue0 -> M0(batch) -> queue1 -> M1(single) -> queue2 -> M2(batch) -> queue3 -> M3(single) -> (ship or inventory)

    Demand is modeled as a separate queue:
      - demand arrivals go into demand queue each step
      - when a finished unit leaves M3:
          if demand exists => ship immediately, earn revenue
          else => store as finished inventory (queue_fin)

    Economics tracked:
      revenue - (production + inventory + backorder)
    """
    ACTIONS = ("Idle", "Run")

    def __init__(self, common_cfg: Dict[str, Any]):
        pt = common_cfg["processing_times"]
        M = int(common_cfg.get("n_machines", 4))
        batch_cfg = common_cfg.get("batch_sizes", {0: 4, 1: 1, 2: 4, 3: 1})
        self.n_machines = M

        # A tool is a batch furnace if its batch size is > 1, else a single-wafer tool.
        self.machines: List[Any] = []
        for m in range(M):
            b = int(batch_cfg[m])
            self.machines.append(
                BatchProcessingMachine(pt[m][0], batch_size=b) if b > 1
                else Machine(pt[m][0])
            )
        # Legacy attribute names — 14 other modules reach for these directly.
        for m in range(min(M, 4)):
            setattr(self, f"machine{m}", self.machines[m])

        self.arrivals_schedule = common_cfg["arrivals_schedule"][0]
        self.demand_schedule = common_cfg["demand_schedule"][0]
        self.H = int(common_cfg["time_horizon"])
        self.demand_interval = int(common_cfg.get("demand_interval", 672))
        self.demand_delay = int(common_cfg.get("demand_delay", 96))

        # queues are simple lists of arrival timestamps
        self.queues: Dict[str, List[int]] = {f"queue{m}": [] for m in range(M)}
        self.queues["queue_fin"] = []    # finished inventory
        self.queues["demand"] = []       # unmet demand

        # Initialize with starting finished goods inventory
        init_fin = common_cfg.get("initial_finished_inventory", {0: 0})
        for _ in range(int(init_fin.get(0, 0))):
            self.queues["queue_fin"].append(-1)  # -1 indicates pre-existing inventory

        # logs
        self.logs: Dict[str, List[Tuple[int, int]]] = {k: [] for k in self.queues}
        self.demand_met_log: List[int] = []        # list of times when demand was met
        self.event_log: List[Dict[str, Any]] = []   # per-step snapshot (optional)

        self.total_demand = sum(self.demand_schedule)

        self.econ = {
            "revenue_per_unit": common_cfg["revenue_per_unit"][0],
            "production_cost": {m: common_cfg["production_cost"][m][0] for m in range(M)},
            "inventory_cost_per_unit": common_cfg["inventory_cost_per_unit"][0],
            "wip_cost_per_unit": {
                m: common_cfg.get("wip_cost_per_unit", {}).get(m, 0.0) for m in range(M)
            },
            "backorder_cost_per_unit": common_cfg["backorder_cost_per_unit"][0],
        }
        self.costs = {"revenue": 0.0, "production": 0.0, "inventory": 0.0, "wip": 0.0, "backorder": 0.0}
        self.cost_log: List[Dict[str, Any]] = []

        self.logging_enabled = False
        self._t = 0  # next decision time

    # ---------------------------
    # queue helpers
    # ---------------------------
    def add_item_to_queue(self, queue_name: str, current_time: int) -> None:
        self.queues[queue_name].append(int(current_time))
        self.logs[queue_name].append((len(self.queues[queue_name]), int(current_time)))

    def remove_item_from_queue(self, queue_name: str, current_time: int) -> None:
        if self.queues[queue_name]:
            self.queues[queue_name].pop(0)
            self.logs[queue_name].append((len(self.queues[queue_name]), int(current_time)))

    def _ship_or_store(self, current_time: int) -> None:
        """
        If demand exists: ship now, earn revenue.
        Else: store in finished inventory.
        """
        if self.queues["demand"]:
            self.queues["demand"].pop(0)
            self.demand_met_log.append(int(current_time))
            self.costs["revenue"] += float(self.econ["revenue_per_unit"])
            self.logs["demand"].append((len(self.queues["demand"]), int(current_time)))
        else:
            self.add_item_to_queue("queue_fin", current_time)

    # ---------------------------
    # economics
    # ---------------------------
    def profit_total(self) -> float:
        c = self.costs
        return c["revenue"] - (c["production"] + c["inventory"] + c["wip"] + c["backorder"])

    # ---------------------------
    # state / obs / mask
    # ---------------------------
    def _state_dict(self, current_time: int) -> Dict[str, Any]:
        d: Dict[str, Any] = {"Time": current_time}
        for m in range(len(self.machines)):
            d[f"Queue {m}"] = len(self.queues[f"queue{m}"])
        d["Queue Fin"] = len(self.queues["queue_fin"])
        d["Demand Queue"] = len(self.queues["demand"])
        for m, mach in enumerate(self.machines):
            d[f"Machine {m} Busy"] = int(mach.busy)
            d[f"Machine {m} TimeRemaining"] = (
                0 if not mach.busy else max(mach.finish_time - current_time, 0)
            )
        return d

    def get_observation(self, current_time: int, norm: bool = True) -> List[float]:
        """
        18-dim observation:
          0-5:   queue counts [q0, q1, q2, q3, qfin, demand]
          6-9:   machine busy flags (0 or 1)
          10-13: machine time_remaining, scaled to [0, 1]
          14:    global progress (t / H)
          15:    period progress (t % demand_interval / demand_interval)
          16:    ticks_until_next_demand (scaled to [0, 1])
          17:    upcoming_demand_qty (scaled to expected max 200)
        """
        max_units = 64.0
        machines = self.machines
        M = len(machines)
        # Batch furnaces run 16 ticks, single-wafer tools 2; caps mirror the
        # original [20, 3, 20, 3] at M=4.
        caps = [20.0 if m.batch_size > 1 else 3.0 for m in machines]

        v: List[float] = []

        # dims 0 .. M+1: queue counts, then finished goods, then demand backlog
        for name in [f"queue{m}" for m in range(M)] + ["queue_fin", "demand"]:
            c = len(self.queues[name])
            v.append((c / max_units) if norm else float(c))

        # next M dims: busy flags
        for m in machines:
            v.append(1.0 if m.busy else 0.0)

        # next M dims: time remaining, scaled
        for i, m in enumerate(machines):
            tl = 0 if not m.busy else max(m.finish_time - current_time, 0)
            v.append((tl / caps[i]) if norm else float(tl))

        # final 4 dims: global progress, period progress, look-ahead to next demand
        gp = (current_time / self.H) if self.H > 0 else 0.0
        v.append(min(gp, 1.0) if norm else float(current_time))

        pp = (current_time % self.demand_interval) / self.demand_interval
        v.append(pp if norm else float(current_time % self.demand_interval))

        next_dem_t = current_time
        while next_dem_t < self.H and self.demand_schedule[next_dem_t] == 0:
            next_dem_t += 1

        ticks_until = next_dem_t - current_time
        v.append((ticks_until / self.demand_interval) if norm else float(ticks_until))

        qty = self.demand_schedule[next_dem_t] if next_dem_t < self.H else 0.0
        v.append((qty / 200.0) if norm else float(qty))

        return v

    def compute_action_mask(self, current_time: int) -> np.ndarray:
        """
        mask shape (4, 2) for 4 machines × {Idle, Run}
        """
        M = len(self.machines)
        mask = np.zeros((M, 2), dtype=np.int8)
        mask[:, 0] = 1  # Idle always allowed

        # `not busy` dominates: a tool completing at this tick is still flagged
        # busy here (update() has not run yet), so it is not offered as runnable.
        for m, mach in enumerate(self.machines):
            if not mach.busy and current_time >= mach.finish_time:
                if len(self.queues[f"queue{m}"]) >= mach.batch_size:
                    mask[m, 1] = 1

        return mask

    # ---------------------------
    # simulation step
    # ---------------------------
    def run_step(
        self,
        current_time: int,
        actions_override: Optional[Dict[str, bool]] = None,
        invalid_penalty: float = 0.0,
    ) -> float:
        """
        Advance the simulation by one discrete time step.

        actions_override: dict like {'machine0': True/False, ...}
        If None, the built-in commander heuristic decides.

        Returns: step_reward = delta_profit + invalid_penalty * invalid_count
        """
        current_time = int(current_time)

        # inject arrivals and demand
        if current_time < self.H:
            for _ in range(int(self.arrivals_schedule[current_time])):
                self.add_item_to_queue("queue0", current_time)
            for _ in range(int(self.demand_schedule[current_time])):
                self.add_item_to_queue("demand", current_time)

            # Match any existing finished inventory with unmet demand instantly
            while self.queues["queue_fin"] and self.queues["demand"]:
                self.queues["queue_fin"].pop(0)
                self.logs["queue_fin"].append((len(self.queues["queue_fin"]), current_time))
                
                self.queues["demand"].pop(0)
                self.logs["demand"].append((len(self.queues["demand"]), current_time))
                
                self.demand_met_log.append(current_time)
                self.costs["revenue"] += float(self.econ["revenue_per_unit"])

        if actions_override is None:
            actions = commander_decide_general(self.machines, self.queues)
        else:
            actions = actions_override

        prev_profit = self.profit_total()
        invalids = 0

        # --- completions (before starts) ---
        # Separate from starts so a tool finishing at t can restart at t, and so a
        # unit deposited by tool m at t is available to tool m+1 at t.
        last = len(self.machines) - 1
        for m, mach in enumerate(self.machines):
            if mach.busy and mach.update(current_time):
                if m == last:
                    for _ in range(mach.batch_size):
                        self._ship_or_store(current_time)
                else:
                    for _ in range(mach.batch_size):
                        self.add_item_to_queue(f"queue{m + 1}", current_time)

        # --- starts ---
        for m, mach in enumerate(self.machines):
            if not actions.get(f"machine{m}", False):
                continue
            if mach.busy or len(self.queues[f"queue{m}"]) < mach.batch_size:
                invalids += 1
            else:
                for _ in range(mach.batch_size):
                    self.remove_item_from_queue(f"queue{m}", current_time)
                self.costs["production"] += (
                    float(self.econ["production_cost"][m]) * mach.batch_size
                )
                mach.process(current_time)

        # end-of-step holding / backorder
        inv_units = len(self.queues["queue_fin"])
        self.costs["inventory"] += float(self.econ["inventory_cost_per_unit"]) * inv_units
        for q_idx in range(len(self.machines)):
            q_name = f"queue{q_idx}"
            self.costs["wip"] += float(self.econ["wip_cost_per_unit"][q_idx]) * len(self.queues[q_name])
        late_demand_count = sum(1 for arrival_t in self.queues["demand"] if current_time >= arrival_t + self.demand_interval)
        self.costs["backorder"] += float(self.econ["backorder_cost_per_unit"]) * late_demand_count

        profit_now = self.profit_total()
        step_reward = (profit_now - prev_profit) + (float(invalid_penalty) * float(invalids))

        if self.logging_enabled:
            snapshot = self._state_dict(current_time)
            for m in range(len(self.machines)):
                snapshot[f"Action M{m}"] = actions.get(f"machine{m}", False)
            self.event_log.append(snapshot)

        self.cost_log.append({
            "Time": current_time,
            "revenue": self.costs["revenue"],
            "production": self.costs["production"],
            "inventory": self.costs["inventory"],
            "wip": self.costs["wip"],
            "backorder": self.costs["backorder"],
            "profit": profit_now,
            "step_reward": step_reward,
            "invalid_actions": invalids,
        })

        self._t = current_time + 1
        return float(step_reward)
