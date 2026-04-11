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

        self.machine0 = BatchProcessingMachine(pt[0][0], batch_size=4)
        self.machine1 = Machine(pt[1][0])
        self.machine2 = BatchProcessingMachine(pt[2][0], batch_size=4)
        self.machine3 = Machine(pt[3][0])

        self.arrivals_schedule = common_cfg["arrivals_schedule"][0]
        self.demand_schedule = common_cfg["demand_schedule"][0]
        self.H = int(common_cfg["time_horizon"])
        self.demand_interval = int(common_cfg.get("demand_interval", 672))
        self.demand_delay = int(common_cfg.get("demand_delay", 96))

        # queues are simple lists of arrival timestamps
        self.queues: Dict[str, List[int]] = {
            "queue0": [],
            "queue1": [],
            "queue2": [],
            "queue3": [],
            "queue_fin": [],    # finished inventory
            "demand": [],       # unmet demand
        }

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
            "production_cost": {
                0: common_cfg["production_cost"][0][0],
                1: common_cfg["production_cost"][1][0],
                2: common_cfg["production_cost"][2][0],
                3: common_cfg["production_cost"][3][0],
            },
            "inventory_cost_per_unit": common_cfg["inventory_cost_per_unit"][0],
            "backorder_cost_per_unit": common_cfg["backorder_cost_per_unit"][0],
        }
        self.costs = {"revenue": 0.0, "production": 0.0, "inventory": 0.0, "backorder": 0.0}
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
        return c["revenue"] - (c["production"] + c["inventory"] + c["backorder"])

    # ---------------------------
    # state / obs / mask
    # ---------------------------
    def _state_dict(self, current_time: int) -> Dict[str, Any]:
        return {
            "Time": current_time,
            "Queue 0": len(self.queues["queue0"]),
            "Queue 1": len(self.queues["queue1"]),
            "Queue 2": len(self.queues["queue2"]),
            "Queue 3": len(self.queues["queue3"]),
            "Queue Fin": len(self.queues["queue_fin"]),
            "Demand Queue": len(self.queues["demand"]),
            "Machine 0 Busy": int(self.machine0.busy),
            "Machine 1 Busy": int(self.machine1.busy),
            "Machine 2 Busy": int(self.machine2.busy),
            "Machine 3 Busy": int(self.machine3.busy),
            "Machine 0 TimeRemaining": 0 if not self.machine0.busy else max(self.machine0.finish_time - current_time, 0),
            "Machine 1 TimeRemaining": 0 if not self.machine1.busy else max(self.machine1.finish_time - current_time, 0),
            "Machine 2 TimeRemaining": 0 if not self.machine2.busy else max(self.machine2.finish_time - current_time, 0),
            "Machine 3 TimeRemaining": 0 if not self.machine3.busy else max(self.machine3.finish_time - current_time, 0),
        }

    def get_observation(self, current_time: int, norm: bool = True) -> List[float]:
        """
        16-dim observation:
          0-5:   queue counts [q0, q1, q2, q3, qfin, demand]
          6-9:   machine busy flags (0 or 1)
          10-13: machine time_remaining, scaled to [0, 1]
          14:    global progress (t / H)
          15:    period progress (t % demand_interval / demand_interval)
        """
        max_units = 64.0
        per_machine_time_cap = {0: 20.0, 1: 3.0, 2: 20.0, 3: 3.0}

        v: List[float] = []

        # dims 0-5: queue counts
        for name in ("queue0", "queue1", "queue2", "queue3", "queue_fin", "demand"):
            c = len(self.queues[name])
            v.append((c / max_units) if norm else float(c))

        # dims 6-9: machine busy flags
        machines = [self.machine0, self.machine1, self.machine2, self.machine3]
        for m in machines:
            v.append(1.0 if m.busy else 0.0)

        # dims 10-13: time remaining
        for i, m in enumerate(machines):
            tl = 0 if not m.busy else max(m.finish_time - current_time, 0)
            cap = per_machine_time_cap[i]
            v.append((tl / cap) if norm else float(tl))

        # dim 14: global progress
        gp = (current_time / self.H) if self.H > 0 else 0.0
        v.append(min(gp, 1.0) if norm else float(current_time))

        # dim 15: period progress
        pp = (current_time % self.demand_interval) / self.demand_interval
        v.append(pp if norm else float(current_time % self.demand_interval))

        return v

    def compute_action_mask(self, current_time: int) -> np.ndarray:
        """
        mask shape (4, 2) for 4 machines × {Idle, Run}
        """
        mask = np.zeros((4, 2), dtype=np.int8)
        mask[:, 0] = 1  # Idle always allowed

        # M0: needs batch_size from queue0
        if not self.machine0.busy and current_time >= self.machine0.finish_time:
            if len(self.queues["queue0"]) >= self.machine0.batch_size:
                mask[0, 1] = 1

        # M1: needs 1 from queue1
        if not self.machine1.busy and current_time >= self.machine1.finish_time:
            if len(self.queues["queue1"]) >= 1:
                mask[1, 1] = 1

        # M2: needs batch_size from queue2
        if not self.machine2.busy and current_time >= self.machine2.finish_time:
            if len(self.queues["queue2"]) >= self.machine2.batch_size:
                mask[2, 1] = 1

        # M3: needs 1 from queue3
        if not self.machine3.busy and current_time >= self.machine3.finish_time:
            if len(self.queues["queue3"]) >= 1:
                mask[3, 1] = 1

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

        if actions_override is None:
            actions = commander_decide(
                self.machine0, self.machine1, self.machine2, self.machine3,
                self.queues,
            )
        else:
            actions = actions_override

        prev_profit = self.profit_total()
        invalids = 0

        # --- completions (before starts) ---
        if self.machine0.busy:
            if self.machine0.update(current_time):
                for _ in range(self.machine0.batch_size):
                    self.add_item_to_queue("queue1", current_time)

        if self.machine1.busy:
            if self.machine1.update(current_time):
                self.add_item_to_queue("queue2", current_time)

        if self.machine2.busy:
            if self.machine2.update(current_time):
                for _ in range(self.machine2.batch_size):
                    self.add_item_to_queue("queue3", current_time)

        if self.machine3.busy:
            if self.machine3.update(current_time):
                self._ship_or_store(current_time)

        # --- starts ---
        # M0 batch
        if actions.get("machine0", False):
            if self.machine0.busy or len(self.queues["queue0"]) < self.machine0.batch_size:
                invalids += 1
            else:
                for _ in range(self.machine0.batch_size):
                    self.remove_item_from_queue("queue0", current_time)
                self.costs["production"] += float(self.econ["production_cost"][0]) * self.machine0.batch_size
                self.machine0.process(current_time)

        # M1 single
        if actions.get("machine1", False):
            if self.machine1.busy or len(self.queues["queue1"]) < 1:
                invalids += 1
            else:
                self.remove_item_from_queue("queue1", current_time)
                self.costs["production"] += float(self.econ["production_cost"][1])
                self.machine1.process(current_time)

        # M2 batch
        if actions.get("machine2", False):
            if self.machine2.busy or len(self.queues["queue2"]) < self.machine2.batch_size:
                invalids += 1
            else:
                for _ in range(self.machine2.batch_size):
                    self.remove_item_from_queue("queue2", current_time)
                self.costs["production"] += float(self.econ["production_cost"][2]) * self.machine2.batch_size
                self.machine2.process(current_time)

        # M3 single
        if actions.get("machine3", False):
            if self.machine3.busy or len(self.queues["queue3"]) < 1:
                invalids += 1
            else:
                self.remove_item_from_queue("queue3", current_time)
                self.costs["production"] += float(self.econ["production_cost"][3])
                self.machine3.process(current_time)

        # end-of-step holding / backorder
        inv_units = len(self.queues["queue_fin"])
        self.costs["inventory"] += float(self.econ["inventory_cost_per_unit"]) * inv_units
        self.costs["backorder"] += float(self.econ["backorder_cost_per_unit"]) * len(self.queues["demand"])

        profit_now = self.profit_total()
        step_reward = (profit_now - prev_profit) + (float(invalid_penalty) * float(invalids))

        if self.logging_enabled:
            snapshot = self._state_dict(current_time)
            snapshot["Action M0"] = actions.get("machine0", False)
            snapshot["Action M1"] = actions.get("machine1", False)
            snapshot["Action M2"] = actions.get("machine2", False)
            snapshot["Action M3"] = actions.get("machine3", False)
            self.event_log.append(snapshot)

        self.cost_log.append({
            "Time": current_time,
            "revenue": self.costs["revenue"],
            "production": self.costs["production"],
            "inventory": self.costs["inventory"],
            "backorder": self.costs["backorder"],
            "profit": profit_now,
            "step_reward": step_reward,
            "invalid_actions": invalids,
        })

        self._t = current_time + 1
        return float(step_reward)
