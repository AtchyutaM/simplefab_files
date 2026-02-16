from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np


class BatchProcessingMachine:
    """
    Batch machine:
      - requires `batch_size` items to start
      - processes a batch as one job
      - on completion, emits `batch_size` units of that product
    Event log tuple format:
      (product, start_time, finish_time, batch_size)
    """
    def __init__(self, processing_times: Dict[int, int], batch_size: int):
        self.processing_times = processing_times
        self.batch_size = int(batch_size)
        self.current_product: Optional[int] = None
        self.finish_time: int = 0
        self.busy: bool = False
        self.event_log: List[Tuple[int, int, int, int]] = []

    def process(self, product: int, current_time: int) -> int:
        self.busy = True
        self.current_product = int(product)
        self.finish_time = int(current_time) + int(self.processing_times[self.current_product])
        self.event_log.append((self.current_product, int(current_time), int(self.finish_time), self.batch_size))
        return self.finish_time

    def update(self, current_time: int) -> Optional[int]:
        if self.busy and current_time >= self.finish_time:
            self.busy = False
            finished = self.current_product
            self.current_product = None
            return finished
        return None


class Machine:
    """
    Single-unit machine with sequence-dependent setup times.
    Event log tuple format:
      (product, decision_time, start_time, finish_time)
    """
    def __init__(self, processing_times: Dict[int, int], setup_times: Dict[Any, Dict[int, int]]):
        self.processing_times = processing_times
        self.setup_times = setup_times
        self.current_product: Optional[int] = None
        self.finish_time: int = 0
        self.busy: bool = False
        self.event_log: List[Tuple[int, int, int, int]] = []
        self.last_finished_product: Optional[int] = None

    def process(self, product: int, current_time: int) -> int:
        # Use last_finished_product (not current_product) so setup time
        # correctly reflects the changeover from the previously completed job,
        # matching how setup COST is calculated in run_step().
        prev = self.last_finished_product
        if prev is None:
            setup_time = int(self.setup_times[None][product])
        else:
            setup_time = int(self.setup_times[prev][product])

        start_time = int(current_time) + setup_time
        finish_time = start_time + int(self.processing_times[product])

        self.current_product = int(product)
        self.finish_time = int(finish_time)
        self.busy = True
        self.event_log.append((int(product), int(current_time), int(start_time), int(finish_time)))
        return self.finish_time

    def update(self, current_time: int) -> Optional[int]:
        if self.busy and current_time >= self.finish_time:
            self.busy = False
            finished = self.current_product
            self.current_product = None
            self.last_finished_product = finished
            return finished
        return None


class Commander:
    """
    Simple demand-driven heuristic:
      - if both products are feasible at a machine, choose the one with higher remaining demand share
    """
    def __init__(self, machine0: BatchProcessingMachine, machine1: Machine, machine2: BatchProcessingMachine, machine3: Machine):
        self.machine0 = machine0
        self.machine1 = machine1
        self.machine2 = machine2
        self.machine3 = machine3

    def decide_actions(self, state: Dict[str, Any]) -> Dict[str, Optional[str]]:
        actions: Dict[str, Optional[str]] = {"machine0": None, "machine1": None, "machine2": None, "machine3": None}

        q0p0 = state["Queue 0 Product 0"]; q0p1 = state["Queue 0 Product 1"]
        q1p0 = state["Queue 1 Product 0"]; q1p1 = state["Queue 1 Product 1"]
        q2p0 = state["Queue 2 Product 0"]; q2p1 = state["Queue 2 Product 1"]
        q3p0 = state["Queue 3 Product 0"]; q3p1 = state["Queue 3 Product 1"]
        dp0  = state["Demand Queue Product 0"]; dp1  = state["Demand Queue Product 1"]

        tot = dp0 + dp1
        share0 = 0.0 if tot == 0 else dp0 / tot
        share1 = 0.0 if tot == 0 else dp1 / tot

        def prefer() -> str:
            return "Prod0" if share0 >= share1 else "Prod1"

        # M0 (batch)
        if (not self.machine0.busy) and (q0p0 >= self.machine0.batch_size or q0p1 >= self.machine0.batch_size):
            if q0p0 >= self.machine0.batch_size and q0p1 >= self.machine0.batch_size:
                actions["machine0"] = prefer()
            elif q0p0 >= self.machine0.batch_size:
                actions["machine0"] = "Prod0"
            else:
                actions["machine0"] = "Prod1"

        # M1 (single)
        if (not self.machine1.busy) and (q1p0 > 0 or q1p1 > 0):
            if q1p0 > 0 and q1p1 > 0:
                actions["machine1"] = prefer()
            elif q1p0 > 0:
                actions["machine1"] = "Prod0"
            else:
                actions["machine1"] = "Prod1"

        # M2 (batch)
        if (not self.machine2.busy) and (q2p0 >= self.machine2.batch_size or q2p1 >= self.machine2.batch_size):
            if q2p0 >= self.machine2.batch_size and q2p1 >= self.machine2.batch_size:
                actions["machine2"] = prefer()
            elif q2p0 >= self.machine2.batch_size:
                actions["machine2"] = "Prod0"
            else:
                actions["machine2"] = "Prod1"

        # M3 (single)
        if (not self.machine3.busy) and (q3p0 > 0 or q3p1 > 0):
            if q3p0 > 0 and q3p1 > 0:
                actions["machine3"] = prefer()
            elif q3p0 > 0:
                actions["machine3"] = "Prod0"
            else:
                actions["machine3"] = "Prod1"

        return actions


class ProductionLine:
    """
    Discrete-time simulation of a 4-machine line:

      queue0 -> M0(batch) -> queue1 -> M1(single) -> queue2 -> M2(batch) -> queue3 -> M3(single) -> (ship or inventory)

    Demand is modeled as a separate queue:
      - demand arrivals go into demand queue each step
      - when a finished unit leaves M3:
          if demand exists for that product => ship immediately, earn revenue
          else => store as finished inventory (queue_fin)

    Economics tracked:
      revenue - (production + setup + inventory + backorder)
    """
    ACTIONS = ("None", "Prod0", "Prod1")

    def __init__(
        self,
        batch_machine0_times: Dict[int, int],
        machine1_times: Dict[int, int],
        machine1_setups: Dict[Any, Dict[int, int]],
        batch_machine2_times: Dict[int, int],
        machine3_times: Dict[int, int],
        machine3_setups: Dict[Any, Dict[int, int]],
        common_cfg: Dict[str, Any],
    ):
        self.machine0 = BatchProcessingMachine(batch_machine0_times, batch_size=4)
        self.machine1 = Machine(machine1_times, machine1_setups)
        self.machine2 = BatchProcessingMachine(batch_machine2_times, batch_size=4)
        self.machine3 = Machine(machine3_times, machine3_setups)

        self.arrivals_schedule = common_cfg["arrivals_schedule"]
        self.demand_schedule = common_cfg["demand_schedule"]
        self.H = int(common_cfg["time_horizon"])

        # queues store timestamps (arrival times) for each unit
        self.queues: Dict[str, List[List[int]]] = {
            "queue0": [[], []],
            "queue1": [[], []],
            "queue2": [[], []],
            "queue3": [[], []],
            "queue_fin": [[], []],   # finished inventory
            "demand": [[], []],      # unmet demand
        }

        # Initialize with starting finished goods inventory (to avoid cold-start backorders)
        init_fin = common_cfg.get("initial_finished_inventory", {0: 0, 1: 0})
        for _ in range(int(init_fin.get(0, 0))):
            self.queues["queue_fin"][0].append(-1)  # -1 indicates pre-existing inventory
        for _ in range(int(init_fin.get(1, 0))):
            self.queues["queue_fin"][1].append(-1)

        # logs
        self.logs: Dict[str, List[Tuple[List[int], int]]] = {k: [] for k in self.queues}
        self.demand_met_log: List[Tuple[int, int]] = []  # (product, time)
        self.event_log: List[Dict[str, Any]] = []         # per-step snapshot (optional)

        self.total_demand = {0: sum(self.demand_schedule[0]), 1: sum(self.demand_schedule[1])}
        self.commander = Commander(self.machine0, self.machine1, self.machine2, self.machine3)

        self.econ = {
            "revenue_per_unit": common_cfg["revenue_per_unit"],
            "production_cost": common_cfg["production_cost"],
            "setup_cost": common_cfg["setup_cost"],
            "inventory_cost_per_unit": common_cfg["inventory_cost_per_unit"],
            "backorder_cost_per_unit": common_cfg["backorder_cost_per_unit"],
        }
        self.costs = {"revenue": 0.0, "production": 0.0, "setup": 0.0, "inventory": 0.0, "backorder": 0.0}
        self.cost_log: List[Dict[str, Any]] = []

        self.logging_enabled = False
        self._t = 0  # next decision time (used for observations)

    # ---------------------------
    # queue helpers
    # ---------------------------
    def add_item_to_queue(self, queue_name: str, product: int, current_time: int) -> None:
        self.queues[queue_name][product].append(int(current_time))
        sizes = [len(q) for q in self.queues[queue_name]]
        self.logs[queue_name].append((sizes, int(current_time)))

    def remove_item_from_queue(self, queue_name: str, product: int, current_time: int) -> None:
        if self.queues[queue_name][product]:
            self.queues[queue_name][product].pop(0)
            sizes = [len(q) for q in self.queues[queue_name]]
            self.logs[queue_name].append((sizes, int(current_time)))

    def _ship_or_store(self, product: int, current_time: int) -> None:
        """
        If demand exists for this product: ship now, earn revenue.
        Else: store in finished inventory.
        """
        if self.queues["demand"][product]:
            self.queues["demand"][product].pop(0)
            self.demand_met_log.append((int(product), int(current_time)))
            self.costs["revenue"] += float(self.econ["revenue_per_unit"][product])
            sizes = [len(q) for q in self.queues["demand"]]
            self.logs["demand"].append((sizes, int(current_time)))
        else:
            self.add_item_to_queue("queue_fin", product, current_time)

    # ---------------------------
    # economics
    # ---------------------------
    def profit_total(self) -> float:
        c = self.costs
        return c["revenue"] - (c["production"] + c["setup"] + c["inventory"] + c["backorder"])

    # ---------------------------
    # state / obs / mask
    # ---------------------------
    def _state_dict(self, current_time: int) -> Dict[str, Any]:
        return {
            "Time": current_time,
            "Queue 0 Product 0": len(self.queues["queue0"][0]),
            "Queue 0 Product 1": len(self.queues["queue0"][1]),
            "Queue 1 Product 0": len(self.queues["queue1"][0]),
            "Queue 1 Product 1": len(self.queues["queue1"][1]),
            "Queue 2 Product 0": len(self.queues["queue2"][0]),
            "Queue 2 Product 1": len(self.queues["queue2"][1]),
            "Queue 3 Product 0": len(self.queues["queue3"][0]),
            "Queue 3 Product 1": len(self.queues["queue3"][1]),
            "Queue Fin Product 0": len(self.queues["queue_fin"][0]),
            "Queue Fin Product 1": len(self.queues["queue_fin"][1]),
            "Demand Queue Product 0": len(self.queues["demand"][0]),
            "Demand Queue Product 1": len(self.queues["demand"][1]),
            "Machine 0 Current Product": None if self.machine0.current_product is None else int(self.machine0.current_product),
            "Machine 1 Current Product": None if self.machine1.current_product is None else int(self.machine1.current_product),
            "Machine 2 Current Product": None if self.machine2.current_product is None else int(self.machine2.current_product),
            "Machine 3 Current Product": None if self.machine3.current_product is None else int(self.machine3.current_product),
            "Machine 0 timeremaining": 0 if self.machine0.current_product is None else max(self.machine0.finish_time - current_time, 0),
            "Machine 1 timeremaining": 0 if self.machine1.current_product is None else max(self.machine1.finish_time - current_time, 0),
            "Machine 2 timeremaining": 0 if self.machine2.current_product is None else max(self.machine2.finish_time - current_time, 0),
            "Machine 3 timeremaining": 0 if self.machine3.current_product is None else max(self.machine3.finish_time - current_time, 0),
        }

    def get_observation(self, current_time: int, norm: bool = True) -> List[float]:
        """
        20-dim observation:
          0-11: queue counts [q0P0,q0P1,q1P0,q1P1,q2P0,q2P1,q3P0,q3P1,qfinP0,qfinP1,demP0,demP1]
          12-15: machine current product codes (None->0, P0->1, P1->2), scaled to [0,1]
          16-19: machine time_remaining, scaled to [0,1]
        """
        def enc_curr(p: Optional[int]) -> int:
            return 0 if p is None else (int(p) + 1)

        max_units = 64.0
        per_machine_time_cap = {0: 20.0, 1: 3.0, 2: 20.0, 3: 3.0}

        v: List[float] = []
        for name in ("queue0", "queue1", "queue2", "queue3", "queue_fin", "demand"):
            c0 = len(self.queues[name][0])
            c1 = len(self.queues[name][1])
            if norm:
                v.append(c0 / max_units)
                v.append(c1 / max_units)
            else:
                v.append(float(c0))
                v.append(float(c1))

        machines = [self.machine0, self.machine1, self.machine2, self.machine3]
        for m in machines:
            code = enc_curr(m.current_product)
            v.append((code / 2.0) if norm else float(code))

        for i, m in enumerate(machines):
            tl = 0 if m.current_product is None else max(m.finish_time - current_time, 0)
            cap = per_machine_time_cap[i]
            v.append((tl / cap) if norm else float(tl))

        return v

    def compute_action_mask(self, current_time: int) -> np.ndarray:
        """
        mask shape (4,3) for 4 machines × {None, Prod0, Prod1}
        """
        mask = np.zeros((4, 3), dtype=np.int8)
        mask[:, 0] = 1  # None always allowed

        # M0: needs batch_size from queue0
        if self.machine0.current_product is None and current_time >= self.machine0.finish_time:
            if len(self.queues["queue0"][0]) >= self.machine0.batch_size: mask[0, 1] = 1
            if len(self.queues["queue0"][1]) >= self.machine0.batch_size: mask[0, 2] = 1

        # M1: needs 1 from queue1
        if self.machine1.current_product is None and current_time >= self.machine1.finish_time:
            if len(self.queues["queue1"][0]) >= 1: mask[1, 1] = 1
            if len(self.queues["queue1"][1]) >= 1: mask[1, 2] = 1

        # M2: needs batch_size from queue2
        if self.machine2.current_product is None and current_time >= self.machine2.finish_time:
            if len(self.queues["queue2"][0]) >= self.machine2.batch_size: mask[2, 1] = 1
            if len(self.queues["queue2"][1]) >= self.machine2.batch_size: mask[2, 2] = 1

        # M3: needs 1 from queue3
        if self.machine3.current_product is None and current_time >= self.machine3.finish_time:
            if len(self.queues["queue3"][0]) >= 1: mask[3, 1] = 1
            if len(self.queues["queue3"][1]) >= 1: mask[3, 2] = 1

        return mask

    # ---------------------------
    # simulation step
    # ---------------------------
    def run_step(
        self,
        current_time: int,
        actions_override: Optional[Dict[str, Optional[str]]] = None,
        invalid_penalty: float = 0.0,
    ) -> float:
        """
        Advance the simulation by one discrete time step.

        actions_override: dict like {'machine0':'None'/'Prod0'/'Prod1', ...}
        If None, the built-in Commander decides.

        Returns: step_reward = delta_profit + invalid_penalty * invalid_count
        """
        current_time = int(current_time)

        # inject arrivals and demand
        if current_time < self.H:
            for j in (0, 1):
                for _ in range(int(self.arrivals_schedule[j][current_time])):
                    self.add_item_to_queue("queue0", j, current_time)
                for _ in range(int(self.demand_schedule[j][current_time])):
                    self.add_item_to_queue("demand", j, current_time)

        state = self._state_dict(current_time)
        actions = self.commander.decide_actions(state) if actions_override is None else actions_override

        prev_profit = self.profit_total()
        invalids = 0

        # --- completions (before starts) ---
        if self.machine0.current_product is not None:
            fin = self.machine0.update(current_time)
            if fin is not None:
                for _ in range(self.machine0.batch_size):
                    self.add_item_to_queue("queue1", fin, current_time)

        if self.machine1.current_product is not None:
            fin = self.machine1.update(current_time)
            if fin is not None:
                self.add_item_to_queue("queue2", fin, current_time)

        if self.machine2.current_product is not None:
            fin = self.machine2.update(current_time)
            if fin is not None:
                for _ in range(self.machine2.batch_size):
                    self.add_item_to_queue("queue3", fin, current_time)

        if self.machine3.current_product is not None:
            fin = self.machine3.update(current_time)
            if fin is not None:
                self._ship_or_store(fin, current_time)

        # --- starts ---
        # M0 batch
        a0 = actions.get("machine0", None)
        if a0 in ("Prod0", "Prod1"):
            k = 0 if a0 == "Prod0" else 1
            if (self.machine0.current_product is not None) or (len(self.queues["queue0"][k]) < self.machine0.batch_size):
                invalids += 1
            else:
                for _ in range(self.machine0.batch_size):
                    self.remove_item_from_queue("queue0", k, current_time)
                self.costs["production"] += float(self.econ["production_cost"][0][k]) * self.machine0.batch_size
                self.machine0.process(k, current_time)

        # M1 single
        a1 = actions.get("machine1", None)
        if a1 in ("Prod0", "Prod1"):
            k = 0 if a1 == "Prod0" else 1
            if (self.machine1.current_product is not None) or (len(self.queues["queue1"][k]) < 1):
                invalids += 1
            else:
                self.remove_item_from_queue("queue1", k, current_time)
                prev = self.machine1.last_finished_product
                self.machine1.process(k, current_time)
                if prev != k:
                    self.costs["setup"] += float(self.econ["setup_cost"][1])
                self.costs["production"] += float(self.econ["production_cost"][1][k])

        # M2 batch
        a2 = actions.get("machine2", None)
        if a2 in ("Prod0", "Prod1"):
            k = 0 if a2 == "Prod0" else 1
            if (self.machine2.current_product is not None) or (len(self.queues["queue2"][k]) < self.machine2.batch_size):
                invalids += 1
            else:
                for _ in range(self.machine2.batch_size):
                    self.remove_item_from_queue("queue2", k, current_time)
                self.costs["production"] += float(self.econ["production_cost"][2][k]) * self.machine2.batch_size
                self.machine2.process(k, current_time)

        # M3 single
        a3 = actions.get("machine3", None)
        if a3 in ("Prod0", "Prod1"):
            k = 0 if a3 == "Prod0" else 1
            if (self.machine3.current_product is not None) or (len(self.queues["queue3"][k]) < 1):
                invalids += 1
            else:
                self.remove_item_from_queue("queue3", k, current_time)
                prev = self.machine3.last_finished_product
                self.machine3.process(k, current_time)
                if prev != k:
                    self.costs["setup"] += float(self.econ["setup_cost"][3])
                self.costs["production"] += float(self.econ["production_cost"][3][k])

        # end-of-step holding / backorder
        for j in (0, 1):
            inv_units = len(self.queues["queue_fin"][j])
            self.costs["inventory"] += float(self.econ["inventory_cost_per_unit"][j]) * inv_units
            self.costs["backorder"] += float(self.econ["backorder_cost_per_unit"][j]) * len(self.queues["demand"][j])

        profit_now = self.profit_total()
        step_reward = (profit_now - prev_profit) + (float(invalid_penalty) * float(invalids))

        if self.logging_enabled:
            snapshot = self._state_dict(current_time)
            snapshot.update({
                "Machine 0 Action": actions.get("machine0"),
                "Machine 1 Action": actions.get("machine1"),
                "Machine 2 Action": actions.get("machine2"),
                "Machine 3 Action": actions.get("machine3"),
            })
            self.event_log.append(snapshot)

        self.cost_log.append({
            "Time": current_time,
            "revenue": self.costs["revenue"],
            "production": self.costs["production"],
            "setup": self.costs["setup"],
            "inventory": self.costs["inventory"],
            "backorder": self.costs["backorder"],
            "profit": profit_now,
            "step_reward": step_reward,
            "invalid_actions": invalids,
        })

        self._t = current_time + 1
        return float(step_reward)
