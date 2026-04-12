from __future__ import annotations

from typing import Dict, Any, List


def make_common_config(
    mode: str = "UNIFORM",
    H: int = 2688,                # 4 weeks (672 ticks/week)
    utilization: float = 0.92,    # target share of the bottleneck capacity for DEMAND (0..1)
    demand_delay: int = 671,      # ticks to delay demand (671 = last tick of each week)
    demand_interval: int = 672,   # ticks between demand batches (672 = 1 week)
    arrival_interval: int = 672,  # ticks between raw material deliveries (672 = 1 week)
) -> Dict[str, Any]:
    """
    Build a single "source of truth" config for the 1-product SimpleFab problem.

    Single product simplification:
      - No alpha (product mix) parameter
      - No setup times or costs (only one product, no changeover)
      - Bottleneck = batch stages 0 and 2, time_per_unit = 4 (16 ticks / batch_size 4)
      - T = u * H / 4

    Arrivals are computed at 100% capacity (fully fed factory).
    Demand is computed at `utilization` capacity (e.g., 92%).
    """
    if H < 0:
        raise ValueError("H must be >= 0")

    common: Dict[str, Any] = {
        "time_horizon": int(H),
        "machines": [0, 1, 2, 3],
        "products": [0],

        "processing_times": {
            0: {0: 16},   # Batch machine 0: 16 ticks per batch
            1: {0: 2},    # Single machine 1: 2 ticks per unit
            2: {0: 16},   # Batch machine 2: 16 ticks per batch
            3: {0: 2},    # Single machine 3: 2 ticks per unit
        },
        "batch_sizes": {0: 4, 1: 1, 2: 4, 3: 1},

        # No setup times or costs with a single product
        "setup_times": {
            0: {0: {0: 0}},
            1: {0: {0: 0}},
            2: {0: {0: 0}},
            3: {0: {0: 0}},
        },

        "revenue_per_unit": {0: 80},
        "production_cost": {
            0: {0: 8},
            1: {0: 4},
            2: {0: 8},
            3: {0: 4},
        },
        "setup_cost": {0: 0, 1: 0, 2: 0, 3: 0},
        "inventory_cost_per_unit": {0: 0.02},
        "backorder_cost_per_unit": {0: 0.10},

        "initial_inventory": {
            0: {0: 0},
            1: {0: 0},
            2: {0: 0},
            3: {0: 0},
            "finished": {0: 0},
        },
        "initial_finished_inventory": {0: 8},

        "arrivals_schedule": {},
        "demand_schedule": {},
    }

    # --- compute totals from (H, utilization) ---
    utilization = float(utilization)
    u = max(0.0, min(1.0, utilization))

    TIME_PER_UNIT = 4.0  # 16 ticks / batch_size 4
    BATCH = int(common["batch_sizes"][0])  # 4

    # Arrivals at 100% capacity (fully fed factory)
    arr_total = (int(H / TIME_PER_UNIT) // BATCH) * BATCH

    # Demand at utilization% capacity
    dem_total = (int((u * H) / TIME_PER_UNIT) // BATCH) * BATCH

    # --- schedule generation ---
    def distribute_to_intervals(total: int, horizon: int, interval: int, delay: int = 0) -> List[int]:
        """Split total units into events spaced by `interval` ticks, with optional delay."""
        schedule = [0] * horizon
        n_events = horizon // interval
        if n_events == 0:
            if delay < horizon:
                schedule[delay] = total
            return schedule
        per_event = total // n_events
        remainder = total % n_events
        for i in range(n_events):
            t = i * interval + delay
            if t < horizon:
                schedule[t] = per_event + (1 if i < remainder else 0)
        return schedule

    mode_u = mode.upper().strip()
    if mode_u == "ALL_AT_T0":
        arr = [0] * H
        dem = [0] * H
        if H > 0:
            arr[0] = arr_total
            dem[0] = dem_total
    elif mode_u == "UNIFORM":
        arr = distribute_to_intervals(arr_total, H, arrival_interval, delay=0)
        dem = distribute_to_intervals(dem_total, H, demand_interval, delay=demand_delay)
    elif mode_u == "ALTERNATING":
        arr = distribute_to_intervals(arr_total, H, arrival_interval, delay=0)
        # Alternating demand pattern
        dem = [0] * H
        pattern = [50, 86, 50, 86]
        for i, amt in enumerate(pattern):
            t = i * demand_interval + demand_delay
            if t < H:
                dem[t] = amt
    else:
        raise ValueError("mode must be ALL_AT_T0, UNIFORM, or ALTERNATING")

    common["arrivals_schedule"] = {0: arr}
    common["demand_schedule"] = {0: dem}
    common["demand"] = {0: dem_total}
    common["arrivals"] = {0: arr_total}
    common["utilization"] = u
    common["demand_interval"] = int(demand_interval)
    common["demand_delay"] = int(demand_delay)

    return common
