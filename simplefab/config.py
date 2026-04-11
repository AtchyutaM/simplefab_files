from __future__ import annotations

from typing import Dict, Any, List


def make_common_config(
    mode: str = "UNIFORM",
    H: int = 2688,                # 4 weeks (672 ticks/week)
    alpha: float = 0.5,           # fraction of Product 0 in the mix (0..1)
    utilization: float = 0.92,    # target share of the bottleneck capacity for DEMAND (0..1)
    demand_delay: int = 96,       # ticks to delay demand after arrivals (96 = 1 day)
    demand_interval: int = 672,   # ticks between demand batches (672 = 1 week)
    arrival_interval: int = 672,  # ticks between raw material deliveries (672 = 1 week)
) -> Dict[str, Any]:
    """
    Build a single "source of truth" config for both MILP and Simulation.

    Capacity math (bottleneck = batch stages 0 and 2):
      time_per_unit: P0=4, P1=5 (because 16/4 and 20/4 with batch size 4)
      4*x0 + 5*x1 ≈ u * H
      with x0 = alpha*T, x1=(1-alpha)*T => T = u*H / (4*alpha + 5*(1-alpha))

    We then round x0 and x1 down to the nearest multiple of the batch size (4),
    so batch machines can always start complete batches.

    Arrivals are computed at 100% capacity (fully fed factory).
    Demand is computed at `utilization` capacity (e.g., 92%).
    """
    if H < 0:
        raise ValueError("H must be >= 0")

    common: Dict[str, Any] = {
        "time_horizon": int(H),
        "machines": [0, 1, 2, 3],
        "products": [0, 1],

        "processing_times": {
            0: {0: 16, 1: 20},
            1: {0: 2,  1: 2},
            2: {0: 16, 1: 20},
            3: {0: 2,  1: 2},
        },
        "batch_sizes": {0: 4, 1: 1, 2: 4, 3: 1},
        "setup_times": {
            0: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}},
            1: {0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}},
            2: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}},
            3: {0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}},
        },

        "revenue_per_unit": {0: 80, 1: 100},
        "production_cost": {
            0: {0: 8,  1: 10},
            1: {0: 4,  1: 4},
            2: {0: 8,  1: 10},
            3: {0: 4,  1: 4},
        },
        "setup_cost": {0: 0, 1: 20, 2: 0, 3: 20},
        "inventory_cost_per_unit": {0: 0.02, 1: 0.025},
        "backorder_cost_per_unit": {0: 0.10, 1: 0.15},

        "initial_inventory": {
            0: {0: 0, 1: 0},
            1: {0: 0, 1: 0},
            2: {0: 0, 1: 0},
            3: {0: 0, 1: 0},
            "finished": {0: 0, 1: 0},
        },
        "initial_finished_inventory": {0: 8, 1: 8},

        "arrivals_schedule": {},
        "demand_schedule": {},
    }

    # --- compute totals from (H, alpha, utilization) ---
    alpha = float(alpha)
    utilization = float(utilization)
    alpha = max(0.0, min(1.0, alpha))
    u = max(0.0, min(1.0, utilization))

    denom = 4.0 * alpha + 5.0 * (1.0 - alpha)
    if denom <= 0.0:
        denom = 4.5

    BATCH = int(common["batch_sizes"][0])  # batch machine size (4)

    # Arrivals at 100% capacity (fully fed factory)
    T_arr = H / denom
    arr_total_0 = (int(alpha * T_arr) // BATCH) * BATCH
    arr_total_1 = (int((1.0 - alpha) * T_arr) // BATCH) * BATCH

    # Demand at utilization% capacity
    T_dem = (u * H) / denom
    total_0 = (int(alpha * T_dem) // BATCH) * BATCH
    total_1 = (int((1.0 - alpha) * T_dem) // BATCH) * BATCH

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
        arr0 = [0] * H
        arr1 = [0] * H
        dem0 = [0] * H
        dem1 = [0] * H
        if H > 0:
            arr0[0] = arr_total_0
            arr1[0] = arr_total_1
            dem0[0] = total_0
            dem1[0] = total_1
    elif mode_u == "UNIFORM":
        arr0 = distribute_to_intervals(arr_total_0, H, arrival_interval, delay=0)
        arr1 = distribute_to_intervals(arr_total_1, H, arrival_interval, delay=0)
        dem0 = distribute_to_intervals(total_0, H, demand_interval, delay=demand_delay)
        dem1 = distribute_to_intervals(total_1, H, demand_interval, delay=demand_delay)
    elif mode_u == "ALTERNATING":
        # Raw materials arrive uniformly at 100% capacity
        arr0 = distribute_to_intervals(arr_total_0, H, arrival_interval, delay=0)
        arr1 = distribute_to_intervals(arr_total_1, H, arrival_interval, delay=0)
        
        # Demand alternates: [50, 86, 50, 86] per product (total 272 = same as 68x4)
        dem0 = [0] * H
        dem1 = [0] * H
        
        pattern = [50, 86, 50, 86]
        for i, amt in enumerate(pattern):
            t = i * demand_interval + demand_delay
            if t < H:
                dem0[t] = amt
                dem1[t] = amt
    else:
        raise ValueError("mode must be ALL_AT_T0, UNIFORM, or ALTERNATING")

    common["arrivals_schedule"] = {0: arr0, 1: arr1}
    common["demand_schedule"] = {0: dem0, 1: dem1}
    common["demand"] = {0: total_0, 1: total_1}
    common["arrivals"] = {0: arr_total_0, 1: arr_total_1}
    common["mix_alpha"] = alpha
    common["utilization"] = u
    common["demand_interval"] = int(demand_interval)
    common["demand_delay"] = int(demand_delay)

    return common
