from __future__ import annotations

from typing import Dict, Any, List


def make_common_config(
    mode: str = "UNIFORM",
    H: int = 2688,                # 4 weeks (672 ticks/week)
    utilization: float = 0.50,    # 50% utilization
    demand_delay: int = 671,      # ticks to delay demand (671 = last tick of each week)
    demand_interval: int = 672,   # ticks between demand batches (672 = 1 week)
    arrival_interval: int = 672,  # ticks between raw material deliveries (672 = 1 week)
    n_pairs: int = 2,             # diffusion+litho pairs; 2 => the original 4-machine line
) -> Dict[str, Any]:
    """
    `n_pairs` controls pipeline depth (experiment family 3). The line is always an
    alternating chain

        raw -> DIFF -> LITHO -> DIFF -> LITHO -> ... -> ship

    with `n_pairs` of each, so n_machines = 2 * n_pairs. n_pairs=2 reproduces the
    original 4-machine configuration exactly, value for value.

    Two economics choices, both deliberate, both reversible here and nowhere else:

    * **Total production cost is held at $24/unit through the line at every
      depth** (diffusion:litho stays 2:1, so per-stage cost is 32/M and 16/M).
      Otherwise a 12-stage line would cost $72/unit against $80 revenue, leaving
      an $8 margin, and the experiment would measure collapsing economics rather
      than scheduling difficulty. Depth is the variable; economics are the control.

    * **WIP holding stays linear from $0 at q0 to $0.0075 at the last queue**
      (wip[m] = m/(M-1) * 0.0075). At M=4 this is exactly [0, .0025, .005, .0075].

    Throughput is unaffected by depth: stages run concurrently, so the bottleneck
    remains a single diffusion tool at 4 units / 16 ticks = 0.25 units/tick, i.e.
    672 units per 2688-tick horizon at any n_pairs. Only *lead time* grows, as
    9 * M ticks (36 / 72 / 108 for n_pairs 2 / 4 / 6).
    """
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
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")

    M = 2 * int(n_pairs)
    # Even index = diffusion (batch furnace), odd index = litho (single wafer).
    is_diff = [(m % 2 == 0) for m in range(M)]

    DIFF_PT, DIFF_BATCH = 16, 4
    LITHO_PT, LITHO_BATCH = 2, 1
    TOTAL_PROD_COST = 24.0   # $/unit through the whole line, held constant
    WIP_MAX = 0.0075         # $/unit/tick at the last queue

    processing_times = {m: {0: (DIFF_PT if is_diff[m] else LITHO_PT)} for m in range(M)}
    batch_sizes = {m: (DIFF_BATCH if is_diff[m] else LITHO_BATCH) for m in range(M)}
    # 2:1 diffusion:litho split, summing to TOTAL_PROD_COST per unit at any depth.
    production_cost = {
        m: {0: (2.0 if is_diff[m] else 1.0) * TOTAL_PROD_COST / (1.5 * M)}
        for m in range(M)
    }
    wip_cost = {m: (0.0 if M == 1 else (m / (M - 1)) * WIP_MAX) for m in range(M)}

    common: Dict[str, Any] = {
        "time_horizon": int(H),
        "machines": list(range(M)),
        "products": [0],
        "n_pairs": int(n_pairs),
        "n_machines": M,
        "machine_kind": ["DIFF" if is_diff[m] else "LITHO" for m in range(M)],

        "processing_times": processing_times,
        "batch_sizes": batch_sizes,

        # No setup times or costs with a single product
        "setup_times": {m: {0: {0: 0}} for m in range(M)},

        "revenue_per_unit": {0: 80},
        "production_cost": production_cost,
        "setup_cost": {m: 0 for m in range(M)},
        "inventory_cost_per_unit": {0: 0.04},
        "wip_cost_per_unit": wip_cost,
        "backorder_cost_per_unit": {0: 0.15},

        "initial_inventory": {**{m: {0: 0} for m in range(M)}, "finished": {0: 0}},
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
        dem = distribute_to_intervals(dem_total, H, demand_interval, delay=0)
    elif mode_u == "ALTERNATING":
        arr = distribute_to_intervals(arr_total, H, arrival_interval, delay=0)
        # Alternating demand pattern
        dem = [0] * H
        pattern = [80, 80, 220, 220]
        for i, amt in enumerate(pattern):
            t = i * demand_interval  # demand at start of each week
            if t < H:
                dem[t] = amt
    elif mode_u == "ALTERNATING_50":
        arr = distribute_to_intervals(arr_total, H, arrival_interval, delay=0)
        dem = [0] * H
        pattern = [56, 56, 112, 112]  # sums to 336 = dem_total at 50% util
        for i, amt in enumerate(pattern):
            t = i * demand_interval
            if t < H:
                dem[t] = amt
    elif mode_u == "ALTERNATING_92":
        arr = distribute_to_intervals(arr_total, H, arrival_interval, delay=0)
        dem = [0] * H
        pattern = [84, 84, 224, 224]  # sums to 616 = dem_total at 92% util
        for i, amt in enumerate(pattern):
            t = i * demand_interval
            if t < H:
                dem[t] = amt
        dem_total = sum(pattern)
        u = dem_total / arr_total if arr_total > 0 else 0.0
    else:
        raise ValueError("mode must be ALL_AT_T0, UNIFORM, ALTERNATING, ALTERNATING_50, or ALTERNATING_92")

    common["arrivals_schedule"] = {0: arr}
    common["demand_schedule"] = {0: dem}
    common["demand"] = {0: dem_total}
    common["arrivals"] = {0: arr_total}
    common["utilization"] = u
    common["demand_interval"] = int(demand_interval)
    common["demand_delay"] = int(demand_delay)

    return common
