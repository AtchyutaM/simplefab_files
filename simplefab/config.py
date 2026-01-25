from __future__ import annotations

from typing import Dict, Any, List


def make_common_config(
    mode: str = "UNIFORM",
    H: int = 1000,
    alpha: float = 0.5,          # fraction of Product 0 in the mix (0..1)
    utilization: float = 0.92     # target share of the bottleneck capacity (0..1)
) -> Dict[str, Any]:
    """
    Build a single "source of truth" config for both MILP and Simulation.

    Capacity math (bottleneck = batch stages 0 and 2):
      time_per_unit: P0=4, P1=5 (because 16/4 and 20/4 with batch size 4)
      4*x0 + 5*x1 ≈ u * H
      with x0 = alpha*T, x1=(1-alpha)*T => T = u*H / (4*alpha + 5*(1-alpha))

    We then round x0 and x1 down to the nearest multiple of the batch size (4),
    so batch machines can always start complete batches.
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
        "inventory_cost_per_unit": {0: 0.5, 1: 0.6},
        "backorder_cost_per_unit": {0: 1.0, 1: 1.0},

        "initial_inventory": {
            0: {0: 0, 1: 0},
            1: {0: 0, 1: 0},
            2: {0: 0, 1: 0},
            3: {0: 0, 1: 0},
            "finished": {0: 0, 1: 0},
        },

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

    T_cont = (u * H) / denom
    x0_cont = alpha * T_cont
    x1_cont = (1.0 - alpha) * T_cont

    BATCH = int(common["batch_sizes"][0])  # batch machine size (4)
    total_0 = (int(x0_cont) // BATCH) * BATCH
    total_1 = (int(x1_cont) // BATCH) * BATCH

    def spread(total: int, horizon: int) -> List[int]:
        if horizon <= 0:
            return []
        base, r = divmod(total, horizon)
        return [base + (1 if t < r else 0) for t in range(horizon)]

    mode_u = mode.upper().strip()
    if mode_u == "ALL_AT_T0":
        arr0 = [0] * H
        arr1 = [0] * H
        dem0 = [0] * H
        dem1 = [0] * H
        if H > 0:
            arr0[0] = total_0
            arr1[0] = total_1
            dem0[0] = total_0
            dem1[0] = total_1
    elif mode_u == "UNIFORM":
        arr0 = spread(total_0, H)
        arr1 = spread(total_1, H)
        dem0 = arr0[:]
        dem1 = arr1[:]
    else:
        raise ValueError("mode must be ALL_AT_T0 or UNIFORM")

    common["arrivals_schedule"] = {0: arr0, 1: arr1}
    common["demand_schedule"] = {0: dem0, 1: dem1}
    common["demand"] = {0: total_0, 1: total_1}
    common["mix_alpha"] = alpha
    common["utilization"] = u

    return common
