"""
CP-SAT optimization for the SimpleFab single-product schedule.

Given full knowledge of arrivals and demand, finds the profit-maximizing
schedule (set of M0/M1/M2/M3 start times) that the line can possibly run.

This module is intentionally independent of env.py and sim.py: importing
optim does NOT change training behavior. Use it via scripts_1p/run_optim.py
or directly:

    from simplefab_1p.optim import solve_optimal_schedule, verify_solution
    from simplefab_1p import make_common_config

    cfg = make_common_config(mode="ALTERNATING_92")
    result = solve_optimal_schedule(cfg, time_limit_sec=300)
    verify_solution(cfg, result["starts"])  # replay through sim.py

Formulation (matches sim.py per-tick semantics exactly):
  - x[m][t] in {0,1}: machine m starts at tick t
  - Capacity: at most 1 start per machine in any pt[m]-tick window
  - Queue recurrences (M0 completes 16 ticks after start, etc.)
  - cum_ship[t] monotone, bounded by cum_demand[t+1] and qfin0+M3_done(t-2)
  - late[t] >= cum_demand_arrived_more_than_672_ticks_ago[t] - cum_ship[t]
  - Maximize: revenue * cum_ship[H-1] - prod - WIP - FG - backorder

Costs ($0.005, $0.04, etc.) are scaled by COST_SCALE=1000 so CP-SAT works
with integer arithmetic. Profit is rescaled on the way out.
"""
from __future__ import annotations
from typing import Dict, Any, List

from ortools.sat.python import cp_model


COST_SCALE = 1000


def commander_schedule(common_cfg: Dict[str, Any]) -> Dict[int, List[int]]:
    """Run the Commander heuristic through sim.py and return per-machine start times.

    Used as a warm-start hint for the MIP solver.
    """
    from .sim import ProductionLine

    H = int(common_cfg["time_horizon"])
    line = ProductionLine(common_cfg=common_cfg)
    line.logging_enabled = False
    for t in range(H):
        line.run_step(current_time=t, actions_override=None)  # None -> Commander
    return {
        0: [s for (s, _e, *_rest) in line.machine0.event_log],
        1: [s for (s, _e) in line.machine1.event_log],
        2: [s for (s, _e, *_rest) in line.machine2.event_log],
        3: [s for (s, _e) in line.machine3.event_log],
    }


def solve_optimal_schedule(
    common_cfg: Dict[str, Any],
    time_limit_sec: float = 300.0,
    num_workers: int = 8,
    verbose: bool = True,
    use_commander_hint: bool = True,
) -> Dict[str, Any]:
    H = int(common_cfg["time_horizon"])
    arr = list(common_cfg["arrivals_schedule"][0])
    dem = list(common_cfg["demand_schedule"][0])
    pt = {0: 16, 1: 2, 2: 16, 3: 2}

    revenue = int(common_cfg["revenue_per_unit"][0])
    prod_cost = [int(common_cfg["production_cost"][m][0]) for m in range(4)]
    inv_cost_per = float(common_cfg["inventory_cost_per_unit"][0])
    wip_costs = [float(common_cfg.get("wip_cost_per_unit", {}).get(m, 0.0)) for m in range(4)]
    bo_cost_per = float(common_cfg["backorder_cost_per_unit"][0])
    demand_interval = int(common_cfg.get("demand_interval", 672))
    qfin0 = int(common_cfg.get("initial_finished_inventory", {0: 0}).get(0, 0))

    cum_arr = [0] * (H + 1)
    cum_dem = [0] * (H + 1)
    for t in range(H):
        cum_arr[t + 1] = cum_arr[t] + int(arr[t])
        cum_dem[t + 1] = cum_dem[t] + int(dem[t])

    if verbose:
        print(f"[optim] H={H}, total_arrivals={cum_arr[H]}, total_demand={cum_dem[H]}, qfin0={qfin0}")

    model = cp_model.CpModel()

    x = [[model.NewBoolVar(f"x{m}_{t}") for t in range(H)] for m in range(4)]

    # Capacity: at most one start per pt[m]-tick window
    for m in range(4):
        for t in range(H):
            window = [x[m][tau] for tau in range(max(0, t - pt[m] + 1), t + 1)]
            model.Add(sum(window) <= 1)

    max_inv = qfin0 + cum_arr[H]
    q0 = [model.NewIntVar(0, max_inv, f"q0_{t}") for t in range(H)]
    q1 = [model.NewIntVar(0, max_inv, f"q1_{t}") for t in range(H)]
    q2 = [model.NewIntVar(0, max_inv, f"q2_{t}") for t in range(H)]
    q3 = [model.NewIntVar(0, max_inv, f"q3_{t}") for t in range(H)]
    qfin = [model.NewIntVar(0, max_inv, f"qfin_{t}") for t in range(H)]
    cum_ship = [model.NewIntVar(0, max_inv, f"cs_{t}") for t in range(H)]
    late = [model.NewIntVar(0, max_inv, f"late_{t}") for t in range(H)]

    # Recurrent queue dynamics — O(1) terms per constraint
    for t in range(H):
        prev_q0 = q0[t - 1] if t > 0 else 0
        model.Add(q0[t] == prev_q0 + arr[t] - 4 * x[0][t])

        prev_q1 = q1[t - 1] if t > 0 else 0
        if t >= 16:
            model.Add(q1[t] == prev_q1 + 4 * x[0][t - 16] - x[1][t])
        else:
            model.Add(q1[t] == prev_q1 - x[1][t])

        prev_q2 = q2[t - 1] if t > 0 else 0
        if t >= 2:
            model.Add(q2[t] == prev_q2 + x[1][t - 2] - 4 * x[2][t])
        else:
            model.Add(q2[t] == prev_q2 - 4 * x[2][t])

        prev_q3 = q3[t - 1] if t > 0 else 0
        if t >= 16:
            model.Add(q3[t] == prev_q3 + 4 * x[2][t - 16] - x[3][t])
        else:
            model.Add(q3[t] == prev_q3 - x[3][t])

        # qfin recurrence: qfin[t] = qfin[t-1] + (M3 done at t) - (cum_ship[t] - cum_ship[t-1])
        prev_qfin = qfin[t - 1] if t > 0 else qfin0
        prev_cs = cum_ship[t - 1] if t > 0 else 0
        m3_done_now = x[3][t - 2] if t >= 2 else 0
        model.Add(qfin[t] == prev_qfin + m3_done_now - cum_ship[t] + prev_cs)

        # Shipping bounds
        model.Add(cum_ship[t] <= cum_dem[t + 1])
        if t > 0:
            model.Add(cum_ship[t] >= cum_ship[t - 1])

        # Backorder: late demand = demand arrived ≥ demand_interval ticks ago and not yet shipped
        cutoff = t - demand_interval + 1
        if cutoff > 0:
            model.Add(late[t] >= cum_dem[cutoff] - cum_ship[t])

    # Objective (all scaled by COST_SCALE=1000)
    prod_cost_total = (
        4 * prod_cost[0] * sum(x[0])
        + prod_cost[1] * sum(x[1])
        + 4 * prod_cost[2] * sum(x[2])
        + prod_cost[3] * sum(x[3])
    )

    wip_scale = [int(round(w * COST_SCALE)) for w in wip_costs]
    inv_scale = int(round(inv_cost_per * COST_SCALE))
    bo_scale = int(round(bo_cost_per * COST_SCALE))

    rev_scaled = revenue * COST_SCALE * cum_ship[H - 1]
    prod_scaled = COST_SCALE * prod_cost_total
    wip_scaled = sum(
        wip_scale[1] * q1[t] + wip_scale[2] * q2[t] + wip_scale[3] * q3[t]
        for t in range(H)
    )
    inv_scaled = inv_scale * sum(qfin[t] for t in range(H))
    bo_scaled = bo_scale * sum(late[t] for t in range(H))

    model.Maximize(rev_scaled - prod_scaled - wip_scaled - inv_scaled - bo_scaled)

    # Warm-start hint from the Commander heuristic — gives CP-SAT a feasible
    # starting solution worth ~$31K so it can refine instead of searching from scratch.
    if use_commander_hint:
        if verbose:
            print("[optim] Computing Commander schedule for hint...")
        cmd_starts = commander_schedule(common_cfg)
        for m in range(4):
            cmd_set = set(cmd_starts[m])
            for t in range(H):
                model.AddHint(x[m][t], 1 if t in cmd_set else 0)
        if verbose:
            counts = {m: len(cmd_starts[m]) for m in range(4)}
            print(f"[optim] Commander starts hinted: M0={counts[0]}, M1={counts[1]}, M2={counts[2]}, M3={counts[3]}")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = int(num_workers)
    if verbose:
        solver.parameters.log_search_progress = True

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": status_name,
            "is_optimal": False,
            "profit": None,
            "objective_bound": solver.BestObjectiveBound() / COST_SCALE,
            "solver_wall_time": solver.WallTime(),
        }

    starts: Dict[int, List[int]] = {m: [] for m in range(4)}
    for m in range(4):
        for t in range(H):
            if solver.Value(x[m][t]) == 1:
                starts[m].append(t)

    shipments: List[int] = []
    prev = 0
    for t in range(H):
        v = solver.Value(cum_ship[t])
        shipments.append(v - prev)
        prev = v

    profit = solver.ObjectiveValue() / COST_SCALE
    bound = solver.BestObjectiveBound() / COST_SCALE

    total_rev = revenue * solver.Value(cum_ship[H - 1])
    total_prod = solver.Value(prod_cost_total)
    total_inv = sum(solver.Value(qfin[t]) for t in range(H)) * inv_cost_per
    total_wip = sum(
        solver.Value(q1[t]) * wip_costs[1]
        + solver.Value(q2[t]) * wip_costs[2]
        + solver.Value(q3[t]) * wip_costs[3]
        for t in range(H)
    )
    total_bo = sum(solver.Value(late[t]) for t in range(H)) * bo_cost_per

    return {
        "status": status_name,
        "is_optimal": status == cp_model.OPTIMAL,
        "profit": profit,
        "objective_bound": bound,
        "starts": starts,
        "shipments": shipments,
        "throughput": int(solver.Value(cum_ship[H - 1])),
        "breakdown": {
            "revenue": float(total_rev),
            "production": float(total_prod),
            "wip": float(total_wip),
            "inventory": float(total_inv),
            "backorder": float(total_bo),
        },
        "queues": {
            "q0": [solver.Value(q0[t]) for t in range(H)],
            "q1": [solver.Value(q1[t]) for t in range(H)],
            "q2": [solver.Value(q2[t]) for t in range(H)],
            "q3": [solver.Value(q3[t]) for t in range(H)],
            "qfin": [solver.Value(qfin[t]) for t in range(H)],
        },
        "solver_wall_time": solver.WallTime(),
    }


def starts_to_actions(starts: Dict[int, List[int]], H: int) -> List[Dict[str, bool]]:
    """Convert per-machine start-time lists to per-tick action dicts (sim.py format)."""
    sets = {m: set(starts[m]) for m in range(4)}
    return [
        {f"machine{m}": (t in sets[m]) for m in range(4)}
        for t in range(H)
    ]


def verify_solution(common_cfg: Dict[str, Any], starts: Dict[int, List[int]]) -> Dict[str, Any]:
    """Replay the optimal schedule through ProductionLine and report its results.

    A mismatch between solver profit and sim.py profit means the formulation
    has drifted from the sim semantics.
    """
    from .sim import ProductionLine

    H = int(common_cfg["time_horizon"])
    actions = starts_to_actions(starts, H)
    line = ProductionLine(common_cfg=common_cfg)
    line.logging_enabled = False
    for t in range(H):
        line.run_step(current_time=t, actions_override=actions[t], invalid_penalty=0.0)

    return {
        "profit": line.profit_total(),
        "revenue": line.costs["revenue"],
        "production": line.costs["production"],
        "inventory": line.costs["inventory"],
        "wip": line.costs["wip"],
        "backorder": line.costs["backorder"],
        "throughput": len(line.demand_met_log),
    }
