"""
CP-SAT optimization for the SimpleFab single-product schedule.

Given the deterministic dynamics of the line, finds the profit-maximizing
schedule (set of M0/M1/M2/M3 start times). Two modes:

  Full-horizon mode (t_start=0, no initial_state): solve from a fresh start.
    cfg = make_common_config(mode="ALTERNATING_92")
    result = solve_optimal_schedule(cfg, time_limit_sec=180)

  Partial-state mode (t_start>0, initial_state provided): solve from any
  point along an in-progress trajectory. This is what the DAgger oracle
  uses — given the agent's current state, query the MIP for the optimal
  next action.
    state = {"q0": 16, "q1": 4, ..., "qfin": 8,
             "in_flight": [(0, 1024)],   # M0 will complete at tick 1024
             "backlog": [0, 0, ...]}     # arrival times of unmet demand
    result = solve_optimal_schedule(cfg, t_start=512, initial_state=state)

This module is independent of env.py and sim.py — importing optim does NOT
change training behavior. Use it via scripts_1p/run_optim.py or via the
expert.py oracle abstraction.

Formulation (matches sim.py semantics exactly):
  - x[m][t] in {0,1}: machine m starts at tick t
  - Capacity: at most 1 start per machine in any pt[m]-tick window
  - Pre-t_start ticks: x[m][t] = 0 (decisions live only in [t_start, H_solve))
  - In-flight machines: x[m][t] = 0 in their busy window; their completion
    deposits product into the next queue at the appropriate tick
  - Queue recurrences (M0 completes 16 ticks after start, etc.)
  - cum_ship monotone, bounded by demand and qfin+production+initial qfin
  - late[t] >= cum_demand_arrived_>=672_ticks_ago[t] - cum_ship[t]
  - Maximize: revenue * (cum_ship[H_solve-1] - already_shipped) - prod - WIP - FG - backorder

Costs ($0.005, $0.04, etc.) are scaled by COST_SCALE=1000 so CP-SAT works
with integer arithmetic. Profit is rescaled on the way out.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

from ortools.sat.python import cp_model


# Costs are scaled to integers for CP-SAT. 1e5 (not 1e3) because deep lines have
# WIP coefficients as small as $0.00068/unit/tick — at 1e3 those round to 1 and
# carry a ~47% coefficient error. At 1e5 they are exact to 5 decimal places, and
# the largest objective term stays around 1e10, far inside int64.
COST_SCALE = 100000

# Default processing times for the original 4-machine line. Any config carrying
# `n_machines` supplies its own; this is only the fallback so old callers work.
PT = {0: 16, 1: 2, 2: 16, 3: 2}


def _line_params(common_cfg: Dict[str, Any]):
    """(M, pt, take, make, route) for any line depth. route[m] = destination
    queue of machine m's output, -1 meaning finished goods."""
    M = int(common_cfg.get("n_machines", 4))
    pt = [int(common_cfg["processing_times"][m][0]) for m in range(M)]
    batch = [int(common_cfg["batch_sizes"][m]) for m in range(M)]
    take = list(batch)
    make = list(batch)
    route = list(range(1, M + 1))
    route[M - 1] = -1
    return M, pt, take, make, route


def commander_schedule_general(common_cfg: Dict[str, Any], horizon: int) -> Dict[int, List[int]]:
    """
    Greedy "run whenever idle and fed" schedule for a line of any depth, in plain
    Python — used as the CP-SAT warm-start hint when the line is not the original
    4-machine one (`commander_schedule` goes through ProductionLine, which is
    hardcoded to 4 machines).

    Only needs to be *feasible* to be a useful hint, and it is so by construction:
    a machine starts only when its register is empty and its input queue holds at
    least `take[m]`.
    """
    M, pt, take, make, route = _line_params(common_cfg)
    arr = list(common_cfg["arrivals_schedule"][0])
    q = [0] * M
    rem = [0] * M                      # ticks until completion; 0 = idle
    starts: Dict[int, List[int]] = {m: [] for m in range(M)}

    for t in range(horizon):
        if t < len(arr):
            q[0] += int(arr[t])

        # `sim.commander_decide` is called BEFORE completions inside run_step, so
        # it sees the pre-completion busy flag: a machine finishing at tick t still
        # reads as busy and is NOT restarted at t, even though the simulator would
        # allow it. Reproducing that here keeps the hint identical to the commander
        # baseline the rest of the project quotes.
        decide = [rem[m] == 0 and q[m] >= take[m] for m in range(M)]

        # Completions. Decrement THEN test zero: a job started at tick s with
        # processing time pt completes at s+pt, matching the MILP's x[m][t-pt[m]]
        # term. Testing `rem == 1` before decrementing completes it a tick early,
        # which makes the hint infeasible — CP-SAT then silently discards it and
        # searches from scratch, which is how this was found.
        for m in range(M):
            if rem[m] > 0:
                rem[m] -= 1
                if rem[m] == 0:
                    dest = route[m]
                    if dest >= 0:
                        q[dest] += make[m]

        # Starts, re-checked against the post-completion state exactly as run_step does.
        for m in range(M):
            if decide[m] and rem[m] == 0 and q[m] >= take[m]:
                q[m] -= take[m]
                rem[m] = pt[m]
                starts[m].append(t)
    return starts


def _default_state(common_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The state at t=0 of a fresh ProductionLine: empty queues, qfin0 from cfg, no in-flight."""
    qfin0 = int(common_cfg.get("initial_finished_inventory", {0: 0}).get(0, 0))
    M = int(common_cfg.get("n_machines", 4))
    state = {"q": [0] * M, "qfin": qfin0, "in_flight": [], "backlog": []}
    # Legacy q0..q3 keys, kept so existing 4-machine callers keep working.
    for m in range(min(M, 4)):
        state[f"q{m}"] = 0
    return state


def _inject_state(line, t_start: int, state: Dict[str, Any]) -> None:
    """Mutate a ProductionLine to match `state` at the start of tick t_start.

    Used by commander_schedule() to compute a hint from a partial state.
    """
    line.queues = {
        "queue0": [], "queue1": [], "queue2": [], "queue3": [],
        "queue_fin": [], "demand": [],
    }
    for _ in range(int(state["q0"])):
        line.queues["queue0"].append(0)
    for _ in range(int(state["q1"])):
        line.queues["queue1"].append(0)
    for _ in range(int(state["q2"])):
        line.queues["queue2"].append(0)
    for _ in range(int(state["q3"])):
        line.queues["queue3"].append(0)
    for _ in range(int(state["qfin"])):
        line.queues["queue_fin"].append(-1)
    for arrival_t in state.get("backlog", []):
        line.queues["demand"].append(int(arrival_t))

    machines = [line.machine0, line.machine1, line.machine2, line.machine3]
    for m in machines:
        m.busy = False
        m.finish_time = 0
        m.event_log = []
    for (m_idx, finish_t) in state.get("in_flight", []):
        machines[int(m_idx)].busy = True
        machines[int(m_idx)].finish_time = int(finish_t)

    line.costs = {"revenue": 0.0, "production": 0.0, "inventory": 0.0, "wip": 0.0, "backorder": 0.0}
    line.cost_log = []
    line.demand_met_log = []
    line.event_log = []
    line.logs = {k: [] for k in line.queues}
    line._t = int(t_start)


def commander_schedule(
    common_cfg: Dict[str, Any],
    t_start: int = 0,
    initial_state: Optional[Dict[str, Any]] = None,
    horizon: Optional[int] = None,
) -> Dict[int, List[int]]:
    """Run Commander heuristic from t_start to horizon, return per-machine start ticks.

    Used as a warm-start hint for the MIP solver. With partial state, the line
    is first reset to `initial_state` so Commander starts from there.
    """
    from .sim import ProductionLine

    H = int(common_cfg["time_horizon"])
    H_end = min(H, int(horizon)) if horizon is not None else H

    line = ProductionLine(common_cfg=common_cfg)
    line.logging_enabled = False
    if initial_state is not None or t_start > 0:
        state = initial_state if initial_state is not None else _default_state(common_cfg)
        _inject_state(line, t_start, state)

    for t in range(t_start, H_end):
        line.run_step(current_time=t, actions_override=None)

    starts: Dict[int, List[int]] = {m: [] for m in range(4)}
    machines = [line.machine0, line.machine1, line.machine2, line.machine3]
    for m_idx in range(4):
        for entry in machines[m_idx].event_log:
            s = entry[0] if isinstance(entry, tuple) else int(entry)
            if t_start <= s < H_end:
                starts[m_idx].append(int(s))
    return starts


def _build_in_flight_arrays(
    in_flight: List[Tuple[int, int]],
    t_start: int,
    H_solve: int,
    M: int = 4,
    pt: Optional[List[int]] = None,
) -> Tuple[Dict[int, List[int]], Dict[int, set]]:
    """Pre-compute, for each machine: ticks at which an in-flight job completes,
    and the set of ticks during which the machine is busy with an in-flight job.
    """
    if pt is None:
        pt = [PT[m] for m in range(M)]
    completes_at = {m: [0] * H_solve for m in range(M)}
    busy_ticks = {m: set() for m in range(M)}
    for (m_idx, c) in in_flight:
        m_idx = int(m_idx)
        c = int(c)
        if t_start <= c < H_solve:
            completes_at[m_idx][c] += 1
        # In-flight machine is busy from (c - pt[m]) to (c - 1) inclusive.
        # Within our solve window, intersect with [t_start, c-1].
        busy_from = max(c - pt[m_idx], t_start)
        busy_to = min(c, H_solve)  # exclusive
        for tick in range(busy_from, busy_to):
            busy_ticks[m_idx].add(tick)
    return completes_at, busy_ticks


def solve_optimal_schedule(
    common_cfg: Dict[str, Any],
    t_start: int = 0,
    initial_state: Optional[Dict[str, Any]] = None,
    lookahead_ticks: Optional[int] = None,
    time_limit_sec: float = 300.0,
    num_workers: int = 8,
    verbose: bool = True,
    use_commander_hint: bool = True,
) -> Dict[str, Any]:
    """Solve the schedule MIP from `t_start` with the given `initial_state`.

    Default args (t_start=0, initial_state=None, lookahead_ticks=None) reproduce
    the original full-horizon solve from a fresh start.

    Args:
        common_cfg: SimpleFab config (output of make_common_config).
        t_start: First tick where the optimizer can decide actions. Pre-t_start
            ticks are constrained to x=0 and treated as historical.
        initial_state: Required if t_start > 0. Dict with keys:
            q0, q1, q2, q3, qfin (int counts), in_flight (list of
            (machine_id, completion_tick)), backlog (list of arrival ticks of
            unmet demand). If None, the default fresh-start state is used.
        lookahead_ticks: If provided, solve over [t_start, t_start+lookahead).
            Useful for fast DAgger oracle queries (smaller MIP). Default: full
            horizon to H.
        time_limit_sec: Wall-time bound for CP-SAT.
        num_workers: Parallel workers for CP-SAT.
        verbose: Print solver progress and summary.
        use_commander_hint: Warm-start the MIP with the Commander policy run
            from `initial_state`. Cuts solve time dramatically.

    Returns dict with status, profit (incremental over the solve window),
    objective_bound, starts (per-machine list of decision ticks in
    [t_start, H_solve)), shipments, throughput (units shipped in window),
    breakdown (incremental cost/revenue), and queues trajectories.
    """
    H = int(common_cfg["time_horizon"])
    if lookahead_ticks is not None:
        H_solve = min(H, t_start + int(lookahead_ticks))
    else:
        H_solve = H

    if initial_state is None:
        if t_start != 0:
            raise ValueError("initial_state is required when t_start > 0")
        initial_state = _default_state(common_cfg)

    arr = list(common_cfg["arrivals_schedule"][0])
    dem = list(common_cfg["demand_schedule"][0])

    M, PTs, TAKE, MAKE, ROUTE = _line_params(common_cfg)
    revenue = int(common_cfg["revenue_per_unit"][0])
    # Float, not int: deep lines have fractional per-stage costs (24/M split 2:1).
    prod_cost = [float(common_cfg["production_cost"][m][0]) for m in range(M)]
    inv_cost_per = float(common_cfg["inventory_cost_per_unit"][0])
    wip_costs = [float(common_cfg.get("wip_cost_per_unit", {}).get(m, 0.0)) for m in range(M)]
    bo_cost_per = float(common_cfg["backorder_cost_per_unit"][0])
    demand_interval = int(common_cfg.get("demand_interval", 672))

    if "q" in initial_state:
        q_init = [int(v) for v in initial_state["q"]]
    else:
        q_init = [int(initial_state[f"q{m}"]) for m in range(M)]
    if len(q_init) != M:
        raise ValueError(f"initial_state queue vector has {len(q_init)} entries, expected {M}")
    qfin_init = int(initial_state["qfin"])
    in_flight = [(int(m), int(c)) for (m, c) in initial_state.get("in_flight", [])]
    backlog = [int(at) for at in initial_state.get("backlog", [])]

    # cum_dem_full[t+1] = total demand arrived by tick t, counting BOTH
    # backlog items (arrival times before t_start) AND new arrivals from t_start onward.
    cum_dem_full = [0] * (H + 1)
    for at in backlog:
        if 0 <= at < H:
            cum_dem_full[at + 1] += 1
    for t in range(t_start, H):
        cum_dem_full[t + 1] += int(dem[t])
    for k in range(1, H + 1):
        cum_dem_full[k] += cum_dem_full[k - 1]

    # Total demand already arrived by tick t_start - 1 (everything before solve window).
    # This equals (already shipped) + (current backlog), so:
    already_shipped = cum_dem_full[t_start] - len(backlog)
    if already_shipped < 0:
        raise ValueError(
            f"Inconsistent state: backlog ({len(backlog)}) exceeds cumulative "
            f"demand arrived before t_start={t_start} ({cum_dem_full[t_start]})."
        )

    completes_at, busy_ticks = _build_in_flight_arrays(
        in_flight, t_start, H_solve, M=M, pt=PTs)

    if verbose:
        print(
            f"[optim] t_start={t_start}, H_solve={H_solve} "
            f"(lookahead={H_solve - t_start}t), already_shipped={already_shipped}, "
            f"backlog={len(backlog)}, in_flight={len(in_flight)}"
        )

    model = cp_model.CpModel()

    # Decision vars over [0, H_solve). Pre-t_start and in-flight-busy ticks are forced to 0.
    x = [[model.NewBoolVar(f"x{m}_{t}") for t in range(H_solve)] for m in range(M)]
    for m in range(M):
        for t in range(H_solve):
            if t < t_start or t in busy_ticks[m]:
                model.Add(x[m][t] == 0)

    # Capacity: at most one start per pt[m]-tick window
    for m in range(M):
        for t in range(t_start, H_solve):
            window = [x[m][tau] for tau in range(max(0, t - PTs[m] + 1), t + 1)]
            model.Add(sum(window) <= 1)

    max_inv = qfin_init + sum(q_init) + sum(arr) + len(backlog) + 16
    q = [{t: model.NewIntVar(0, max_inv, f"q{m}_{t}") for t in range(t_start, H_solve)}
         for m in range(M)]
    qfin = {t: model.NewIntVar(0, max_inv, f"qfin_{t}") for t in range(t_start, H_solve)}
    cum_ship = {t: model.NewIntVar(already_shipped, max_inv, f"cs_{t}") for t in range(t_start, H_solve)}
    late = {t: model.NewIntVar(0, max_inv, f"late_{t}") for t in range(t_start, H_solve)}

    # Inbound edges per queue, precomputed: queue j is fed by every machine whose
    # route lands on j. For the serial line that is exactly machine j-1, but the
    # loop form is what makes reentrancy (family 5) a config change, not a rewrite.
    feeders = {j: [m for m in range(M) if ROUTE[m] == j] for j in range(M)}
    shippers = [m for m in range(M) if ROUTE[m] < 0]

    def _completed_units(m, t):
        """Units machine m delivers at tick t: a start pt[m] ticks ago, plus any
        in-flight job carried in from the initial state."""
        terms = []
        if t - PTs[m] >= 0:
            terms.append(MAKE[m] * x[m][t - PTs[m]])
        c = completes_at[m][t]
        if c:
            terms.append(MAKE[m] * c)
        return terms

    for t in range(t_start, H_solve):
        for j in range(M):
            prev = q[j][t - 1] if t > t_start else q_init[j]
            inflow = [arr[t]] if j == 0 else []
            for src in feeders[j]:
                inflow += _completed_units(src, t)
            model.Add(q[j][t] == prev + sum(inflow) - TAKE[j] * x[j][t])

        prev_qfin = qfin[t - 1] if t > t_start else qfin_init
        prev_cs = cum_ship[t - 1] if t > t_start else already_shipped
        fg_in = []
        for src in shippers:
            fg_in += _completed_units(src, t)
        model.Add(qfin[t] == prev_qfin + sum(fg_in) - cum_ship[t] + prev_cs)

        # Shipping: monotone, bounded by total demand arrived
        model.Add(cum_ship[t] <= cum_dem_full[t + 1])
        if t > t_start:
            model.Add(cum_ship[t] >= cum_ship[t - 1])

        # Backorder
        cutoff = t - demand_interval + 1
        if cutoff > 0:
            model.Add(late[t] >= cum_dem_full[cutoff] - cum_ship[t])

    # Objective — INCREMENTAL profit over the solve window only.
    # Pre-t_start production / shipping is sunk and not counted here.
    # Scale to integers once, at COST_SCALE resolution. Production cost per start
    # is take[m] units at prod_cost[m] each; both may be fractional on deep lines.
    prod_scale = [int(round(TAKE[m] * prod_cost[m] * COST_SCALE)) for m in range(M)]
    wip_scale = [int(round(w * COST_SCALE)) for w in wip_costs]
    inv_scale = int(round(inv_cost_per * COST_SCALE))
    bo_scale = int(round(bo_cost_per * COST_SCALE))

    rev_scaled = revenue * COST_SCALE * (cum_ship[H_solve - 1] - already_shipped)
    prod_scaled = sum(
        prod_scale[m] * sum(x[m][t] for t in range(t_start, H_solve)) for m in range(M)
    )
    wip_scaled = sum(
        wip_scale[m] * q[m][t]
        for m in range(M) if wip_scale[m]
        for t in range(t_start, H_solve)
    )
    inv_scaled = inv_scale * sum(qfin[t] for t in range(t_start, H_solve))
    bo_scaled = bo_scale * sum(late[t] for t in range(t_start, H_solve))

    model.Maximize(rev_scaled - prod_scaled - wip_scaled - inv_scaled - bo_scaled)

    if use_commander_hint:
        if verbose:
            print("[optim] Computing Commander schedule for hint...")
        if M == 4:
            cmd_starts = commander_schedule(common_cfg, t_start=t_start,
                                            initial_state=initial_state, horizon=H_solve)
        else:
            # ProductionLine is hardcoded to 4 machines; use the depth-agnostic
            # greedy schedule instead. Fresh starts only.
            if t_start != 0:
                raise ValueError("commander hint at t_start>0 requires the 4-machine line")
            cmd_starts = commander_schedule_general(common_cfg, H_solve)
        for m in range(M):
            cmd_set = set(cmd_starts[m])
            for t in range(t_start, H_solve):
                if t not in busy_ticks[m]:
                    model.AddHint(x[m][t], 1 if t in cmd_set else 0)
        if verbose:
            counts = [len(cmd_starts[m]) for m in range(M)]
            print(f"[optim] Commander hint starts per machine: {counts}")

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
            "t_start": t_start,
            "H_solve": H_solve,
        }

    starts: Dict[int, List[int]] = {m: [] for m in range(M)}
    for m in range(M):
        for t in range(t_start, H_solve):
            if solver.Value(x[m][t]) == 1:
                starts[m].append(t)

    shipments: List[int] = []
    prev = already_shipped
    for t in range(t_start, H_solve):
        v = solver.Value(cum_ship[t])
        shipments.append(v - prev)
        prev = v

    profit = solver.ObjectiveValue() / COST_SCALE
    bound = solver.BestObjectiveBound() / COST_SCALE

    units_shipped = solver.Value(cum_ship[H_solve - 1]) - already_shipped
    total_rev = revenue * units_shipped
    total_prod = sum(
        TAKE[m] * prod_cost[m] * sum(solver.Value(x[m][t]) for t in range(t_start, H_solve))
        for m in range(M)
    )
    total_inv = sum(solver.Value(qfin[t]) for t in range(t_start, H_solve)) * inv_cost_per
    total_wip = sum(
        solver.Value(q[m][t]) * wip_costs[m]
        for m in range(M) if wip_costs[m]
        for t in range(t_start, H_solve)
    )
    total_bo = sum(solver.Value(late[t]) for t in range(t_start, H_solve)) * bo_cost_per

    return {
        "status": status_name,
        "is_optimal": status == cp_model.OPTIMAL,
        "profit": profit,
        "objective_bound": bound,
        "starts": starts,
        "shipments": shipments,
        "throughput": int(units_shipped),
        "breakdown": {
            "revenue": float(total_rev),
            "production": float(total_prod),
            "wip": float(total_wip),
            "inventory": float(total_inv),
            "backorder": float(total_bo),
        },
        "queues": {
            **{f"q{m}": [solver.Value(q[m][t]) for t in range(t_start, H_solve)]
               for m in range(M)},
            "qfin": [solver.Value(qfin[t]) for t in range(t_start, H_solve)],
        },
        "n_machines": M,
        "solver_wall_time": solver.WallTime(),
        "t_start": t_start,
        "H_solve": H_solve,
    }


def starts_to_actions(starts: Dict[int, List[int]], H: int) -> List[Dict[str, bool]]:
    """Convert per-machine start-time lists to per-tick action dicts (sim.py format).

    The returned list has length H; ticks not in any start list become all-Idle.
    """
    sets = {m: set(starts[m]) for m in range(4)}
    return [
        {f"machine{m}": (t in sets[m]) for m in range(4)}
        for t in range(H)
    ]


def verify_solution(
    common_cfg: Dict[str, Any],
    starts: Dict[int, List[int]],
    t_start: int = 0,
    initial_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Replay starts through ProductionLine and report sim's accounting.

    For full-horizon (t_start=0, no initial_state), this confirms the MIP and
    sim agree on profit. For partial-state, the sim is first injected with
    initial_state, then the starts are applied from t_start onward; the
    reported costs are only those accumulated within the solve window.
    """
    from .sim import ProductionLine

    H = int(common_cfg["time_horizon"])
    actions = starts_to_actions(starts, H)
    line = ProductionLine(common_cfg=common_cfg)
    line.logging_enabled = False

    if initial_state is not None or t_start > 0:
        state = initial_state if initial_state is not None else _default_state(common_cfg)
        _inject_state(line, t_start, state)

    for t in range(t_start, H):
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
