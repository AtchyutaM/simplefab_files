from __future__ import annotations

from typing import Callable, Dict, Any, Optional, Tuple, Sequence
import numpy as np
import random

from .sim import ProductionLine

ACTIONS = ["None", "Prod0", "Prod1"]


def policy_commander(_: ProductionLine) -> Optional[Dict[str, str]]:
    """Return None to let ProductionLine use the built-in Commander heuristic."""
    return None


def make_agent_policy(model) -> Callable[[ProductionLine], Dict[str, str]]:
    def _policy(line: ProductionLine) -> Dict[str, str]:
        obs = np.array(line.get_observation(current_time=line._t, norm=True), dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        a0, a1, a2, a3 = map(int, action)
        return {
            "machine0": ACTIONS[a0],
            "machine1": ACTIONS[a1],
            "machine2": ACTIONS[a2],
            "machine3": ACTIONS[a3],
        }
    return _policy


def run_episode(
    common_cfg: Dict[str, Any],
    policy_fn: Callable[[ProductionLine], Optional[Dict[str, str]]],
    H: Optional[int] = None,
    log_first_n: int = 0,
) -> Tuple[Dict[str, Any], ProductionLine]:
    pt = common_cfg["processing_times"]
    setups = {None: {0: 1, 1: 1}, 0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}}

    line = ProductionLine(
        {0: pt[0][0], 1: pt[0][1]},
        {0: pt[1][0], 1: pt[1][1]},
        setups,
        {0: pt[2][0], 1: pt[2][1]},
        {0: pt[3][0], 1: pt[3][1]},
        setups,
        common_cfg=common_cfg,
    )
    line.logging_enabled = False

    horizon = int(H if H is not None else common_cfg["time_horizon"])
    for t in range(horizon):
        override = policy_fn(line)  # None -> Commander
        r = line.run_step(current_time=t, actions_override=override)

        if t < log_first_n:
            last = line.cost_log[-1]
            print(f"t={t:03d} r={r:+8.2f} profit={last['profit']:+10.2f} invalid_sim={last['invalid_actions']}")

    c = line.costs
    throughput_p0 = sum(1 for p, _ in line.demand_met_log if p == 0)
    throughput_p1 = sum(1 for p, _ in line.demand_met_log if p == 1)

    metrics = {
        "profit": line.profit_total(),
        "revenue": c["revenue"],
        "production": c["production"],
        "setup": c["setup"],
        "inventory": c["inventory"],
        "backorder": c["backorder"],
        "throughput_p0": throughput_p0,
        "throughput_p1": throughput_p1,
        "throughput_total": throughput_p0 + throughput_p1,
    }
    return metrics, line


def compare_vs_commander(
    common_cfg: Dict[str, Any],
    model,
    H: Optional[int] = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> Tuple[Dict[str, float], Dict[str, float], ProductionLine]:
    def avg_metrics(runs):
        keys = runs[0].keys()
        return {k: float(np.mean([m[k] for m in runs])) for k in keys}

    agent_policy = make_agent_policy(model)

    commander_runs, agent_runs = [], []
    last_agent_line = None

    for s in seeds:
        np.random.seed(s); random.seed(s)

        m_comm, _ = run_episode(common_cfg, policy_commander, H=H)
        commander_runs.append(m_comm)

        m_agent, last_agent_line = run_episode(common_cfg, agent_policy, H=H)
        agent_runs.append(m_agent)

    commander_avg = avg_metrics(commander_runs)
    agent_avg = avg_metrics(agent_runs)

    print("\n=== Commander vs Agent (avg over seeds) ===")
    def fmt(name, m):
        print(
            f"{name:10s} | Profit {m['profit']:+10.2f} | Rev {m['revenue']:+9.2f} | Prod {m['production']:+8.2f} "
            f"| Setup {m['setup']:+7.2f} | Inv {m['inventory']:+7.2f} | BO {m['backorder']:+7.2f} "
            f"| Thru(P0,P1)=({int(m['throughput_p0'])},{int(m['throughput_p1'])})"
        )
    fmt("Commander", commander_avg)
    fmt("Agent", agent_avg)

    return commander_avg, agent_avg, last_agent_line
