from __future__ import annotations

from typing import Callable, Dict, Any, Optional, Tuple, Sequence
import numpy as np
import random

from .sim import ProductionLine

ACTIONS = ["Idle", "Run"]


def policy_commander(_: ProductionLine) -> None:
    """Return None to let ProductionLine use the built-in commander heuristic."""
    return None


def make_agent_policy(model) -> Callable[[ProductionLine], Dict[str, bool]]:
    def _policy(line: ProductionLine) -> Dict[str, bool]:
        obs = np.array(line.get_observation(current_time=line._t, norm=True), dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        a0, a1, a2, a3 = map(int, action)
        return {
            "machine0": bool(a0),
            "machine1": bool(a1),
            "machine2": bool(a2),
            "machine3": bool(a3),
        }
    return _policy


def run_episode(
    common_cfg: Dict[str, Any],
    policy_fn: Callable[[ProductionLine], Optional[Dict[str, bool]]],
    H: Optional[int] = None,
    log_first_n: int = 0,
) -> Tuple[Dict[str, Any], ProductionLine]:
    line = ProductionLine(common_cfg=common_cfg)
    line.logging_enabled = False

    horizon = int(H if H is not None else common_cfg["time_horizon"])
    for t in range(horizon):
        override = policy_fn(line)  # None -> Commander
        r = line.run_step(current_time=t, actions_override=override)

        if t < log_first_n:
            last = line.cost_log[-1]
            print(f"t={t:03d} r={r:+8.2f} profit={last['profit']:+10.2f} invalid_sim={last['invalid_actions']}")

    c = line.costs
    throughput = len(line.demand_met_log)

    metrics = {
        "profit": line.profit_total(),
        "revenue": c["revenue"],
        "production": c["production"],
        "inventory": c["inventory"],
        "backorder": c["backorder"],
        "throughput": throughput,
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
            f"| Inv {m['inventory']:+7.2f} | BO {m['backorder']:+7.2f} "
            f"| Throughput={int(m['throughput'])}"
        )
    fmt("Commander", commander_avg)
    fmt("Agent", agent_avg)

    return commander_avg, agent_avg, last_agent_line
