"""Replay the saved MIP-optimal schedule through sim with logging on, then
render it through generate_dashboards_compare.generate_dashboard_for_sim.

Produces output_data_1p/reports/<exp_name>_optimum.png/svg with the same
visual style as the Commander/Agent dashboards.
"""
from __future__ import annotations
import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplefab_1p.env import FabEnv
from simplefab_1p.optim import starts_to_actions
from scripts_1p.generate_dashboards_compare import generate_dashboard_for_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=str, required=True,
                        help="Path to optim solution pickle (e.g., output_data_1p/optim/alternating_92_optim.pkl)")
    parser.add_argument("--exp-name", type=str, default="alternating_92pct",
                        help="Output filename prefix")
    args = parser.parse_args()

    with open(args.solution, "rb") as f:
        payload = pickle.load(f)

    cfg = payload["config"]
    starts = payload["result"]["starts"]
    H = cfg["time_horizon"]

    env = FabEnv(common_cfg=cfg, normalize_obs=True, invalid_action_penalty=0.0)
    env.reset()
    env.line.logging_enabled = True

    actions = starts_to_actions(starts, H)
    for t in range(H):
        env.line.run_step(current_time=t, actions_override=actions[t], invalid_penalty=0.0)

    profit = env.line.profit_total()
    shipped = len(env.line.demand_met_log)
    print(f"MIP-optimum replay: profit=${profit:,.2f}, shipped={shipped}/{cfg['demand'][0]}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output_data_1p", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.exp_name}_optimum.png")
    generate_dashboard_for_sim(env, H, out_path)
    print(f"Wrote dashboard: {out_path}")


if __name__ == "__main__":
    main()
