"""
Evaluate a trained RL agent against the Commander heuristic.
Usage:
  python scripts/evaluate_agent.py [--model PATH] [--vecnorm PATH]
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from simplefab import make_common_config
from simplefab.eval import compare_vs_commander


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent vs Commander")
    parser.add_argument("--model", default="output_data/models/ppo_fab_policy",
                        help="Path to saved model (without .zip)")
    parser.add_argument("--vecnorm", default="output_data/models/vecnorm_stats.pkl",
                        help="Path to VecNormalize stats")
    args = parser.parse_args()

    common = make_common_config()

    print(f"Loading model from: {args.model}")
    model = MaskablePPO.load(args.model)

    print(f"Config: H={common['time_horizon']}, demand={common['demand']}, "
          f"arrivals={common['arrivals']}")
    print()

    commander_avg, agent_avg, _ = compare_vs_commander(common, model, seeds=(0, 1, 2, 3, 4))

    # Summary
    profit_diff = agent_avg["profit"] - commander_avg["profit"]
    pct = (profit_diff / abs(commander_avg["profit"]) * 100) if commander_avg["profit"] != 0 else 0
    print(f"\nAgent profit delta: {profit_diff:+.2f} ({pct:+.1f}% vs Commander)")


if __name__ == "__main__":
    main()
