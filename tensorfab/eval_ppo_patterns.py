"""
Evaluate the retrained Maskable PPO agents on their own demand patterns.

Uses the project's established evaluation methodology verbatim — temperature
scaled sampling at T=0.7, best of 20 episodes, via `run_single_episode` from
scripts_1p/generate_dashboards_compare.py — so the agent numbers are produced the
same way as every historical result in CLAUDE.md. Deterministic argmax is also
reported, since the documented 5-8% temperature gain is worth re-checking.

Everything runs under the CURRENT (halved) WIP costs, matching
tensorfab/compare_patterns.py, so the agent column drops straight into the
cross-method table.

Usage:  python -m tensorfab.eval_ppo_patterns
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from tensorfab.logbook import Run

MODELS = Path("output_data_1p/models")


def evaluate(model_path: Path, cfg, n_eval: int, temperature: float) -> Dict[str, Any]:
    from sb3_contrib import MaskablePPO

    from scripts_1p.generate_dashboards_compare import run_single_episode

    model = MaskablePPO.load(str(model_path), device="cpu")

    _, det = run_single_episode(model, cfg, temperature=None)
    profits = []
    best = None
    for _ in range(n_eval):
        _, s = run_single_episode(model, cfg, temperature=temperature)
        profits.append(s["profit"])
        if best is None or s["profit"] > best["profit"]:
            best = s
    profits = np.asarray(profits, dtype=np.float64)
    return {
        "deterministic": float(det["profit"]),
        "temp_best": float(best["profit"]),
        "temp_mean": float(profits.mean()),
        "temp_std": float(profits.std()),
        "shipped": int(best["shipped"]),
        "total_demand": int(best["total_demand"]),
        "revenue": float(best["revenue"]),
        "prod_cost": float(best["prod_cost"]),
        "inv_cost": float(best["inv_cost"]),
        "wip_cost": float(best["wip_cost"]),
        "bo_cost": float(best["bo_cost"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+",
                    default=["UNIFORM", "ALTERNATING_50", "ALTERNATING"])
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    from simplefab_1p.config import make_common_config

    with Run("eval_ppo_patterns", params=vars(args).copy(),
             note="T=0.7 best-of-20 evaluation of the retrained PPO agents, "
                  "current (halved) WIP costs") as run:
        results: Dict[str, Any] = {}
        for p in args.patterns:
            mp = MODELS / f"ppo_mp_{p}.zip"
            if not mp.exists():
                run.log(f"  {p}: MISSING {mp} — training may not have finished")
                continue
            cfg = make_common_config(mode=p)
            r = evaluate(mp, cfg, args.n_eval, args.temperature)
            results[p] = r
            run.log(f"  {p:<16} det=${r['deterministic']:>10,.0f}   "
                    f"T={args.temperature} best=${r['temp_best']:>10,.0f}   "
                    f"mean=${r['temp_mean']:>10,.0f} +/- {r['temp_std']:,.0f}   "
                    f"shipped {r['shipped']}/{r['total_demand']}")

        (run.dir / "ppo_results.json").write_text(json.dumps(results, indent=2))
        run.set_metrics(results=results)
        if results:
            gains = [r["temp_best"] - r["deterministic"] for r in results.values()]
            run.headline(
                f"PPO retrained on {len(results)} patterns; temperature sampling "
                f"adds ${np.mean(gains):,.0f} on average over argmax")
    return 0


if __name__ == "__main__":
    sys.exit(main())
