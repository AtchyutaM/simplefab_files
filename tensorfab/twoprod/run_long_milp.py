"""
Long CP-SAT solves on the two-product line, at a budget matched to the
single-product depth study.

The first two-product solves used 900 seconds — an arbitrary choice made to get
a result inside a reasonable wait. The single-product depth study used 1800.
Comparing gaps across different time budgets is not sound, so this re-runs every
mix at 1800 seconds with the same worker count, making the only difference
between the two studies the problem itself.

Runs sequentially: the models are large and concurrent solves would contend for
cores, which would itself distort the comparison.

Every schedule is replayed through the exact tensor simulator and checked against
CP-SAT's own objective, so a formulation error cannot masquerade as a result.

Usage:
    python -m tensorfab.twoprod.run_long_milp
    python -m tensorfab.twoprod.run_long_milp --seconds 1800 --alphas 0.5 0.9 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from tensorfab import device

device.configure()

import jax.numpy as jnp  # noqa: E402

from simplefab.config import make_common_config  # noqa: E402
from tensorfab.logbook import Run  # noqa: E402
from tensorfab.twoprod.optim import solve_2p  # noqa: E402
from tensorfab.twoprod.spec import spec2p_from_common_config  # noqa: E402
from tensorfab.twoprod.tensor_sim import make_line_2p, profit_breakdown_2p  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.9, 0.1])
    ap.add_argument("--seconds", type=float, default=1800.0)
    ap.add_argument("--util", type=float, default=0.80)
    ap.add_argument("--mode", default="UNIFORM")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with Run("twoprod_long_milp", params=vars(args).copy(),
             note="CP-SAT at a budget matched to the single-product depth study "
                  "(1800s, 8 workers), so the only difference is the problem.") as run:
        run.log(device.describe())
        results = {}

        for alpha in args.alphas:
            common = make_common_config(mode=args.mode, alpha=alpha,
                                        utilization=args.util)
            spec = spec2p_from_common_config(common, mode=args.mode)
            line = make_line_2p(spec)

            run.log(f"\n--- alpha={alpha}  util={args.util}  "
                    f"demand={spec.demand.sum(axis=0).tolist()}  "
                    f"limit={args.seconds:.0f}s ---")
            t0 = time.perf_counter()
            sol = solve_2p(common, time_limit_sec=args.seconds,
                           num_workers=args.workers, verbose=False)
            wall = time.perf_counter() - t0

            inc, bound = sol.get("profit"), sol["objective_bound"]
            gap = (bound - inc) / bound if inc is not None and bound else None
            rec = {"alpha": alpha, "status": sol["status"],
                   "is_optimal": bool(sol["is_optimal"]),
                   "incumbent": inc, "bound": bound, "gap": gap,
                   "changeovers": sol.get("changeovers"),
                   "solver_wall": sol["solver_wall_time"], "wall": wall,
                   "n_y_vars": sol["n_y_vars"], "n_c_vars": sol["n_c_vars"]}

            if sol.get("starts") is not None:
                tape = jnp.asarray(sol["starts"][:, None, :, :])
                bd = profit_breakdown_2p(spec, line.rollout(tape))
                rec["replay"] = float(bd["profit"][0])
                rec["replay_diff"] = abs(rec["replay"] - inc)
                rec["cross_validated"] = rec["replay_diff"] <= 0.05

            results[str(alpha)] = rec
            run.log(f"  status      {sol['status']} (optimal={sol['is_optimal']})")
            run.log(f"  incumbent   ${inc:,.2f}" if inc is not None else "  incumbent   none")
            run.log(f"  bound       ${bound:,.2f}")
            if gap is not None:
                run.log(f"  gap         {gap:.2%}")
            run.log(f"  changeovers {sol.get('changeovers')}")
            if "replay" in rec:
                run.log(f"  replay      ${rec['replay']:,.2f}  |d| ${rec['replay_diff']:.4f}  "
                        f"{'PASS' if rec['cross_validated'] else 'FAIL'}")
            run.log(f"  wall        {wall:.0f}s")

        (run.dir / "milp_1800.json").write_text(json.dumps(results, indent=2))
        run.set_metrics(results=results)
        gaps = [r["gap"] for r in results.values() if r.get("gap") is not None]
        proven = sum(1 for r in results.values() if r["is_optimal"])
        run.headline(
            f"2-product CP-SAT at {args.seconds:.0f}s: gaps "
            + " / ".join(f"{g:.2%}" for g in gaps)
            + f"; proven optimal {proven}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
