"""
Long CP-SAT solves across pipeline depths (experiment family 3).

Run SEQUENTIALLY, not in parallel: the M=12 model peaked at 6.6 GB in testing and
an earlier concurrent 40 s attempt was OOM-killed. Each depth gets its own
generous time limit, and both the incumbent and the dual bound are recorded —
the bound is a rigorous ceiling even when the solver never finds a good solution,
which at these sizes it often does not.

Every schedule returned is replayed through the exact tensor simulator and the
profit compared against CP-SAT's own objective. A mismatch means optim.py and
tensor_sim.py disagree about the dynamics, which would invalidate the result.

Usage:
    python -m tensorfab.run_long_milp
    python -m tensorfab.run_long_milp --pairs 4 6 --seconds 3600
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

from simplefab_1p.config import make_common_config  # noqa: E402
from tensorfab import policies  # noqa: E402
from tensorfab.logbook import Run  # noqa: E402
from tensorfab.spec import spec_from_common_config  # noqa: E402
from tensorfab.tensor_sim import make_line, profit_breakdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--seconds", type=float, default=1800.0)
    ap.add_argument("--mode", default="UNIFORM")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from simplefab_1p.optim import solve_optimal_schedule

    with Run("long_milp_depth", params=vars(args).copy(),
             note="Family 3: CP-SAT incumbent + dual bound vs pipeline depth. "
                  "Sequential — the M=12 model is memory-hungry.") as run:
        run.log(device.describe())
        results = {}

        for n_pairs in args.pairs:
            common = make_common_config(mode=args.mode, n_pairs=n_pairs)
            spec = spec_from_common_config(common, mode=args.mode)
            line = make_line(spec)
            M, H = spec.n_machines, spec.H

            cmd = line.rollout_policy(policies.commander(line), batch=1)
            cmd_profit = float(profit_breakdown(spec, cmd)["profit"][0])

            run.log(f"\n--- n_pairs={n_pairs}  M={M}  lead={int(spec.pt.sum())}t  "
                    f"limit={args.seconds:.0f}s ---")
            t0 = time.perf_counter()
            sol = solve_optimal_schedule(common, time_limit_sec=args.seconds,
                                         verbose=False, num_workers=args.workers)
            wall = time.perf_counter() - t0

            rec = {
                "n_pairs": n_pairs, "M": M, "lead": int(spec.pt.sum()),
                "status": sol["status"], "is_optimal": bool(sol["is_optimal"]),
                "incumbent": sol["profit"], "bound": sol["objective_bound"],
                "solver_wall": sol["solver_wall_time"], "wall": wall,
                "commander": cmd_profit,
            }

            # Cross-validate: replay the schedule through the exact tensor sim.
            if sol.get("starts") is not None:
                tape = np.zeros((H, 1, M), dtype=np.int32)
                for m in range(M):
                    for t in sol["starts"][m]:
                        tape[t, 0, m] = 1
                final = line.rollout(jnp.asarray(tape))
                replay = float(profit_breakdown(spec, final)["profit"][0])
                rec["replay"] = replay
                rec["replay_diff"] = abs(replay - sol["profit"])
                rec["cross_validated"] = rec["replay_diff"] <= 0.05
            else:
                rec["cross_validated"] = None

            gap = ((sol["objective_bound"] - sol["profit"]) / sol["objective_bound"]
                   if sol["profit"] is not None and sol["objective_bound"] else None)
            rec["gap"] = gap
            results[str(n_pairs)] = rec

            inc = sol["profit"]
            run.log(f"  status      {sol['status']}  (optimal={sol['is_optimal']})")
            run.log(f"  incumbent   ${inc:,.2f}" if inc is not None else "  incumbent   none found")
            run.log(f"  dual bound  ${sol['objective_bound']:,.2f}")
            if gap is not None:
                run.log(f"  gap         {gap:.2%}")
            run.log(f"  commander   ${cmd_profit:,.2f}")
            if rec.get("cross_validated") is not None:
                run.log(f"  replay      ${rec['replay']:,.2f}  |diff| ${rec['replay_diff']:.4f}  "
                        f"{'PASS' if rec['cross_validated'] else 'FAIL'}")
            run.log(f"  wall        {wall:.0f}s")

        (run.dir / "milp_depth.json").write_text(json.dumps(results, indent=2))
        run.set_metrics(results=results)
        solved = [r for r in results.values() if r.get("incumbent") is not None]
        run.headline(
            f"CP-SAT vs depth: " + " | ".join(
                f"M={r['M']} inc ${r['incumbent']:,.0f} bound ${r['bound']:,.0f} "
                f"gap {r['gap']:.1%}" for r in solved) if solved
            else "CP-SAT found no incumbent at any depth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
