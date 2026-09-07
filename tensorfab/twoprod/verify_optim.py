"""
Cross-validation: the CP-SAT model against the tensor simulator.

`simplefab/sim.py` gates the tensor simulator, and this gates the MILP against
the tensor simulator — so the MILP is transitively checked against the reference
implementation without needing a third hand-written replay.

The test: take the schedule CP-SAT returns, replay it through the exact integer
simulator, and require the resulting profit to equal CP-SAT's own objective.
These are two completely independent encodings of the same dynamics — an integer
program and a batched tensor program — so a disagreement in the changeover
timing, the occupancy windows, or the queue recurrences shows up here as a
dollar gap.

It also reports the *shape* of the model, because model size and solve time are
first-class results for this experiment, not diagnostics.

Usage:
    python -m tensorfab.twoprod.verify_optim --horizon 336 --seconds 30
    python -m tensorfab.twoprod.verify_optim --horizon 2688 --seconds 600
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from tensorfab import device

device.configure()

import jax.numpy as jnp  # noqa: E402

from simplefab.config import make_common_config  # noqa: E402
from tensorfab.twoprod.optim import solve_2p  # noqa: E402
from tensorfab.twoprod.spec import spec2p_from_common_config  # noqa: E402
from tensorfab.twoprod.tensor_sim import make_line_2p, profit_breakdown_2p  # noqa: E402

TOL = 0.05  # dollars; CP-SAT rounds costs at 1/COST_SCALE


def run(horizon: int, seconds: float, alpha: float, util: float,
        mode: str, workers: int, hint: bool) -> bool:
    common = make_common_config(mode=mode, H=horizon, alpha=alpha, utilization=util)
    spec = spec2p_from_common_config(common, mode=mode)
    line = make_line_2p(spec)

    print(f"\n=== {mode}  H={horizon}  alpha={alpha}  util={util}  "
          f"demand={spec.demand.sum(axis=0).tolist()} ===")

    sol = solve_2p(common, time_limit_sec=seconds, num_workers=workers,
                   verbose=True, use_hint=hint)
    print(f"  status      {sol['status']}   (optimal={sol['is_optimal']})")
    print(f"  bound       ${sol['objective_bound']:,.2f}")
    if sol.get("profit") is None:
        print("  no feasible solution found — cannot cross-validate")
        return False
    print(f"  objective   ${sol['profit']:,.2f}")
    print(f"  changeovers {sol['changeovers']}   shipped {sol['units_shipped']}")
    print(f"  solve       {sol['solver_wall_time']:.1f}s   build {sol['build_seconds']:.1f}s")

    tape = jnp.asarray(sol["starts"][:, None, :, :])       # [H, 1, M, P]
    final = line.rollout(tape)
    bd = profit_breakdown_2p(spec, final)
    replay = float(bd["profit"][0])
    d = abs(replay - sol["profit"])
    ok = d <= TOL
    print(f"  replayed    ${replay:,.2f}   |diff| ${d:.4f}   "
          f"{'PASS' if ok else 'FAIL — optim and tensor_sim disagree'}")
    if not ok:
        print(f"    breakdown: rev ${bd['revenue'][0]:,.2f}  prod ${bd['production'][0]:,.2f}  "
              f"setup ${bd['setup'][0]:,.2f}  inv ${bd['inventory'][0]:,.2f}  "
              f"wip ${bd['wip'][0]:,.2f}  bo ${bd['backorder'][0]:,.2f}")
        print(f"    changeovers: milp {sol['changeovers']}  "
              f"replay {int(bd['changeovers'][0].sum())}")
        print(f"    shipped:     milp {sol['units_shipped']}  "
              f"replay {bd['units_shipped'][0].tolist()}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=336)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--util", type=float, default=0.80)
    ap.add_argument("--mode", default="UNIFORM")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-hint", action="store_true")
    args = ap.parse_args()

    ok = run(args.horizon, args.seconds, args.alpha, args.util,
             args.mode, args.workers, not args.no_hint)
    print("\n" + "=" * 60)
    print("MILP CROSS-VALIDATION PASSED" if ok else "MILP CROSS-VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
