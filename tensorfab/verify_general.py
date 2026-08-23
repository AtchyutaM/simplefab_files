"""
Verification for the generalized N-machine line.

`sim.py` is hardcoded to 4 machines, so it can only gate the n_pairs=2 case. For
deeper lines we rely on three independent checks, run here:

  1. REGRESSION — at n_pairs=2 the generalized code must reproduce the original
     4-machine results exactly (`verify.py` still gates this against sim.py).

  2. CROSS-VALIDATION — take the schedule CP-SAT returns, replay it through the
     tensor simulator, and require the resulting profit to match CP-SAT's own
     objective. These are two completely independent implementations of the same
     dynamics: an integer program and a batched tensor program. If the queue
     recurrences in optim.py disagree with the routing in tensor_sim.py, this
     catches it. Works at any depth.

  3. INVARIANTS — physical properties that must hold for any schedule at any
     depth: unit conservation, non-negative queues, one job per tool at a time,
     no shipment before the pipeline can possibly deliver.

Usage:
    python -m tensorfab.verify_general
    python -m tensorfab.verify_general --pairs 2 4 6 --milp-seconds 30
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import numpy as np

from tensorfab import device

device.configure()

import jax.numpy as jnp  # noqa: E402

from simplefab_1p.config import make_common_config  # noqa: E402
from tensorfab import policies  # noqa: E402
from tensorfab.spec import spec_from_common_config  # noqa: E402
from tensorfab.tensor_sim import make_line, profit_breakdown  # noqa: E402

PROFIT_TOL = 0.02  # dollars; CP-SAT rounds costs at 1/COST_SCALE resolution


def starts_to_tape(starts: Dict[int, List[int]], H: int, M: int) -> np.ndarray:
    """[H, 1, M] action tape from CP-SAT's per-machine start times."""
    tape = np.zeros((H, 1, M), dtype=np.int32)
    for m in range(M):
        for t in starts[m]:
            tape[t, 0, m] = 1
    return tape


def check_invariants(line, spec, tape: np.ndarray) -> List[str]:
    """Replay a tape with tracing and assert physical properties."""
    fails = []
    final, tr = line.rollout_trace(jnp.asarray(tape))
    q = np.asarray(tr.q)            # [H, B, M]
    busy = np.asarray(tr.busy)      # [H, B, M]
    started = np.asarray(tr.started)
    cum_ship = np.asarray(tr.cum_ship)

    if (q < 0).any():
        fails.append(f"negative queue count (min {q.min()})")
    if (busy > 1).any():
        fails.append("a tool ran more than one job at once")
    if not np.all(np.diff(cum_ship[:, 0]) >= 0):
        fails.append("cumulative shipments decreased")

    # Unit conservation: every unit that entered q0 is either still somewhere in
    # the line, in flight, finished, or shipped.
    f = final
    consumed = np.asarray(f.started)[0] * spec.take         # units taken per machine
    produced = np.asarray(f.started)[0] * spec.make         # units emitted per machine
    in_q = np.asarray(f.q)[0].sum()
    in_flight = int(np.asarray(f.reg)[0].sum())
    shipped = int(np.asarray(f.cum_ship)[0])
    qfin = int(np.asarray(f.qfin)[0])
    arrivals = int(spec.arrivals.sum())
    # units still inside tools = sum over machines of make[m] for each in-flight job
    inflight_units = int((np.asarray(f.reg)[0].sum(axis=-1) * spec.make).sum())
    balance = arrivals + spec.init_qfin - (in_q + inflight_units + qfin + shipped)
    # Consumed-but-not-yet-emitted is exactly the in-flight units, already counted.
    if balance != 0:
        fails.append(
            f"unit conservation off by {balance} "
            f"(arrivals {arrivals} + init {spec.init_qfin} vs queues {in_q} + "
            f"in-flight {inflight_units} + FG {qfin} + shipped {shipped})")

    # Nothing can ship before the pipeline lead time has elapsed, except from
    # initial finished stock.
    lead = int(spec.pt.sum())
    early = cum_ship[:lead, 0].max() if lead < len(cum_ship) else 0
    if early > spec.init_qfin:
        fails.append(f"shipped {early} units before lead time {lead} "
                     f"(only {spec.init_qfin} initial stock available)")
    return fails


def run_pairs(n_pairs: int, milp_seconds: float, mode: str) -> bool:
    from simplefab_1p.optim import solve_optimal_schedule

    common = make_common_config(mode=mode, n_pairs=n_pairs)
    spec = spec_from_common_config(common, mode=mode)
    line = make_line(spec)
    M, H = spec.n_machines, spec.H
    ok = True

    print(f"\n=== n_pairs={n_pairs}  M={M}  lead={int(spec.pt.sum())}t  "
          f"demand={spec.total_demand} ===")

    # --- commander invariants -------------------------------------------
    from tensorfab.tensor_sim import LineState
    cmd_final, cmd_tr = line.rollout_policy(policies.commander(line), batch=1, trace=True)
    cmd_profit = float(profit_breakdown(spec, cmd_final)["profit"][0])
    d_started = np.diff(np.concatenate(
        [np.zeros((1, 1, M)), np.asarray(cmd_tr.started)], axis=0), axis=0)
    cmd_tape = d_started.astype(np.int32)
    fails = check_invariants(line, spec, cmd_tape)
    print(f"  commander profit ${cmd_profit:>12,.2f}   invariants: "
          f"{'PASS' if not fails else 'FAIL ' + '; '.join(fails)}")
    ok &= not fails

    # --- CP-SAT cross-validation ----------------------------------------
    sol = solve_optimal_schedule(common, time_limit_sec=milp_seconds,
                                 verbose=False, num_workers=8)
    tape = starts_to_tape(sol["starts"], H, M)
    final = line.rollout(jnp.asarray(tape))
    replay = float(profit_breakdown(spec, final)["profit"][0])
    d = abs(replay - sol["profit"])
    good = d <= PROFIT_TOL
    ok &= good
    print(f"  CP-SAT objective ${sol['profit']:>12,.2f}   bound ${sol['objective_bound']:>12,.2f}"
          f"   ({sol['status']}, {sol['solver_wall_time']:.0f}s)")
    print(f"  replayed in sim  ${replay:>12,.2f}   |diff| = ${d:.4f}   "
          f"{'PASS' if good else 'FAIL — optim.py and tensor_sim.py disagree'}")

    fails = check_invariants(line, spec, tape)
    if fails:
        ok = False
        print(f"  CP-SAT schedule invariants: FAIL — {'; '.join(fails)}")
    else:
        print(f"  CP-SAT schedule invariants: PASS")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--milp-seconds", type=float, default=30.0)
    ap.add_argument("--mode", default="UNIFORM")
    args = ap.parse_args()

    print(device.describe())
    results = {n: run_pairs(n, args.milp_seconds, args.mode) for n in args.pairs}
    print("\n" + "=" * 62)
    for n, ok in results.items():
        print(f"  n_pairs={n}  {'PASS' if ok else 'FAIL'}")
    every = all(results.values())
    print("=" * 62)
    print("GENERAL VERIFICATION PASSED" if every else "GENERAL VERIFICATION FAILED")
    return 0 if every else 1


if __name__ == "__main__":
    sys.exit(main())
