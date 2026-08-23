"""
Equivalence harness: tensor sim vs. the reference `simplefab_1p/sim.py`.

This is the gate on Phase 1. The tensor sim is only useful if it is the SAME
dynamics; a fast simulator of the wrong line is worthless. So we drive both
implementations with identical action sequences and assert agreement on every
integer quantity at every one of the 2688 ticks:

    q0..q3, qfin, backlog, cumulative shipments, machine busy flags,
    machine time-remaining, cumulative starts, invalid-action counts,
    and the three cost accumulators (Σqfin, Σq_i, Σlate)

Integer state is compared with `==` (exact). Profit is compared in float64 with
a tight tolerance: the reference accumulates dollars incrementally over 2688
ticks while we multiply exact integer accumulators by prices once at the end, so
the two differ only by the reference's accumulated rounding (~1e-10 on $33k).

Action regimes covered:
    all-run     — every machine requested every tick; maximal invalid-action load
    all-idle    — nothing ever runs; exercises pure backorder accounting
    bernoulli   — unmasked random requests at several rates; mixes valid/invalid
    commander   — the greedy heuristic, driven by state rather than a fixed tape

Usage:
    python -m tensorfab.verify
    python -m tensorfab.verify --mode UNIFORM --seed 3
    python -m tensorfab.verify --all-modes
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from simplefab_1p.config import make_common_config  # noqa: E402
from simplefab_1p.sim import ProductionLine  # noqa: E402
from tensorfab import policies  # noqa: E402
from tensorfab.spec import LineSpec, spec_from_common_config  # noqa: E402
from tensorfab.tensor_sim import make_line, profit_breakdown  # noqa: E402

PROFIT_TOL = 1e-6  # absolute dollars


# ----------------------------------------------------------------------
# reference rollout
# ----------------------------------------------------------------------
def reference_rollout(common: Dict, actions: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Run `ProductionLine` for the full horizon and record the same quantities the
    tensor sim traces.

    actions: [H, 4] of 0/1, or None to let the built-in commander heuristic drive.
    """
    line = ProductionLine(common)
    H = line.H

    rec: Dict[str, List] = {k: [] for k in (
        "q", "qfin", "backlog", "cum_ship", "busy", "rem",
        "started", "sum_qfin", "sum_q", "sum_late", "n_invalid",
    )}
    M = line.n_machines
    started = np.zeros(M, dtype=np.int64)
    sum_qfin = 0
    sum_q = np.zeros(M, dtype=np.int64)
    sum_late = 0
    n_invalid = 0

    machines = None
    for t in range(H):
        override = None
        if actions is not None:
            override = {f"machine{i}": bool(actions[t, i]) for i in range(actions.shape[1])}

        # Count starts by watching each machine's event log grow.
        machines = line.machines
        pre_events = np.array([len(m.event_log) for m in machines], dtype=np.int64)

        line.run_step(t, actions_override=override)

        machines = line.machines
        post_events = np.array([len(m.event_log) for m in machines], dtype=np.int64)
        started = started + (post_events - pre_events)

        q = np.array([len(line.queues[f"queue{i}"]) for i in range(M)], dtype=np.int64)
        qfin = len(line.queues["queue_fin"])
        backlog = len(line.queues["demand"])
        # Every shipment (instant match or M3 completion) appends here.
        cum_ship = len(line.demand_met_log)

        busy = np.array([int(m.busy) for m in machines], dtype=np.int64)
        rem = np.array(
            [max(m.finish_time - t, 0) if m.busy else 0 for m in machines], dtype=np.int64
        )

        # Reproduce the accumulators the tensor sim carries.
        sum_qfin += qfin
        sum_q = sum_q + q
        late = sum(
            1 for a in line.queues["demand"] if t >= a + line.demand_interval
        )
        sum_late += late
        n_invalid += int(line.cost_log[-1]["invalid_actions"])

        rec["q"].append(q)
        rec["qfin"].append(qfin)
        rec["backlog"].append(backlog)
        rec["cum_ship"].append(cum_ship)
        rec["busy"].append(busy)
        rec["rem"].append(rem)
        rec["started"].append(started.copy())
        rec["sum_qfin"].append(sum_qfin)
        rec["sum_q"].append(sum_q.copy())
        rec["sum_late"].append(sum_late)
        rec["n_invalid"].append(n_invalid)

    out = {k: np.asarray(v, dtype=np.int64) for k, v in rec.items()}
    out["profit"] = np.float64(line.profit_total())
    out["costs"] = line.costs
    return out


# ----------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------
def _compare(ref: Dict[str, np.ndarray], got: Dict[str, np.ndarray], label: str) -> bool:
    ok = True
    rows = []
    for key in ("q", "qfin", "backlog", "cum_ship", "busy", "rem",
                "started", "sum_qfin", "sum_q", "sum_late", "n_invalid"):
        a, b = ref[key], got[key]
        if a.shape != b.shape:
            rows.append((key, "SHAPE", f"{a.shape} vs {b.shape}"))
            ok = False
            continue
        diff = np.abs(a.astype(np.int64) - b.astype(np.int64))
        bad = int((diff != 0).sum())
        if bad:
            first = int(np.argmax(diff.reshape(diff.shape[0], -1).any(axis=1)))
            rows.append((key, "MISMATCH", f"{bad} elems, first tick {first}, max |d|={diff.max()}"))
            ok = False
        else:
            rows.append((key, "ok", ""))

    dp = abs(float(ref["profit"]) - float(got["profit"]))
    if dp > PROFIT_TOL:
        rows.append(("profit", "MISMATCH", f"ref={ref['profit']:.6f} got={got['profit']:.6f} d={dp:.3e}"))
        ok = False
    else:
        rows.append(("profit", "ok", f"${float(got['profit']):,.4f}  (|d|={dp:.2e})"))

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    for key, st, note in rows:
        if st != "ok":
            print(f"          {key:<12} {st}  {note}")
    profit_row = [r for r in rows if r[0] == "profit"][0]
    if ok:
        print(f"          profit       {profit_row[2]}")
    return ok


def _tensor_to_ref_dict(trace, final, spec: LineSpec, b: int) -> Dict[str, np.ndarray]:
    """Slice batch element b out of a Trace and shape it like reference_rollout's output."""
    out = {
        "q": np.asarray(trace.q[:, b], dtype=np.int64),
        "qfin": np.asarray(trace.qfin[:, b], dtype=np.int64),
        "backlog": np.asarray(trace.backlog[:, b], dtype=np.int64),
        "cum_ship": np.asarray(trace.cum_ship[:, b], dtype=np.int64),
        "busy": np.asarray(trace.busy[:, b], dtype=np.int64),
        "rem": np.asarray(trace.rem[:, b], dtype=np.int64),
        "started": np.asarray(trace.started[:, b], dtype=np.int64),
        "sum_qfin": np.asarray(trace.sum_qfin[:, b], dtype=np.int64),
        "sum_q": np.asarray(trace.sum_q[:, b], dtype=np.int64),
        "sum_late": np.asarray(trace.sum_late[:, b], dtype=np.int64),
        "n_invalid": np.asarray(trace.n_invalid[:, b], dtype=np.int64),
    }
    out["profit"] = float(profit_breakdown(spec, final)["profit"][b])
    return out


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def build_action_bank(H: int, seed: int, n_random: int = 4,
                      M: int = 4) -> tuple[np.ndarray, List[str]]:
    """[H, B, M] of 0/1 plus a label per batch column."""
    rng = np.random.default_rng(seed)
    cols, labels = [], []

    cols.append(np.ones((H, M), dtype=np.int32))
    labels.append("all-run")
    cols.append(np.zeros((H, M), dtype=np.int32))
    labels.append("all-idle")

    for i in range(n_random):
        p = [0.9, 0.5, 0.25, 0.1][i % 4]
        cols.append((rng.random((H, M)) < p).astype(np.int32))
        labels.append(f"bernoulli p={p}")

    return np.stack(cols, axis=1), labels


def run_verification(mode: str = "ALTERNATING_92", seed: int = 0, n_random: int = 4,
                     n_pairs: int = 2) -> bool:
    common = make_common_config(mode=mode, n_pairs=n_pairs)
    spec = spec_from_common_config(common, mode=mode)
    line = make_line(spec)

    print(f"\n=== mode {mode}  n_pairs={n_pairs}  M={spec.n_machines}  "
          f"H={spec.H}  demand={spec.total_demand} ===")

    all_ok = True

    # ---- fixed action tapes, run as one batch ------------------------
    actions, labels = build_action_bank(spec.H, seed, n_random, M=spec.n_machines)
    final, trace = line.rollout_trace(jnp.asarray(actions))
    jax.block_until_ready(final)

    for b, label in enumerate(labels):
        ref = reference_rollout(common, actions[:, b])
        got = _tensor_to_ref_dict(trace, final, spec, b)
        all_ok &= _compare(ref, got, f"{label}  (batch col {b})")

    # ---- state-driven policy: the commander heuristic ----------------
    final_c, trace_c = line.rollout_policy(
        policies.commander(line), batch=1, trace=True
    )
    jax.block_until_ready(final_c)
    ref_c = reference_rollout(common, None)  # None -> built-in commander_decide
    got_c = _tensor_to_ref_dict(trace_c, final_c, spec, 0)
    all_ok &= _compare(ref_c, got_c, "commander heuristic (state-driven)")

    return bool(all_ok)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify tensor sim == simplefab_1p/sim.py")
    ap.add_argument("--mode", default="ALTERNATING_92")
    ap.add_argument("--all-modes", action="store_true",
                    help="verify UNIFORM, ALTERNATING, ALTERNATING_50, ALTERNATING_92")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--n-pairs", type=int, nargs="+", default=[2],
                    help="pipeline depths to verify; 2 = the original 4-machine line")
    args = ap.parse_args()

    print(device.describe())

    modes = (["UNIFORM", "ALTERNATING", "ALTERNATING_50", "ALTERNATING_92"]
             if args.all_modes else [args.mode])

    results = {}
    for np_ in args.n_pairs:
        for m in modes:
            results[f"{m} (n_pairs={np_})"] = run_verification(
                m, args.seed, args.n_random, n_pairs=np_)

    print("\n" + "=" * 60)
    for m, ok in results.items():
        print(f"  {m:<30} {'PASS' if ok else 'FAIL'}")
    every = all(results.values())
    print("=" * 60)
    print("VERIFICATION PASSED" if every else "VERIFICATION FAILED")
    return 0 if every else 1


if __name__ == "__main__":
    sys.exit(main())
