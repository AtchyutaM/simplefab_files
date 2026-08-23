"""
Equivalence harness: tensor_sim_2p vs the reference `simplefab/sim.py`.

Same gate as `verify.py`, extended for the product axis and changeover state.
Compared exactly at every tick:

    per-product queue counts, finished goods, backlog, cumulative shipments,
    machine busy flags, ticks-to-completion, the product in each tool,
    each tool's LAST COMPLETED product, cumulative starts, cumulative
    changeovers, invalid-action counts, and all three cost accumulators

Profit is compared in float64 to a tight tolerance: the reference accumulates
dollars incrementally while we multiply exact integer accumulators by prices once
at the end, so the two differ only by the reference's accumulated rounding.

The changeover columns matter most. Setup time is sized from the tool's last
*completed* product and setup cost is charged on the same comparison, so an
off-by-one there would shift both the schedule and the bill without necessarily
breaking anything visible.

Usage:
    python -m tensorfab.verify_2p
    python -m tensorfab.verify_2p --alpha 0.9 --util 0.8
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np

from tensorfab import device

device.configure()

import jax.numpy as jnp  # noqa: E402

from simplefab.config import make_common_config  # noqa: E402
from simplefab.sim import ProductionLine  # noqa: E402
from tensorfab.spec_2p import spec2p_from_common_config  # noqa: E402
from tensorfab.tensor_sim_2p import make_line_2p, profit_breakdown_2p  # noqa: E402

PROFIT_TOL = 1e-6
ACT_NAMES = {0: "Prod0", 1: "Prod1"}


def build_line(common: Dict) -> ProductionLine:
    pt, su = common["processing_times"], common["setup_times"]
    return ProductionLine(
        {0: pt[0][0], 1: pt[0][1]},
        {0: pt[1][0], 1: pt[1][1]}, su[1],
        {0: pt[2][0], 1: pt[2][1]},
        {0: pt[3][0], 1: pt[3][1]}, su[3],
        common_cfg=common,
    )


def reference_rollout(common: Dict, actions: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """actions: [H, M, P] one-hot-ish, or None to let the Commander drive."""
    line = build_line(common)
    H, M, P = line.H, 4, 2
    machines = [line.machine0, line.machine1, line.machine2, line.machine3]

    rec: Dict[str, List] = {k: [] for k in (
        "q", "qfin", "backlog", "cum_ship", "busy", "rem", "curp", "lastp",
        "started", "setup_cost", "sum_qfin", "sum_q", "sum_late", "n_invalid")}
    started = np.zeros((M, P), dtype=np.int64)
    sum_qfin = np.zeros(P, dtype=np.int64)
    sum_q = np.zeros((M, P), dtype=np.int64)
    sum_late = np.zeros(P, dtype=np.int64)
    n_invalid = 0

    for t in range(H):
        override = None
        if actions is not None:
            override = {}
            for m in range(M):
                sel = np.argmax(actions[t, m]) if actions[t, m].sum() > 0 else None
                override[f"machine{m}"] = ACT_NAMES[int(sel)] if sel is not None else None

        pre_events = np.array([len(mm.event_log) for mm in machines], dtype=np.int64)
        line.run_step(current_time=t, actions_override=override)
        post_events = np.array([len(mm.event_log) for mm in machines], dtype=np.int64)

        # which product each machine started this tick (from its event log tail)
        for m in range(M):
            for e in machines[m].event_log[pre_events[m]:post_events[m]]:
                started[m, int(e[0])] += 1

        q = np.array([[len(line.queues[f"queue{m}"][p]) for p in range(P)]
                      for m in range(M)], dtype=np.int64)
        qfin = np.array([len(line.queues["queue_fin"][p]) for p in range(P)], dtype=np.int64)
        backlog = np.array([len(line.queues["demand"][p]) for p in range(P)], dtype=np.int64)
        cum_ship = np.zeros(P, dtype=np.int64)
        for prod, _ in line.demand_met_log:
            cum_ship[int(prod)] += 1

        busy = np.array([int(mm.current_product is not None) for mm in machines], dtype=np.int64)
        rem = np.array([max(mm.finish_time - t, 0) if mm.current_product is not None else 0
                        for mm in machines], dtype=np.int64)
        curp = np.array([0 if mm.current_product is None else int(mm.current_product) + 1
                         for mm in machines], dtype=np.int64)
        lastp = np.array([0 if getattr(mm, "last_finished_product", None) is None
                          else int(mm.last_finished_product) + 1
                          for mm in (machines[1], machines[3])], dtype=np.int64)

        sum_qfin += qfin
        sum_q += q
        late = np.array([
            sum(1 for a in line.queues["demand"][p] if t >= a + line.demand_interval)
            for p in range(P)], dtype=np.int64)
        sum_late += late
        n_invalid += int(line.cost_log[-1]["invalid_actions"])

        for k, v in (("q", q), ("qfin", qfin), ("backlog", backlog), ("cum_ship", cum_ship),
                     ("busy", busy), ("rem", rem), ("curp", curp), ("lastp", lastp),
                     ("started", started.copy()),
                     ("setup_cost", line.costs["setup"]),
                     ("sum_qfin", sum_qfin.copy()), ("sum_q", sum_q.copy()),
                     ("sum_late", sum_late.copy()), ("n_invalid", n_invalid)):
            rec[k].append(v)

    out = {k: np.asarray(v, dtype=np.int64) for k, v in rec.items() if k != "setup_cost"}
    out["setup_cost"] = np.asarray(rec["setup_cost"], dtype=np.float64)
    out["profit"] = np.float64(line.profit_total())
    return out


def _compare(ref: Dict, got: Dict, label: str) -> bool:
    ok, bad = True, []
    for key in ("q", "qfin", "backlog", "cum_ship", "busy", "rem", "curp", "lastp",
                "started", "setup_cost", "sum_qfin", "sum_q", "sum_late", "n_invalid"):
        a, b = ref[key], got[key]
        if a.shape != b.shape:
            bad.append(f"{key}: shape {a.shape} vs {b.shape}"); ok = False; continue
        diff = np.abs(a - b)
        if diff.any():
            first = int(np.argmax(diff.reshape(diff.shape[0], -1).any(axis=1)))
            bad.append(f"{key}: {int((diff != 0).sum())} elems, first tick {first}, "
                       f"max |d|={diff.max()}")
            ok = False
    dp = abs(float(ref["profit"]) - float(got["profit"]))
    if dp > PROFIT_TOL:
        bad.append(f"profit: ref={ref['profit']:.6f} got={got['profit']:.6f} d={dp:.3e}")
        ok = False
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    for b in bad:
        print(f"          {b}")
    if ok:
        print(f"          profit ${float(got['profit']):,.4f}  (|d|={dp:.2e})")
    return ok


def _slice(tr, final, spec, b: int) -> Dict:
    out = {k: np.asarray(getattr(tr, k)[:, b], dtype=np.int64)
           for k in ("q", "qfin", "backlog", "cum_ship", "busy", "rem", "curp",
                     "started", "sum_qfin", "sum_q", "sum_late", "n_invalid")}
    out["lastp"] = np.asarray(tr.lastp[:, b][:, [1, 3]], dtype=np.int64)
    out["setup_cost"] = np.asarray(tr.setup_cost[:, b], dtype=np.float64)
    out["profit"] = float(profit_breakdown_2p(spec, final)["profit"][b])
    return out


def build_actions(H: int, M: int, P: int, seed: int, n: int):
    """[H, B, M, P] one-hot-ish tapes plus labels."""
    rng = np.random.default_rng(seed)
    cols, labels = [], []
    # always request P0 / always P1 / never run
    for p, name in ((0, "always-P0"), (1, "always-P1")):
        a = np.zeros((H, M, P), dtype=np.int32); a[:, :, p] = 1
        cols.append(a); labels.append(name)
    cols.append(np.zeros((H, M, P), dtype=np.int32)); labels.append("all-idle")
    # random choices among {idle, P0, P1} at several run rates
    for rate in (0.9, 0.5, 0.25):
        choice = rng.random((H, M))
        pick = rng.integers(0, P, size=(H, M))
        a = np.zeros((H, M, P), dtype=np.int32)
        run = choice < rate
        a[np.arange(H)[:, None].repeat(M, 1)[run], np.arange(M)[None].repeat(H, 0)[run],
          pick[run]] = 1
        cols.append(a); labels.append(f"random p={rate}")
    return np.stack(cols, axis=1), labels


def run(alpha: float, util: float, mode: str, seed: int) -> bool:
    common = make_common_config(mode=mode, alpha=alpha, utilization=util)
    spec = spec2p_from_common_config(common, mode=mode)
    line = make_line_2p(spec)
    H, M, P = spec.H, spec.n_machines, spec.n_products

    print(f"\n=== {mode}  alpha={alpha}  util={util}  demand={spec.demand.sum(axis=0).tolist()} ===")
    acts, labels = build_actions(H, M, P, seed, 3)
    final, tr = line.rollout_trace(jnp.asarray(acts))
    ok = True
    for b, lbl in enumerate(labels):
        ref = reference_rollout(common, acts[:, b])
        ok &= _compare(ref, _slice(tr, final, spec, b), f"{lbl}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.5, 0.9])
    ap.add_argument("--util", type=float, default=0.92)
    ap.add_argument("--mode", default="UNIFORM")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(device.describe())
    results = {a: run(a, args.util, args.mode, args.seed) for a in args.alpha}
    print("\n" + "=" * 60)
    for a, ok in results.items():
        print(f"  alpha={a}  {'PASS' if ok else 'FAIL'}")
    every = all(results.values())
    print("=" * 60)
    print("2-PRODUCT VERIFICATION PASSED" if every else "2-PRODUCT VERIFICATION FAILED")
    return 0 if every else 1


if __name__ == "__main__":
    sys.exit(main())
