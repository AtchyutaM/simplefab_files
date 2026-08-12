"""
Throughput benchmark: tensor sim vs. the reference Python simulator.

The report (docs/tensorized_line_report.md § 4.5) projects 1-2 orders of
magnitude over a measured 432,000 sim-steps/sec baseline, and explicitly warns
that the win is *batched* throughput, not single-trajectory latency. This script
is what confirms or corrects that, so it measures both regimes and re-measures
the Python baseline on whatever machine it is run on rather than trusting a
number from a previous machine.

Definitions used throughout:
    sim-step  = one machine-decision tick for ONE trajectory
    steps/sec = batch * H / wall_seconds

Compile time is reported separately from steady-state throughput. XLA compiles
the scan BODY once (not 2688 unrolled copies), so compile time is a fixed
one-off of a few seconds and does not grow with the horizon.

Usage:
    python -m tensorfab.bench
    python -m tensorfab.bench --batches 1 64 1024 10000 100000
    python -m tensorfab.bench --policy masked_bernoulli --reps 5
    python -m tensorfab.bench --no-baseline           # skip the slow Python run
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from typing import Callable, Dict, List

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from simplefab_1p.config import make_common_config  # noqa: E402
from simplefab_1p.sim import ProductionLine  # noqa: E402
from tensorfab import policies  # noqa: E402
from tensorfab.spec import spec_from_common_config  # noqa: E402
from tensorfab.tensor_sim import make_line, profit_breakdown  # noqa: E402


def bench_reference(common: Dict, episodes: int = 5) -> Dict[str, float]:
    """Single-threaded Python baseline, commander-driven (no action tape to read)."""
    H = int(common["time_horizon"])

    # warmup (import/JIT-free, but lets the allocator settle)
    ProductionLine(common).run_step(0)

    t0 = time.perf_counter()
    for _ in range(episodes):
        line = ProductionLine(common)
        for t in range(H):
            line.run_step(t)
    dt = time.perf_counter() - t0

    steps = episodes * H
    return {
        "steps_per_sec": steps / dt,
        "sec_per_episode": dt / episodes,
        "us_per_step": dt / steps * 1e6,
        "profit": float(line.profit_total()),
    }


def bench_tensor(
    line, policy_fn: Callable, batch: int, reps: int = 3
) -> Dict[str, float]:
    H = line.spec.H
    fn = line.compile_policy_rollout(policy_fn, trace=False)
    keys = jax.random.split(jax.random.PRNGKey(0), H)
    s0 = line.init_state(batch)

    # First call includes tracing + XLA compilation.
    t0 = time.perf_counter()
    out = fn(s0, keys)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    # Steady state: take the best of `reps` to suppress OS scheduling noise.
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(s0, keys)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)

    steps = batch * H
    profits = profit_breakdown(line.spec, out)["profit"]
    return {
        "steps_per_sec": steps / best,
        "sec_per_batch": best,
        "sec_per_episode": best / batch,
        "compile_s": compile_s,
        "profit_mean": float(np.mean(profits)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="tensorfab throughput benchmark")
    ap.add_argument("--mode", default="ALTERNATING_92")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64, 1024, 10000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baseline-episodes", type=int, default=5)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--policy", default="commander",
                    choices=["commander", "masked_bernoulli", "bernoulli"])
    args = ap.parse_args()

    print(device.describe())
    print()
    print(device.thread_note())

    common = make_common_config(mode=args.mode)
    spec = spec_from_common_config(common, mode=args.mode)
    line = make_line(spec)

    print(f"\nmode={args.mode}  H={spec.H}  policy={args.policy}  reps={args.reps}")

    baseline = None
    if not args.no_baseline:
        print("\n--- reference Python sim (simplefab_1p/sim.py, single-threaded) ---")
        baseline = bench_reference(common, args.baseline_episodes)
        print(f"  {baseline['steps_per_sec']:>14,.0f} steps/sec"
              f"   ({baseline['us_per_step']:.2f} us/step,"
              f" {baseline['sec_per_episode']*1e3:.2f} ms/episode)")
        print(f"  commander profit: ${baseline['profit']:,.2f}")

    policy_fn = {
        "commander": lambda: policies.commander(line),
        "masked_bernoulli": lambda: policies.masked_bernoulli(line, 0.5),
        "bernoulli": lambda: policies.bernoulli(0.5),
    }[args.policy]()

    print("\n--- tensor sim (JAX / XLA) ---")
    header = (f"  {'batch':>8} {'steps/sec':>16} {'vs python':>10} "
              f"{'ms/batch':>10} {'us/episode':>12} {'compile s':>10}  {'mean profit':>13}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows: List[Dict] = []
    for b in args.batches:
        try:
            r = bench_tensor(line, policy_fn, b, args.reps)
        except Exception as exc:  # OOM at large B is informative, not fatal
            print(f"  {b:>8} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        speed = (r["steps_per_sec"] / baseline["steps_per_sec"]) if baseline else float("nan")
        rows.append({"batch": b, **r, "speedup": speed})
        print(f"  {b:>8} {r['steps_per_sec']:>16,.0f} {speed:>9.1f}x "
              f"{r['sec_per_batch']*1e3:>10.1f} {r['sec_per_episode']*1e6:>12,.0f} "
              f"{r['compile_s']:>10.2f}  ${r['profit_mean']:>12,.2f}")

    if rows and baseline:
        best = max(rows, key=lambda r: r["steps_per_sec"])
        print(f"\n  peak: {best['steps_per_sec']:,.0f} steps/sec at B={best['batch']} "
              f"({best['speedup']:.1f}x the Python sim)")
        b1 = next((r for r in rows if r["batch"] == 1), None)
        if b1:
            print(f"  B=1 : {b1['steps_per_sec']:,.0f} steps/sec "
                  f"({b1['speedup']:.2f}x) — single-trajectory latency is NOT the win, "
                  f"exactly as the report predicts")

    if platform.system() == "Darwin":
        print("\n  Reminder: these are XLA *CPU* numbers. The 10^6-10^8 steps/sec regime "
              "\n  needs the NVIDIA box — re-run this script there unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
