"""
Throughput of the tensorized env wrapper vs. the reference `FabEnv`.

bench.py measures the bare dynamics. This measures what PPO actually calls: obs
+ action mask + shaping + reward on every step. That path was measured at 66,553
steps/s in Python — 5.4x slower than the bare sim — so it, not the dynamics, is
the cost that Phase 2a has to remove.

Usage:  python -m tensorfab.bench_env
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from simplefab_1p.config import make_common_config  # noqa: E402
from simplefab_1p.env import FabEnv  # noqa: E402
from tensorfab.env_tensor import Shaping, make_env  # noqa: E402
from tensorfab.logbook import Run  # noqa: E402
from tensorfab.spec import spec_from_common_config  # noqa: E402
from tensorfab.tensor_sim import make_line  # noqa: E402


def bench_reference(common, episodes: int, seed: int) -> float:
    env = FabEnv(common, invalid_action_penalty=-2.0)
    rng = np.random.default_rng(seed)
    H = int(common["time_horizon"])
    env.reset()
    n, t0 = 0, time.perf_counter()
    for _ in range(episodes):
        env.reset()
        for _ in range(H):
            env.action_masks()          # PPO calls this every step
            env.step(rng.integers(0, 2, size=4))
            n += 1
    return n / (time.perf_counter() - t0)


def bench_tensor(tenv, batch: int, H: int, reps: int, seed: int):
    fn = jax.jit(tenv.rollout)
    key = jax.random.PRNGKey(seed)
    acts = (jax.random.uniform(key, (H, batch, 4)) < 0.5).astype(jnp.int32)

    t0 = time.perf_counter()
    out = fn(acts)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(acts)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return batch * H / best, compile_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ALTERNATING_92")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64, 1024, 4096])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with Run("bench_env", params=vars(args).copy(),
             note="Tensorized env wrapper (obs+mask+shaping+reward) vs FabEnv") as run:
        run.log(device.describe())
        common = make_common_config(mode=args.mode)
        spec = spec_from_common_config(common, mode=args.mode)
        line = make_line(spec)
        tenv = make_env(line, shaping=Shaping(), invalid_action_penalty=-2.0)

        run.log(f"\nmode={args.mode} H={spec.H}")
        ref = bench_reference(common, args.episodes, args.seed)
        run.log(f"\n  reference FabEnv.step + action_masks : {ref:>14,.0f} steps/s")

        run.log(f"\n  {'batch':>8} {'steps/sec':>16} {'vs FabEnv':>11} {'compile s':>10}")
        run.log("  " + "-" * 49)
        rows = []
        for b in args.batches:
            sps, comp = bench_tensor(tenv, b, spec.H, args.reps, args.seed)
            rows.append((b, sps, sps / ref))
            run.log(f"  {b:>8} {sps:>16,.0f} {sps / ref:>10.1f}x {comp:>10.2f}")
            run.record(batch=b, steps_per_sec=sps, speedup=sps / ref)

        best = max(rows, key=lambda r: r[1])
        run.set_metrics(reference_steps_per_sec=ref,
                        peak_steps_per_sec=best[1],
                        peak_batch=best[0], peak_speedup=best[2])
        run.headline(f"tensor env peak {best[1]:,.0f} steps/s at B={best[0]} "
                     f"({best[2]:.0f}x FabEnv's {ref:,.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
