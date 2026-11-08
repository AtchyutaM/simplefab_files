"""
Throughput of the tensor Mini-Fab against the reference simulator.

Runs the default dispatching rule (LBFS, full batches) closed-loop for the full
300-shift horizon at several batch sizes, with every row on the same constant
release rate. Compile time is reported separately and excluded from the rate.

    python -m tensorfab.device                     # confirm the backend first
    python -m tensorfab.minifab.bench
    python -m tensorfab.minifab.bench --batches 1 64 1024 10000 --pm on
"""

from __future__ import annotations

import argparse
import time

from tensorfab import device

device.configure()

import jax  # noqa: E402

from minifab.config import make_minifab_config  # noqa: E402
from minifab.sim import MiniFab  # noqa: E402
from tensorfab.minifab.policies import make_rule  # noqa: E402
from tensorfab.minifab.spec import spec_from_config  # noqa: E402
from tensorfab.minifab.tensor_sim import TensorMiniFab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64, 1024, 10000])
    ap.add_argument("--rate", type=float, default=6.36)
    ap.add_argument("--pm", choices=("off", "on"), default="off")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    cfg = make_minifab_config(release_rate_per_shift=args.rate, pm=args.pm == "on")
    H = cfg["horizon"]
    print(f"backend {jax.default_backend()}  ·  {H:,} ticks  ·  release {args.rate}/shift  ·  PM {args.pm}")

    t0 = time.perf_counter()
    for _ in range(args.repeats):
        MiniFab(cfg).run()
    ref = (time.perf_counter() - t0) / args.repeats
    print(f"reference (Python, one run)   {ref:7.3f} s   {H / ref:>14,.0f} ticks/s")

    sp = spec_from_config(cfg)
    fab = TensorMiniFab(sp)
    rule = make_rule(sp, "lbfs-full")
    run = fab.compile_policy_rollout(rule)
    xs = (jax.numpy.arange(H, dtype=fab.dtype), fab.releases_for())
    for B in args.batches:
        s0, p0 = fab.init_state(B), rule.init(fab, B)
        t0 = time.perf_counter()
        jax.block_until_ready(run(s0, p0, xs))
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            out = run(s0, p0, xs)
        jax.block_until_ready(out)
        dt = (time.perf_counter() - t0) / args.repeats
        rate = B * H / dt
        print(f"tensor  B={B:<6d}  compile+run {first:6.2f} s   run {dt:7.3f} s   "
              f"{rate:>14,.0f} ticks/s   {rate / (H / ref):8.1f}x reference")


if __name__ == "__main__":
    main()
