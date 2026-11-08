"""
How the tensor-vs-reference speed ratio moves as the fab's time scale grows.

The tensor tick does dense work over its whole delay register every tick, busy
or not; the reference keeps one finish time per machine. So the register length
L (longest job in ticks, + 1) should cost the tensor sim linearly and the
reference nothing. L grows with longer operations or a finer tick, so this is
the axis along which a bigger, more realistic fab moves.

The probe pads the register of the unchanged Mini-Fab from 64 to 128, 256 and
512 slots. Padding adds only empty slots, so results must be identical (checked:
outs, WIP-ticks, runs and setups per row); only the cost changes. Timing is
interleaved rounds with the median, as in `speed_study`. The reference is also
timed at two loads, to show its per-tick cost tracks events rather than state.

    python -m tensorfab.minifab.scaling_probe            # ~5 min on CPU
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import time

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from minifab.config import make_minifab_config  # noqa: E402
from minifab.sim import MiniFab  # noqa: E402
from tensorfab.logbook import Run  # noqa: E402
from tensorfab.minifab.policies import make_rule  # noqa: E402
from tensorfab.minifab.spec import spec_from_config  # noqa: E402
from tensorfab.minifab.tensor_sim import TensorMiniFab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg-lens", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 1024])
    ap.add_argument("--shifts", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    params = vars(args) | {"rate": 6.36, "rule": "lbfs-full", "pm": False}
    with Run("minifab_scaling_probe", params=params) as run:
        cfg = make_minifab_config(release_rate_per_shift=6.36, pm=False)
        base = spec_from_config(cfg)
        T = args.shifts * base.shift_ticks

        # --- reference: per-tick cost at two loads -------------------------
        ref = {}
        for rate in (3.0, 6.36):
            c = make_minifab_config(release_rate_per_shift=rate, pm=False)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                MiniFab(c).run(horizon=T)
                ts.append(time.perf_counter() - t0)
            ref[rate] = T / statistics.median(ts)
            run.log(f"reference, release {rate}/shift: {ref[rate]:,.0f} ticks/s "
                    f"({1e6 / ref[rate]:.2f} us per tick)")

        # --- tensor: register-length sweep ---------------------------------
        fns, finals = {}, {}
        for L in args.reg_lens:
            sp = dataclasses.replace(base, reg_len=L)
            fab = TensorMiniFab(sp)
            rule = make_rule(sp, "lbfs-full")
            fn = fab.compile_policy_rollout(rule)
            xs = (jnp.arange(T, dtype=fab.dtype), fab.releases_for()[:T])
            for B in args.batches:
                s0, p0 = fab.init_state(B), rule.init(fab, B)
                fns[L, B] = (lambda fn=fn, s0=s0, p0=p0, xs=xs: fn(s0, p0, xs))
                f = jax.block_until_ready(fns[L, B]())          # compile + exactness
                finals[L, B] = tuple(np.asarray(a) for a in (f.outs, f.sum_wip, f.runs, f.setups))

        ok = True
        for (L, B), got in finals.items():
            want = finals[args.reg_lens[0], B]
            same = all(np.array_equal(g, w) for g, w in zip(got, want))
            ok &= same
        run.log(f"padding changes no result (outs, WIP-ticks, runs, setups, every row): {'EXACT' if ok else 'MISMATCH'}")

        times = {k: [] for k in fns}
        keys = list(fns)
        for r in range(args.rounds):
            order = keys[r % len(keys):] + keys[:r % len(keys)]
            for k in order:
                t0 = time.perf_counter()
                jax.block_until_ready(fns[k]())
                times[k].append(time.perf_counter() - t0)

        run.log(f"\ntensor, closed-loop lbfs-full, {T:,} ticks, median of {args.rounds} interleaved rounds")
        run.log(f"  {'reg_len':>7} {'ints/row':>9}" + "".join(f"{'B=' + str(B):>26}" for B in args.batches))
        rates = {}
        for L in args.reg_lens:
            cells = []
            for B in args.batches:
                rates[L, B] = B * T / statistics.median(times[L, B])
                rel = rates[L, B] / rates[args.reg_lens[0], B]
                cells.append(f"{rates[L, B]:>13,.0f} ({rel:4.2f}x)  {rates[L, B] / ref[6.36]:5.1f}x ref")
            run.log(f"  {L:>7} {5 * 2 * L:>9}" + "".join(f"{c:>26}" for c in cells))

        Bmax = max(args.batches)
        lo, hi = args.reg_lens[0], args.reg_lens[-1]
        run.set_metrics(exact=ok, ref_ticks_per_s={str(k): v for k, v in ref.items()},
                        tensor_row_ticks_per_s={f"L{L}_B{B}": v for (L, B), v in rates.items()})
        run.headline(f"reg {lo}->{hi}: tensor B={Bmax} {rates[lo, Bmax] / 1e6:.2f}M -> "
                     f"{rates[hi, Bmax] / 1e6:.2f}M row-ticks/s ({rates[lo, Bmax] / ref[6.36]:.0f}x -> "
                     f"{rates[hi, Bmax] / ref[6.36]:.1f}x ref); reference unaffected; padding exact={ok}")


if __name__ == "__main__":
    main()
