"""
Is the gradient's low-load shortfall an optimisation budget problem or a bias?

At 60% utilisation grad-ST releases 116 units more than the order book and
strands 128 of them mid-line, which accounts for two thirds of its gap to
CP-SAT. Two explanations fit that:

  budget  the run stopped at 400 Adam steps before it had finished pulling
          the release rate down, in which case more steps fix it

  bias    the straight-through surrogate systematically under-reports the cost
          of a release, in which case more steps converge to the same wrong
          answer and the stranded units persist

They make opposite predictions, so this runs the same optimisation at
increasing step counts and watches profit and stranded WIP together. A budget
problem shows profit still climbing; a bias shows both plateauing.

Runs the gradient only -- no solver -- so it is cheap.

    python -m tensorfab.twoprod.converge_test --util 0.60
    python -m tensorfab.twoprod.converge_test --util 0.60 0.92 --steps 400 1200 3000
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from simplefab.config import make_common_config  # noqa: E402
from tensorfab.logbook import Run  # noqa: E402
from tensorfab.twoprod.compare import adam_ascend, make_evaluator, sel_to_tape  # noqa: E402
from tensorfab.twoprod.make_schedules import _Q  # noqa: E402
from tensorfab.twoprod.diffsim import DiffLine2P, Relax2P  # noqa: E402
from tensorfab.twoprod.spec import spec2p_from_common_config  # noqa: E402
from tensorfab.twoprod.tensor_sim import make_line_2p, profit_breakdown_2p  # noqa: E402

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "presentation_data.json"


def one(spec, line, ev, steps: int, batch: int, seed: int, lr: float = 0.1):
    H, M, P = spec.H, spec.n_machines, spec.n_products
    dl = DiffLine2P(spec, Relax2P(mode="st"))
    theta0 = jax.random.normal(jax.random.PRNGKey(seed), (batch, H, M, P + 1))
    t0 = time.perf_counter()
    _, sel = adam_ascend(dl, theta0, steps, lr, ev, 10, 1000.0, _Q(), P)
    wall = time.perf_counter() - t0

    scores = np.asarray(ev(sel))
    best = int(np.argmax(scores))
    tape = sel_to_tape(jnp.asarray(np.asarray(sel)[best:best + 1]),
                       P).astype(line.dtype)
    final, tr = line.rollout_trace(tape)
    bd = profit_breakdown_2p(spec, final)
    q = np.asarray(tr.q)[:, 0]
    started = np.asarray(tr.started)[:, 0]
    released = int(np.diff(started[:, 0, :], axis=0, prepend=0).sum()
                   * int(spec.take[0]))
    return {
        "steps": steps, "batch": batch, "seed": seed, "wall": round(wall, 1),
        "profit": float(np.asarray(bd["profit"])[0]),
        "released": released,
        "stranded": int(q[-1, 1:].sum()),
        "shipped": int(np.asarray(bd["units_shipped"])[0].sum()),
        "production": float(np.asarray(bd["production"])[0]),
        "batch_best": float(scores.max()), "batch_mean": float(scores.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--util", type=float, nargs="+", default=[0.60])
    ap.add_argument("--steps", type=int, nargs="+", default=[400, 1200, 3000])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    with Run("twoprod_converge", params=vars(args).copy(),
             note="Does more optimisation close the gradient's low-load gap, "
                  "or does the over-release persist?") as run:
        run.log(device.describe())
        out = {}
        for util in args.util:
            common = make_common_config(mode="UNIFORM", alpha=args.alpha,
                                        utilization=util)
            spec = spec2p_from_common_config(common)
            line = make_line_2p(spec)
            ev = make_evaluator(line)
            demand = int(spec.demand.sum())
            rows = []
            run.log(f"\n=== util {util:.0%}   demand {demand} units ===")
            run.log(f"{'steps':>7s}{'seed':>6s}{'profit':>11s}{'released':>10s}"
                    f"{'vs demand':>11s}{'stranded':>10s}{'wall':>8s}")
            for steps in args.steps:
                for seed in args.seeds:
                    r = one(spec, line, ev, steps, args.batch, seed)
                    r["demand"] = demand
                    rows.append(r)
                    run.log(f"{steps:>7d}{seed:>6d}{r['profit']:>11,.0f}"
                            f"{r['released']:>10d}"
                            f"{r['released'] / demand:>10.2f}x"
                            f"{r['stranded']:>10d}{r['wall']:>7.0f}s")
            out[f"{util:.2f}"] = {"demand": demand, "rows": rows}

        d = json.loads(DATA.read_text())
        d.setdefault("twoprod", {})["converge"] = out
        DATA.write_text(json.dumps(d, indent=2))
        first = out[f"{args.util[0]:.2f}"]["rows"]
        run.headline(
            f"util {args.util[0]:.0%}: profit "
            + " -> ".join(f"${r['profit']:,.0f}" for r in first)
            + "; stranded " + " -> ".join(str(r["stranded"]) for r in first))
    return 0


if __name__ == "__main__":
    sys.exit(main())
