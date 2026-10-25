"""
Is the low-load gap a bad basin rather than a bad gradient?

The gate-temperature sweep killed the surrogate-bias hypothesis: sharpening the
input gate makes the over-release worse, not better. What remains is the shape
of the search itself.

Every tick carries its own three-way logit (idle / P0 / P1) drawn from N(0,1),
so a random start runs something about two thirds of the time. At 92% load that
is nearly the right answer already. At 50% the optimum needs roughly half the
ticks idle, and reaching it means coordinating thousands of independent
parameters in the same direction -- a long way from where the optimiser starts,
in a landscape where each release looks locally cheap.

If that is what is happening, biasing the initial logits toward idle should
help, and help most at low load. If the surrogate simply cannot see the cost of
over-release, the starting point will not matter.

    python -m tensorfab.twoprod.init_test --util 0.60
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time
import numpy as np

from tensorfab import device
device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from simplefab.config import make_common_config  # noqa: E402
from tensorfab.logbook import Run  # noqa: E402
from tensorfab.twoprod.compare import adam_ascend, make_evaluator, sel_to_tape  # noqa: E402
from tensorfab.twoprod.diffsim import DiffLine2P, Relax2P  # noqa: E402
from tensorfab.twoprod.make_schedules import _Q  # noqa: E402
from tensorfab.twoprod.naming import sched_name  # noqa: E402
from tensorfab.twoprod.spec import spec2p_from_common_config  # noqa: E402
from tensorfab.twoprod.tensor_sim import make_line_2p, profit_breakdown_2p  # noqa: E402

HERE = pathlib.Path(__file__).parent
SCH = HERE / "schedules"
DATA = HERE.parent / "presentation_data.json"


def run_one(spec, line, ev, bias: float, steps: int, batch: int, seed: int):
    H, M, P = spec.H, spec.n_machines, spec.n_products
    dl = DiffLine2P(spec, Relax2P(mode="st"))
    theta0 = jax.random.normal(jax.random.PRNGKey(seed), (batch, H, M, P + 1))
    # slot 0 is idle; lifting it makes "do nothing" the default action
    theta0 = theta0.at[..., 0].add(bias)
    t0 = time.perf_counter()
    _, sel = adam_ascend(dl, theta0, steps, 0.1, ev, 10, 1000.0, _Q(), P)
    wall = time.perf_counter() - t0

    scores = np.asarray(ev(sel))
    best = int(np.argmax(scores))
    tape = sel_to_tape(jnp.asarray(np.asarray(sel)[best:best + 1]), P).astype(line.dtype)
    final, tr = line.rollout_trace(tape)
    bd = profit_breakdown_2p(spec, final)
    q = np.asarray(tr.q)[:, 0]
    started = np.asarray(tr.started)[:, 0]
    return {"bias": bias, "wall": round(wall, 1),
            "profit": float(np.asarray(bd["profit"])[0]),
            "released": int(np.diff(started[:, 0, :], axis=0, prepend=0).sum()
                            * int(spec.take[0])),
            "stranded": int(q[-1, 1:].sum()),
            "midline_wip": round(float(q[:, 1:3].sum(-1).mean()), 1),
            "shipped": int(np.asarray(bd["units_shipped"])[0].sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--util", type=float, nargs="+", default=[0.60])
    ap.add_argument("--biases", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with Run("twoprod_init", params=vars(args).copy(),
             note="Does biasing the start toward idle close the low-load gap?") as run:
        run.log(device.describe())
        out = {}
        for util in args.util:
            common = make_common_config(mode="UNIFORM", alpha=args.alpha, utilization=util)
            spec = spec2p_from_common_config(common)
            line = make_line_2p(spec); ev = make_evaluator(line)
            demand = int(spec.demand.sum())
            f = SCH / sched_name(args.alpha, util)
            cp = float(np.load(f)["cpsat__profit"]) if f.exists() else None
            run.log(f"\n=== util {util:.0%}   demand {demand}"
                    + (f"   CP-SAT ${cp:,.0f}" if cp else "") + " ===")
            run.log(f"{'idle bias':>10s}{'profit':>11s}{'released':>10s}{'vs dem':>9s}"
                    f"{'stranded':>10s}{'midWIP':>9s}" + ("{:>12s}".format("% of CP-SAT") if cp else ""))
            rows = []
            for b in args.biases:
                r = run_one(spec, line, ev, b, args.steps, args.batch, args.seed)
                r["demand"] = demand; r["cpsat"] = cp
                rows.append(r)
                run.log(f"{b:>10.1f}{r['profit']:>11,.0f}{r['released']:>10d}"
                        f"{r['released']/demand:>8.2f}x{r['stranded']:>10d}"
                        f"{r['midline_wip']:>9.1f}"
                        + (f"{r['profit']/cp:>11.1%}" if cp else ""))
            out[f"{util:.2f}"] = {"demand": demand, "cpsat": cp, "rows": rows}
        d = json.loads(DATA.read_text())
        d.setdefault("twoprod", {})["init_sweep"] = out
        DATA.write_text(json.dumps(d, indent=2))
        first = out[f"{args.util[0]:.2f}"]["rows"]
        run.headline(f"util {args.util[0]:.0%}: profit "
                     + " -> ".join(f"${r['profit']:,.0f}" for r in first)
                     + " as idle bias " + " -> ".join(f"{r['bias']:g}" for r in first))
    return 0


if __name__ == "__main__":
    sys.exit(main())
