"""
Does the queue-gate surrogate cause the over-release?

Everything else has been ruled out. The objective charges production cost on
every started job with a direct gradient path to the release decision, and the
shipping cap is a `jnp.minimum` whose gradient is exactly zero once supply
exceeds backlog -- so the true objective cannot be crediting an unsellable unit
with revenue. More optimisation does not help. A post-hoc trim of trailing
releases recovers only a small share.

The one remaining path by which a release can be credited with something it does
not earn is the input-availability gate. In the exact dynamics a tool runs only
if `q >= take`; in the backward pass that step is replaced by a sigmoid of
temperature tau_q. Under that surrogate, adding material to an upstream queue
always looks like it raises the chance of a downstream run -- even when the
extra material will never be converted or sold.

That predicts something specific and testable: the spurious credit is largest
when queues sit near the threshold, which is exactly the low-load regime, and it
should shrink as the gate is sharpened. So sweep tau_q and watch the release
volume. If releases fall as tau_q falls, the gate is the culprit.

    python -m tensorfab.twoprod.tau_test --util 0.60
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
from tensorfab.twoprod.diffsim import DiffLine2P, Relax2P  # noqa: E402
from tensorfab.twoprod.make_schedules import _Q  # noqa: E402
from tensorfab.twoprod.naming import sched_name  # noqa: E402
from tensorfab.twoprod.spec import spec2p_from_common_config  # noqa: E402
from tensorfab.twoprod.tensor_sim import make_line_2p, profit_breakdown_2p  # noqa: E402

HERE = pathlib.Path(__file__).parent
SCH = HERE / "schedules"
DATA = HERE.parent / "presentation_data.json"


def run_one(spec, line, ev, tau_q: float, steps: int, batch: int, seed: int):
    H, M, P = spec.H, spec.n_machines, spec.n_products
    dl = DiffLine2P(spec, Relax2P(mode="st", tau_q=tau_q))
    theta0 = jax.random.normal(jax.random.PRNGKey(seed), (batch, H, M, P + 1))
    t0 = time.perf_counter()
    _, sel = adam_ascend(dl, theta0, steps, 0.1, ev, 10, 1000.0, _Q(), P)
    wall = time.perf_counter() - t0

    scores = np.asarray(ev(sel))
    best = int(np.argmax(scores))
    tape = sel_to_tape(jnp.asarray(np.asarray(sel)[best:best + 1]),
                       P).astype(line.dtype)
    final, tr = line.rollout_trace(tape)
    bd = profit_breakdown_2p(spec, final)
    q = np.asarray(tr.q)[:, 0]
    started = np.asarray(tr.started)[:, 0]
    return {
        "tau_q": tau_q, "wall": round(wall, 1),
        "profit": float(np.asarray(bd["profit"])[0]),
        "released": int(np.diff(started[:, 0, :], axis=0, prepend=0).sum()
                        * int(spec.take[0])),
        "stranded": int(q[-1, 1:].sum()),
        "midline_wip": round(float(q[:, 1:3].sum(-1).mean()), 1),
        "shipped": int(np.asarray(bd["units_shipped"])[0].sum()),
        "production": float(np.asarray(bd["production"])[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--util", type=float, nargs="+", default=[0.60])
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[4.0, 1.0, 0.25, 0.1, 0.05])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with Run("twoprod_tau", params=vars(args).copy(),
             note="Sweep the queue-gate temperature to test whether the gate "
                  "surrogate is what drives the gradient's over-release.") as run:
        run.log(device.describe())
        out = {}
        for util in args.util:
            common = make_common_config(mode="UNIFORM", alpha=args.alpha,
                                        utilization=util)
            spec = spec2p_from_common_config(common)
            line = make_line_2p(spec)
            ev = make_evaluator(line)
            demand = int(spec.demand.sum())
            f = SCH / sched_name(args.alpha, util)
            cp = float(np.load(f)["cpsat__profit"]) if f.exists() else None

            run.log(f"\n=== util {util:.0%}   demand {demand} units"
                    + (f"   CP-SAT ${cp:,.0f}" if cp else "") + " ===")
            run.log(f"{'tau_q':>7s}{'profit':>11s}{'released':>10s}{'vs dem':>9s}"
                    f"{'stranded':>10s}{'midWIP':>9s}{'wall':>7s}")
            rows = []
            for tau in args.taus:
                r = run_one(spec, line, ev, tau, args.steps, args.batch, args.seed)
                r["demand"] = demand
                r["cpsat"] = cp
                rows.append(r)
                run.log(f"{tau:>7.2f}{r['profit']:>11,.0f}{r['released']:>10d}"
                        f"{r['released'] / demand:>8.2f}x{r['stranded']:>10d}"
                        f"{r['midline_wip']:>9.1f}{r['wall']:>6.0f}s")
            out[f"{util:.2f}"] = {"demand": demand, "cpsat": cp, "rows": rows}

        d = json.loads(DATA.read_text())
        d.setdefault("twoprod", {})["tau_sweep"] = out
        DATA.write_text(json.dumps(d, indent=2))
        first = out[f"{args.util[0]:.2f}"]["rows"]
        run.headline(
            f"util {args.util[0]:.0%}: released "
            + " -> ".join(f"{r['released']}" for r in first)
            + " as tau_q " + " -> ".join(f"{r['tau_q']:g}" for r in first))
    return 0


if __name__ == "__main__":
    sys.exit(main())
