"""
Is the gradient's over-release a subtle trade-off or simply money left behind?

The convergence test ruled out an optimisation budget: 400, 1,200 and 3,000
Adam steps all converge on releasing 468 units against 352 of demand and all
strand exactly 128 units mid-line. The objective itself is not at fault --
production cost is charged on every started job and has a direct gradient path
back to the release decision.

That leaves the surrogate's landscape. This tests how far off it is with the
crudest possible repair: take the gradient's own schedule and delete its last
few releases at the gating tool, changing nothing else, then replay through the
exact simulator. Deleting a release that was going to strand costs nothing --
the unit was never going to ship -- and saves its production and holding cost.

If a trim of this kind recovers a large share of the gap, the surrogate is not
making a subtle trade-off; it is over-releasing in a way that is visible from
outside and trivially removable. If it recovers little, the extra releases were
doing something useful and the gap is elsewhere.

    python -m tensorfab.twoprod.trim_test --util 0.60
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from tensorfab import device

device.configure()

import jax.numpy as jnp  # noqa: E402

from simplefab.config import make_common_config  # noqa: E402
from tensorfab.twoprod.compare import sel_to_tape  # noqa: E402
from tensorfab.twoprod.naming import sched_name  # noqa: E402
from tensorfab.twoprod.spec import spec2p_from_common_config  # noqa: E402
from tensorfab.twoprod.tensor_sim import make_line_2p, profit_breakdown_2p  # noqa: E402

HERE = pathlib.Path(__file__).parent
SCH = HERE / "schedules"
DATA = HERE.parent / "presentation_data.json"


def tape_of(z, method: str) -> np.ndarray:
    cum = z[f"{method}__started_jobs"]
    per = np.diff(cum, axis=0, prepend=0)
    H, M, P = per.shape
    sel = np.zeros((H, M), dtype=np.int32)
    for p in range(P):
        sel[per[:, :, p] == 1] = p + 1
    return sel


def evaluate(line, spec, sels: np.ndarray) -> np.ndarray:
    """sels: [K, H, M] -> exact-sim profit [K]."""
    tape = sel_to_tape(jnp.asarray(sels), spec.n_products).astype(line.dtype)
    return np.asarray(profit_breakdown_2p(spec, line.rollout(tape))["profit"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--utils", type=float, nargs="+",
                    default=[0.50, 0.60, 0.80, 0.92])
    ap.add_argument("--method", default="gradst")
    ap.add_argument("--max-trim", type=int, default=60,
                    help="most gating-tool jobs to consider deleting")
    args = ap.parse_args()

    out = {}
    for util in args.utils:
        f = SCH / sched_name(args.alpha, util)
        if not f.exists():
            print(f"  skip {util:.0%}: not generated yet")
            continue
        z = np.load(f)
        common = make_common_config(mode="UNIFORM", alpha=args.alpha,
                                    utilization=util)
        spec = spec2p_from_common_config(common)
        line = make_line_2p(spec)

        sel = tape_of(z, args.method)
        m0_starts = np.flatnonzero(sel[:, 0] > 0)          # ticks M0 begins a job
        K = min(args.max_trim, len(m0_starts))

        # variant k deletes the last k gating-tool releases and nothing else
        variants = np.repeat(sel[None, ...], K + 1, axis=0)
        for k in range(1, K + 1):
            variants[k][m0_starts[-k:], 0] = 0

        profits = evaluate(line, spec, variants)
        best_k = int(np.argmax(profits))
        base, best = float(profits[0]), float(profits[best_k])
        cp = float(z["cpsat__profit"])
        gap = cp - base
        rec = {"util": util, "base": base, "best": best, "best_k": best_k,
               "trimmed_units": best_k * int(spec.take[0]),
               "recovered": best - base, "cpsat": cp, "gap": gap,
               "recovered_pct": (best - base) / gap if gap > 0 else None,
               "still_behind": cp - best}
        out[f"{util:.2f}"] = rec
        print(f"\n{util:.0%} utilisation  ({args.method})")
        print(f"   as optimised          ${base:>10,.0f}")
        print(f"   best simple trim      ${best:>10,.0f}   "
              f"(delete the last {best_k} gating-tool jobs = "
              f"{best_k * int(spec.take[0])} units)")
        print(f"   recovered             ${best - base:>10,.0f}   "
              + (f"= {(best - base) / gap:.0%} of the ${gap:,.0f} gap to CP-SAT"
                 if gap > 0 else ""))
        print(f"   CP-SAT                ${cp:>10,.0f}   "
              f"still ahead by ${cp - best:,.0f}")

    d = json.loads(DATA.read_text())
    d.setdefault("twoprod", {})["trim"] = out
    DATA.write_text(json.dumps(d, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
