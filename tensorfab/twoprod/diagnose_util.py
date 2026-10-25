"""
Why does the gradient look worse against the bound at low load?

The utilisation sweep showed grad-ST at 86.2% of the certified bound at 60%
against 91.8% at 92%, which reads like the gradient degrading as the line
empties -- the opposite of the single-product finding. This decomposes the
comparison so the claim can be checked rather than believed.

Three separate things are folded into "% of bound" and they move differently:

  1. bound looseness  -- CP-SAT itself reaches only 95.7% of the bound at 60%
                         and 97.3% at 92%, so the yardstick is not equally
                         tight at both ends
  2. the profit base  -- a fixed dollar shortfall is a larger percentage of a
                         $22k optimum than of a $34k one
  3. real schedule quality -- what is left after the first two

and then splits the remaining dollar gap into the cost lines that produce it,
so "the gradient over-releases" is a measurement rather than a story.

    python -m tensorfab.twoprod.diagnose_util
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
from tensorfab.twoprod.naming import sched_name  # noqa: E402
from tensorfab.twoprod.spec import spec2p_from_common_config  # noqa: E402
from tensorfab.twoprod.compare import sel_to_tape  # noqa: E402
from tensorfab.twoprod.tensor_sim import make_line_2p, profit_breakdown_2p  # noqa: E402

HERE = pathlib.Path(__file__).parent
SCH = HERE / "schedules"
DATA = HERE.parent / "presentation_data.json"
METHODS = ("cpsat", "gradst", "commander")


def tape_of(z, method: str) -> np.ndarray:
    """Rebuild the [1, H, M] selection tape from the stored cumulative starts."""
    cum = z[f"{method}__started_jobs"]                  # [H, M, P] cumulative
    per = np.diff(cum, axis=0, prepend=0)               # [H, M, P] 0/1
    H, M, P = per.shape
    sel = np.zeros((1, H, M), dtype=np.int32)
    for p in range(P):
        sel[0][per[:, :, p] == 1] = p + 1
    return sel


def breakdown(alpha: float, util: float) -> dict:
    z = np.load(SCH / sched_name(alpha, util))
    common = make_common_config(mode="UNIFORM", alpha=alpha, utilization=util)
    spec = spec2p_from_common_config(common)
    line = make_line_2p(spec)

    out = {"util": util, "bound": float(z["cpsat_bound"]),
           "demand": int(z["demand"].sum()), "methods": {}}
    for m in METHODS:
        if f"{m}__profit" not in z.files:
            continue
        sel = tape_of(z, m)
        tape = sel_to_tape(jnp.asarray(sel), spec.n_products).astype(line.dtype)
        bd = profit_breakdown_2p(spec, line.rollout(tape))
        # some entries are per-product or per-tool vectors; sum those
        rec = {}
        for k, v in bd.items():
            a = np.asarray(v)[0]
            rec[k] = float(a) if a.ndim == 0 else float(a.sum())
        # the replay must agree with the stored profit or the tape is wrong
        rec["stored"] = float(z[f"{m}__profit"])
        rec["replay_ok"] = abs(rec["profit"] - rec["stored"]) < 0.05
        q = z[f"{m}__q"]
        rec["stranded"] = int(q[-1, 1:].sum())
        rec["fg_left"] = int(z[f"{m}__qfin"][-1].sum())
        rec["released"] = int(np.diff(z[f"{m}__started_jobs"][:, 0, :], axis=0,
                                      prepend=0).sum() * int(z["take"][0]))
        rec["shipped"] = int(z[f"{m}__cum_ship"][-1].sum())
        out["methods"][m] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--utils", type=float, nargs="+",
                    default=[0.50, 0.60, 0.80, 0.92])
    args = ap.parse_args()

    runs = [u for u in sorted(args.utils)
            if (SCH / sched_name(args.alpha, u)).exists()]
    for u in args.utils:
        if u not in runs:
            print(f"  skip {u:.0%}: not generated yet")

    all_out = {}
    for u in runs:
        b = breakdown(args.alpha, u)
        all_out[f"{u:.2f}"] = b
        cp, gs = b["methods"].get("cpsat"), b["methods"].get("gradst")
        print(f"\n{'='*76}\n{u:.0%} utilisation   demand {b['demand']}   "
              f"bound ${b['bound']:,.0f}\n{'='*76}")
        keys = [k for k in ("revenue", "production", "setup", "inventory",
                            "wip", "backorder", "profit")
                if cp and k in cp]
        hdr = f"{'component':>16s}" + "".join(f"{m:>13s}" for m in b["methods"])
        print(hdr + f"{'CP-SAT - ST':>14s}")
        for k in keys:
            line = f"{k:>16s}"
            for m in b["methods"]:
                line += f"{b['methods'][m][k]:>13,.0f}"
            if cp and gs:
                line += f"{cp[k] - gs[k]:>+14,.0f}"
            print(line)
        for k in ("released", "shipped", "stranded", "fg_left"):
            line = f"{k:>16s}"
            for m in b["methods"]:
                line += f"{b['methods'][m][k]:>13,d}"
            print(line)
        bad = [m for m, r in b["methods"].items() if not r["replay_ok"]]
        print(f"{'replay check':>16s}" +
              ("  all tapes reproduce the stored profit" if not bad
               else f"  MISMATCH: {bad}"))

    d = json.loads(DATA.read_text())
    d.setdefault("twoprod", {})["util_diagnosis"] = all_out
    DATA.write_text(json.dumps(d, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
