"""
Cross-pattern method comparison.

Runs the full tensorfab method stack on every demand pattern, under the CURRENT
(halved) WIP costs, so that all numbers are mutually comparable and comparable to
the freshly retrained PPO agents.

Methods, all evaluated on the EXACT integer simulator:
    commander    the greedy heuristic in sim.py
    random       random search at the gradient's budget (the floor)
    hillclimb    (1+1) ES on the binary tape, flip-rate ladder (zeroth-order control)
    grad-soft    Adam through the fully relaxed forward
    grad-st      Adam through straight-through gradients  <- the method under test
    milp         CP-SAT, time-limited: a real optimality certificate

The MILP column matters historically: CLAUDE.md's "Theoretical Max" was a naive
bound (demand x (revenue - production) minus a small term), not a solve, so every
"% of max" figure in that table was pessimistic. This computes the real thing.

Usage:
    python -m tensorfab.compare_patterns
    python -m tensorfab.compare_patterns --patterns UNIFORM ALTERNATING_50
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from simplefab_1p.config import make_common_config  # noqa: E402
from tensorfab import policies  # noqa: E402
from tensorfab.diffsim import DiffLine, Relax, hard_actions  # noqa: E402
from tensorfab.logbook import Run  # noqa: E402
from tensorfab.spec import spec_from_common_config  # noqa: E402
from tensorfab.spike_openloop import (  # noqa: E402
    adam_ascend, exact_profit_f64, hill_climb, make_exact_evaluator, random_search,
)
from tensorfab.tensor_sim import make_line, profit_breakdown  # noqa: E402

# Historical numbers from CLAUDE.md, recorded under the PRE-May-15 WIP costs
# (0.005/0.01/0.015 — double the current values) and possibly with the
# since-removed -5.0 pipeline nudge. Shown for context only; the live comparison
# is against the freshly retrained agents.
HISTORICAL = {
    "UNIFORM":        {"agent": 17607, "commander": -159,  "claimed_max": 18786},
    "ALTERNATING_50": {"agent": 16337, "commander": -2813, "claimed_max": 18786},
    "ALTERNATING":    {"agent": 29965, "commander": 29480, "claimed_max": 33546},
    "ALTERNATING_92": {"agent": 31493, "commander": 31301, "claimed_max": None},
}


def run_pattern(pattern: str, args, run: Run, key) -> Dict[str, Any]:
    common = make_common_config(mode=pattern, n_pairs=args.n_pairs,
                                utilization=args.utilization)
    spec = spec_from_common_config(common, mode=pattern)
    line = make_line(spec)
    ev = make_exact_evaluator(line)
    B, H, M = args.batch, spec.H, spec.n_machines

    util = spec.total_demand / max(1, int(spec.arrivals.sum()))
    run.log(f"\n{'=' * 70}")
    run.log(f"{pattern}   n_pairs={args.n_pairs}  M={spec.n_machines}  "
            f"lead={int(spec.pt.sum())}t  demand={spec.total_demand}  util={util:.1%}")
    run.log(f"{'=' * 70}")

    k_init, k_hc, k_rs = jax.random.split(key, 3)
    theta0 = jax.random.normal(k_init, (B, H, M)) * args.init_scale
    acts0 = hard_actions(theta0)
    budget = 3 * args.steps * B
    zeroth_iters = budget // B
    run.log(f"  budget {budget:,} rollout-equivalents "
            f"({args.steps} grad steps x B={B} == {zeroth_iters:,} zeroth-order iters)")

    res: Dict[str, Any] = {"demand": spec.total_demand, "utilization": util, "H": H}

    cmd = line.rollout_policy(policies.commander(line), batch=1)
    res["commander"] = float(profit_breakdown(spec, cmd)["profit"][0])

    grad_extra = {}
    for mode in ("st", "soft"):
        dl = DiffLine(spec, Relax(mode=mode))
        out = adam_ascend(dl, theta0, args.steps, args.lr, ev,
                          args.eval_every, run, f"grad-{mode}", clip=args.clip)
        res[f"grad-{mode}"] = float(np.max(exact_profit_f64(line, out["best_acts"])))
        grad_extra[mode] = out

    ladder = np.array([1, 4, 16, 64], dtype=np.float64)
    p_flip = jnp.asarray(np.repeat(ladder, max(1, B // 4))[:B] / (H * M), dtype=jnp.float32)
    hc = hill_climb(ev, acts0, zeroth_iters, p_flip, k_hc)
    res["hillclimb"] = float(np.max(exact_profit_f64(line, hc["best_acts"])))
    res["random"] = random_search(ev, (B, H, M), zeroth_iters, k_rs)["best_profit"]

    if not args.no_milp:
        from simplefab_1p.optim import solve_optimal_schedule

        sol = solve_optimal_schedule(common, time_limit_sec=args.milp_seconds,
                                     verbose=False, num_workers=8)
        res["milp"] = float(sol["profit"])
        res["milp_bound"] = float(sol["objective_bound"])
        res["milp_optimal"] = bool(sol["is_optimal"])

    gbt = grad_extra["st"]["grad_norm_by_t"]
    if gbt is not None and gbt.sum() > 0:
        rev_cum = np.cumsum(gbt[::-1]) / gbt.sum()
        res["grad_reach_frac"] = float((np.searchsorted(rev_cum, 0.9) + 1) / H)
        res["grad_mass_first_quarter"] = float(gbt[:H // 4].sum() / gbt.sum())
    res["grad_norm_initial"] = float(grad_extra["st"]["grad_norm_curve"][0])
    res["n_nonfinite"] = int(grad_extra["st"]["n_nonfinite"])

    # The CP-SAT *incumbent* is only a valid ceiling if the solve converged. On
    # these instances it often does not within the time limit — on UNIFORM the
    # 180 s incumbent came in below plain random search. The *dual bound*
    # (objective_bound) is a rigorous upper bound on the true optimum regardless
    # of convergence, so that is what we normalise against.
    bound = res.get("milp_bound")
    incumbent = res.get("milp")
    methods = ("random", "commander", "hillclimb", "grad-soft", "grad-st")
    best_found = max((res[k] for k in methods if k in res), default=None)
    if incumbent is not None and best_found is not None and incumbent < best_found:
        res["milp_beaten"] = True
        res["milp_best_known"] = best_found

    run.log(f"\n  {'method':<14}{'profit':>14}{'vs commander':>15}{'% of bound':>13}")
    for k in methods + ("milp",):
        if k not in res:
            continue
        v = res[k]
        pct = f"{v / bound:>12.1%}" if bound else " " * 13
        run.log(f"  {k:<14}{v:>14,.2f}{v - res['commander']:>+15,.2f}{pct}")
    if bound is not None:
        status = "PROVEN OPTIMAL" if res.get("milp_optimal") else "NOT converged"
        run.log(f"  {'milp bound':<14}{bound:>14,.2f}{'':>15}{'':>13}  <- CP-SAT dual "
                f"bound ({status} in {args.milp_seconds:.0f}s)")
    if res.get("milp_beaten"):
        run.log(f"  ** CP-SAT incumbent ${incumbent:,.0f} is BELOW the best method "
                f"found (${best_found:,.0f}) — the solver did not converge and its "
                f"incumbent is not a valid ceiling here.")

    hist = HISTORICAL.get(pattern)
    if hist:
        run.log(f"\n  historical (CLAUDE.md, OLD doubled WIP costs — context only):")
        cm = hist["claimed_max"]
        run.log(f"    agent {hist['agent']:>10,}   commander {hist['commander']:>10,}"
                f"   claimed 'max' {cm if cm is not None else 'n/a':>10}")
        if cm is not None and bound is not None:
            run.log(f"    -> that 'max' was a naive hand-computed bound; the rigorous "
                    f"CP-SAT dual bound is ${bound:,.0f} ({bound - cm:+,.0f})")

    run.save_series(**{f"{pattern}_grad_norm_by_t": gbt if gbt is not None else np.zeros(1),
                       f"{pattern}_hillclimb_curve": hc["curve"],
                       f"{pattern}_st_exact_curve": grad_extra["st"]["exact_curve"]})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+",
                    default=["UNIFORM", "ALTERNATING_50", "ALTERNATING", "ALTERNATING_92"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--milp-seconds", type=float, default=180.0)
    ap.add_argument("--clip", type=float, default=0.0,
                    help="global-norm gradient clip; 0 disables")
    ap.add_argument("--utilization", type=float, default=0.50,
                    help="target demand as a fraction of line capacity; ignored by the "
                         "ALTERNATING_* modes, which hardcode their own demand pattern")
    ap.add_argument("--n-pairs", type=int, default=2,
                    help="diffusion+litho pairs; 2 = the original 4-machine line")
    ap.add_argument("--no-milp", action="store_true")
    args = ap.parse_args()

    with Run("compare_patterns", params=vars(args).copy(),
             note="Cross-pattern method comparison under current (halved) WIP "
                  "costs; adds a real CP-SAT optimality certificate per pattern") as run:
        run.log(device.describe())
        key = jax.random.PRNGKey(args.seed)
        results: Dict[str, Any] = {}
        for p in args.patterns:
            key, sub = jax.random.split(key)
            results[p] = run_pattern(p, args, run, sub)

        run.log("\n" + "=" * 96)
        run.log("SUMMARY — exact-sim profit, best of batch, current (halved) WIP costs")
        run.log("=" * 96)
        cols = ["commander", "random", "hillclimb", "grad-soft", "grad-st",
                "milp", "milp_bound"]
        run.log(f"  {'pattern':<16}{'util':>7}" + "".join(f"{c:>12}" for c in cols)
                + f"{'grad %bnd':>11}{'grad-hc':>12}")
        for p, r in results.items():
            row = f"  {p:<16}{r['utilization']:>6.0%} "
            row += "".join(f"{r.get(c, float('nan')):>12,.0f}" for c in cols)
            if r.get("milp_bound"):
                row += f"{r['grad-st'] / r['milp_bound']:>10.1%}"
            row += f"{r['grad-st'] - r['hillclimb']:>+12,.0f}"
            run.log(row)

        out = run.dir / "results.json"
        out.write_text(json.dumps(results, indent=2, default=str))
        run.set_metrics(results=results)

        wins = sum(1 for r in results.values() if r["grad-st"] > r["hillclimb"])
        pcts = [r["grad-st"] / r["milp_bound"]
                for r in results.values() if r.get("milp_bound")]
        margins = [r["grad-st"] - r["hillclimb"] for r in results.values()]
        beaten = sum(1 for r in results.values() if r.get("milp_beaten"))
        run.set_metrics(gradient_wins_vs_hillclimb=wins,
                        milp_not_converged=beaten,
                        margins_vs_hillclimb=margins)
        run.headline(
            f"grad-st beats hillclimb on {wins}/{len(results)} patterns "
            f"(margin ${min(margins):,.0f} to ${max(margins):,.0f}); "
            f"mean {np.mean(pcts):.1%} of CP-SAT dual bound; "
            f"CP-SAT failed to converge on {beaten}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
