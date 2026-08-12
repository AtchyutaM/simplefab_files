"""
Phase 2b de-risking spike: do surrogate gradients through the discontinuous line
point anywhere useful?

--------------------------------------------------------------------------------
The question
--------------------------------------------------------------------------------
docs/tensorized_line_report.md § 5.2 stakes the strategic thesis on a
*differentiable* simulator giving a better optimisation signal than model-free RL.
Suh et al. (ICML 2022) is the standing warning: through stiff or discontinuous
dynamics, pathwise gradients can be higher-variance or more biased than plain
score-function estimators. Our batch completions, threshold gates, and
ship-vs-store branches are the analog of contacts in physics.

So before building a closed-loop policy trainer on top of this, answer the
cheapest possible version of the question. We optimise an OPEN-LOOP action tape
theta[B, H, 4] — no policy network, no RL, no value function. Just: does gradient
ascent through the surrogate beat a zeroth-order method that sees only the exact
simulator, at equal budget?

This isolates the gradient. If it loses here, it will not win with a policy
network stacked on top, and Phase 2b should be abandoned or redesigned.

--------------------------------------------------------------------------------
The comparison
--------------------------------------------------------------------------------
All methods start from the SAME random initial tapes and get the SAME budget,
measured in forward-rollout-equivalents (one gradient step = 1 forward + 1
backward ~= 3 forwards).

  grad-st     Adam on theta, straight-through gradients (exact forward)
  grad-soft   Adam on theta, fully relaxed forward (smooth but wrong system)
  hillclimb   (1+1) evolution strategy on the binary tape, flipping bits.
              Given a ladder of flip rates across the batch (1/4/16/64) so the
              zeroth-order baseline is genuinely competitive, not a strawman.
  random      Pure random search — the floor.
  commander   The greedy heuristic in sim.py.
  milp        CP-SAT, time-limited. Near-optimal reference ceiling (opt-in).

EVERY number reported is the profit of the rounded tape run through the EXACT
integer simulator. Relaxed-forward profit is never reported as a result; it is
not achievable.

--------------------------------------------------------------------------------
The diagnostic that matters most
--------------------------------------------------------------------------------
`grad_norm_by_t`: ||d(profit)/d(theta[t, :])|| as a function of tick t. This is
the direct measurement of whether credit assignment survives backpropagation
through H steps of threshold gates. If it collapses for early t, the gradient
cannot tell the optimiser to pre-build for a demand spike 600 ticks later — which
is precisely the decision PPO's local reward gets wrong, and the entire reason
this approach was proposed.

Usage:
    python -m tensorfab.spike_openloop                    # default sweep
    python -m tensorfab.spike_openloop --horizons 672 --steps 800
    python -m tensorfab.spike_openloop --milp --milp-seconds 60
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from simplefab_1p.config import make_common_config  # noqa: E402
from tensorfab import policies  # noqa: E402
from tensorfab.diffsim import DiffLine, Relax, hard_actions  # noqa: E402
from tensorfab.logbook import Run  # noqa: E402
from tensorfab.spec import LineSpec, spec_from_common_config  # noqa: E402
from tensorfab.tensor_sim import TensorLine, make_line, profit_breakdown  # noqa: E402


# ----------------------------------------------------------------------
# problem construction — self-similar across horizons
# ----------------------------------------------------------------------
def make_spike_problem(
    H: int, utilization: float = 0.9, problem: str = "selfsimilar"
) -> Tuple[Dict, LineSpec]:
    """
    problem="selfsimilar":
        A scaled-down version of the problem, self-similar across H. UNIFORM
        demand with the arrival and demand intervals set to H/4 gives four
        delivery events and four demand events at every horizon, so only the BPTT
        *depth* changes across the sweep, not the character of the problem.

        Caveat learned from run 20260802-081...: this family is EASY. Commander
        lands within ~2.5% of the MILP and random search within ~0.3% of the best
        method found, so it cannot discriminate between optimisers. Use it for the
        depth sweep, not for method comparison.

    problem=<mode name>:
        One of the real configured problems (ALTERNATING_92 etc.). These have
        bursty demand where the commander is ~6.5% below the MILP, so there is
        real room for an optimiser to demonstrate skill. H is ignored (the mode
        fixes its own schedule).
    """
    if problem == "selfsimilar":
        interval = max(1, H // 4)
        common = make_common_config(
            mode="UNIFORM", H=H, utilization=utilization,
            demand_interval=interval, arrival_interval=interval,
        )
        return common, spec_from_common_config(common, mode=f"UNIFORM_H{H}")

    common = make_common_config(mode=problem, H=H)
    return common, spec_from_common_config(common, mode=problem)


# ----------------------------------------------------------------------
# exact evaluation — the only thing we ever report
# ----------------------------------------------------------------------
def make_exact_evaluator(line: TensorLine):
    @jax.jit
    def ev(acts: jnp.ndarray) -> jnp.ndarray:
        """acts [B, H, 4] int -> profit [B] (exact integer dynamics)."""
        s0 = line.init_state(acts.shape[0])
        final = line._rollout_actions(s0, jnp.swapaxes(acts, 0, 1).astype(line.dtype))
        return line.profit_jax(final)

    return ev


def exact_profit_f64(line: TensorLine, acts: np.ndarray) -> np.ndarray:
    """Host-side float64 profit — used for final reported numbers."""
    s0 = line.init_state(acts.shape[0])
    final = line._rollout_actions(
        s0, jnp.swapaxes(jnp.asarray(acts), 0, 1).astype(line.dtype)
    )
    return profit_breakdown(line.spec, final)["profit"]


# ----------------------------------------------------------------------
# method 1: Adam on the relaxed objective
# ----------------------------------------------------------------------
def adam_ascend(
    dl: DiffLine,
    theta0: jnp.ndarray,
    steps: int,
    lr: float,
    ev,
    eval_every: int,
    run: Run,
    tag: str,
) -> Dict[str, Any]:
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def update(theta, m, v, t):
        val, g = dl.value_and_grad(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        theta = theta + lr * mh / (jnp.sqrt(vh) + eps)   # ASCENT on profit
        return theta, m, v, val, g

    theta = theta0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)

    hist_relaxed, hist_exact, hist_gnorm, eval_steps = [], [], [], []
    best_profit = -np.inf
    best_acts = hard_actions(theta)
    n_nonfinite = 0
    grad_by_t_last = None

    for step in range(1, steps + 1):
        theta, m, v, val, g = update(theta, m, v, step)
        gn = float(jnp.linalg.norm(g))
        if not np.isfinite(gn):
            n_nonfinite += 1
        hist_relaxed.append(float(jnp.mean(val)))
        hist_gnorm.append(gn)

        if step % eval_every == 0 or step == steps:
            acts = hard_actions(theta)
            p = np.asarray(ev(acts))
            hist_exact.append(float(np.max(p)))
            eval_steps.append(step)
            if float(np.max(p)) > best_profit:
                best_profit = float(np.max(p))
                best_acts = acts
            grad_by_t_last = np.asarray(
                jnp.mean(jnp.linalg.norm(g, axis=-1), axis=0)
            )  # [H]

    run.log(f"    {tag}: {steps} Adam steps, "
            f"grad norm {hist_gnorm[0]:.3g} -> {hist_gnorm[-1]:.3g}, "
            f"non-finite grads: {n_nonfinite}")

    return {
        "best_profit": best_profit,
        "best_acts": np.asarray(best_acts),
        "relaxed_curve": np.asarray(hist_relaxed),
        "exact_curve": np.asarray(hist_exact),
        "eval_steps": np.asarray(eval_steps),
        "grad_norm_curve": np.asarray(hist_gnorm),
        "grad_norm_by_t": grad_by_t_last,
        "n_nonfinite": n_nonfinite,
    }


# ----------------------------------------------------------------------
# method 2: (1+1) hill climbing — the zeroth-order control
# ----------------------------------------------------------------------
def hill_climb(
    ev, acts0: jnp.ndarray, iters: int, p_flip: jnp.ndarray, key: jnp.ndarray
) -> Dict[str, Any]:
    """
    p_flip: [B] per-chain bit-flip probability, so different chains explore at
    different neighbourhood radii. Reported result is the best chain.
    """
    fit0 = ev(acts0)
    pf = p_flip[:, None, None]

    def body(i, carry):
        acts, fit, key, trace = carry
        key, k = jax.random.split(key)
        flip = jax.random.uniform(k, acts.shape) < pf
        prop = jnp.where(flip, 1 - acts, acts)
        f = ev(prop)
        better = f > fit
        acts = jnp.where(better[:, None, None], prop, acts)
        fit = jnp.where(better, f, fit)
        return acts, fit, key, trace.at[i].set(jnp.max(fit))

    trace = jnp.zeros((iters,), dtype=jnp.float32)
    acts, fit, _, trace = lax.fori_loop(0, iters, body, (acts0, fit0, key, trace))
    return {
        "best_profit": float(jnp.max(fit)),
        "best_acts": np.asarray(acts),
        "curve": np.asarray(trace),
        "fit": np.asarray(fit),
    }


# ----------------------------------------------------------------------
# method 3: random search — the floor
# ----------------------------------------------------------------------
def random_search(ev, shape: Tuple[int, int, int], iters: int, key: jnp.ndarray) -> Dict[str, Any]:
    def body(i, carry):
        best, key, trace = carry
        key, k1, k2 = jax.random.split(key, 3)
        # vary the run-rate across the batch; a single rate would be a strawman
        p = jax.random.uniform(k1, (shape[0], 1, 1), minval=0.1, maxval=0.95)
        acts = (jax.random.uniform(k2, shape) < p).astype(jnp.int32)
        f = jnp.max(ev(acts))
        best = jnp.maximum(best, f)
        return best, key, trace.at[i].set(best)

    trace = jnp.zeros((iters,), dtype=jnp.float32)
    best, _, trace = lax.fori_loop(0, iters, body, (jnp.float32(-1e9), key, trace))
    return {"best_profit": float(best), "curve": np.asarray(trace)}


# ----------------------------------------------------------------------
# the sweep
# ----------------------------------------------------------------------
def run_horizon(
    H: int, args, run: Run, key: jnp.ndarray
) -> Dict[str, Any]:
    common, spec = make_spike_problem(H, args.utilization, args.problem)
    line = make_line(spec)
    ev = make_exact_evaluator(line)
    B = args.batch

    run.log(f"\n--- {args.problem}  H={H}  demand={spec.total_demand} units  "
            f"util={spec.utilization:.2f} ---")

    # identical starting tapes for every method
    k_init, k_hc, k_rs = jax.random.split(key, 3)
    theta0 = jax.random.normal(k_init, (B, H, 4)) * args.init_scale
    acts0 = hard_actions(theta0)

    # budget: one gradient step ~= 3 forward rollouts (fwd + bwd)
    budget_rollouts = 3 * args.steps * B
    zeroth_iters = budget_rollouts // B
    run.log(f"    budget: {budget_rollouts:,} rollout-equivalents "
            f"({args.steps} grad steps x B={B}  ==  {zeroth_iters:,} zeroth-order iters)")

    results: Dict[str, Any] = {}

    # --- baselines -----------------------------------------------------
    cmd_final = line.rollout_policy(policies.commander(line), batch=1)
    results["commander"] = float(profit_breakdown(spec, cmd_final)["profit"][0])
    results["init_random"] = float(np.max(exact_profit_f64(line, np.asarray(acts0))))

    # --- gradient methods ----------------------------------------------
    grad_out: Dict[str, Any] = {}
    for mode in ("st", "soft"):
        dl = DiffLine(spec, Relax(mode=mode, tau_q=args.tau_q, tau_a=args.tau_a))
        out = adam_ascend(dl, theta0, args.steps, args.lr, ev,
                          args.eval_every, run, f"grad-{mode}")
        results[f"grad-{mode}"] = float(
            np.max(exact_profit_f64(line, out["best_acts"]))
        )
        grad_out[mode] = out

    # --- zeroth-order controls -----------------------------------------
    ladder = np.array([1, 4, 16, 64], dtype=np.float64)
    p_flip = jnp.asarray(
        np.repeat(ladder, max(1, B // len(ladder)))[:B] / (H * 4), dtype=jnp.float32
    )
    hc = hill_climb(ev, acts0, zeroth_iters, p_flip, k_hc)
    results["hillclimb"] = float(np.max(exact_profit_f64(line, hc["best_acts"])))

    rs = random_search(ev, (B, H, 4), zeroth_iters, k_rs)
    results["random"] = rs["best_profit"]

    # --- optional MILP ceiling -----------------------------------------
    if args.milp:
        from simplefab_1p.optim import solve_optimal_schedule

        sol = solve_optimal_schedule(
            common, time_limit_sec=args.milp_seconds, verbose=False, num_workers=8
        )
        results["milp"] = float(sol["profit"])
        results["milp_bound"] = float(sol["objective_bound"])
        results["milp_optimal"] = bool(sol["is_optimal"])

    # --- credit-assignment depth diagnostic -----------------------------
    gbt = grad_out["st"]["grad_norm_by_t"]
    if gbt is not None and gbt.sum() > 0:
        # fraction of total gradient mass in the first / last quarter of the tape
        n4 = max(1, H // 4)
        head = float(gbt[:n4].sum() / gbt.sum())
        tail = float(gbt[-n4:].sum() / gbt.sum())
        # "reach": how far back from the end we must go to capture 90% of mass
        rev_cum = np.cumsum(gbt[::-1]) / gbt.sum()
        reach = int(np.searchsorted(rev_cum, 0.9) + 1)
        results["grad_mass_first_quarter"] = head
        results["grad_mass_last_quarter"] = tail
        results["grad_reach_ticks"] = reach
        results["grad_reach_frac"] = reach / H
        run.log(f"    credit assignment: {head:.1%} of gradient mass in first "
                f"quarter, {tail:.1%} in last; 90% mass within {reach}/{H} "
                f"ticks of the end ({reach / H:.0%})")

    # --- report ---------------------------------------------------------
    order = ["random", "init_random", "commander", "hillclimb", "grad-soft", "grad-st"]
    if args.milp:
        order.append("milp")
    run.log(f"    {'method':<14} {'exact profit':>14}   vs commander")
    for k in order:
        if k not in results:
            continue
        v = results[k]
        delta = v - results["commander"]
        run.log(f"    {k:<14} {v:>14,.2f}   {delta:>+12,.2f}")

    run.save_series(**{
        f"H{H}_grad_norm_by_t": gbt if gbt is not None else np.zeros(1),
        f"H{H}_st_exact_curve": grad_out["st"]["exact_curve"],
        f"H{H}_st_relaxed_curve": grad_out["st"]["relaxed_curve"],
        f"H{H}_st_grad_norm": grad_out["st"]["grad_norm_curve"],
        f"H{H}_soft_exact_curve": grad_out["soft"]["exact_curve"],
        f"H{H}_hillclimb_curve": hc["curve"],
        f"H{H}_random_curve": rs["curve"],
    })
    results["n_nonfinite_st"] = grad_out["st"]["n_nonfinite"]
    results["n_nonfinite_soft"] = grad_out["soft"]["n_nonfinite"]
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2b open-loop gradient spike")
    ap.add_argument("--horizons", type=int, nargs="+", default=[336, 672, 1344, 2688])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400, help="Adam steps")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--tau-q", type=float, default=1.0)
    ap.add_argument("--tau-a", type=float, default=1.0)
    ap.add_argument("--utilization", type=float, default=0.9)
    ap.add_argument("--problem", default="selfsimilar",
                    help="'selfsimilar' (scaled UNIFORM, for the depth sweep) or a "
                         "configured mode name such as ALTERNATING_92 (the hard, "
                         "bursty-demand problem used for method comparison)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--milp", action="store_true", help="add a CP-SAT reference")
    ap.add_argument("--milp-seconds", type=float, default=60.0)
    args = ap.parse_args()

    params = vars(args).copy()
    with Run("openloop_spike", params=params,
             note="Phase 2b de-risk: do surrogate gradients beat zeroth-order "
                  "search at equal budget, and how deep does credit reach?") as run:
        run.log(device.describe())
        key = jax.random.PRNGKey(args.seed)

        all_results: Dict[int, Dict[str, Any]] = {}
        for i, H in enumerate(args.horizons):
            key, sub = jax.random.split(key)
            all_results[H] = run_horizon(H, args, run, sub)

        # ---- summary ----------------------------------------------------
        run.log("\n" + "=" * 78)
        run.log("SUMMARY — exact-sim profit, best of batch")
        run.log("=" * 78)
        cols = ["commander", "random", "hillclimb", "grad-soft", "grad-st"]
        if args.milp:
            cols.append("milp")
        run.log(f"  {'H':>6} " + "".join(f"{c:>13}" for c in cols)
                + f"{'grad reach':>13}")
        for H, r in all_results.items():
            row = f"  {H:>6} " + "".join(f"{r.get(c, float('nan')):>13,.0f}" for c in cols)
            row += f"{r.get('grad_reach_frac', float('nan')):>12.0%}"
            run.log(row)

        # ---- the verdict -------------------------------------------------
        run.log("")
        wins = sum(1 for r in all_results.values()
                   if max(r["grad-st"], r["grad-soft"]) > r["hillclimb"])
        beats_cmd = sum(1 for r in all_results.values()
                        if max(r["grad-st"], r["grad-soft"]) > r["commander"])
        n = len(all_results)
        run.log(f"  gradient beats zeroth-order at equal budget: {wins}/{n} horizons")
        run.log(f"  gradient beats the commander heuristic:      {beats_cmd}/{n} horizons")

        reaches = [r.get("grad_reach_frac", np.nan) for r in all_results.values()]
        run.set_metrics(results={str(k): v for k, v in all_results.items()},
                        gradient_wins_vs_zeroth=wins,
                        gradient_wins_vs_commander=beats_cmd,
                        n_horizons=n)

        verdict = ("gradient WINS vs zeroth-order" if wins > n / 2
                   else "gradient LOSES vs zeroth-order")
        run.headline(
            f"{verdict} ({wins}/{n} horizons); beats commander {beats_cmd}/{n}; "
            f"credit reach {np.nanmean(reaches):.0%} of horizon"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
