"""
Two-product method comparison.

Every method optimises the same object — a categorical tape `sel[B, H, M]` with
values {0 = idle, 1 = P0, 2 = P1} — from the same random starting point, at the
same budget in forward-rollout-equivalents. Every profit figure is that tape
replayed through the EXACT integer simulator; the relaxed objective a method
optimised internally is never reported as a result.

    commander    the reference demand-share heuristic
    random       random search over categorical tapes (the floor)
    hillclimb    (1+1) evolution strategy: re-draw the choice at random
                 positions, keep improvements. Mutation-rate ladder across the
                 batch so it is not a strawman.
    grad-st      Adam on theta[B, H, M, 3] through softmax straight-through
    milp         CP-SAT incumbent and dual bound (opt-in; slow)

Budget: one gradient step counts as three forward rollouts (forward + backward),
so the zeroth-order methods receive 3 * steps * batch rollouts.

Usage:
    python -m tensorfab.twoprod.compare --alphas 0.5 0.9 0.1 --util 0.80
    python -m tensorfab.twoprod.compare --alphas 0.5 --milp --milp-seconds 900
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from simplefab.config import make_common_config  # noqa: E402
from tensorfab.logbook import Run  # noqa: E402
from tensorfab.twoprod.diffsim import DiffLine2P, Relax2P  # noqa: E402
from tensorfab.twoprod.spec import spec2p_from_common_config  # noqa: E402
from tensorfab.twoprod.tensor_sim import make_line_2p, profit_breakdown_2p  # noqa: E402


def sel_to_tape(sel: jnp.ndarray, P: int) -> jnp.ndarray:
    """[B, H, M] categorical -> [H, B, M, P] one-hot, idle dropped."""
    oh = jax.nn.one_hot(sel, P + 1, dtype=jnp.int32)[..., 1:]
    return jnp.swapaxes(oh, 0, 1)


def make_evaluator(line):
    P = line.spec.n_products

    @jax.jit
    def ev(sel):
        final = line._rollout(line.init_state(sel.shape[0]),
                              sel_to_tape(sel, P).astype(line.dtype))
        sp = line.spec
        return (
            (jnp.asarray(sp.revenue) * final.cum_ship).sum(-1)
            - (jnp.asarray(sp.prod_cost * sp.take[:, None]) * final.started).sum((-2, -1))
            - (jnp.asarray(sp.setup_cost) * final.setups).sum(-1)
            - (jnp.asarray(sp.fg_holding) * final.sum_qfin).sum(-1)
            - (jnp.asarray(sp.wip_holding) * final.sum_q).sum((-2, -1))
            - (jnp.asarray(sp.backorder) * final.sum_late).sum(-1)
        )
    return ev


def exact_profit(line, sel) -> np.ndarray:
    """Float64 profit for reporting."""
    final = line._rollout(line.init_state(sel.shape[0]),
                          sel_to_tape(jnp.asarray(sel), line.spec.n_products).astype(line.dtype))
    return profit_breakdown_2p(line.spec, final)["profit"]


def commander_profit(common) -> float:
    from simplefab.sim import ProductionLine
    pt, su = common["processing_times"], common["setup_times"]
    l = ProductionLine({0: pt[0][0], 1: pt[0][1]}, {0: pt[1][0], 1: pt[1][1]}, su[1],
                       {0: pt[2][0], 1: pt[2][1]}, {0: pt[3][0], 1: pt[3][1]}, su[3],
                       common_cfg=common)
    for t in range(common["time_horizon"]):
        l.run_step(current_time=t, actions_override=None)
    return float(l.profit_total())


def adam_ascend(dl, theta0, steps, lr, ev, eval_every, clip, run, P):
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def upd(theta, m, v, t):
        val, g = dl.value_and_grad(theta)
        if clip > 0:
            g = g * jnp.minimum(1.0, clip / (jnp.linalg.norm(g) + 1e-12))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta + lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
        return theta, m, v, val, g

    theta, m, v = theta0, jnp.zeros_like(theta0), jnp.zeros_like(theta0)
    best, best_sel, gn0, nonfinite = -np.inf, jnp.argmax(theta0, -1), None, 0
    for step in range(1, steps + 1):
        theta, m, v, val, g = upd(theta, m, v, step)
        gn = float(jnp.linalg.norm(g))
        if gn0 is None:
            gn0 = gn
        if not np.isfinite(gn):
            nonfinite += 1
        if step % eval_every == 0 or step == steps:
            sel = jnp.argmax(theta, axis=-1)
            p = float(np.max(np.asarray(ev(sel))))
            if p > best:
                best, best_sel = p, sel
    run.log(f"    grad-st: {steps} steps, |g| {gn0:.3g} -> {gn:.3g}, "
            f"non-finite {nonfinite}")
    return best, best_sel


def hill_climb(ev, sel0, iters, p_mut, key, n_opt):
    fit0 = ev(sel0)
    pm = p_mut[:, None, None]

    def body(i, carry):
        sel, fit, key = carry
        key, k1, k2 = jax.random.split(key, 3)
        mutate = jax.random.uniform(k1, sel.shape) < pm
        fresh = jax.random.randint(k2, sel.shape, 0, n_opt)
        prop = jnp.where(mutate, fresh, sel)
        f = ev(prop)
        better = f > fit
        return (jnp.where(better[:, None, None], prop, sel),
                jnp.where(better, f, fit), key)

    sel, fit, _ = lax.fori_loop(0, iters, body, (sel0, fit0, key))
    return sel, fit


def random_search(ev, shape, iters, key, n_opt):
    def body(i, carry):
        best, best_sel, key = carry
        key, k1, k2 = jax.random.split(key, 3)
        # vary how often a tool is asked to run at all
        idle_p = jax.random.uniform(k1, (shape[0], 1, 1), minval=0.05, maxval=0.6)
        r = jax.random.uniform(k2, shape)
        prod = (r < idle_p * 0.5).astype(jnp.int32) + (r < idle_p).astype(jnp.int32)
        f = ev(prod)
        i_best = jnp.argmax(f)
        better = f[i_best] > best
        return (jnp.where(better, f[i_best], best),
                jnp.where(better, prod[i_best], best_sel), key)

    z = jnp.zeros(shape[1:], dtype=jnp.int32)
    best, best_sel, _ = lax.fori_loop(0, iters, body, (jnp.float32(-1e9), z, key))
    return best, best_sel


def run_cell(alpha, args, run, key) -> Dict[str, Any]:
    common = make_common_config(mode=args.mode, alpha=alpha, utilization=args.util)
    spec = spec2p_from_common_config(common, mode=args.mode)
    line = make_line_2p(spec)
    ev = make_evaluator(line)
    B, H, M, P = args.batch, spec.H, spec.n_machines, spec.n_products
    n_opt = P + 1

    run.log(f"\n--- alpha={alpha}  util={args.util}  mode={args.mode}  "
            f"demand={spec.demand.sum(axis=0).tolist()} ---")
    k_init, k_hc, k_rs = jax.random.split(key, 3)
    theta0 = jax.random.normal(k_init, (B, H, M, n_opt)) * args.init_scale
    sel0 = jnp.argmax(theta0, axis=-1)

    budget = 3 * args.steps * B
    iters = budget // B
    run.log(f"    budget {budget:,} rollout-equivalents ({args.steps} grad steps x B={B} "
            f"== {iters:,} zeroth-order iters)")

    res: Dict[str, Any] = {"alpha": alpha, "demand": spec.demand.sum(axis=0).tolist()}
    res["commander"] = commander_profit(common)

    dl = DiffLine2P(spec, Relax2P(mode="st"))
    _, g_sel = adam_ascend(dl, theta0, args.steps, args.lr, ev,
                           args.eval_every, args.clip, run, P)
    res["grad-st"] = float(np.max(exact_profit(line, np.asarray(g_sel))))

    ladder = np.array([1, 4, 16, 64], dtype=np.float64)
    p_mut = jnp.asarray(np.repeat(ladder, max(1, B // 4))[:B] / (H * M), dtype=jnp.float32)
    hc_sel, _ = hill_climb(ev, sel0, iters, p_mut, k_hc, n_opt)
    res["hillclimb"] = float(np.max(exact_profit(line, np.asarray(hc_sel))))

    _, rs_sel = random_search(ev, (B, H, M), iters, k_rs, n_opt)
    res["random"] = float(exact_profit(line, np.asarray(rs_sel)[None])[0])

    if args.milp:
        from tensorfab.twoprod.optim import solve_2p
        sol = solve_2p(common, time_limit_sec=args.milp_seconds,
                       num_workers=8, verbose=False)
        res["milp"] = sol.get("profit")
        res["milp_bound"] = sol["objective_bound"]
        res["milp_changeovers"] = sol.get("changeovers")

    bound = res.get("milp_bound")
    run.log(f"    {'method':<12}{'profit':>13}{'vs commander':>15}"
            + (f"{'% of bound':>12}" if bound else ""))
    for k in ("random", "commander", "hillclimb", "grad-st", "milp"):
        if k not in res or res[k] is None:
            continue
        v = res[k]
        pct = f"{v / bound:>11.1%}" if bound else ""
        run.log(f"    {k:<12}{v:>13,.2f}{v - res['commander']:>+15,.2f}{pct}")
    if bound:
        run.log(f"    {'bound':<12}{bound:>13,.2f}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.9, 0.1])
    ap.add_argument("--util", type=float, default=0.80)
    ap.add_argument("--mode", default="UNIFORM")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--milp", action="store_true")
    ap.add_argument("--milp-seconds", type=float, default=900.0)
    args = ap.parse_args()

    with Run("twoprod_compare", params=vars(args).copy(),
             note="Two-product method comparison: commander / random / hill-climb / "
                  "softmax straight-through gradient / CP-SAT, equal budget") as run:
        run.log(device.describe())
        key = jax.random.PRNGKey(args.seed)
        results = {}
        for a in args.alphas:
            key, sub = jax.random.split(key)
            results[str(a)] = run_cell(a, args, run, sub)

        run.log("\n" + "=" * 84)
        run.log("SUMMARY — exact-sim profit, best of batch")
        cols = ["commander", "random", "hillclimb", "grad-st"] + (["milp", "milp_bound"] if args.milp else [])
        run.log(f"  {'alpha':>6}" + "".join(f"{c:>13}" for c in cols) + f"{'ST - best0th':>14}")
        for a, r in results.items():
            row = f"  {a:>6}" + "".join(
                f"{(r.get(c) if r.get(c) is not None else float('nan')):>13,.0f}" for c in cols)
            row += f"{r['grad-st'] - max(r['random'], r['hillclimb']):>+14,.0f}"
            run.log(row)

        (run.dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
        run.set_metrics(results=results)
        wins = sum(1 for r in results.values()
                   if r["grad-st"] > max(r["random"], r["hillclimb"]))
        margins = [r["grad-st"] - max(r["random"], r["hillclimb"]) for r in results.values()]
        run.headline(f"grad-st beats zeroth-order on {wins}/{len(results)} mixes "
                     f"(margin ${min(margins):,.0f} to ${max(margins):,.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
