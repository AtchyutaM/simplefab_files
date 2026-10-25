"""
Generate and cache the schedules each method produces, for side-by-side study.

The metrics tables say which method scores higher. They say nothing about *what
the schedules actually do differently* — when material is released, where it
queues, how long tools sit idle, when changeovers happen. This dumps the full
per-tick trace of each method's schedule so that can be looked at directly.

    python -m tensorfab.twoprod.make_schedules --milp-seconds 600
"""
from __future__ import annotations
import argparse, json, time, pathlib
import numpy as np
from tensorfab import device
device.configure()
import jax, jax.numpy as jnp
from simplefab.config import make_common_config
from tensorfab.twoprod.naming import sched_name
from tensorfab.twoprod.spec import spec2p_from_common_config
from tensorfab.twoprod.tensor_sim import make_line_2p, profit_breakdown_2p
from tensorfab.twoprod.diffsim import DiffLine2P, Relax2P
from tensorfab.twoprod.compare import make_evaluator, adam_ascend, sel_to_tape

OUT = pathlib.Path(__file__).parent / "schedules"


class _Q:
    def log(self, *a, **k): pass


def trace_of(line, spec, sel):
    """Replay a categorical tape and pull out every per-tick quantity."""
    tape = sel_to_tape(jnp.asarray(sel), spec.n_products).astype(line.dtype)
    final, tr = line.rollout_trace(tape)
    take = spec.take.astype(np.int64)
    make = spec.make.astype(np.int64)
    started = np.asarray(tr.started)[:, 0]            # [H, M, P] cumulative jobs
    busy = np.asarray(tr.busy)[:, 0]                  # [H, M]
    q = np.asarray(tr.q)[:, 0]                        # [H, M, P]

    # cumulative UNITS started / completed at each tool
    started_u = (started * take[None, :, None]).sum(-1)          # [H, M]
    inflight_u = busy * make[None, :]                            # [H, M]
    done_u = started_u - inflight_u                              # [H, M]

    return {
        "q": q, "busy": busy, "curp": np.asarray(tr.curp)[:, 0],
        "lastp": np.asarray(tr.lastp)[:, 0],
        "started_jobs": started, "started_units": started_u, "done_units": done_u,
        "qfin": np.asarray(tr.qfin)[:, 0], "backlog": np.asarray(tr.backlog)[:, 0],
        "cum_ship": np.asarray(tr.cum_ship)[:, 0], "setups": np.asarray(tr.setups)[:, 0],
        "profit": float(profit_breakdown_2p(spec, final)["profit"][0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--util", type=float, default=0.80)
    ap.add_argument("--mode", default="UNIFORM")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--milp-seconds", type=float, default=600.0)
    args = ap.parse_args()

    common = make_common_config(mode=args.mode, alpha=args.alpha, utilization=args.util)
    spec = spec2p_from_common_config(common, mode=args.mode)
    line = make_line_2p(spec); ev = make_evaluator(line)
    H, M, P = spec.H, spec.n_machines, spec.n_products
    OUT.mkdir(exist_ok=True)
    store = {"H": H, "M": M, "P": P,
             "arrivals": spec.arrivals, "demand": spec.demand,
             "pt": spec.pt, "take": spec.take, "make": spec.make}

    # ---- commander -------------------------------------------------------
    from simplefab.sim import ProductionLine
    pt, su = common["processing_times"], common["setup_times"]
    ref = ProductionLine({0: pt[0][0], 1: pt[0][1]}, {0: pt[1][0], 1: pt[1][1]}, su[1],
                         {0: pt[2][0], 1: pt[2][1]}, {0: pt[3][0], 1: pt[3][1]}, su[3],
                         common_cfg=common)
    for t in range(H):
        ref.run_step(current_time=t, actions_override=None)
    cmd_sel = np.zeros((1, H, M), dtype=np.int32)
    for m, mach in enumerate([ref.machine0, ref.machine1, ref.machine2, ref.machine3]):
        for e in mach.event_log:
            st = int(e[1]) if m in (0, 2) else int(e[2])
            if 0 <= st < H:
                cmd_sel[0, st, m] = int(e[0]) + 1
    store["commander"] = trace_of(line, spec, cmd_sel)
    print(f"commander  ${store['commander']['profit']:,.2f}")

    # ---- grad-ST ---------------------------------------------------------
    dl = DiffLine2P(spec, Relax2P(mode="st"))
    theta0 = jax.random.normal(jax.random.PRNGKey(0), (args.batch, H, M, P + 1))
    t0 = time.perf_counter()
    _, g_sel = adam_ascend(dl, theta0, args.steps, 0.1, ev, 10, 1000.0, _Q(), P)
    g_wall = time.perf_counter() - t0
    best = int(np.argmax(np.asarray(ev(g_sel))))
    store["gradst"] = trace_of(line, spec, np.asarray(g_sel)[best:best + 1])
    store["gradst"]["wall"] = g_wall
    print(f"grad-ST    ${store['gradst']['profit']:,.2f}   ({g_wall:.1f}s)")

    # ---- CP-SAT ----------------------------------------------------------
    from tensorfab.twoprod.optim import solve_2p
    t0 = time.perf_counter()
    sol = solve_2p(common, time_limit_sec=args.milp_seconds, num_workers=8, verbose=False)
    c_wall = time.perf_counter() - t0
    starts = sol["starts"]                                   # [H, M, P]
    cp_sel = np.zeros((1, H, M), dtype=np.int32)
    for p in range(P):
        cp_sel[0][starts[:, :, p] == 1] = p + 1
    store["cpsat"] = trace_of(line, spec, cp_sel)
    store["cpsat"]["wall"] = c_wall
    store["cpsat_bound"] = sol["objective_bound"]
    print(f"CP-SAT     ${store['cpsat']['profit']:,.2f}   ({c_wall:.0f}s, "
          f"bound ${sol['objective_bound']:,.2f})")

    flat = {}
    for k, v in store.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}__{kk}"] = np.asarray(vv)
        else:
            flat[k] = np.asarray(v)
    name = sched_name(args.alpha, args.util)
    np.savez_compressed(OUT / name, **flat)
    print(f"\nwrote {OUT}/{name}")


if __name__ == "__main__":
    main()
