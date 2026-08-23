"""
LineSpec2P — static description of the TWO-product line.

Derived from `simplefab.config.make_common_config` so the parameters have one
source of truth, exactly as `spec.py` does for the single-product line.

What is genuinely new relative to the 1-product spec:

  * every queue, cost and schedule carries a product axis
  * processing time is per (machine, product) — P1 runs 20 ticks on the batch
    tools against P0's 16
  * `setup_time[m, prev, p]` is sequence dependent, indexed by the machine's
    LAST COMPLETED product. Index 0 of the middle axis is the cold-start case
    ("never run"), which the reference simulator treats as a changeover.
  * `setup_cost[m]` is charged whenever the started product differs from the
    last completed one — including on cold start.

Register length must cover setup + processing, not just processing: a job
started at tick t finishes at t + setup + pt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

N_PRODUCTS = 2


@dataclass(frozen=True)
class LineSpec2P:
    # ---- structure ----
    H: int
    n_machines: int
    n_products: int
    reg_len: int                 # max(setup + pt) + 1
    demand_interval: int

    # ---- per machine / product ----
    pt: np.ndarray               # [M, P] processing ticks
    take: np.ndarray             # [M]    units consumed per start
    make: np.ndarray             # [M]    units emitted per completion
    route: np.ndarray            # [M]    destination queue, -1 = ship
    kind: Tuple[str, ...]        # "DIFF"/"LITHO", reporting only

    # ---- changeover ----
    setup_time: np.ndarray       # [M, P+1, P]; middle index 0 = cold start
    setup_cost: np.ndarray       # [M]

    # ---- schedules, per tick and product ----
    arrivals: np.ndarray         # [H, P]
    demand: np.ndarray           # [H, P]
    demand_lag_cum: np.ndarray   # [H, P]  D(t - demand_interval) per product

    # ---- economics ----
    revenue: np.ndarray          # [P]
    prod_cost: np.ndarray        # [M, P] $ per unit at that stage
    fg_holding: np.ndarray       # [P]
    wip_holding: np.ndarray      # [M, P]
    backorder: np.ndarray        # [P]

    init_qfin: np.ndarray        # [P]

    mode: str
    alpha: float
    utilization: float

    @property
    def total_demand(self) -> int:
        return int(self.demand.sum())

    def summary(self) -> str:
        return (
            f"LineSpec2P(mode={self.mode}, alpha={self.alpha}, "
            f"util={self.utilization:.2f}, H={self.H})\n"
            f"  M={self.n_machines} P={self.n_products} kinds={'/'.join(self.kind)}\n"
            f"  pt=\n{self.pt}\n"
            f"  take={self.take.tolist()} make={self.make.tolist()} "
            f"route={self.route.tolist()}  reg_len={self.reg_len}\n"
            f"  setup_cost={self.setup_cost.tolist()}\n"
            f"  setup_time[M1]=\n{self.setup_time[1]}\n"
            f"  demand per product={self.demand.sum(axis=0).tolist()}  "
            f"arrivals={self.arrivals.sum(axis=0).tolist()}\n"
            f"  revenue={self.revenue.tolist()}  fg={self.fg_holding.tolist()}  "
            f"backorder={self.backorder.tolist()}\n"
            f"  init_qfin={self.init_qfin.tolist()}"
        )


def spec2p_from_common_config(common: Dict[str, Any], mode: str = "?") -> LineSpec2P:
    H = int(common["time_horizon"])
    machines = list(common["machines"])
    products = list(common["products"])
    M, P = len(machines), len(products)

    batch = np.array([int(common["batch_sizes"][m]) for m in range(M)], dtype=np.int32)
    pt = np.array([[int(common["processing_times"][m][p]) for p in range(P)]
                   for m in range(M)], dtype=np.int32)

    # Serial chain, same as the 1-product line: machine m feeds queue m+1.
    route = np.arange(1, M + 1, dtype=np.int32)
    route[M - 1] = -1
    kind = tuple("DIFF" if int(batch[m]) > 1 else "LITHO" for m in range(M))

    # setup_time[m, prev, p]; prev index 0 == cold start (the config's None key).
    st_cfg = common["setup_times"]
    setup_time = np.zeros((M, P + 1, P), dtype=np.int32)
    for m in range(M):
        for p in range(P):
            setup_time[m, 0, p] = int(st_cfg[m][None][p])
            for prev in range(P):
                setup_time[m, prev + 1, p] = int(st_cfg[m][prev][p])
    setup_cost = np.array([float(common["setup_cost"][m]) for m in range(M)], dtype=np.float64)

    arrivals = np.stack([np.asarray(common["arrivals_schedule"][p], dtype=np.int32)
                         for p in range(P)], axis=-1)                      # [H, P]
    demand = np.stack([np.asarray(common["demand_schedule"][p], dtype=np.int32)
                       for p in range(P)], axis=-1)                        # [H, P]
    if arrivals.shape[0] != H or demand.shape[0] != H:
        raise ValueError("arrivals/demand schedules must have length H")

    demand_interval = int(common.get("demand_interval", 672))
    # Same FIFO-lateness accounting as the 1-product line, per product.
    cum_demand = np.cumsum(demand, axis=0)
    demand_lag_cum = np.zeros((H, P), dtype=np.int32)
    if H > demand_interval:
        demand_lag_cum[demand_interval:] = cum_demand[: H - demand_interval]

    revenue = np.array([float(common["revenue_per_unit"][p]) for p in range(P)])
    prod_cost = np.array([[float(common["production_cost"][m][p]) for p in range(P)]
                          for m in range(M)])
    fg = np.array([float(common["inventory_cost_per_unit"][p]) for p in range(P)])
    wip_cfg = common.get("wip_cost_per_unit", {})
    wip = np.array([[float(wip_cfg.get(m, {}).get(p, 0.0)) for p in range(P)]
                    for m in range(M)])
    bo = np.array([float(common["backorder_cost_per_unit"][p]) for p in range(P)])
    init_fin = common.get("initial_finished_inventory", {})
    init_qfin = np.array([int(init_fin.get(p, 0)) for p in range(P)], dtype=np.int32)

    # A job occupies its tool for setup + processing ticks.
    reg_len = int((setup_time.max(axis=1) + pt).max()) + 1

    return LineSpec2P(
        H=H, n_machines=M, n_products=P, reg_len=reg_len,
        demand_interval=demand_interval,
        pt=pt, take=batch.copy(), make=batch.copy(), route=route, kind=kind,
        setup_time=setup_time, setup_cost=setup_cost,
        arrivals=arrivals, demand=demand, demand_lag_cum=demand_lag_cum,
        revenue=revenue, prod_cost=prod_cost, fg_holding=fg, wip_holding=wip,
        backorder=bo, init_qfin=init_qfin,
        mode=mode, alpha=float(common.get("mix_alpha", 0.5)),
        utilization=float(common.get("utilization", 0.0)),
    )


def spec2p_for(mode: str = "UNIFORM", **kwargs: Any) -> LineSpec2P:
    from simplefab.config import make_common_config

    common = make_common_config(mode=mode, **kwargs)
    return spec2p_from_common_config(common, mode=mode)


if __name__ == "__main__":
    print(spec2p_for("UNIFORM", alpha=0.5, utilization=0.92).summary())
