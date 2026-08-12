"""
LineSpec — the static description of the production line, derived from the
existing `simplefab_1p.config.make_common_config` so there is exactly one
source of truth for parameters.

Everything here is plain numpy / Python. No JAX. The spec is closed over at
trace time by `tensor_sim.make_line`, so XLA bakes these values in as
constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

N_MACHINES = 4


@dataclass(frozen=True)
class LineSpec:
    # ---- static structure (plain ints; safe as jit static arguments) ----
    H: int                      # horizon in ticks
    n_machines: int
    reg_len: int                # delay-register length = max(pt) + 1
    demand_interval: int        # ticks after which unmet demand becomes "late"

    # ---- per-machine, shape [4] ----
    pt: np.ndarray              # processing time in ticks
    take: np.ndarray            # units consumed from the input queue per start
    make: np.ndarray            # units emitted into the next queue per completion

    # ---- per-tick, shape [H] ----
    arrivals: np.ndarray        # raw material into q0 at tick t
    demand: np.ndarray          # demand units arriving at tick t
    demand_lag_cum: np.ndarray  # D(t - demand_interval): cumulative demand that
                                # has been outstanding long enough to be "late"

    # ---- economics ----
    revenue_per_unit: float
    prod_cost_per_unit: np.ndarray   # [4], $ per unit processed at that stage
    fg_holding: float                # $/unit/tick on finished goods
    wip_holding: np.ndarray          # [4], $/unit/tick on q0..q3
    backorder: float                 # $/unit/tick on late demand

    # ---- initial condition ----
    init_qfin: int

    # ---- provenance ----
    mode: str
    utilization: float

    @property
    def total_demand(self) -> int:
        return int(self.demand.sum())

    def summary(self) -> str:
        return (
            f"LineSpec(mode={self.mode}, H={self.H}, util={self.utilization:.3f})\n"
            f"  pt={self.pt.tolist()}  take={self.take.tolist()}  make={self.make.tolist()}\n"
            f"  reg_len={self.reg_len}  demand_interval={self.demand_interval}\n"
            f"  arrivals total={int(self.arrivals.sum())}  demand total={self.total_demand}\n"
            f"  revenue=${self.revenue_per_unit}/unit  "
            f"prod=${self.prod_cost_per_unit.tolist()}/unit/stage\n"
            f"  fg_holding=${self.fg_holding}  wip={self.wip_holding.tolist()}  "
            f"backorder=${self.backorder}\n"
            f"  init_qfin={self.init_qfin}"
        )


def spec_from_common_config(common: Dict[str, Any], mode: str = "?") -> LineSpec:
    """Translate a `make_common_config` dict into a LineSpec."""
    H = int(common["time_horizon"])
    pt = np.array([int(common["processing_times"][m][0]) for m in range(N_MACHINES)], dtype=np.int32)
    batch = np.array([int(common["batch_sizes"][m]) for m in range(N_MACHINES)], dtype=np.int32)

    # For this serial line, units consumed == units emitted at every stage:
    # the batch machines take 4 and emit 4, the single machines take 1 and emit 1.
    take = batch.copy()
    make = batch.copy()

    arrivals = np.asarray(common["arrivals_schedule"][0], dtype=np.int32)
    demand = np.asarray(common["demand_schedule"][0], dtype=np.int32)
    if arrivals.shape[0] != H or demand.shape[0] != H:
        raise ValueError("arrivals/demand schedules must have length H")

    demand_interval = int(common.get("demand_interval", 672))

    # D(tau) = cumulative demand arrived at ticks <= tau.
    # A demand unit that arrived at tick a is "late" at tick t iff t >= a + interval,
    # so the number of *arrived-long-enough* units at tick t is D(t - interval).
    # Because the demand queue is strictly FIFO, the units still outstanding are
    # the most recently arrived ones, hence
    #     late(t) = max(0, D(t - interval) - cumulative_shipped(t)).
    # This is the same accounting the CP-SAT model in optim.py uses.
    cum_demand = np.cumsum(demand)
    demand_lag_cum = np.zeros(H, dtype=np.int32)
    if H > demand_interval:
        demand_lag_cum[demand_interval:] = cum_demand[: H - demand_interval]

    wip_cfg = common.get("wip_cost_per_unit", {})
    wip = np.array([float(wip_cfg.get(i, 0.0)) for i in range(N_MACHINES)], dtype=np.float64)

    prod = np.array(
        [float(common["production_cost"][m][0]) for m in range(N_MACHINES)], dtype=np.float64
    )

    return LineSpec(
        H=H,
        n_machines=N_MACHINES,
        reg_len=int(pt.max()) + 1,
        demand_interval=demand_interval,
        pt=pt,
        take=take,
        make=make,
        arrivals=arrivals,
        demand=demand,
        demand_lag_cum=demand_lag_cum,
        revenue_per_unit=float(common["revenue_per_unit"][0]),
        prod_cost_per_unit=prod,
        fg_holding=float(common["inventory_cost_per_unit"][0]),
        wip_holding=wip,
        backorder=float(common["backorder_cost_per_unit"][0]),
        init_qfin=int(common.get("initial_finished_inventory", {0: 0}).get(0, 0)),
        mode=mode,
        utilization=float(common.get("utilization", 0.0)),
    )


def spec_for_mode(mode: str = "ALTERNATING_92", **kwargs: Any) -> LineSpec:
    """Convenience: build a LineSpec straight from a demand mode name."""
    from simplefab_1p.config import make_common_config

    common = make_common_config(mode=mode, **kwargs)
    return spec_from_common_config(common, mode=mode)


if __name__ == "__main__":
    print(spec_for_mode("ALTERNATING_92").summary())
