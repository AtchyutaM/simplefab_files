from __future__ import annotations

from simplefab import make_common_config
from simplefab.sim import ProductionLine


def build_line(H: int = 30):
    common = make_common_config(mode="UNIFORM", H=H, alpha=0.5, utilization=0.92)

    pt = common["processing_times"]
    setups = {None: {0: 1, 1: 1}, 0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}}

    line = ProductionLine(
        {0: pt[0][0], 1: pt[0][1]},
        {0: pt[1][0], 1: pt[1][1]},
        setups,
        {0: pt[2][0], 1: pt[2][1]},
        {0: pt[3][0], 1: pt[3][1]},
        setups,
        common_cfg=common,
    )
    line.logging_enabled = False
    return common, line


def _started_product_batch(machine, t: int):
    # batch event tuple: (product, start_time, finish_time, batch_size)
    for e in reversed(machine.event_log):
        if e[1] == t:
            return int(e[0])
        if e[1] < t:
            break
    return None


def _started_product_single(machine, t: int):
    # single event tuple: (product, decision_time, start_time, finish_time)
    for e in reversed(machine.event_log):
        if e[1] == t:
            return int(e[0])
        if e[1] < t:
            break
    return None


def started_actions_at_t(line: ProductionLine, t: int):
    a = {}
    p = _started_product_batch(line.machine0, t)
    a["M0"] = "None" if p is None else f"Prod{p}"

    p = _started_product_single(line.machine1, t)
    a["M1"] = "None" if p is None else f"Prod{p}"

    p = _started_product_batch(line.machine2, t)
    a["M2"] = "None" if p is None else f"Prod{p}"

    p = _started_product_single(line.machine3, t)
    a["M3"] = "None" if p is None else f"Prod{p}"
    return a


def main():
    H = 30
    common, line = build_line(H=H)

    print(f"=== Reward breakdown for H={H} ===")
    print(
        "t |  M0   M1   M2   M3  |  dRev   dProd  dSetup   dInv    dBO  | dProfit | step_reward | "
        "dem(P0,P1) fin(P0,P1)"
    )
    print("-" * 120)

    for t in range(H):
        prev_costs = dict(line.costs)
        prev_profit = line.profit_total()

        # run one step using Commander (heuristic)
        step_reward = line.run_step(current_time=t, actions_override=None)

        after_costs = dict(line.costs)
        after_profit = line.profit_total()

        d_rev  = after_costs["revenue"]     - prev_costs["revenue"]
        d_prod = after_costs["production"]  - prev_costs["production"]
        d_set  = after_costs["setup"]       - prev_costs["setup"]
        d_inv  = after_costs["inventory"]   - prev_costs["inventory"]
        d_bo   = after_costs["backorder"]   - prev_costs["backorder"]

        d_profit = after_profit - prev_profit

        acts = started_actions_at_t(line, t)

        dem0 = len(line.queues["demand"][0])
        dem1 = len(line.queues["demand"][1])
        fin0 = len(line.queues["queue_fin"][0])
        fin1 = len(line.queues["queue_fin"][1])

        # Note: in this sim configuration, step_reward should equal d_profit (invalid penalty default is 0.0)
        print(
            f"{t:2d} | {acts['M0']:>4} {acts['M1']:>4} {acts['M2']:>4} {acts['M3']:>4} | "
            f"{d_rev:6.1f} {d_prod:7.1f} {d_set:7.1f} {d_inv:7.1f} {d_bo:7.1f} | "
            f"{d_profit:7.1f} | {step_reward:11.1f} | "
            f"({dem0:2d},{dem1:2d})   ({fin0:2d},{fin1:2d})"
        )

    print("\nDone.")
    print("Tip: If step_reward != d_profit, you're applying a nonzero invalid penalty somewhere (e.g., in the Gym env).")


if __name__ == "__main__":
    main()
