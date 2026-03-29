"""Verify the new config produces correct schedules and run a quick simulation."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplefab.config import make_common_config
from simplefab.env import FabEnv

def verify():
    cfg = make_common_config()

    # Check totals
    arr_p0 = sum(cfg["arrivals_schedule"][0])
    arr_p1 = sum(cfg["arrivals_schedule"][1])
    dem_p0 = sum(cfg["demand_schedule"][0])
    dem_p1 = sum(cfg["demand_schedule"][1])

    print("=" * 60)
    print("CONFIG VERIFICATION")
    print("=" * 60)
    print(f"H = {cfg['time_horizon']}")
    print(f"Arrivals total: P0={arr_p0}, P1={arr_p1} (expected 296, 296)")
    print(f"Demand total:   P0={dem_p0}, P1={dem_p1} (expected 272, 272)")
    print(f"Initial fin inv: {cfg['initial_finished_inventory']}")
    print()

    # Check weekly pattern
    H = cfg["time_horizon"]
    WEEK = 672
    DAY = 96
    print("Arrivals schedule (non-zero ticks):")
    for t in range(H):
        a0 = cfg["arrivals_schedule"][0][t]
        a1 = cfg["arrivals_schedule"][1][t]
        if a0 > 0 or a1 > 0:
            print(f"  t={t:5d} (day {t/DAY:5.1f}, week {t//WEEK+1}): P0={a0}, P1={a1}")

    print("\nDemand schedule (non-zero ticks):")
    for t in range(H):
        d0 = cfg["demand_schedule"][0][t]
        d1 = cfg["demand_schedule"][1][t]
        if d0 > 0 or d1 > 0:
            print(f"  t={t:5d} (day {t/DAY:5.1f}, week {t//WEEK+1}): P0={d0}, P1={d1}")

    # Run simulation
    print("\n" + "=" * 60)
    print("SIMULATION RUN")
    print("=" * 60)
    env = FabEnv(common_cfg=cfg)
    obs, info = env.reset()

    commander = env.line.commander
    for t in range(cfg["time_horizon"]):
        state_dict = env.line._state_dict(t)
        actions = commander.decide_actions(state_dict)
        env.line.run_step(current_time=t, actions_override=actions)

    c = env.line.costs
    print(f"Revenue:        {c['revenue']:.2f}")
    print(f"Production:     {c['production']:.2f}")
    print(f"Setup:          {c['setup']:.2f}")
    print(f"Inventory:      {c['inventory']:.2f}")
    print(f"Backorder:      {c['backorder']:.2f}")
    print(f"Profit:         {env.line.profit_total():.2f}")
    print(f"Units shipped:  {len(env.line.demand_met_log)}")
    print(f"Unmet P0:       {len(env.line.queues['demand'][0])}")
    print(f"Unmet P1:       {len(env.line.queues['demand'][1])}")

    # Check early ticks for backorders
    early_bo = False
    for entry in env.line.cost_log[:96]:
        if entry["backorder"] > 0:
            early_bo = True
            break
    print(f"\nBackorders in first 96 ticks (1 day): {'YES' if early_bo else 'NO (good!)'}")

if __name__ == "__main__":
    verify()
