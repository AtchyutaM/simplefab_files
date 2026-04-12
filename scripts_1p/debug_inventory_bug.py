"""Demonstrate the finished-inventory vs demand coexistence bug."""
from simplefab_1p import make_common_config
from simplefab_1p.sim import ProductionLine

cfg = make_common_config()  # demand_delay=671
line = ProductionLine(common_cfg=cfg)

key_ticks = [670, 671, 672, 673, 674, 680, 700, 750, 1000, 1343, 1344, 2687]

for t in range(cfg["time_horizon"]):
    line.run_step(current_time=t)
    if t in key_ticks:
        fin = len(line.queues["queue_fin"])
        bo = len(line.queues["demand"])
        shipped = len(line.demand_met_log)
        dem_this = cfg["demand_schedule"][0][t]
        arr_this = cfg["arrivals_schedule"][0][t]
        events = ""
        if dem_this > 0:
            events += f" [+{dem_this} DEMAND]"
        if arr_this > 0:
            events += f" [+{arr_this} ARRIVALS]"
        print(f"t={t:5d} (day {t/96:5.1f}) | Fin.Inv={fin:4d} | Backlog={bo:4d} | Shipped={shipped:4d}{events}")
        if fin > 0 and bo > 0:
            print(f"         ^^^ BUG: {fin} units on shelf + {bo} customers waiting!")

print()
print("=== EXPLANATION ===")
print("The simulation only ships when M3 completes a NEW unit (_ship_or_store).")
print("When demand arrives, it does NOT check existing finished inventory.")
print("So 156 units built before demand hit sit in queue_fin permanently.")
print("They never get matched to the 154-unit demand that arrives at t=671.")
