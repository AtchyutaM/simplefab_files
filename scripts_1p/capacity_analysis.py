"""Capacity analysis for the single-product SimpleFab system."""
from simplefab_1p import make_common_config

cfg = make_common_config()
H = cfg["time_horizon"]
u = cfg["utilization"]

print("=== TIME HORIZON ===")
print(f"H = {H} ticks")
print(f"  = {H//96} days  (96 ticks/day)")
print(f"  = {H//672} weeks (672 ticks/week)")

print("\n=== MACHINE PARAMETERS ===")
pt = cfg["processing_times"]
bs = cfg["batch_sizes"]
for m in range(4):
    mtype = "Batch" if bs[m] > 1 else "Single"
    proc = pt[m][0]
    effective = proc / bs[m]
    cap = H / effective
    print(f"  M{m} ({mtype:6s}): proc_time={proc:2d} ticks, batch_size={bs[m]}, effective_time/unit={effective:.1f}, max_capacity={cap:.0f} units")

print("\n=== BOTTLENECK ANALYSIS ===")
for m in range(4):
    effective = pt[m][0] / bs[m]
    cap = H / effective
    print(f"  M{m}: {cap:.0f} units over horizon (rate = 1 unit every {effective:.1f} ticks)")

bottleneck_rate = pt[0][0] / bs[0]  # 16/4 = 4 ticks/unit
bottleneck_cap = H / bottleneck_rate
print(f"\n  Bottleneck machines: M0 and M2 (batch, {bottleneck_rate:.0f} ticks/unit)")
print(f"  Bottleneck capacity: {bottleneck_cap:.0f} units over {H//672} weeks")
print(f"  Single machines M1,M3: {H // (pt[1][0]//bs[1]):.0f} units (much faster, NOT bottleneck)")

print("\n=== ARRIVALS (raw material releases) ===")
arr = cfg["arrivals_schedule"][0]
total_arr = cfg["arrivals"][0]
nonzero = [(t, arr[t]) for t in range(H) if arr[t] > 0]
print(f"  Total arrivals: {total_arr} units")
print(f"  Computed as: floor(H / time_per_unit) rounded to batch multiple")
print(f"    = floor({H} / {bottleneck_rate:.0f}) // 4 * 4 = {total_arr}")
print(f"  Number of release events: {len(nonzero)}")
print(f"  Release schedule (tick: qty):")
for t, qty in nonzero:
    day = t / 96
    week = t / 672
    print(f"    t={t:5d} (day {day:5.1f}, week {week:.1f}): {qty} units")

print(f"\n=== DEMAND ===")
dem = cfg["demand_schedule"][0]
total_dem = cfg["demand"][0]
nonzero_d = [(t, dem[t]) for t in range(H) if dem[t] > 0]
print(f"  Utilization target: {u*100:.0f}%")
print(f"  Total demand: {total_dem} units")
print(f"  Computed as: floor(u * H / time_per_unit) rounded to batch multiple")
print(f"    = floor({u} * {H} / {bottleneck_rate:.0f}) // 4 * 4 = {total_dem}")
print(f"  Demand delay: {cfg['demand_delay']} ticks ({cfg['demand_delay']//96} day)")
print(f"  Demand interval: {cfg['demand_interval']} ticks ({cfg['demand_interval']//672} week)")
print(f"  Number of demand events: {len(nonzero_d)}")
print(f"  Demand schedule (tick: qty):")
for t, qty in nonzero_d:
    day = t / 96
    week = t / 672
    print(f"    t={t:5d} (day {day:5.1f}, week {week:.1f}): {qty} units")

print(f"\n=== CAPACITY vs DEMAND GAP ===")
print(f"  Arrivals (100% capacity): {total_arr} units")
print(f"  Demand   ({u*100:.0f}% capacity):  {total_dem} units")
print(f"  Surplus raw material:     {total_arr - total_dem} units (will become excess WIP/inventory)")
print(f"  Demand as % of arrivals:  {total_dem/total_arr*100:.1f}%")

print(f"\n=== ECONOMICS ===")
rev = cfg["revenue_per_unit"][0]
pc = cfg["production_cost"]
costs_per_unit = [pc[m][0] for m in range(4)]
total_prod_cost = sum(c * bs[m] for m, c in enumerate(costs_per_unit)) // 1  # per unit through chain
# Actually: M0 costs $8 per unit in batch (8*4=32 per batch), M1 costs $4 per unit, etc.
# Per unit through whole chain: $8 + $4 + $8 + $4 = $24
per_unit_cost = sum(costs_per_unit)
inv_cost = cfg["inventory_cost_per_unit"][0]
bo_cost = cfg["backorder_cost_per_unit"][0]
print(f"  Revenue per unit shipped:  ${rev}")
print(f"  Production cost per unit:  ${per_unit_cost} (M0=${costs_per_unit[0]} + M1=${costs_per_unit[1]} + M2=${costs_per_unit[2]} + M3=${costs_per_unit[3]})")
print(f"  Net margin per unit:       ${rev - per_unit_cost}")
print(f"  Inventory holding cost:    ${inv_cost}/unit/tick  (=${inv_cost*96:.2f}/unit/day)")
print(f"  Backorder penalty:         ${bo_cost}/unit/tick  (=${bo_cost*96:.2f}/unit/day)")
print(f"  ")
print(f"  Max revenue (all demand met):    ${total_dem} * ${rev} = ${total_dem * rev:,.0f}")
print(f"  Min production cost (all made):  ${total_dem} * ${per_unit_cost} = ${total_dem * per_unit_cost:,.0f}")
print(f"  Theoretical max profit:          ${total_dem * (rev - per_unit_cost):,.0f} (zero inv/BO)")

print(f"\n=== INITIAL FINISHED INVENTORY ===")
print(f"  Starting stock: {cfg['initial_finished_inventory'][0]} units")
print(f"  Purpose: buffer against cold-start backorders before factory ramps up")

print(f"\n=== TIMING ANALYSIS ===")
print(f"  Pipeline latency (time for 1 unit to traverse all 4 machines):")
lat = pt[0][0] + pt[1][0] + pt[2][0] + pt[3][0]
print(f"    = {pt[0][0]} + {pt[1][0]} + {pt[2][0]} + {pt[3][0]} = {lat} ticks ({lat/96:.1f} days)")
print(f"  First demand arrives at: t={cfg['demand_delay']} (day {cfg['demand_delay']/96:.0f})")
print(f"  First arrivals at: t=0")
print(f"  Time to process first batch through M0: {pt[0][0]} ticks")
print(f"  First unit exits factory (M3) at earliest: t={lat}")
print(f"  Gap: demand at t={cfg['demand_delay']}, first output at t~{lat} => {'OK, output before demand' if lat <= cfg['demand_delay'] else 'PROBLEM: demand arrives before first output'}")
