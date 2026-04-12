"""Compare demand schedules with different delay values."""
from simplefab_1p import make_common_config

# Current setup: demand_delay=96 (day 1)
cfg_now = make_common_config(demand_delay=96)
dem_now = cfg_now["demand_schedule"][0]
events_now = [(t, dem_now[t]) for t in range(cfg_now["time_horizon"]) if dem_now[t] > 0]
print("=== CURRENT: demand_delay=96 (day 1) ===")
print(f"Total demand configured: {cfg_now['demand'][0]} units")
print(f"Total demand scheduled:  {sum(dem_now)} units")
print(f"Events ({len(events_now)}):")
for t, qty in events_now:
    print(f"  t={t:5d}  (day {t/96:5.1f}, week {t/672:.1f}): {qty} units")

print()

# Proposed: demand_delay=672 (end of week 1)
cfg_new = make_common_config(demand_delay=672)
dem_new = cfg_new["demand_schedule"][0]
events_new = [(t, dem_new[t]) for t in range(cfg_new["time_horizon"]) if dem_new[t] > 0]
total_scheduled = sum(dem_new)
print("=== PROPOSED: demand_delay=672 (end of week 1) ===")
print(f"Total demand configured: {cfg_new['demand'][0]} units")
print(f"Total demand scheduled:  {total_scheduled} units")
print(f"Events ({len(events_new)}):")
for t, qty in events_new:
    print(f"  t={t:5d}  (day {t/96:5.1f}, week {t/672:.1f}): {qty} units")

print()
H = cfg_new["time_horizon"]
print("=== ISSUE ===")
print(f"With delay=672 and interval=672, event placement: i*672 + 672")
print(f"  i=0: t = {0*672+672}")
print(f"  i=1: t = {1*672+672}")
print(f"  i=2: t = {2*672+672}")
print(f"  i=3: t = {3*672+672}  <-- equals H={H}, so t < H is False => DROPPED!")
print(f"Result: only {total_scheduled} of {cfg_new['demand'][0]} units placed ({cfg_new['demand'][0] - total_scheduled} lost)")
