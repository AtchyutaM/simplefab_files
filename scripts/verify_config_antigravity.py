import sys
import os

# Add the parent directory to sys.path so we can import simplefab
sys.path.append(os.path.abspath("c:/Users/bhara/RL/SimpleFab_Arch/simplefab_files"))

from simplefab.config import make_common_config
from simplefab.sim import ProductionLine

def verify_config():
    print("--- Verifying Config ---")
    cfg = make_common_config(mode="UNIFORM", H=10)
    
    # 1. Check Stochasticity
    print(f"\n[Stochastic Check] Mode: UNIFORM")
    print(f"Arrivals P0: {cfg['arrivals_schedule'][0][:10]}")
    print(f"Arrivals P1: {cfg['arrivals_schedule'][1][:10]}")
    # Run again to see if it changes (it shouldn't based on code reading, but good to prove)
    cfg2 = make_common_config(mode="UNIFORM", H=10)
    print(f"Arrivals P0 (Run 2): {cfg2['arrivals_schedule'][0][:10]}")
    if cfg['arrivals_schedule'] == cfg2['arrivals_schedule']:
        print("-> Deterministic Schedule (UNIFORM)")
    else:
        print("-> Stochastic Schedule")

    # 2. Check Setup Times & Keys
    print(f"\n[Setup Times Check]")
    setup_m1 = cfg['setup_times'][1]
    print(f"Machine 1 Setup Dict Keys: {list(setup_m1.keys())}")
    
    # Check if None key exists
    if None in setup_m1:
        print("-> 'None' key exists in setup_times.")
    else:
        print("-> 'None' key MISSING in setup_times.")
        # Check if sim.py handles this or if it crashes
        print("   Warning: sim.py accesses setup_times[None] for first item.")

    # 3. Print all Parameters for Infographic
    print(f"\n[Parameters for Infographic]")
    print(f"Batch Sizes: {cfg['batch_sizes']}")
    print(f"Processing Times: {cfg['processing_times']}")
    print(f"Setup Costs: {cfg['setup_cost']}")
    print(f"Prod Costs: {cfg['production_cost']}")
    print(f"Revenue: {cfg['revenue_per_unit']}")
    print(f"Inventory Cost: {cfg['inventory_cost_per_unit']}")
    print(f"Backorder Cost: {cfg['backorder_cost_per_unit']}")

if __name__ == "__main__":
    verify_config()
