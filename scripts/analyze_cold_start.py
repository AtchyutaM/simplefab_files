import sys
import os

# Ensure simplefab is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplefab.config import make_common_config

def calculate_min_lead_times():
    cfg = make_common_config(mode="UNIFORM", H=1000, alpha=0.5, utilization=0.92)
    
    # Calculate demand arrival ticks
    dem_p0_ticks = [t for t, count in enumerate(cfg['demand_schedule'][0]) if count > 0]
    dem_p1_ticks = [t for t, count in enumerate(cfg['demand_schedule'][1]) if count > 0]
    
    print(f"--- Cold Start Analysis ---")
    print(f"First P0 demand arrives at tick: {dem_p0_ticks[0]}")
    print(f"First P1 demand arrives at tick: {dem_p1_ticks[0]}")
    print(f"10th P0 demand arrives at tick: {dem_p0_ticks[9]}")
    
    # Math limits
    print("\n--- Theoretical Minimum Lead Times ---")
    # P0 minimum time
    m0_p0 = 16
    m1_p0_batch = 2 * 4 # 4 units taking 2 ticks each to form a batch for M2
    m2_p0 = 16
    m3_p0_first = 2
    min_p0 = m0_p0 + m1_p0_batch + m2_p0 + m3_p0_first
    
    # P1 minimum time
    m0_p1 = 20
    m1_p1_batch = 2 * 4 
    m2_p1 = 20
    m3_p1_first = 2
    min_p1 = m0_p1 + m1_p1_batch + m2_p1 + m3_p1_first
    
    print(f"Absolute minimum ticks to produce 1st P0 unit: {min_p0}")
    print(f"Absolute minimum ticks to produce 1st P1 unit: {min_p1}")
    
    # Conclusion
    print("\n--- Conclusion ---")
    if dem_p0_ticks[0] < min_p0:
        print(f"UNAVOIDABLE P0 BACKORDERS: Demand arrives at {dem_p0_ticks[0]}, but earliest production is {min_p0}.")
    else:
        print("P0 backorders are theoretically avoidable.")
        
    if dem_p1_ticks[0] < min_p1:
        print(f"UNAVOIDABLE P1 BACKORDERS: Demand arrives at {dem_p1_ticks[0]}, but earliest production is {min_p1}.")
    else:
        print("P1 backorders are theoretically avoidable.")

if __name__ == "__main__":
    calculate_min_lead_times()
