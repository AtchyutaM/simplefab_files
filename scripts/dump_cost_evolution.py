import sys
import os
import pandas as pd

# Ensure simplefab is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplefab.config import make_common_config
from simplefab.env import FabEnv
from simplefab.sim import Commander

def generate_cost_evolution():
    # 1. Setup config
    cfg = make_common_config(mode="UNIFORM", H=1000, alpha=0.5, utilization=0.92)
    
    # 2. Run simulation using the Environment (which correctly initializes the sim)
    env = FabEnv(common_cfg=cfg)
    obs, info = env.reset()
    
    cost_snapshots = []
    
    # We will just let the built-in Commander heuristic run it by passing invalid actions
    # to the env, or we can instantiate the commander directly
    commander = env.line.commander
    
    for t in range(cfg["time_horizon"]):
        # Get heuristic actions
        state_dict = env.line._state_dict(env.line._t)
        dict_actions = commander.decide_actions(state_dict)
        
        # We step the internal line directly to use dictate_actions
        reward = env.line.run_step(current_time=t, actions_override=dict_actions)
        
        # Capture the snapshot of costs exactly at the end of this tick
        c = env.line.costs
        cost_snapshots.append({
            "Tick": t,
            "Profit": env.line.profit_total(),
            "Revenue": c["revenue"],
            "Production": c["production"],
            "Setup": c["setup"],
            "Inventory": c["inventory"],
            "Backorder": c["backorder"],
            "Unmet_P0": len(env.line.queues["demand"][0]),
            "Unmet_P1": len(env.line.queues["demand"][1]),
            "Fin_P0": len(env.line.queues["queue_fin"][0]),
            "Fin_P1": len(env.line.queues["queue_fin"][1]),
        })
        
    df = pd.DataFrame(cost_snapshots)
    
    # Output important milestones
    print("--- Time Evolution of Costs (Cold Start) ---")
    print(df.loc[[0, 10, 20, 30, 42, 50, 60, 100, 200, 500, 999]].to_string(index=False))
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cost_evolution_cold_start.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved full tick-by-tick breakdown to '{out_path}'")

if __name__ == "__main__":
    generate_cost_evolution()
