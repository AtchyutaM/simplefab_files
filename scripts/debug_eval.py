"""Compare stochastic vs deterministic evaluation"""
from sb3_contrib import MaskablePPO
from simplefab.config import make_common_config
from simplefab.env import FabEnv
import numpy as np

model = MaskablePPO.load('output_data/models/ppo_fab_checkpoint_3500000_steps.zip')
common_cfg = make_common_config()

for det, label in [(True, "DETERMINISTIC"), (False, "STOCHASTIC (5 runs)")]:
    profits = []
    n_runs = 1 if det else 5
    for run in range(n_runs):
        env = FabEnv(common_cfg=common_cfg, normalize_obs=True, invalid_action_penalty=0.0)
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=det, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        c = env.line.costs
        tp0 = sum(1 for p, _ in env.line.demand_met_log if p == 0)
        tp1 = sum(1 for p, _ in env.line.demand_met_log if p == 1)
        profit = env.line.profit_total()
        profits.append(profit)
        if det:
            print(f"\n{label}:")
            print(f"  Profit: {profit:.2f}")
            print(f"  Revenue: {c['revenue']:.2f}, Backorder: {c['backorder']:.2f}")
            print(f"  Throughput: P0={tp0}, P1={tp1}, Total={tp0+tp1}")
    
    if not det:
        print(f"\n{label}:")
        print(f"  Profits: {[f'{p:.0f}' for p in profits]}")
        print(f"  Mean profit: {np.mean(profits):.2f}")
        print(f"  Best: {max(profits):.2f}, Worst: {min(profits):.2f}")
