import os
import argparse
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from simplefab_1p.env import FabEnv, ShapingConfig
from simplefab_1p.config import make_common_config
def heuristic_policy(obs, line, action_mask):
    action = [0, 0, 0, 0]
    if action_mask[0, 1] == 1: action[0] = 1
    if action_mask[1, 1] == 1: action[1] = 1
    if action_mask[2, 1] == 1: action[2] = 1
    if action_mask[3, 1] == 1: action[3] = 1
    return action

def generate_data(num_episodes=500, out_path="output_data_1p/expert_data.npz"):
    print(f"Generating expert data using Commander Heuristic for {num_episodes} episodes...")
    # We don't need shaping for the heuristic, but we match the state obs normalizations
    common_cfg = make_common_config()
    shaping = ShapingConfig()
    
    # Needs to match the architecture EXACTLY
    env = FabEnv(common_cfg=common_cfg, shaping=shaping, normalize_obs=True)
    
    all_obs = []
    all_actions = []
    all_masks = []
    
    obs, info = env.reset()
    episode_count = 0
    total_profit = 0
    
    while episode_count < num_episodes:
        # Get action from Commander Heuristic
        mask = info.get("action_mask", np.ones((4,2)))
        action = heuristic_policy(obs, env.unwrapped.line, mask) # Assuming unvec
        
        all_obs.append(obs)
        all_actions.append(action)
        all_masks.append(mask)
        
        obs, reward, done, trunc, info = env.step(action)
        
        if done or trunc:
            profit = env.unwrapped.line.profit_total()
            total_profit += profit
            episode_count += 1
            if episode_count % 50 == 0:
                print(f"  Completed {episode_count}/{num_episodes} episodes... (Avg Profit: ${total_profit/episode_count:.2f})")
            
            obs, info = env.reset()
            
    # Save the Dataset (converting lists to numpy arrays)
    np.savez_compressed(
        out_path,
        obs=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.int64),
        masks=np.array(all_masks, dtype=np.int64)
    )
    print(f"\nDone! Dataset saved to {out_path} with {len(all_actions)} total transitions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to generate")
    parser.add_argument("--out", type=str, default="output_data_1p/expert_data.npz", help="Output path")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    generate_data(num_episodes=args.episodes, out_path=args.out)
