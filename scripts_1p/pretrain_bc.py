import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from simplefab_1p.env import FabEnv, ShapingConfig
from simplefab_1p.config import make_common_config

def pretrain_behavioral_cloning(data_path="output_data_1p/expert_data.npz", 
                                out_path="output_data_1p/models/ppo_pretrained_bc",
                                epochs=30, batch_size=256):
    
    print(f"Loading expert dataset from {data_path}...")
    dataset = np.load(data_path)
    obs_all = dataset['obs']
    act_all = dataset['actions']
    
    dataset_size = len(obs_all)
    print(f"Loaded {dataset_size} transitions.")
    
    # 1. Initialize fresh PPO Model
    print("Initializing fresh MaskablePPO neural network...")
    common_cfg = make_common_config()
    shaping = ShapingConfig()
    env = FabEnv(common_cfg=common_cfg, shaping=shaping, normalize_obs=True)
    
    model = MaskablePPO(
        "MlpPolicy", env, verbose=0, 
        gamma=0.999, # Ensure architectures match EXACTLY
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    )
    
    # 2. Setup Supervised Learning Optimizer
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    
    print(f"Starting Supervised Behavioral Cloning for {epochs} epochs...")
    model.policy.train()
    
    # PyTorch Training Loop
    indices = np.arange(dataset_size)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        batches = 0
        
        for start_idx in range(0, dataset_size, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            
            # Convert to PyTorch Tensors
            obs_tensor = torch.tensor(obs_all[batch_idx], dtype=torch.float32).to(model.device)
            act_tensor = torch.tensor(act_all[batch_idx], dtype=torch.float32).to(model.device)
            
            # stable_baselines3 provides a magic method `evaluate_actions`
            # which returns (log_prob, values, entropy)
            # Maximizing log_prob (Negative Log Likelihood) is exact Behavioral Cloning!
            log_prob, _, _ = model.policy.evaluate_actions(obs_tensor, act_tensor)
            
            # Cross-Entropy Loss
            loss = -log_prob.mean()
            
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        print(f"  Epoch {epoch+1}/{epochs} | Loss Object (NLL): {total_loss/batches:.4f}")
        
    print(f"\nPre-Training complete! Saving cloned brain to {out_path}.zip...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print("Optimization Handoff Ready!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="output_data_1p/expert_data.npz", help="Dataset path")
    parser.add_argument("--out", type=str, default="output_data_1p/models/ppo_pretrained_bc", help="Output path")
    parser.add_argument("--epochs", type=int, default=30, help="Training Epochs")
    args = parser.parse_args()
    
    pretrain_behavioral_cloning(data_path=args.data, out_path=args.out, epochs=args.epochs)
