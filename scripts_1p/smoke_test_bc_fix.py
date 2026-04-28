"""Quick smoke test for the BC-loading hyperparameter fix.

Loads a BC model into a freshly-configured PPO, runs 1 rollout, and reports
fps. Vanilla PPO runs at ~13K fps; the buggy BC-loaded version was ~3-4K.
We expect ~13K with the fix.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, SubprocVecEnv

from simplefab_1p import make_common_config
from simplefab_1p.env import FabEnv, ShapingConfig


def main():
    bc_path = "output_data_1p/models/bc_alternating_92.zip"
    if not os.path.exists(bc_path):
        print(f"BC model not found at {bc_path}, aborting smoke test")
        sys.exit(2)

    cfg = make_common_config(mode="ALTERNATING_92")
    shaping = ShapingConfig(
        enabled=True, beta=0.5, gamma=0.999,
        w_backlog=1.0, w_wip=0.0, w_finished=0.5,
    )

    def make_env():
        return FabEnv(
            common_cfg=cfg, invalid_action_penalty=-2.0,
            normalize_obs=True, shaping=shaping,
        )

    env = make_vec_env(make_env, n_envs=16, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Mirror train_ppo.py's --pretrained path
    model = MaskablePPO(
        "MlpPolicy", env, device="cpu",
        n_steps=2688, batch_size=1344,
        learning_rate=3e-4, clip_range=0.2, ent_coef=0.05,
        gae_lambda=0.95, gamma=0.999, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=0,
    )
    bc_temp = MaskablePPO.load(bc_path, device="cpu")
    model.policy.load_state_dict(bc_temp.policy.state_dict())
    del bc_temp

    print(f"Hyperparams: n_steps={model.n_steps}, batch_size={model.batch_size}, "
          f"n_epochs={model.n_epochs}, lr={model.learning_rate}")
    assert model.n_steps == 2688
    assert model.batch_size == 1344
    assert model.n_epochs == 10
    print("Hyperparam check: PASS")

    # 2 rollouts so we discard the warmup
    target_steps = 16 * 2688 * 2  # 2 iterations
    t0 = time.time()
    model.learn(total_timesteps=target_steps, log_interval=10000, use_masking=True)
    elapsed = time.time() - t0
    sps = target_steps / elapsed
    print(f"Smoke test: {target_steps} steps in {elapsed:.1f}s = {sps:.0f} steps/sec")

    # Vanilla was ~13K fps. Buggy BC was ~3-4K. Anything above 8K means the fix worked.
    if sps > 8000:
        print("RESULT: PASS (>8K sps, fix is working)")
        sys.exit(0)
    else:
        print(f"RESULT: FAIL (sps={sps:.0f}, expected >8000)")
        sys.exit(1)


if __name__ == "__main__":
    main()
