"""
Benchmark training throughput for different n_envs values.

Measures wall-clock time to complete a fixed number of PPO rollouts
(rollout = n_envs * n_steps env steps + 1 update).
Reports steps/sec and seconds-per-rollout.
"""
from __future__ import annotations

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, SubprocVecEnv

from simplefab_1p import make_common_config
from simplefab_1p.env import FabEnv, ShapingConfig


def bench_one(n_envs: int, n_rollouts: int = 2, n_steps_per_env: int = 2688) -> dict:
    """Train for n_rollouts and time it. Returns timing dict."""
    common = make_common_config(mode="ALTERNATING")
    shaping = ShapingConfig(enabled=True, beta=0.5, gamma=0.999,
                            w_backlog=1.0, w_wip=0.0, w_finished=0.5)

    def make_env():
        return FabEnv(common_cfg=common, invalid_action_penalty=-2.0,
                      normalize_obs=True, shaping=shaping)

    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # batch_size must divide n_envs*n_steps
    total_buffer = n_envs * n_steps_per_env
    # pick a batch size that divides total_buffer and is <= 4096
    batch_size = total_buffer // 4
    while total_buffer % batch_size != 0:
        batch_size -= 1

    model = MaskablePPO(
        "MlpPolicy", env, device="cpu",
        n_steps=n_steps_per_env, batch_size=batch_size,
        learning_rate=3e-4, clip_range=0.2, ent_coef=0.05,
        gae_lambda=0.95, gamma=0.999, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=0,
    )

    total_steps = n_envs * n_steps_per_env * n_rollouts

    # Warmup: 1 rollout that we discard from timing (subprocess spawn / JIT etc.)
    t_warm = time.time()
    model.learn(total_timesteps=n_envs * n_steps_per_env, log_interval=1000)
    warmup_time = time.time() - t_warm

    # Timed: n_rollouts more
    t0 = time.time()
    model.learn(total_timesteps=total_steps, log_interval=1000, reset_num_timesteps=False)
    elapsed = time.time() - t0

    env.close()

    sps = total_steps / elapsed
    return {
        "n_envs": n_envs,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "elapsed_s": elapsed,
        "warmup_s": warmup_time,
        "steps_per_sec": sps,
        "sec_per_rollout": elapsed / n_rollouts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, nargs="+", default=[4, 8, 12, 16, 24],
                        help="List of n_envs to benchmark")
    parser.add_argument("--rollouts", type=int, default=2,
                        help="Rollouts per benchmark run (after warmup)")
    args = parser.parse_args()

    results = []
    for n in args.n_envs:
        print(f"\n=== Benchmarking n_envs={n} (warmup + {args.rollouts} timed rollouts) ===")
        r = bench_one(n, n_rollouts=args.rollouts)
        results.append(r)
        print(f"  warmup: {r['warmup_s']:.1f}s  |  timed: {r['elapsed_s']:.1f}s "
              f"|  {r['steps_per_sec']:.0f} steps/s  |  {r['sec_per_rollout']:.1f}s/rollout "
              f"|  batch={r['batch_size']}")

    print("\n" + "=" * 80)
    print(f"{'n_envs':>8} {'batch':>8} {'total_steps':>12} {'elapsed_s':>10} "
          f"{'steps/s':>10} {'s/rollout':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['n_envs']:>8} {r['batch_size']:>8} {r['total_steps']:>12} "
              f"{r['elapsed_s']:>10.1f} {r['steps_per_sec']:>10.0f} {r['sec_per_rollout']:>10.1f}")

    best = max(results, key=lambda x: x["steps_per_sec"])
    print(f"\n=> FASTEST: n_envs={best['n_envs']} at {best['steps_per_sec']:.0f} steps/s")
    # Estimate full 8M-step training time at the best setting
    full_train_s = 8_000_000 / best["steps_per_sec"]
    print(f"   Estimated 8M-step training time: {full_train_s/60:.1f} min  ({full_train_s/3600:.2f} h)")


if __name__ == "__main__":
    main()
