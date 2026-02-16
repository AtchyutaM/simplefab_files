from __future__ import annotations

import numpy as np
from typing import List, Dict, Any

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from simplefab import make_common_config
from simplefab.env import FabEnv, ShapingConfig


class FabTBCallback(BaseCallback):
    """
    Reads per-episode summaries from infos[i]["fab_stats"] (emitted by FabEnv.step at done)
    and logs rollout-mean values to TensorBoard under `fab/*`.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._buf: List[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        if infos is None or dones is None:
            return True

        for i, d in enumerate(dones):
            if not d:
                continue
            stats = infos[i].get("fab_stats")
            if stats is not None:
                self._buf.append(stats)
        return True

    def _on_rollout_end(self) -> None:
        if not self._buf:
            return
        keys = self._buf[0].keys()
        for k in keys:
            vals = [float(s[k]) for s in self._buf if k in s]
            if vals:
                self.logger.record(f"fab/{k}_mean", float(np.mean(vals)))
        self._buf.clear()


def main():
    common_train = make_common_config(mode="UNIFORM", H=500, alpha=0.5, utilization=0.92)
    
    # Add initial finished inventory to cover cold-start ramp-up period
    # This reduces backorder costs during the first ~80 timesteps before production can ship
    common_train["initial_finished_inventory"] = {0: 8, 1: 9}  # P0: 8, P1: 9

    PPO_GAMMA = 0.99
    shaping = ShapingConfig(
        enabled=True,
        beta=0.5,          # start here
        gamma=PPO_GAMMA,   # MUST match PPO gamma
        w_backlog=2.0,
        w_wip=0.2,
        w_finished=0.05,
    )

    def make_env():
        return FabEnv(
            common_cfg=common_train,
            invalid_action_penalty=-2.0,
            normalize_obs=True,
            shaping=shaping,
        )

    env = make_vec_env(make_env, n_envs=8)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        device="cpu",
        n_steps=2048,           # Longer rollouts for better value estimates
        batch_size=512,         # Smaller batches for more gradient updates
        learning_rate=3e-4,     # Lower LR for stability
        clip_range=0.2,         # Standard PPO clip range
        ent_coef=0.05,          # Higher entropy to prevent collapse
        gae_lambda=0.95,
        gamma=PPO_GAMMA,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tb_fab/",
        verbose=1,
    )


    tb_name = "PPO_warmstart_8P0_9P1"
    callback = FabTBCallback()

    model.learn(
        total_timesteps=3_000_000,
        tb_log_name=tb_name,
        log_interval=1,
        callback=callback,
        use_masking=True,
    )

    model.save("ppo_fab_policy")
    env.save("vecnorm_stats.pkl")

    print("✅ Training complete.")
    print("   Model:            ppo_fab_policy")
    print("   VecNormalize:     vecnorm_stats.pkl")
    print("ℹ️  Start TensorBoard:  python -m tensorboard.main --logdir ./tb_fab --reload_interval 2")


if __name__ == "__main__":
    main()
