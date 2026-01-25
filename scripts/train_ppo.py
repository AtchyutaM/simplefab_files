from __future__ import annotations

import numpy as np
from typing import List, Dict, Any

from stable_baselines3 import PPO
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

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        n_steps=1024,
        batch_size=2048,
        learning_rate=1e-3,
        clip_range=0.3,
        ent_coef=0.01,
        gae_lambda=0.95,
        gamma=PPO_GAMMA,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tb_fab/",
        verbose=1,
    )

    tb_name = "PPO_CPU_Run_shaping_beta0p5"
    callback = FabTBCallback()

    model.learn(
        total_timesteps=3_000_000,
        tb_log_name=tb_name,
        log_interval=1,
        callback=callback,
    )

    model.save("ppo_fab_policy")
    env.save("vecnorm_stats.pkl")

    print("✅ Training complete.")
    print("   Model:            ppo_fab_policy")
    print("   VecNormalize:     vecnorm_stats.pkl")
    print("ℹ️  Start TensorBoard:  python -m tensorboard.main --logdir ./tb_fab --reload_interval 2")


if __name__ == "__main__":
    main()
