from __future__ import annotations

import os
import numpy as np
from typing import List, Dict, Any

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from simplefab import make_common_config
from simplefab.env import FabEnv, ShapingConfig


ACTION_NAMES = ["None", "Prod0", "Prod1"]


class FabTBCallback(BaseCallback):
    """
    Logs per-episode fab_stats AND per-rollout action distributions to TensorBoard.
    Action distribution: what % of the time each machine picks None/Prod0/Prod1.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._buf: List[Dict[str, Any]] = []
        # Track actions across the rollout: [step][machine] = action_id
        self._action_log: List[Any] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        actions = self.locals.get("actions", None)

        # Log actions from first env for distribution tracking
        if actions is not None:
            self._action_log.append(actions[0].copy())  # shape (4,)

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
        # Log fab_stats
        if self._buf:
            keys = self._buf[0].keys()
            for k in keys:
                vals = [float(s[k]) for s in self._buf if k in s]
                if vals:
                    self.logger.record(f"fab/{k}_mean", float(np.mean(vals)))
            self._buf.clear()

        # Log action distributions per machine
        if self._action_log:
            actions_arr = np.array(self._action_log)  # shape (T, 4)
            T = len(actions_arr)
            for m in range(4):
                for a in range(3):
                    pct = float(np.sum(actions_arr[:, m] == a)) / T * 100
                    self.logger.record(f"actions/m{m}_{ACTION_NAMES[a]}_pct", pct)
            self._action_log.clear()


class EvalCallback(BaseCallback):
    """
    Every `eval_freq` rollouts, run one deterministic evaluation episode
    and log the results to TensorBoard under `eval/*`.
    """
    def __init__(self, eval_cfg: Dict[str, Any], shaping: ShapingConfig,
                 eval_freq: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.eval_cfg = eval_cfg
        self.shaping = shaping
        self.eval_freq = eval_freq
        self._rollout_count = 0

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.eval_freq != 0:
            return

        # Build a fresh eval env
        eval_env = FabEnv(
            common_cfg=self.eval_cfg,
            invalid_action_penalty=0.0,
            normalize_obs=True,
            shaping=self.shaping,
        )
        obs, info = eval_env.reset()
        done = False
        total_reward = 0.0
        action_counts = np.zeros((4, 3), dtype=int)  # 4 machines × 3 actions
        step_count = 0
        while not done:
            mask = eval_env.action_masks()
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            for m in range(4):
                action_counts[m, int(action[m])] += 1
            step_count += 1
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        stats = info.get("fab_stats", {})
        self.logger.record("eval/profit", stats.get("profit", 0.0))
        self.logger.record("eval/revenue", stats.get("revenue", 0.0))
        self.logger.record("eval/cost_backorder", stats.get("cost_backorder", 0.0))
        self.logger.record("eval/cost_inventory", stats.get("cost_inventory", 0.0))
        self.logger.record("eval/throughput_total", stats.get("throughput_total", 0))
        self.logger.record("eval/episode_reward", total_reward)

        # Log eval action distributions
        if step_count > 0:
            for m in range(4):
                for a in range(3):
                    pct = float(action_counts[m, a]) / step_count * 100
                    self.logger.record(f"eval_actions/m{m}_{ACTION_NAMES[a]}_pct", pct)

    def _on_step(self) -> bool:
        return True


def main():
    # Use default config: H=2688 (4 weeks), 92% utilization, weekly arrivals/demand
    common_train = make_common_config()

    PPO_GAMMA = 0.99
    shaping = ShapingConfig(
        enabled=True,
        beta=0.5,
        gamma=PPO_GAMMA,
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
        n_steps=2688,           # = episode length, so each rollout = 1 full episode
        batch_size=672,         # divides 2688 evenly (4 minibatches)
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.05,
        gae_lambda=0.95,
        gamma=PPO_GAMMA,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tb_fab/",
        verbose=1,
    )

    tb_name = "PPO_weekly_2688"
    callbacks = [
        FabTBCallback(),
        EvalCallback(eval_cfg=common_train, shaping=shaping, eval_freq=5),
    ]

    model.learn(
        total_timesteps=5_000_000,
        tb_log_name=tb_name,
        log_interval=1,
        callback=callbacks,
        use_masking=True,
    )

    # Save model and normalization stats
    save_dir = os.path.join(os.getcwd(), "output_data", "models")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "ppo_fab_policy"))
    env.save(os.path.join(save_dir, "vecnorm_stats.pkl"))

    print("Training complete.")
    print(f"   Model:        {save_dir}/ppo_fab_policy")
    print(f"   VecNormalize: {save_dir}/vecnorm_stats.pkl")
    print("   TensorBoard:  python -m tensorboard.main --logdir ./tb_fab --reload_interval 2")


if __name__ == "__main__":
    main()
