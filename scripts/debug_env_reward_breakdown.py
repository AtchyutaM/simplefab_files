from __future__ import annotations

import numpy as np

from simplefab import make_common_config
from simplefab.env import FabEnv, ACTION_MAP


def main():
    H = 30
    common = make_common_config(mode="UNIFORM", H=H, alpha=0.5, utilization=0.92)

    # IMPORTANT: your env projects invalid picks to None and then applies invalid_action_penalty per invalid *selection*
    env = FabEnv(common_cfg=common, invalid_action_penalty=-2.0, normalize_obs=True)

    obs, info = env.reset(seed=0)

    print(f"=== Env reward breakdown (H={H}, invalid_action_penalty={env.invalid_action_penalty}) ===")
    print(
        "t | action(idx)->(name)                      | inv_sel | penalty | sim_dProfit | env_reward | "
        "profit    dem(P0,P1) fin(P0,P1)"
    )
    print("-" * 140)

    rng = np.random.default_rng(0)

    for t in range(H):
        # ------------------------------------------------------------------
        # Choose an action to test.
        # For debugging invalids, random actions are useful.
        # If you want to test a trained PPO model, replace this block.
        # ------------------------------------------------------------------
        action = np.array([rng.integers(0, 3) for _ in range(4)], dtype=np.int64)

        # Pre-step profit so we can compute sim delta-profit ourselves
        prev_profit = env.line.profit_total()

        # Step environment
        obs, reward, terminated, truncated, step_info = env.step(action)

        # Post-step profit
        new_profit = env.line.profit_total()
        sim_dprofit = new_profit - prev_profit

        # Env penalty is based on invalid_selected (what agent attempted)
        invalid_selected = int(step_info.get("invalid_selected", 0))
        penalty = env.invalid_action_penalty * invalid_selected

        # This should match: env_reward ≈ sim_dprofit + penalty
        env_reward = float(reward)

        # Read queue state
        dem0 = len(env.line.queues["demand"][0])
        dem1 = len(env.line.queues["demand"][1])
        fin0 = len(env.line.queues["queue_fin"][0])
        fin1 = len(env.line.queues["queue_fin"][1])

        # Human-readable action names (what the agent picked)
        picked_names = [ACTION_MAP[int(a)] for a in action]

        profit = step_info.get("profit", env.line.cost_log[-1]["profit"] if env.line.cost_log else env.line.profit_total())

        print(
            f"{t:2d} | {action.tolist()} -> {picked_names!s:35s} | "
            f"{invalid_selected:7d} | {penalty:7.1f} | {sim_dprofit:10.1f} | {env_reward:10.1f} | "
            f"{profit:8.1f}  ({dem0:2d},{dem1:2d})   ({fin0:2d},{fin1:2d})"
        )

        # quick consistency check
        if abs(env_reward - (sim_dprofit + penalty)) > 1e-6:
            print("   ⚠️  mismatch: env_reward != sim_dprofit + penalty (check code paths)")

        if terminated or truncated:
            break

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
