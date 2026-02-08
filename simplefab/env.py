from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks

from .sim import ProductionLine

ACTION_MAP = ["None", "Prod0", "Prod1"]


def _build_default_setups():
    # Same as your original Mac1_setups / Mac3_setups
    return {
        None: {0: 1, 1: 1},
        0:    {0: 0, 1: 1},
        1:    {0: 1, 1: 0},
    }


@dataclass
class ShapingConfig:
    """
    Potential-based shaping configuration.

    Reward becomes:
      reward = base_reward + beta * (gamma * Phi(s') - Phi(s)) + invalid_action_penalty * invalid_selected

    IMPORTANT:
      - Set gamma equal to PPO's gamma (e.g., 0.99) for clean shaping behavior.
      - Use beta to control shaping strength relative to delta-profit.
    """
    enabled: bool = True
    beta: float = 1.0
    gamma: float = 0.99
    w_backlog: float = 2.0
    w_wip: float = 0.2
    w_finished: float = 0.05


class FabEnv(gym.Env):
    """
    Gymnasium wrapper around ProductionLine.

    Observation: 20-dim vector in [0,1] (by default normalized).
    Action: MultiDiscrete([3,3,3,3]) for 4 machines × {None, Prod0, Prod1}.

    Invalid actions handling:
      - We compute an action mask each step.
      - If the chosen action is invalid, we project it to 'None'.
      - Optionally apply a penalty per invalid *selection* via invalid_action_penalty.

    NEW (Potential-based shaping):
      - Add beta * (gamma*Phi(s') - Phi(s)) to the base reward (delta-profit).
      - Phi(s) is a negative weighted sum of backlog, WIP, and finished inventory.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        common_cfg: Dict[str, Any],
        invalid_action_penalty: float = 0.0,
        normalize_obs: bool = True,
        shaping: Optional[ShapingConfig] = None,
    ):
        super().__init__()
        self.common = common_cfg
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.normalize_obs = bool(normalize_obs)
        self.shaping = shaping if shaping is not None else ShapingConfig()

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(20,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])

        self._max_steps = int(self.common["time_horizon"])
        self._t = 0
        self._build_line()

    def _build_line(self) -> None:
        pt = self.common["processing_times"]
        setups = _build_default_setups()

        self.line = ProductionLine(
            {0: pt[0][0], 1: pt[0][1]},
            {0: pt[1][0], 1: pt[1][1]},
            setups,
            {0: pt[2][0], 1: pt[2][1]},
            {0: pt[3][0], 1: pt[3][1]},
            setups,
            common_cfg=self.common,
        )
        self.line.logging_enabled = False
        self._t = 0

    # -----------------------------
    # NEW: Potential function Phi(s)
    # -----------------------------
    def _phi(self) -> float:
        """
        Phi(s) = -(w_backlog*backlog + w_wip*wip + w_finished*finished_inventory)

        backlog: total units waiting in demand queues
        wip: total units in queues 0..3 (in-process buffers)
        finished_inventory: units in queue_fin (finished but not necessarily shipped)
        """
        q = self.line.queues

        backlog = len(q["demand"][0]) + len(q["demand"][1])

        wip = 0
        for name in ("queue0", "queue1", "queue2", "queue3"):
            wip += len(q[name][0]) + len(q[name][1])

        finished = len(q["queue_fin"][0]) + len(q["queue_fin"][1])

        cfg = self.shaping
        return -(
            cfg.w_backlog * backlog
            + cfg.w_wip * wip
            + cfg.w_finished * finished
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._build_line()
        obs = np.array(self.line.get_observation(current_time=self._t, norm=self.normalize_obs), dtype=np.float32)
        info = {
            "action_mask": self.line.compute_action_mask(current_time=self._t),
            "profit": float(self.line.profit_total()) if hasattr(self.line, "profit_total") else 0.0,
        }
        return obs, info

    def step(self, action):
        a0, a1, a2, a3 = [int(x) for x in action]
        mask = self.line.compute_action_mask(current_time=self._t)  # (4,3)

        # compute invalid selections FIRST
        invalid_selected = int(
            (mask[0, a0] == 0) + (mask[1, a1] == 0) + (mask[2, a2] == 0) + (mask[3, a3] == 0)
        )

        # optional debug check (now safe)
        if invalid_selected > 0:
            # With correct MaskablePPO + action_masks(), this should be extremely rare.
            # Uncomment to hard-fail if you want:
            # raise RuntimeError(f"Mask mismatch: invalid_selected={invalid_selected}, action={action}, mask={mask}")
            pass

        # project invalid picks to None
        a0 = a0 if mask[0, a0] else 0
        a1 = a1 if mask[1, a1] else 0
        a2 = a2 if mask[2, a2] else 0
        a3 = a3 if mask[3, a3] else 0

        actions = {
            "machine0": ACTION_MAP[a0],
            "machine1": ACTION_MAP[a1],
            "machine2": ACTION_MAP[a2],
            "machine3": ACTION_MAP[a3],
        }

        # --- rest of your code unchanged ---
        phi_before = self._phi() if self.shaping.enabled else 0.0
        base_reward = self.line.run_step(current_time=self._t, actions_override=actions, invalid_penalty=0.0)

        invalid_pen = 0.0
        if self.invalid_action_penalty != 0.0 and invalid_selected > 0:
            invalid_pen = self.invalid_action_penalty * invalid_selected

        shaping_term = 0.0
        phi_after = 0.0
        if self.shaping.enabled:
            phi_after = self._phi()
            shaping_term = self.shaping.beta * (self.shaping.gamma * phi_after - phi_before)

        reward = float(base_reward + invalid_pen + shaping_term)

        self._t += 1
        obs = np.array(self.line.get_observation(current_time=self._t, norm=self.normalize_obs), dtype=np.float32)

        terminated = (self._t >= self._max_steps)
        truncated = False

        info = {
            "action_mask": self.line.compute_action_mask(current_time=self._t),
            "profit": self.line.cost_log[-1]["profit"] if self.line.cost_log else 0.0,
            "invalid_selected": invalid_selected,
            "invalid_sim": self.line.cost_log[-1]["invalid_actions"] if self.line.cost_log else 0,
            "base_reward": float(base_reward),
            "invalid_penalty": float(invalid_pen),
            "shaping": float(shaping_term),
            "phi_before": float(phi_before),
            "phi_after": float(phi_after),
        }

        if terminated or truncated:
            ln = self.line
            c = ln.costs
            info["fab_stats"] = {
                "profit": float(ln.profit_total()),
                "revenue": float(c.get("revenue", 0.0)),
                "cost_production": float(c.get("production", 0.0)),
                "cost_setup": float(c.get("setup", 0.0)),
                "cost_inventory": float(c.get("inventory", 0.0)),
                "cost_backorder": float(c.get("backorder", 0.0)),
                "throughput_p0": int(sum(1 for p, _ in ln.demand_met_log if p == 0)),
                "throughput_p1": int(sum(1 for p, _ in ln.demand_met_log if p == 1)),
                "throughput_total": int(len(ln.demand_met_log)),
                "starts_m0": int(len(ln.machine0.event_log)),
                "starts_m1": int(len(ln.machine1.event_log)),
                "starts_m2": int(len(ln.machine2.event_log)),
                "starts_m3": int(len(ln.machine3.event_log)),
            }

        return obs, reward, terminated, truncated, info


    def action_masks(self) -> np.ndarray:
        """
        Action mask for sb3-contrib MaskablePPO.

        For MultiDiscrete([3,3,3,3]), return a flattened boolean array of length 12:
        [m0_a0, m0_a1, m0_a2, m1_a0, ..., m3_a2]
        True = valid, False = invalid.
        """
        mask_4x3 = self.line.compute_action_mask(current_time=self._t)  # shape (4,3), 0/1
        return mask_4x3.astype(bool).reshape(-1)

    
    def render(self):
        pass

    def close(self):
        pass
