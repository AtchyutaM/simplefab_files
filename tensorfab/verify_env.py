"""
Verify `env_tensor.TensorEnv` against the reference `simplefab_1p.env.FabEnv`.

Same discipline as verify.py: identical requested actions into both, then compare
every per-tick output the RL algorithm actually consumes —

    obs (all 18 dims), reward, base_reward (delta-profit), shaping term,
    invalid_selected, and the action mask

Obs/mask/invalid are compared exactly or to float32 round-off. Reward is compared
to a small absolute tolerance because the reference accumulates dollars
incrementally in float64 while we reconstruct the per-tick delta from integer
increments.

Usage:  python -m tensorfab.verify_env
"""

from __future__ import annotations

import sys

import numpy as np

from tensorfab import device

device.configure()

import jax.numpy as jnp  # noqa: E402

from simplefab_1p.config import make_common_config  # noqa: E402
from simplefab_1p.env import FabEnv, ShapingConfig  # noqa: E402
from tensorfab.env_tensor import Shaping, make_env  # noqa: E402
from tensorfab.spec import spec_from_common_config  # noqa: E402
from tensorfab.tensor_sim import make_line  # noqa: E402

REWARD_TOL = 2e-3
OBS_TOL = 1e-5


def reference_env_rollout(common, actions, shaping: Shaping, penalty: float):
    env = FabEnv(
        common,
        invalid_action_penalty=penalty,
        shaping=ShapingConfig(
            enabled=shaping.enabled, beta=shaping.beta, gamma=shaping.gamma,
            w_backlog=shaping.w_backlog, w_wip=shaping.w_wip,
            w_finished=shaping.w_finished,
        ),
    )
    obs0, info0 = env.reset()
    H = actions.shape[0]

    rec = {k: [] for k in ("obs", "reward", "base", "shaping", "invalid", "mask")}
    for t in range(H):
        m = info0["action_mask"] if t == 0 else None
        mask_now = env.line.compute_action_mask(current_time=env._t)[:, 1]
        obs, r, term, trunc, info = env.step(actions[t])
        rec["obs"].append(obs)
        rec["reward"].append(r)
        rec["base"].append(info["base_reward"])
        rec["shaping"].append(info["shaping"])
        rec["invalid"].append(info["invalid_selected"])
        rec["mask"].append(mask_now)

    return {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}, np.asarray(obs0)


def check(mode: str, seed: int, shaping: Shaping, penalty: float) -> bool:
    common = make_common_config(mode=mode)
    spec = spec_from_common_config(common, mode=mode)
    line = make_line(spec)
    tenv = make_env(line, shaping=shaping, invalid_action_penalty=penalty)

    rng = np.random.default_rng(seed)
    B = 3
    acts = np.stack(
        [rng.random((spec.H, 4)) < p for p in (0.9, 0.5, 0.2)], axis=1
    ).astype(np.int32)                                   # [H, B, 4]

    _, outs = tenv.rollout(jnp.asarray(acts))
    got = {
        "obs": np.asarray(outs.obs, dtype=np.float64),
        "reward": np.asarray(outs.reward, dtype=np.float64),
        "base": np.asarray(outs.base_reward, dtype=np.float64),
        "shaping": np.asarray(outs.shaping, dtype=np.float64),
        "invalid": np.asarray(outs.invalid_selected, dtype=np.float64),
        "mask": np.asarray(outs.mask, dtype=np.float64),
    }

    ok = True
    # reset obs
    ref0 = reference_env_rollout(common, acts[:, 0], shaping, penalty)[1]
    d0 = float(np.max(np.abs(np.asarray(tenv.reset_obs(1))[0] - ref0)))
    ok0 = d0 < OBS_TOL
    ok &= ok0
    print(f"  [{'PASS' if ok0 else 'FAIL'}] reset obs                 max|d| = {d0:.2e}")

    for b in range(B):
        ref, _ = reference_env_rollout(common, acts[:, b], shaping, penalty)
        for key, tol in (("obs", OBS_TOL), ("mask", 0.0), ("invalid", 0.0),
                         ("base", REWARD_TOL), ("shaping", REWARD_TOL),
                         ("reward", REWARD_TOL)):
            a = ref[key]
            g = got[key][:, b] if got[key].ndim > 1 else got[key]
            if g.ndim > a.ndim:
                g = g.reshape(a.shape)
            d = float(np.max(np.abs(a - g)))
            good = d <= tol
            ok &= good
            if not good:
                bad_t = int(np.argmax(np.abs(a - g).reshape(a.shape[0], -1).max(axis=1)))
                print(f"  [FAIL] col{b} {key:<9} max|d| = {d:.3e} (tol {tol:g}) "
                      f"first bad tick {bad_t}")
        print(f"  [{'PASS' if ok else 'FAIL'}] col{b} p={(0.9, 0.5, 0.2)[b]}  "
              f"obs/mask/invalid/base/shaping/reward")
    return ok


def main() -> int:
    print(device.describe())
    all_ok = True
    cases = [
        ("ALTERNATING_92", Shaping(), 0.0),
        ("ALTERNATING_92", Shaping(beta=0.5, gamma=0.999, w_backlog=1.0,
                                   w_wip=0.0, w_finished=0.5), -1.0),
        ("UNIFORM", Shaping(enabled=False), 0.0),
    ]
    for mode, sh, pen in cases:
        print(f"\n=== {mode}  shaping={sh}  penalty={pen} ===")
        all_ok &= check(mode, seed=1, shaping=sh, penalty=pen)
    print("\n" + "=" * 60)
    print("ENV VERIFICATION PASSED" if all_ok else "ENV VERIFICATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
