"""
Verify that the relaxed line in `diffsim.py` is a faithful surrogate.

Three properties must hold, or the spike's numbers mean nothing:

  1. hard-mode forward == the exact integer sim in tensor_sim.py
     (same dynamics, just written in float)

  2. st-mode forward == hard-mode forward, EXACTLY (bit-for-bit)
     This is the "exact-forward, approximate-backward" claim. It is a real
     numerical risk in float32, which is why the straight-through estimator is
     written as `hard + soft - stop_gradient(soft)`.

  3. hard-mode gradient == 0 everywhere
     The control condition: naive autodiff through the true dynamics gives
     nothing. If this is nonzero, something is not actually a hard threshold.

Usage:  python -m tensorfab.verify_diffsim
"""

from __future__ import annotations

import sys

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tensorfab.diffsim import DiffLine, Relax, hard_actions  # noqa: E402
from tensorfab.spec import spec_for_mode  # noqa: E402
from tensorfab.tensor_sim import make_line, profit_breakdown  # noqa: E402


def check(spec, batch: int = 8, seed: int = 0) -> bool:
    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(0.0, 2.0, size=(batch, spec.H, 4)), dtype=jnp.float32)

    exact = make_line(spec)
    acts = hard_actions(theta)                       # [B, H, 4]
    final_exact = exact.rollout(jnp.swapaxes(acts, 0, 1))
    p_exact = profit_breakdown(spec, final_exact)["profit"]

    ok = True

    # 1. hard-mode forward vs the exact integer sim
    p_hard = np.asarray(DiffLine(spec, Relax(mode="hard")).profit_of(theta))
    d1 = float(np.max(np.abs(p_hard - p_exact)))
    # float32 accumulation over H ticks on ~$30k quantities: ~1e-2 is expected.
    ok1 = d1 < 0.5
    ok &= ok1
    print(f"  [{'PASS' if ok1 else 'FAIL'}] hard-mode forward == exact integer sim   "
          f"max|d profit| = ${d1:.4f}")

    # 2. st-mode forward vs hard-mode forward — must be EXACT
    p_st = np.asarray(DiffLine(spec, Relax(mode="st")).profit_of(theta))
    d2 = float(np.max(np.abs(p_st - p_hard)))
    ok2 = d2 == 0.0
    ok &= ok2
    print(f"  [{'PASS' if ok2 else 'FAIL'}] st-mode forward == hard-mode forward     "
          f"max|d profit| = {d2:.1e}  (must be exactly 0)")

    # 3. hard-mode gradient must be identically zero
    _, g_hard = DiffLine(spec, Relax(mode="hard")).value_and_grad(theta)
    gmax = float(jnp.max(jnp.abs(g_hard)))
    ok3 = gmax == 0.0
    ok &= ok3
    print(f"  [{'PASS' if ok3 else 'FAIL'}] hard-mode gradient == 0 (the control)     "
          f"max|grad| = {gmax:.1e}")

    # informational: the surrogate does produce a signal
    _, g_st = DiffLine(spec, Relax(mode="st")).value_and_grad(theta)
    finite = bool(jnp.all(jnp.isfinite(g_st)))
    print(f"         st-mode gradient: max|g|={float(jnp.max(jnp.abs(g_st))):.4g}  "
          f"mean|g|={float(jnp.mean(jnp.abs(g_st))):.4g}  finite={finite}")
    ok &= finite

    return ok


def main() -> int:
    print(device.describe())
    all_ok = True
    for mode, H in [("ALTERNATING_92", 2688), ("UNIFORM", 672)]:
        spec = spec_for_mode(mode, H=H) if H != 2688 else spec_for_mode(mode)
        print(f"\n=== {mode}  H={spec.H} ===")
        all_ok &= check(spec)
    print("\n" + "=" * 60)
    print("DIFFSIM VERIFICATION PASSED" if all_ok else "DIFFSIM VERIFICATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
