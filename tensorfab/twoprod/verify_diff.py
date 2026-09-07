"""
Verify the two-product relaxation is a faithful surrogate.

Three properties, the same three the single-product line is gated on:

  1. hard-mode forward == the exact integer simulator
     (same dynamics, written in float, with a softmax argmax instead of an
      explicit action tape)

  2. st-mode forward == hard-mode forward, EXACTLY
     Straight-through must not perturb the forward pass. This is a real float32
     risk, which is why the estimator is written `hard + (soft - sg(soft))`.

  3. hard-mode gradient == 0 everywhere
     The control: differentiating the true dynamics gives nothing, which is the
     entire reason a surrogate exists.

Plus a two-product-specific check: the softmax straight-through must produce a
gradient that distinguishes the two products. A relaxation that collapsed to
"run something" rather than "run P1 rather than P0" would pass 1-3 and still be
useless, so we report the gradient split across the three action slots.

Usage:  python -m tensorfab.twoprod.verify_diff
"""

from __future__ import annotations

import sys

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tensorfab.twoprod.diffsim import DiffLine2P, Relax2P, hard_actions_2p  # noqa: E402
from tensorfab.twoprod.spec import spec2p_for  # noqa: E402
from tensorfab.twoprod.tensor_sim import make_line_2p, profit_breakdown_2p  # noqa: E402


def check(spec, batch: int = 8, seed: int = 0) -> bool:
    rng = np.random.default_rng(seed)
    M, P = spec.n_machines, spec.n_products
    theta = jnp.asarray(rng.normal(0.0, 2.0, size=(batch, spec.H, M, P + 1)),
                        dtype=jnp.float32)

    exact = make_line_2p(spec)
    p_exact = profit_breakdown_2p(spec, exact.rollout(hard_actions_2p(theta)))["profit"]

    ok = True
    p_hard = np.asarray(DiffLine2P(spec, Relax2P(mode="hard")).profit_of(theta))
    d1 = float(np.max(np.abs(p_hard - p_exact)))
    ok1 = d1 < 0.5          # float32 accumulation over H ticks on ~$30k
    ok &= ok1
    print(f"  [{'PASS' if ok1 else 'FAIL'}] hard forward == exact integer sim    "
          f"max|d| = ${d1:.4f}")

    p_st = np.asarray(DiffLine2P(spec, Relax2P(mode="st")).profit_of(theta))
    d2 = float(np.max(np.abs(p_st - p_hard)))
    ok2 = d2 == 0.0
    ok &= ok2
    print(f"  [{'PASS' if ok2 else 'FAIL'}] st forward == hard forward           "
          f"max|d| = {d2:.1e}  (must be exactly 0)")

    _, g_hard = DiffLine2P(spec, Relax2P(mode="hard")).value_and_grad(theta)
    gmax = float(jnp.max(jnp.abs(g_hard)))
    ok3 = gmax == 0.0
    ok &= ok3
    print(f"  [{'PASS' if ok3 else 'FAIL'}] hard gradient == 0 (the control)     "
          f"max|g| = {gmax:.1e}")

    _, g_st = DiffLine2P(spec, Relax2P(mode="st")).value_and_grad(theta)
    finite = bool(jnp.all(jnp.isfinite(g_st)))
    ok &= finite
    per_slot = np.abs(np.asarray(g_st)).mean(axis=(0, 1, 2))   # [3]
    spread = float(per_slot[1:].max() / max(per_slot[1:].min(), 1e-12))
    discriminates = spread > 1.02
    ok &= discriminates
    print(f"         st gradient: max|g|={float(jnp.max(jnp.abs(g_st))):.4g}  finite={finite}")
    print(f"  [{'PASS' if discriminates else 'FAIL'}] gradient distinguishes the products   "
          f"mean|g| per slot idle/P0/P1 = "
          f"{per_slot[0]:.4g} / {per_slot[1]:.4g} / {per_slot[2]:.4g}  "
          f"(P0:P1 ratio {spread:.3f})")
    return ok


def main() -> int:
    print(device.describe())
    all_ok = True
    for alpha, util, mode in ((0.5, 0.92, "UNIFORM"), (0.9, 0.80, "UNIFORM")):
        spec = spec2p_for(mode, alpha=alpha, utilization=util)
        print(f"\n=== {mode}  alpha={alpha}  util={util}  "
              f"demand={spec.demand.sum(axis=0).tolist()} ===")
        all_ok &= check(spec)
    print("\n" + "=" * 62)
    print("2-PRODUCT DIFFSIM VERIFICATION PASSED" if all_ok
          else "2-PRODUCT DIFFSIM VERIFICATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
