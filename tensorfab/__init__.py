"""
tensorfab — a batched, differentiable JAX reimplementation of the SimpleFab line.

Phase 1 scope: exact-forward tensor simulator + equivalence proof against
`simplefab_1p/sim.py` + throughput benchmark. Nothing here is learned; this is
the same dynamics, rewritten as array arithmetic.

See tensorfab/README.md for full context and the Windows/CUDA port guide.
"""

from tensorfab.spec import LineSpec, spec_from_common_config, spec_for_mode
from tensorfab.tensor_sim import LineState, TensorLine, make_line, profit_breakdown

__all__ = [
    "LineSpec",
    "spec_from_common_config",
    "spec_for_mode",
    "LineState",
    "TensorLine",
    "make_line",
    "profit_breakdown",
]
