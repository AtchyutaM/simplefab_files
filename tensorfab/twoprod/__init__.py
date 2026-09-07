"""
tensorfab.twoprod — everything for the TWO-product line, in one place.

    spec.py          LineSpec2P: static description built from simplefab.config
    tensor_sim.py    batched JAX simulator; product axis, product-tagged delay
                     register, per-tool changeover state
    verify_sim.py    equivalence gate vs the reference simplefab/sim.py
    optim.py         CP-SAT model with sequence-dependent changeovers
    verify_optim.py  cross-validation: a CP-SAT schedule replayed through the
                     tensor simulator must reproduce CP-SAT's own objective

The single-product stack lives one level up in `tensorfab/` and is untouched by
anything here.
"""

from tensorfab.twoprod.spec import LineSpec2P, spec2p_for, spec2p_from_common_config
from tensorfab.twoprod.tensor_sim import (
    State2P, TensorLine2P, make_line_2p, profit_breakdown_2p,
)

__all__ = [
    "LineSpec2P", "spec2p_for", "spec2p_from_common_config",
    "State2P", "TensorLine2P", "make_line_2p", "profit_breakdown_2p",
]
