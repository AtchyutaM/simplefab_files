"""
Action sources for the tensor sim.

A policy has signature `policy(state, key) -> [B, 4]` in {0, 1}, and is called on
the post-`inject`, pre-completion state — the same point at which the reference
`sim.commander_decide` is invoked inside `run_step`.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from tensorfab.tensor_sim import LineState, TensorLine


def commander(line: TensorLine) -> Callable[[LineState, jnp.ndarray], jnp.ndarray]:
    """
    The greedy baseline from `sim.commander_decide`: run every machine that is
    idle and has enough input.

    Note this is NOT the same predicate as `TensorLine.action_mask`. The
    commander runs inside `run_step` and sees the pre-completion `machine.busy`
    flag, so a machine finishing this tick still reads as busy and is not
    restarted. The env's action mask is computed one step earlier and treats
    such a machine as free. Both are faithful to the reference; they simply
    observe at different instants.
    """

    def policy(s: LineState, key: jnp.ndarray) -> jnp.ndarray:
        idle = s.reg.sum(axis=-1) == 0
        canrun = s.q >= line._take
        return (idle & canrun).astype(line.dtype)

    return policy


def bernoulli(p: float = 0.5) -> Callable[[LineState, jnp.ndarray], jnp.ndarray]:
    """Request each machine independently with probability p, ignoring the mask.
    Exercises the invalid-action path heavily — useful for verification and for
    a worst-case throughput measurement."""

    def policy(s: LineState, key: jnp.ndarray) -> jnp.ndarray:
        return (jax.random.uniform(key, s.q.shape) < p).astype(s.q.dtype)

    return policy


def masked_bernoulli(
    line: TensorLine, p: float = 0.5
) -> Callable[[LineState, jnp.ndarray], jnp.ndarray]:
    """Bernoulli, but only where a start is actually possible. Closer to what a
    trained masked policy does, so it keeps the line producing and gives a more
    representative throughput number."""

    def policy(s: LineState, key: jnp.ndarray) -> jnp.ndarray:
        idle = s.reg.sum(axis=-1) == 0
        canrun = s.q >= line._take
        draw = jax.random.uniform(key, s.q.shape) < p
        return (idle & canrun & draw).astype(line.dtype)

    return policy
