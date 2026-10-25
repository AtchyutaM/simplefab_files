"""
diffsim — the line with *differentiable* gates.

`tensor_sim.py` is the exact reference: integer state, hard thresholds, and
therefore a derivative that is zero almost everywhere and undefined at the jumps.
Naive autodiff through it returns a gradient of exactly zero, which is correct
and useless. This module is where the surrogate lives.

--------------------------------------------------------------------------------
Where the discontinuities actually are
--------------------------------------------------------------------------------
Only three places in the tick. Everything else is linear or has a usable
subgradient:

  (a) the binary action                a ∈ {0, 1}
  (b) input availability               canrun = 1[q >= take]
  (c) machine availability             idle   = 1[reg.sum(-1) == 0]

`min(qfin, backlog)`, `min(newfin, backlog)` and `max(0, ...)` are NOT problems —
`jnp.minimum`/`jnp.maximum` have well-defined subgradients that route to whichever
argument is active, which is exactly the credit assignment we want.

(c) is nearly free: the one-job-at-a-time invariant means `reg.sum(-1) ∈ {0,1}`,
so `idle = 1 - reg.sum(-1)` is *exactly* correct AND linear. The gradient
d(idle)/d(reg) = -1 carries the "starting now blocks me for pt ticks" signal with
no relaxation error at all. Only (a) and (b) need surrogates.

--------------------------------------------------------------------------------
Three modes
--------------------------------------------------------------------------------
  "hard"  Exact forward, zero gradient. The control condition — run this to show
          that naive autodiff through the true dynamics gives nothing.

  "st"    Straight-through. Forward is bit-identical to "hard"; the backward pass
          substitutes a sigmoid for each hard gate. This is the design of record
          in docs/tensorized_line_report.md § 4.4: exact-forward, biased-backward.
          You cannot have exact gradients of a discontinuous function.

  "soft"  Fully relaxed forward: fractional jobs, fractional queue occupancy. The
          landscape is smooth and the gradient is a true gradient — but *of a
          different system*. Profit reported by a soft rollout is not achievable;
          it must be re-evaluated by rounding the actions and running the exact
          integer sim.

--------------------------------------------------------------------------------
Numerical note on the straight-through estimator
--------------------------------------------------------------------------------
Written as `hard + soft - stop_gradient(soft)`, NOT the more common
`soft + stop_gradient(hard - soft)`. The two are algebraically identical but the
second is not exact in float32: `0.7 + (1 - 0.7)` can evaluate to 0.99999994,
which silently destroys the "forward is exactly the hard sim" guarantee. In the
form used here `soft - stop_gradient(soft)` is exactly 0.0 by construction.
`verify_diffsim.py` asserts this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tensorfab.spec import LineSpec
from tensorfab.tensor_sim import LineState


@dataclass(frozen=True)
class Relax:
    """Relaxation settings for the differentiable line."""

    mode: str = "st"        # "hard" | "st" | "soft"
    tau_q: float = 1.0      # temperature of the input-availability gate (units: parts)
    tau_a: float = 1.0      # temperature of the action sigmoid (units: logits)

    def __post_init__(self) -> None:
        if self.mode not in ("hard", "st", "soft"):
            raise ValueError(f"mode must be hard|st|soft, got {self.mode!r}")


def straight_through(hard: jnp.ndarray, soft: jnp.ndarray) -> jnp.ndarray:
    """
    Forward exactly `hard`, backward the derivative of `soft`.

    The parenthesisation is load-bearing. `hard + soft - stop_gradient(soft)`
    parses as `(hard + soft) - soft`, which in float32 is NOT exactly `hard`:
    (1 + 0.7) - 0.7 evaluates to 0.99999994. Grouping the difference first makes
    `soft - stop_gradient(soft)` identically +0.0, so the result is exactly
    `hard`. verify_diffsim.py caught this; it cost ~$230 of phantom profit at
    H=2688 before the fix.
    """
    return hard + (soft - lax.stop_gradient(soft))


class DiffLine:
    """
    Differentiable rollout of the line, driven by an open-loop action tape.

    The tape is `theta[B, H, 4]` of real-valued logits. B independent tapes are
    optimised in parallel, which costs nothing extra (the batch axis already
    exists) and turns a single optimisation run into a distribution over random
    restarts — the only honest way to talk about local optima.
    """

    def __init__(self, spec: LineSpec, relax: Relax = Relax(), dtype: Any = jnp.float32):
        self.spec = spec
        self.relax = relax
        self.dtype = dtype

        L, M = spec.reg_len, spec.n_machines
        self._take = jnp.asarray(spec.take, dtype=dtype)
        self._make = jnp.asarray(spec.make, dtype=dtype)

        onehot = np.zeros((M, L), dtype=np.float64)
        onehot[np.arange(M), spec.pt] = 1.0
        self._slot_onehot = jnp.asarray(onehot, dtype=dtype)

        # Routing matrix, mirroring tensor_sim: R[m, j] = 1 when machine m's
        # completions land in queue j; to_fg[m] = 1 when they ship instead.
        # This replaced four hardcoded `q.at[:, k].add(emit[:, k-1])` lines that
        # silently assumed M == 4. At M = 8 or 12 the out-of-range indices were
        # dropped or clamped by XLA rather than raising, so the surrogate
        # optimised a different line than the one being measured.
        route_mat = np.zeros((M, M), dtype=np.float64)
        to_fg = np.zeros((M,), dtype=np.float64)
        for m, dest in enumerate(np.asarray(spec.route)):
            if dest < 0:
                to_fg[m] = 1.0
            else:
                if not (0 <= dest < M):
                    raise ValueError(f"route[{m}]={dest} out of range for M={M}")
                route_mat[m, int(dest)] = 1.0
        if to_fg.sum() == 0:
            raise ValueError("no machine ships to finished goods (route has no -1)")
        self._route_mat = jnp.asarray(route_mat, dtype=dtype)
        self._to_fg = jnp.asarray(to_fg, dtype=dtype)

        self._arrivals = jnp.asarray(spec.arrivals, dtype=dtype)
        self._demand = jnp.asarray(spec.demand, dtype=dtype)
        self._demand_lag = jnp.asarray(spec.demand_lag_cum, dtype=dtype)

        # Prices, for the in-graph objective.
        self._rev = dtype(spec.revenue_per_unit)
        self._pcost = jnp.asarray(spec.prod_cost_per_unit * spec.take, dtype=dtype)
        self._fg = dtype(spec.fg_holding)
        self._wip = jnp.asarray(spec.wip_holding, dtype=dtype)
        self._back = dtype(spec.backorder)

    # ------------------------------------------------------------------
    def init_state(self, batch: int) -> LineState:
        dt = self.dtype
        z1 = jnp.zeros((batch,), dtype=dt)
        z4 = jnp.zeros((batch, self.spec.n_machines), dtype=dt)
        return LineState(
            q=z4,
            reg=jnp.zeros((batch, self.spec.n_machines, self.spec.reg_len), dtype=dt),
            qfin=jnp.full((batch,), self.spec.init_qfin, dtype=dt),
            backlog=z1,
            cum_ship=z1,
            started=z4,
            sum_qfin=z1,
            sum_q=z4,
            sum_late=z1,
            n_invalid=z1,
        )

    # ------------------------------------------------------------------
    # the three gates
    # ------------------------------------------------------------------
    def _gate_action(self, logit: jnp.ndarray) -> jnp.ndarray:
        """(a) binary action from a real-valued logit."""
        soft = jax.nn.sigmoid(logit / self.relax.tau_a)
        hard = (logit > 0).astype(self.dtype)
        if self.relax.mode == "hard":
            return hard
        if self.relax.mode == "st":
            return straight_through(hard, soft)
        return soft

    def _gate_canrun(self, q: jnp.ndarray) -> jnp.ndarray:
        """(b) input availability, 1[q >= take]."""
        # +0.5 centres the sigmoid between the integers take-1 and take, so the
        # surrogate is symmetric about the true threshold.
        soft = jax.nn.sigmoid((q - self._take + 0.5) / self.relax.tau_q)
        hard = (q >= self._take).astype(self.dtype)
        if self.relax.mode == "hard":
            return hard
        if self.relax.mode == "st":
            return straight_through(hard, soft)
        return soft

    def _gate_idle(self, reg: jnp.ndarray) -> jnp.ndarray:
        """
        (c) machine availability. Exact and linear whenever reg.sum(-1) ∈ {0,1},
        which the one-job-at-a-time invariant guarantees in hard/st mode. No
        surrogate needed, so none is applied — this gate contributes zero
        relaxation bias.
        """
        occupied = reg.sum(axis=-1)
        if self.relax.mode == "soft":
            return jnp.clip(1.0 - occupied, 0.0, 1.0)
        hard = (occupied == 0).astype(self.dtype)
        if self.relax.mode == "hard":
            return hard
        return straight_through(hard, 1.0 - occupied)

    # ------------------------------------------------------------------
    # one tick
    # ------------------------------------------------------------------
    def _tick(self, s: LineState, logit: jnp.ndarray, arr, dem, dlag) -> LineState:
        # --- inject + instant FG/demand match ---------------------------
        q = s.q.at[:, 0].add(arr)
        backlog = s.backlog + dem
        matched = jnp.minimum(s.qfin, backlog)
        qfin = s.qfin - matched
        backlog = backlog - matched
        cum_ship = s.cum_ship + matched

        # --- completions -------------------------------------------------
        done = s.reg[:, :, 0]
        reg = s.reg.at[:, :, 0].set(0.0)
        emit = self._make * done
        q = q + emit @ self._route_mat
        newfin = emit @ self._to_fg

        shipped = jnp.minimum(newfin, backlog)
        backlog = backlog - shipped
        cum_ship = cum_ship + shipped
        qfin = qfin + newfin - shipped

        # --- starts (all three gates) ------------------------------------
        action = self._gate_action(logit)
        ok = self._gate_idle(reg) * self._gate_canrun(q)
        start = action * ok

        # max(.,0) is a no-op in hard/st mode (the gate guarantees q >= take) and
        # a safety clamp in soft mode where fractional starts can overshoot.
        q = jnp.maximum(q - self._take * start, 0.0)
        reg = reg + start[:, :, None] * self._slot_onehot

        # --- cost accruals -------------------------------------------------
        late = jnp.maximum(dlag - cum_ship, 0.0)

        s = LineState(
            q=q,
            reg=reg,
            qfin=qfin,
            backlog=backlog,
            cum_ship=cum_ship,
            started=s.started + start,
            sum_qfin=s.sum_qfin + qfin,
            sum_q=s.sum_q + q,
            sum_late=s.sum_late + late,
            n_invalid=s.n_invalid + (action * (1.0 - ok)).sum(axis=-1),
        )
        # --- shift ---------------------------------------------------------
        reg = jnp.concatenate([s.reg[..., 1:], jnp.zeros_like(s.reg[..., :1])], axis=-1)
        return s._replace(reg=reg)

    # ------------------------------------------------------------------
    def profit_jax(self, s: LineState) -> jnp.ndarray:
        return (
            self._rev * s.cum_ship
            - (self._pcost * s.started).sum(axis=-1)
            - self._fg * s.sum_qfin
            - (self._wip * s.sum_q).sum(axis=-1)
            - self._back * s.sum_late
        )

    def rollout_tape(self, theta: jnp.ndarray) -> Tuple[jnp.ndarray, LineState]:
        """
        theta: [B, H, 4] logits. Returns (profit [B], final state).
        """
        b = theta.shape[0]

        def body(s, x):
            logit, arr, dem, dlag = x
            return self._tick(s, logit, arr, dem, dlag), None

        xs = (jnp.swapaxes(theta, 0, 1), self._arrivals, self._demand, self._demand_lag)
        final, _ = lax.scan(body, self.init_state(b), xs)
        return self.profit_jax(final), final

    def profit_of(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self.rollout_tape(theta)[0]

    # ------------------------------------------------------------------
    def value_and_grad(self, theta: jnp.ndarray):
        """
        (profit [B], dprofit/dtheta [B, H, 4]).

        Summing over the batch is what makes the B tapes independent: tape i's
        profit does not depend on tape j's parameters, so d(Σ profit)/d(theta_i)
        is exactly d(profit_i)/d(theta_i).
        """
        f = lambda th: self.profit_of(th).sum()
        grad = jax.grad(f)(theta)
        return self.profit_of(theta), grad


def hard_actions(theta: jnp.ndarray) -> jnp.ndarray:
    """Round a tape to the {0,1} actions the exact integer sim will execute."""
    return (theta > 0).astype(jnp.int32)
