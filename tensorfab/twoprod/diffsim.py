"""
Differentiable two-product line — the surrogate-gradient path.

--------------------------------------------------------------------------------
What is new relative to the single-product relaxation
--------------------------------------------------------------------------------
The single-product action was binary (run / idle), so a sigmoid straight-through
sufficed. Here every tool faces a **three-way categorical** choice —
{idle, run P0, run P1} — so the relaxation becomes a **softmax** straight-through
over three logits. That is a genuine redesign, not a port: with a sigmoid the
gradient answers "should I run harder?", whereas here it has to answer "which
product is worth the tool time?", which is the decision the two-product problem
is actually about.

Parameterisation: `theta[B, H, M, 3]`, slot 0 = idle, slot p+1 = product p.

--------------------------------------------------------------------------------
Where the gradient flows, and where it does not
--------------------------------------------------------------------------------
Three gates, as before:

  (a) the categorical action          -> softmax straight-through
  (b) input availability  1[q >= take] -> sigmoid straight-through
  (c) tool availability                -> `1 - reg.sum()` is exact AND linear,
                                          because a tool holds at most one job.
                                          No relaxation, no bias.

One genuine limitation, stated plainly. A job's register slot is
`setup_time[m, lastp, p] + pt[m, p]`, and `lastp` is discrete state. The one-hot
that writes into that slot is a hard comparison whose derivative is zero, so the
gradient does **not** see "switching product costs an extra tick of tool time".
It is held under `stop_gradient`.

That omission is small and the important half survives: the changeover **cost**
enters as `setup_cost * start * changed`, where `changed` is a coefficient on
`start`, so the $20 penalty does reach the action gradient. Since setup time is
1 tick against 2-20 ticks of processing while setup cost is $20 against a ~$56-72
unit margin, the differentiable half carries most of the changeover signal.

--------------------------------------------------------------------------------
Why there is no "soft" mode
--------------------------------------------------------------------------------
The single-product study ran a fully relaxed forward pass (`grad-soft`) alongside
straight-through on every demand pattern and pipeline depth. It lost to the
greedy commander on most configurations and collapsed outright at depth
(-$11,794 at 8 tools, -$21,996 at 12). Optimising a smoothed system optimises the
wrong system. It is not reimplemented here; `mode` is "hard" or "st".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tensorfab.twoprod.spec import LineSpec2P
from tensorfab.twoprod.tensor_sim import State2P


@dataclass(frozen=True)
class Relax2P:
    mode: str = "st"       # "hard" | "st"
    tau_q: float = 1.0     # temperature of the input-availability gate (parts)
    tau_a: float = 1.0     # temperature of the action softmax (logits)

    def __post_init__(self) -> None:
        if self.mode not in ("hard", "st"):
            raise ValueError(f"mode must be hard|st, got {self.mode!r}")


def straight_through(hard: jnp.ndarray, soft: jnp.ndarray) -> jnp.ndarray:
    """
    Forward exactly `hard`, backward the derivative of `soft`.

    Grouping the difference first is load-bearing: `hard + soft - sg(soft)`
    parses as `(hard + soft) - soft`, which is not exact in float32 and silently
    invented $230 of profit in the single-product line before it was caught.
    """
    return hard + (soft - lax.stop_gradient(soft))


class DiffLine2P:
    """Differentiable rollout driven by an open-loop tape `theta[B, H, M, 3]`."""

    def __init__(self, spec: LineSpec2P, relax: Relax2P = Relax2P(),
                 dtype: Any = jnp.float32):
        self.spec = spec
        self.relax = relax
        self.dtype = dtype
        M, P, L = spec.n_machines, spec.n_products, spec.reg_len

        self._take = jnp.asarray(spec.take, dtype=dtype)[:, None]
        self._make = jnp.asarray(spec.make, dtype=dtype)[:, None]
        self._pt = jnp.asarray(spec.pt, dtype=jnp.int32)
        self._setup_time = jnp.asarray(spec.setup_time, dtype=jnp.int32)

        route_mat = np.zeros((M, M), dtype=np.float64)
        to_fg = np.zeros((M,), dtype=np.float64)
        for m, dest in enumerate(np.asarray(spec.route)):
            if dest < 0:
                to_fg[m] = 1.0
            else:
                route_mat[m, int(dest)] = 1.0
        self._route_mat = jnp.asarray(route_mat, dtype=dtype)
        self._to_fg = jnp.asarray(to_fg, dtype=dtype)

        self._arrivals = jnp.asarray(spec.arrivals, dtype=dtype)
        self._demand = jnp.asarray(spec.demand, dtype=dtype)
        self._demand_lag = jnp.asarray(spec.demand_lag_cum, dtype=dtype)

        self._slots = jnp.arange(L, dtype=jnp.int32)
        self._pidx = jnp.arange(P, dtype=jnp.int32)
        self._midx = jnp.arange(M, dtype=jnp.int32)

        # prices, for the in-graph objective
        self._rev = jnp.asarray(spec.revenue, dtype=dtype)
        self._pcost = jnp.asarray(spec.prod_cost * spec.take[:, None], dtype=dtype)
        self._scost = jnp.asarray(spec.setup_cost, dtype=dtype)
        self._fg = jnp.asarray(spec.fg_holding, dtype=dtype)
        self._wip = jnp.asarray(spec.wip_holding, dtype=dtype)
        self._back = jnp.asarray(spec.backorder, dtype=dtype)

    # ------------------------------------------------------------------
    def init_state(self, batch: int) -> State2P:
        sp, dt = self.spec, self.dtype
        M, P, L = sp.n_machines, sp.n_products, sp.reg_len
        z_mp = jnp.zeros((batch, M, P), dtype=dt)
        z_p = jnp.zeros((batch, P), dtype=dt)
        return State2P(
            q=z_mp,
            reg=jnp.zeros((batch, M, P, L), dtype=dt),
            qfin=jnp.broadcast_to(jnp.asarray(sp.init_qfin, dtype=dt), (batch, P)).copy(),
            backlog=z_p,
            lastp=jnp.zeros((batch, M), dtype=jnp.int32),
            cum_ship=z_p, started=z_mp,
            setups=jnp.zeros((batch, M), dtype=dt),
            sum_qfin=z_p, sum_q=z_mp, sum_late=z_p,
            n_invalid=jnp.zeros((batch,), dtype=dt),
        )

    # ------------------------------------------------------------------
    def _gate_action(self, logit: jnp.ndarray) -> jnp.ndarray:
        """[B, M, 3] logits -> [B, M, P] product selection (idle slot dropped)."""
        soft = jax.nn.softmax(logit / self.relax.tau_a, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(logit, axis=-1), logit.shape[-1],
                              dtype=self.dtype)
        gated = hard if self.relax.mode == "hard" else straight_through(hard, soft)
        return gated[..., 1:]

    def _gate_canrun(self, q: jnp.ndarray) -> jnp.ndarray:
        soft = jax.nn.sigmoid((q - self._take + 0.5) / self.relax.tau_q)
        hard = (q >= self._take).astype(self.dtype)
        return hard if self.relax.mode == "hard" else straight_through(hard, soft)

    def _gate_idle(self, reg: jnp.ndarray) -> jnp.ndarray:
        """Exact and linear: a tool holds at most one job, so occupancy is 0 or 1."""
        occupied = reg.sum(axis=(-2, -1))
        hard = (occupied == 0).astype(self.dtype)
        return hard if self.relax.mode == "hard" else straight_through(hard, 1.0 - occupied)

    # ------------------------------------------------------------------
    def _tick(self, s: State2P, logit, arr, dem, dlag) -> State2P:
        # --- inject + match finished goods against demand ------------------
        q = s.q.at[:, 0, :].add(arr)
        backlog = s.backlog + dem
        matched = jnp.minimum(s.qfin, backlog)
        qfin = s.qfin - matched
        backlog = backlog - matched
        cum_ship = s.cum_ship + matched

        # --- completions ----------------------------------------------------
        done = s.reg[:, :, :, 0]
        reg = s.reg.at[:, :, :, 0].set(0.0)
        emit = self._make * done
        q = q + jnp.einsum("bmp,mj->bjp", emit, self._route_mat)
        newfin = jnp.einsum("bmp,m->bp", emit, self._to_fg)

        shipped = jnp.minimum(newfin, backlog)
        backlog = backlog - shipped
        cum_ship = cum_ship + shipped
        qfin = qfin + newfin - shipped

        completed = done.sum(axis=-1)
        newlast = (done * (self._pidx + 1)).sum(axis=-1)
        lastp = jnp.where(completed > 0.5,
                          lax.stop_gradient(newlast).astype(jnp.int32), s.lastp)

        # --- starts -----------------------------------------------------------
        action = self._gate_action(logit)                    # [B, M, P]
        ok = self._gate_idle(reg)[:, :, None] * self._gate_canrun(q)
        start = action * ok

        q = jnp.maximum(q - self._take * start, 0.0)

        # The write slot depends on `lastp`, which is discrete state; the
        # comparison below has zero derivative and is held constant. The
        # changeover COST below is differentiable and carries most of the signal.
        st = self._setup_time[self._midx[None, :, None], lastp[:, :, None],
                              self._pidx[None, None, :]]
        slot = lax.stop_gradient(st + self._pt[None, :, :])
        slot_oh = (self._slots[None, None, None, :] == slot[..., None]).astype(self.dtype)
        reg = reg + start[..., None] * slot_oh

        changed = (lastp[:, :, None] != (self._pidx + 1)[None, None, :]).astype(self.dtype)
        setups = s.setups + (start * changed).sum(axis=-1)

        late = jnp.maximum(dlag - cum_ship, 0.0)

        s = State2P(
            q=q, reg=reg, qfin=qfin, backlog=backlog, lastp=lastp,
            cum_ship=cum_ship, started=s.started + start, setups=setups,
            sum_qfin=s.sum_qfin + qfin, sum_q=s.sum_q + q,
            sum_late=s.sum_late + late,
            n_invalid=s.n_invalid + (action * (1.0 - ok)).sum(axis=(-2, -1)),
        )
        reg = jnp.concatenate([s.reg[..., 1:], jnp.zeros_like(s.reg[..., :1])], axis=-1)
        return s._replace(reg=reg)

    # ------------------------------------------------------------------
    def profit_jax(self, s: State2P) -> jnp.ndarray:
        return (
            (self._rev * s.cum_ship).sum(axis=-1)
            - (self._pcost * s.started).sum(axis=(-2, -1))
            - (self._scost * s.setups).sum(axis=-1)
            - (self._fg * s.sum_qfin).sum(axis=-1)
            - (self._wip * s.sum_q).sum(axis=(-2, -1))
            - (self._back * s.sum_late).sum(axis=-1)
        )

    def rollout_tape(self, theta: jnp.ndarray) -> Tuple[jnp.ndarray, State2P]:
        """theta: [B, H, M, 3] -> (profit [B], final state)."""
        def body(s, x):
            logit, arr, dem, dlag = x
            return self._tick(s, logit, arr, dem, dlag), None
        xs = (jnp.swapaxes(theta, 0, 1), self._arrivals, self._demand, self._demand_lag)
        final, _ = lax.scan(body, self.init_state(theta.shape[0]), xs)
        return self.profit_jax(final), final

    def profit_of(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self.rollout_tape(theta)[0]

    def value_and_grad(self, theta: jnp.ndarray):
        g = jax.grad(lambda th: self.profit_of(th).sum())(theta)
        return self.profit_of(theta), g


def hard_actions_2p(theta: jnp.ndarray) -> jnp.ndarray:
    """[B, H, M, 3] logits -> [H, B, M, P] integer tape for the exact simulator."""
    sel = jnp.argmax(theta, axis=-1)                       # [B, H, M], 0 = idle
    P = theta.shape[-1] - 1
    oh = jax.nn.one_hot(sel, P + 1, dtype=jnp.int32)[..., 1:]
    return jnp.swapaxes(oh, 0, 1)
