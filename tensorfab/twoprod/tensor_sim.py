"""
tensor_sim_2p — the TWO-product line as a batched JAX program.

Same discipline as `tensor_sim.py`: exact forward, integer state, one `lax.scan`
over the horizon. Three things are genuinely new.

1. **A product axis on everything.** `q[B, M, P]`, `qfin[B, P]`, `backlog[B, P]`.

2. **The delay register carries a product tag.** `reg[B, M, P, L]`, where
   `reg[b, m, p, k] == 1` means machine m is running product p and completes in k
   ticks. A tool runs one job at a time, so `reg.sum((-2, -1)) <= 1` — which
   makes "busy" free again, and makes the running product recoverable from the
   register itself rather than a separate variable.

   The register is longer than the 1-product case because a job occupies its tool
   for **setup + processing** ticks, not just processing.

3. **Changeover state.** `lastp[B, M]` holds the machine's last *completed*
   product, encoded 0 = never run, p+1 = product p. This mirrors the reference's
   `last_finished_product`, and the choice of *completed* rather than *started*
   is load-bearing: `Machine.process` reads it to size the setup, and `run_step`
   reads the same value to decide whether to charge the $20.

   Cold start counts as a changeover in the reference (`prev != k` is true when
   prev is None), so it is charged here too.

Batch tools have zero setup time and zero setup cost in the config, so tracking
`lastp` uniformly for all machines costs nothing and keeps the tick branch-free.

The tick order matches `simplefab/sim.py` exactly:
    inject -> match finished goods against demand -> completions -> starts ->
    accrue costs -> shift the register
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tensorfab.twoprod.spec import LineSpec2P


class State2P(NamedTuple):
    """Carry for the tick loop. Leading batch dimension B throughout."""

    q: jnp.ndarray          # [B, M, P] queue counts
    reg: jnp.ndarray        # [B, M, P, L] delay register, product-tagged
    qfin: jnp.ndarray       # [B, P] finished goods
    backlog: jnp.ndarray    # [B, P] outstanding demand
    lastp: jnp.ndarray      # [B, M] last COMPLETED product; 0 = never run
    cum_ship: jnp.ndarray   # [B, P]
    started: jnp.ndarray    # [B, M, P] cumulative job starts
    setups: jnp.ndarray     # [B, M] cumulative changeovers
    sum_qfin: jnp.ndarray   # [B, P]
    sum_q: jnp.ndarray      # [B, M, P]
    sum_late: jnp.ndarray   # [B, P]
    n_invalid: jnp.ndarray  # [B]


class Trace2P(NamedTuple):
    q: jnp.ndarray          # [H, B, M, P]
    qfin: jnp.ndarray       # [H, B, P]
    backlog: jnp.ndarray    # [H, B, P]
    cum_ship: jnp.ndarray   # [H, B, P]
    busy: jnp.ndarray       # [H, B, M]
    rem: jnp.ndarray        # [H, B, M] ticks to completion, 0 if idle
    curp: jnp.ndarray       # [H, B, M] product in the tool; 0 = idle, p+1 = product p
    lastp: jnp.ndarray      # [H, B, M]
    started: jnp.ndarray    # [H, B, M, P]
    setups: jnp.ndarray     # [H, B, M]
    setup_cost: jnp.ndarray  # [H, B] cumulative changeover cost
    sum_qfin: jnp.ndarray   # [H, B, P]
    sum_q: jnp.ndarray      # [H, B, M, P]
    sum_late: jnp.ndarray   # [H, B, P]
    n_invalid: jnp.ndarray  # [H, B]


class TensorLine2P:
    def __init__(self, spec: LineSpec2P, dtype: Any = jnp.int32):
        self.spec = spec
        self.dtype = dtype
        M, P, L = spec.n_machines, spec.n_products, spec.reg_len

        self._take = jnp.asarray(spec.take, dtype=dtype)[:, None]        # [M, 1]
        self._make = jnp.asarray(spec.make, dtype=dtype)[:, None]        # [M, 1]
        self._pt = jnp.asarray(spec.pt, dtype=jnp.int32)                 # [M, P]
        self._setup_time = jnp.asarray(spec.setup_time, dtype=jnp.int32)  # [M, P+1, P]

        route_mat = np.zeros((M, M), dtype=np.int32)
        to_fg = np.zeros((M,), dtype=np.int32)
        for m, dest in enumerate(np.asarray(spec.route)):
            if dest < 0:
                to_fg[m] = 1
            else:
                route_mat[m, int(dest)] = 1
        if to_fg.sum() == 0:
            raise ValueError("no machine ships to finished goods")
        self._setup_cost = jnp.asarray(spec.setup_cost, dtype=jnp.float32)  # [M]
        self._route_mat = jnp.asarray(route_mat, dtype=dtype)            # [M, M]
        self._to_fg = jnp.asarray(to_fg, dtype=dtype)                    # [M]

        self._arrivals = jnp.asarray(spec.arrivals, dtype=dtype)         # [H, P]
        self._demand = jnp.asarray(spec.demand, dtype=dtype)             # [H, P]
        self._demand_lag = jnp.asarray(spec.demand_lag_cum, dtype=dtype)  # [H, P]

        self._slots = jnp.arange(L, dtype=jnp.int32)
        self._pidx = jnp.arange(P, dtype=jnp.int32)
        self._midx = jnp.arange(M, dtype=jnp.int32)

        self._rollout = jax.jit(lambda s, a: self._scan(s, a, trace=False))
        self._rollout_tr = jax.jit(lambda s, a: self._scan(s, a, trace=True))

    # ------------------------------------------------------------------
    def init_state(self, batch: int = 1) -> State2P:
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
            cum_ship=z_p,
            started=z_mp,
            setups=jnp.zeros((batch, M), dtype=dt),
            sum_qfin=z_p,
            sum_q=z_mp,
            sum_late=z_p,
            n_invalid=jnp.zeros((batch,), dtype=dt),
        )

    # ------------------------------------------------------------------
    def inject(self, s: State2P, arr: jnp.ndarray, dem: jnp.ndarray) -> State2P:
        """Arrivals into q0, demand into the backlog, then match finished goods
        against demand per product — the loop added to the reference on
        2026-08-22 without which finished inventory was stranded forever."""
        q = s.q.at[:, 0, :].add(arr.astype(self.dtype))
        backlog = s.backlog + dem.astype(self.dtype)
        matched = jnp.minimum(s.qfin, backlog)
        return s._replace(q=q, qfin=s.qfin - matched, backlog=backlog - matched,
                          cum_ship=s.cum_ship + matched)

    def advance(self, s: State2P, action: jnp.ndarray, dlag: jnp.ndarray) -> State2P:
        """
        action: [B, M, P] in {0,1}, at most one product selected per machine.
        """
        dt = self.dtype
        action = action.astype(dt)

        # --- completions -------------------------------------------------
        done = s.reg[:, :, :, 0]                              # [B, M, P]
        reg = s.reg.at[:, :, :, 0].set(0)
        emit = self._make * done                              # [B, M, P]

        q = s.q + jnp.einsum("bmp,mj->bjp", emit, self._route_mat)
        newfin = jnp.einsum("bmp,m->bp", emit, self._to_fg)   # [B, P]

        shipped = jnp.minimum(newfin, s.backlog)
        backlog = s.backlog - shipped
        cum_ship = s.cum_ship + shipped
        qfin = s.qfin + newfin - shipped

        # A completion updates the tool's last-finished product.
        completed = done.sum(axis=-1)                                     # [B, M]
        newlast = (done * (self._pidx + 1)).sum(axis=-1).astype(jnp.int32)
        lastp = jnp.where(completed > 0, newlast, s.lastp)                # [B, M]

        # --- starts --------------------------------------------------------
        idle = reg.sum(axis=(-2, -1)) == 0                    # [B, M]
        canrun = q >= self._take                              # [B, M, P]
        ok = (idle[:, :, None] & canrun).astype(dt)
        start = action * ok
        invalid = action * (1 - ok)

        q = q - self._take * start

        # setup ticks for this (machine, last-completed, target) triple
        st = self._setup_time[self._midx[None, :, None],
                              lastp[:, :, None],
                              self._pidx[None, None, :]]      # [B, M, P]
        slot = st + self._pt[None, :, :]                      # [B, M, P]
        slot_oh = (self._slots[None, None, None, :] == slot[..., None]).astype(dt)
        reg = reg + start[..., None] * slot_oh

        # Changeover is charged whenever the started product differs from the
        # last COMPLETED one — cold start (lastp == 0) always differs.
        changed = (lastp[:, :, None] != (self._pidx + 1)[None, None, :]).astype(dt)
        setups = s.setups + (start * changed).sum(axis=-1)

        # --- cost accruals ----------------------------------------------------
        late = jnp.maximum(dlag.astype(dt) - cum_ship, 0)

        return State2P(
            q=q, reg=reg, qfin=qfin, backlog=backlog, lastp=lastp,
            cum_ship=cum_ship,
            started=s.started + start,
            setups=setups,
            sum_qfin=s.sum_qfin + qfin,
            sum_q=s.sum_q + q,
            sum_late=s.sum_late + late,
            n_invalid=s.n_invalid + invalid.sum(axis=(-2, -1)),
        )

    @staticmethod
    def shift(s: State2P) -> State2P:
        reg = jnp.concatenate([s.reg[..., 1:], jnp.zeros_like(s.reg[..., :1])], axis=-1)
        return s._replace(reg=reg)

    # ------------------------------------------------------------------
    def action_mask(self, s: State2P) -> jnp.ndarray:
        """[B, M, P] — may machine m start product p? Mirrors the reference mask,
        which is computed before arrivals and completions."""
        free = s.reg[..., 1:].sum(axis=(-2, -1)) == 0
        return free[:, :, None] & (s.q >= self._take)

    # ------------------------------------------------------------------
    def _snapshot(self, s: State2P) -> Trace2P:
        occ = s.reg.sum(axis=(-2, -1))                      # [B, M]
        busy = (occ > 0).astype(jnp.int32)
        # slot index of the in-flight job, and which product it is
        per_slot = s.reg.sum(axis=2)                        # [B, M, L]
        rem = jnp.argmax(per_slot, axis=-1).astype(jnp.int32) * busy
        per_prod = s.reg.sum(axis=-1)                       # [B, M, P]
        curp = ((per_prod * (self._pidx + 1)).sum(axis=-1)).astype(jnp.int32) * busy
        return Trace2P(
            q=s.q, qfin=s.qfin, backlog=s.backlog, cum_ship=s.cum_ship,
            busy=busy, rem=rem, curp=curp, lastp=s.lastp,
            started=s.started, setups=s.setups,
            setup_cost=(s.setups.astype(jnp.float32) * self._setup_cost).sum(axis=-1),
            sum_qfin=s.sum_qfin,
            sum_q=s.sum_q, sum_late=s.sum_late, n_invalid=s.n_invalid,
        )

    def _scan(self, s0: State2P, actions: jnp.ndarray, trace: bool):
        def body(s, x):
            act, arr, dem, dlag = x
            s = self.inject(s, arr, dem)
            s = self.advance(s, act, dlag)
            out = self._snapshot(s) if trace else None
            return self.shift(s), out

        xs = (actions, self._arrivals, self._demand, self._demand_lag)
        final, ys = lax.scan(body, s0, xs)
        return (final, ys) if trace else final

    def rollout(self, actions: jnp.ndarray) -> State2P:
        """actions: [H, B, M, P]."""
        return self._rollout(self.init_state(actions.shape[1]), actions)

    def rollout_trace(self, actions: jnp.ndarray) -> Tuple[State2P, Trace2P]:
        return self._rollout_tr(self.init_state(actions.shape[1]), actions)

    def compile_policy_rollout(self, policy: Callable, trace: bool = False):
        def scan_policy(s0, keys):
            def body(s, x):
                arr, dem, dlag, key = x
                s = self.inject(s, arr, dem)
                s = self.advance(s, policy(self, s, key), dlag)
                out = self._snapshot(s) if trace else None
                return self.shift(s), out
            xs = (self._arrivals, self._demand, self._demand_lag, keys)
            final, ys = lax.scan(body, s0, xs)
            return (final, ys) if trace else final
        return jax.jit(scan_policy)

    def rollout_policy(self, policy: Callable, batch: int = 1,
                       key: Optional[jnp.ndarray] = None, trace: bool = False):
        keys = (jax.random.split(key, self.spec.H) if key is not None
                else jnp.zeros((self.spec.H, 2), dtype=jnp.uint32))
        return self.compile_policy_rollout(policy, trace)(self.init_state(batch), keys)


def profit_breakdown_2p(spec: LineSpec2P, s: State2P) -> dict:
    """Exact profit from the integer accumulators, in float64 on the host."""
    cum_ship = np.asarray(s.cum_ship, dtype=np.int64)      # [B, P]
    started = np.asarray(s.started, dtype=np.int64)        # [B, M, P]
    setups = np.asarray(s.setups, dtype=np.int64)          # [B, M]
    sum_qfin = np.asarray(s.sum_qfin, dtype=np.int64)
    sum_q = np.asarray(s.sum_q, dtype=np.int64)
    sum_late = np.asarray(s.sum_late, dtype=np.int64)

    revenue = (cum_ship * spec.revenue).sum(axis=-1)
    production = (started * spec.take[None, :, None] * spec.prod_cost[None]).sum(axis=(1, 2))
    setup = (setups * spec.setup_cost[None]).sum(axis=-1)
    inventory = (sum_qfin * spec.fg_holding).sum(axis=-1)
    wip = (sum_q * spec.wip_holding[None]).sum(axis=(1, 2))
    backorder = (sum_late * spec.backorder).sum(axis=-1)
    return {
        "revenue": revenue, "production": production, "setup": setup,
        "inventory": inventory, "wip": wip, "backorder": backorder,
        "profit": revenue - (production + setup + inventory + wip + backorder),
        "units_shipped": cum_ship,
        "changeovers": setups,
    }


def make_line_2p(spec: LineSpec2P, dtype: Any = jnp.int32) -> TensorLine2P:
    return TensorLine2P(spec, dtype=dtype)
