"""
tensor_sim — the 4-machine SimpleFab line as a batched, differentiable JAX program.

This is `simplefab_1p/sim.py` re-expressed as fixed-size state tensors updated by
pure array arithmetic, carrying a leading batch dimension B so that B trajectories
advance in lockstep. It is NOT a neural network: nothing is learned, there are no
weights, and the forward pass is exact.

--------------------------------------------------------------------------------
The one-tick contract (must stay in lockstep with sim.py's `run_step`)
--------------------------------------------------------------------------------
`run_step(t)` in the reference simulator does, in this exact order:

  1. inject arrivals into q0 and demand into the demand queue
  2. instantly match any finished-goods inventory against any outstanding demand
     (the `while queue_fin and demand` loop) -> revenue
  3. completions, machine by machine: M0->q1 (x4), M1->q2, M2->q3 (x4),
     M3 -> ship if demand outstanding else store in queue_fin
  4. starts, machine by machine: if requested but busy/starved -> count invalid
     and do nothing; else consume input, charge production, set finish_time
  5. accrue FG holding, WIP holding, and backorder costs on the END-OF-TICK state

Steps 3 and 4 are separated, which means a machine completing at tick t can be
restarted at tick t, and a unit deposited into q1 by M0 at tick t is available to
M1 at tick t. We preserve that.

The policy/commander observes the state AFTER step 2 but BEFORE step 3, which is
why `step` is split into `inject` and `advance`.

--------------------------------------------------------------------------------
State representation
--------------------------------------------------------------------------------
The reference sim stores each queue as a Python list of arrival timestamps. Only
counts and countdowns affect the dynamics, so the entire state becomes:

  q        [B, 4]      queue counts q0..q3
  reg      [B, 4, L]   delay register; reg[b,m,k] == 1 means machine m has a job
                       completing in k ticks. L = max(pt) + 1 = 17.
  qfin     [B]         finished-goods inventory
  backlog  [B]         outstanding (unshipped) demand

The delay register replaces the (busy, finish_time) pair. A start at tick t writes
a 1 into slot pt; the register shifts left by one at the end of every tick; a job
reaching slot 0 completes. "Busy" is just `reg.sum(-1) > 0`, which enforces the
one-job-at-a-time capacity constraint for free.

Why a register and not an integer countdown: `done = reg[..., 0]` is a pure
*selection*, so in Phase 2b the gradient flows from a start decision straight
through to its completion event. An integer countdown would need `rem == 1`,
a comparison whose derivative is zero almost everywhere.

Backorder aging never needs per-unit timestamps: the demand queue is FIFO, so
    late(t) = max(0, D(t - demand_interval) - cumulative_shipped(t))
where D is cumulative arrived demand. Same accounting as the CP-SAT model.

--------------------------------------------------------------------------------
Costs
--------------------------------------------------------------------------------
We do NOT accumulate dollars per tick. We accumulate the underlying INTEGER
quantities (units shipped, starts per machine, Σqfin, Σq_i, Σlate) and multiply by
prices once at the end. This is exactly equivalent, avoids 2688 rounding steps of
float error, keeps the whole carry integer-valued, and makes profit a linear
function of the accumulators — convenient for the pathwise gradient work later.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tensorfab.spec import LineSpec


class LineState(NamedTuple):
    """Carry for the tick loop. All fields share a leading batch dimension B."""

    q: jnp.ndarray          # [B, 4]  queue counts q0..q3
    reg: jnp.ndarray        # [B, 4, L] delay register
    qfin: jnp.ndarray       # [B]     finished-goods inventory
    backlog: jnp.ndarray    # [B]     outstanding demand
    cum_ship: jnp.ndarray   # [B]     cumulative units shipped
    started: jnp.ndarray    # [B, 4]  cumulative job starts per machine
    sum_qfin: jnp.ndarray   # [B]     Σ_t qfin(t)      -> FG holding cost
    sum_q: jnp.ndarray      # [B, 4]  Σ_t q_i(t)       -> WIP holding cost
    sum_late: jnp.ndarray   # [B]     Σ_t late(t)      -> backorder cost
    n_invalid: jnp.ndarray  # [B]     cumulative invalid action requests


class Trace(NamedTuple):
    """Per-tick snapshot, emitted only by the tracing rollout (verification)."""

    q: jnp.ndarray          # [H, B, 4]
    qfin: jnp.ndarray       # [H, B]
    backlog: jnp.ndarray    # [H, B]
    cum_ship: jnp.ndarray   # [H, B]
    busy: jnp.ndarray       # [H, B, 4]  1 if a job is in flight at end of tick
    rem: jnp.ndarray        # [H, B, 4]  finish_time - t at end of tick (0 if idle)
    started: jnp.ndarray    # [H, B, 4]
    sum_qfin: jnp.ndarray   # [H, B]
    sum_q: jnp.ndarray      # [H, B, 4]
    sum_late: jnp.ndarray   # [H, B]
    n_invalid: jnp.ndarray  # [H, B]


class TensorLine:
    """
    A compiled tensor simulator bound to one LineSpec.

    The spec's arrays are closed over, so XLA constant-folds them. Build one of
    these per (spec, dtype) pair and reuse it; each new pair triggers a
    recompilation.
    """

    def __init__(self, spec: LineSpec, dtype: Any = jnp.int32):
        self.spec = spec
        self.dtype = dtype

        L = spec.reg_len
        M = spec.n_machines

        self._take = jnp.asarray(spec.take, dtype=dtype)          # [4]
        self._make = jnp.asarray(spec.make, dtype=dtype)          # [4]

        # slot_onehot[m, pt[m]] = 1 — where a start on machine m is written.
        onehot = np.zeros((M, L), dtype=np.int32)
        onehot[np.arange(M), spec.pt] = 1
        self._slot_onehot = jnp.asarray(onehot, dtype=dtype)      # [4, L]

        self._arrivals = jnp.asarray(spec.arrivals, dtype=dtype)          # [H]
        self._demand = jnp.asarray(spec.demand, dtype=dtype)              # [H]
        self._demand_lag = jnp.asarray(spec.demand_lag_cum, dtype=dtype)  # [H]

        # Compiled entry points. `trace` is baked in at trace time so the fast
        # path emits nothing per tick (no [H, B, ...] buffers).
        self._rollout_actions = jax.jit(
            lambda s, a: self._scan_actions(s, a, trace=False)
        )
        self._rollout_actions_trace = jax.jit(
            lambda s, a: self._scan_actions(s, a, trace=True)
        )

    # ------------------------------------------------------------------
    # initial state
    # ------------------------------------------------------------------
    def init_state(self, batch: int = 1) -> LineState:
        spec, dt = self.spec, self.dtype
        z1 = jnp.zeros((batch,), dtype=dt)
        z4 = jnp.zeros((batch, spec.n_machines), dtype=dt)
        return LineState(
            q=z4,
            reg=jnp.zeros((batch, spec.n_machines, spec.reg_len), dtype=dt),
            qfin=jnp.full((batch,), spec.init_qfin, dtype=dt),
            backlog=z1,
            cum_ship=z1,
            started=z4,
            sum_qfin=z1,
            sum_q=z4,
            sum_late=z1,
            n_invalid=z1,
        )

    # ------------------------------------------------------------------
    # one tick, split at the point where a policy observes the state
    # ------------------------------------------------------------------
    def inject(self, s: LineState, arrivals: jnp.ndarray, demand: jnp.ndarray) -> LineState:
        """
        sim.py steps 1-2: raw material into q0, demand into the backlog, then
        instantly match existing finished goods against outstanding demand.

        A policy sees the state returned by this function (the reference
        `commander_decide` is called at this exact point in `run_step`).
        """
        q = s.q.at[:, 0].add(arrivals.astype(self.dtype))
        backlog = s.backlog + demand.astype(self.dtype)

        matched = jnp.minimum(s.qfin, backlog)
        return s._replace(
            q=q,
            qfin=s.qfin - matched,
            backlog=backlog - matched,
            cum_ship=s.cum_ship + matched,
        )

    def advance(
        self, s: LineState, action: jnp.ndarray, demand_lag: jnp.ndarray
    ) -> LineState:
        """
        sim.py steps 3-5 plus the end-of-tick register shift.

        action: [B, 4] in {0, 1} — 1 means "start this machine now".
        """
        dt = self.dtype
        action = action.astype(dt)

        # --- 3. completions (slot 0 of the register) --------------------
        done = s.reg[:, :, 0]                                   # [B, 4]
        reg = s.reg.at[:, :, 0].set(0)

        emit = self._make * done                                # [B, 4] units emitted
        q = s.q
        q = q.at[:, 1].add(emit[:, 0])                          # M0 -> q1
        q = q.at[:, 2].add(emit[:, 1])                          # M1 -> q2
        q = q.at[:, 3].add(emit[:, 2])                          # M2 -> q3
        newfin = emit[:, 3]                                     # M3 -> ship or store

        # _ship_or_store: ship against outstanding demand, otherwise inventory.
        shipped = jnp.minimum(newfin, s.backlog)
        backlog = s.backlog - shipped
        cum_ship = s.cum_ship + shipped
        qfin = s.qfin + newfin - shipped

        # --- 4. starts ---------------------------------------------------
        idle = reg.sum(axis=-1) == 0                            # [B, 4]
        canrun = q >= self._take                                # [B, 4]
        ok = (idle & canrun).astype(dt)
        start = action * ok
        invalid = action * (1 - ok)                             # requested but refused

        q = q - self._take * start
        reg = reg + start[:, :, None] * self._slot_onehot       # write into slot pt

        # --- 5. end-of-tick cost accruals (on the post-start state) -------
        late = jnp.maximum(demand_lag.astype(dt) - cum_ship, 0)

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
            n_invalid=s.n_invalid + invalid.sum(axis=-1),
        )
        return s

    @staticmethod
    def shift(s: LineState) -> LineState:
        """
        Advance the delay register one tick: slot k -> slot k-1, slot L-1 -> 0.
        A fixed gather, so it is exactly differentiable.

        Kept separate from `advance` so that the pre-shift register is available
        for verification (pre-shift slot index == the reference sim's
        `finish_time - current_time`).
        """
        reg = jnp.concatenate(
            [s.reg[..., 1:], jnp.zeros_like(s.reg[..., :1])], axis=-1
        )
        return s._replace(reg=reg)

    def step(
        self,
        s: LineState,
        action: jnp.ndarray,
        arrivals: jnp.ndarray,
        demand: jnp.ndarray,
        demand_lag: jnp.ndarray,
    ) -> LineState:
        """One complete tick with an externally supplied action."""
        s = self.inject(s, arrivals, demand)
        s = self.advance(s, action, demand_lag)
        return self.shift(s)

    # ------------------------------------------------------------------
    # action mask
    # ------------------------------------------------------------------
    def action_mask(self, s: LineState) -> jnp.ndarray:
        """
        [B, 4] boolean: may machine m be started this tick?

        Matches `sim.compute_action_mask`, which the Gymnasium env calls BEFORE
        `run_step` — i.e. on the pre-arrival, pre-completion state. So this is
        evaluated on the raw carry, not on the post-`inject` state.

        The reference predicate is `not machine.busy and t >= finish_time`. The
        `not busy` conjunct dominates: `busy` stays True until `update()` fires
        *inside* `run_step`, so a machine completing at tick t is still busy when
        the mask is taken and is NOT offered as runnable — even though the sim
        would happily restart it at tick t. Hence `reg.sum(-1)`, including slot 0.
        (`t >= finish_time` is vacuous: for an idle machine finish_time is always
        in the past.)

        The mask is therefore conservative in two separate ways, both of which
        are real env behaviour and both of which `env_tensor.py` reproduces:
        a machine finishing this tick is hidden, and input arriving this tick
        (from arrivals or an upstream completion) is not visible.
        """
        free = s.reg.sum(axis=-1) == 0
        canrun = s.q >= self._take
        return free & canrun

    # ------------------------------------------------------------------
    # rollouts
    # ------------------------------------------------------------------
    def _scan_actions(self, s0: LineState, actions: jnp.ndarray, trace: bool):
        """actions: [H, B, 4]."""

        def body(s, x):
            action, arr, dem, dlag = x
            s = self.inject(s, arr, dem)
            s = self.advance(s, action, dlag)
            out = self._snapshot(s) if trace else None
            return self.shift(s), out

        xs = (actions, self._arrivals, self._demand, self._demand_lag)
        final, ys = lax.scan(body, s0, xs)
        return (final, ys) if trace else final

    def _scan_policy(
        self,
        s0: LineState,
        policy: Callable[[LineState, jnp.ndarray], jnp.ndarray],
        keys: Optional[jnp.ndarray],
        trace: bool,
    ):
        def body(s, x):
            arr, dem, dlag, key = x
            s = self.inject(s, arr, dem)
            action = policy(s, key)
            s = self.advance(s, action, dlag)
            out = self._snapshot(s) if trace else None
            return self.shift(s), out

        if keys is None:
            keys = jnp.zeros((self.spec.H, 2), dtype=jnp.uint32)
        xs = (self._arrivals, self._demand, self._demand_lag, keys)
        final, ys = lax.scan(body, s0, xs)
        return (final, ys) if trace else final

    @staticmethod
    def _snapshot(s: LineState) -> Trace:
        """End-of-tick snapshot, taken BEFORE the register shift so that the
        slot index equals the reference sim's `finish_time - current_time`."""
        busy = (s.reg.sum(axis=-1) > 0).astype(jnp.int32)
        rem = jnp.argmax(s.reg, axis=-1).astype(jnp.int32) * busy
        return Trace(
            q=s.q,
            qfin=s.qfin,
            backlog=s.backlog,
            cum_ship=s.cum_ship,
            busy=busy,
            rem=rem,
            started=s.started,
            sum_qfin=s.sum_qfin,
            sum_q=s.sum_q,
            sum_late=s.sum_late,
            n_invalid=s.n_invalid,
        )

    def rollout(self, actions: jnp.ndarray, batch: Optional[int] = None) -> LineState:
        """Run a full horizon from a fresh state. actions: [H, B, 4]."""
        b = batch if batch is not None else actions.shape[1]
        return self._rollout_actions(self.init_state(b), actions)

    def rollout_trace(
        self, actions: jnp.ndarray, batch: Optional[int] = None
    ) -> Tuple[LineState, Trace]:
        """As `rollout`, but also returns the per-tick Trace. Verification only —
        this allocates [H, B, ...] buffers."""
        b = batch if batch is not None else actions.shape[1]
        return self._rollout_actions_trace(self.init_state(b), actions)

    def compile_policy_rollout(
        self,
        policy: Callable[[LineState, jnp.ndarray], jnp.ndarray],
        trace: bool = False,
    ) -> Callable:
        """
        Return a jitted `(state, keys) -> final [, trace]` function.

        Build this ONCE and call it many times. `policy` is captured in the
        closure, so every call to this method produces a distinct function that
        JAX must compile from scratch — benchmarks and training loops must hoist
        it out of the loop.
        """
        return jax.jit(lambda s, k: self._scan_policy(s, policy, k, trace=trace))

    def rollout_policy(
        self,
        policy: Callable[[LineState, jnp.ndarray], jnp.ndarray],
        batch: int = 1,
        key: Optional[jnp.ndarray] = None,
        trace: bool = False,
    ):
        """
        Run a full horizon with actions chosen on the fly.

        policy(state, key) -> [B, 4]. The state passed in is post-`inject` and
        pre-completion, matching where `sim.commander_decide` is called.

        Convenience wrapper: recompiles on every call. For repeated use, call
        `compile_policy_rollout` once and reuse the result.
        """
        keys = jax.random.split(key, self.spec.H) if key is not None else None
        return self.compile_policy_rollout(policy, trace=trace)(
            self.init_state(batch), keys
        )

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    def profit_jax(self, s: LineState) -> jnp.ndarray:
        """[B] profit, computed in-graph. Use this for gradient work."""
        spec = self.spec
        revenue = spec.revenue_per_unit * s.cum_ship
        production = (jnp.asarray(spec.prod_cost_per_unit) * jnp.asarray(spec.take)
                      * s.started).sum(axis=-1)
        inventory = spec.fg_holding * s.sum_qfin
        wip = (jnp.asarray(spec.wip_holding) * s.sum_q).sum(axis=-1)
        backorder = spec.backorder * s.sum_late
        return revenue - (production + inventory + wip + backorder)


def profit_breakdown(spec: LineSpec, s: LineState) -> Dict[str, np.ndarray]:
    """
    Host-side profit decomposition in float64.

    The accumulators are exact integers, so this is the exact profit regardless
    of the dtype the simulation ran in — no float32 drift to argue about.
    """
    cum_ship = np.asarray(s.cum_ship, dtype=np.int64)
    started = np.asarray(s.started, dtype=np.int64)
    sum_qfin = np.asarray(s.sum_qfin, dtype=np.int64)
    sum_q = np.asarray(s.sum_q, dtype=np.int64)
    sum_late = np.asarray(s.sum_late, dtype=np.int64)

    revenue = spec.revenue_per_unit * cum_ship.astype(np.float64)
    production = (started.astype(np.float64)
                  * spec.take.astype(np.float64)
                  * spec.prod_cost_per_unit).sum(axis=-1)
    inventory = spec.fg_holding * sum_qfin.astype(np.float64)
    wip = (sum_q.astype(np.float64) * spec.wip_holding).sum(axis=-1)
    backorder = spec.backorder * sum_late.astype(np.float64)

    return {
        "revenue": revenue,
        "production": production,
        "inventory": inventory,
        "wip": wip,
        "backorder": backorder,
        "profit": revenue - (production + inventory + wip + backorder),
        "units_shipped": cum_ship,
        "units_produced": (started * spec.take.astype(np.int64)).sum(axis=-1),
        "invalid_actions": np.asarray(s.n_invalid, dtype=np.int64),
    }


def make_line(spec: LineSpec, dtype: Any = jnp.int32) -> TensorLine:
    return TensorLine(spec, dtype=dtype)
