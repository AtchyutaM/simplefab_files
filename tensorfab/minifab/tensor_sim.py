"""
tensor_sim — the Intel Mini-Fab (Tier 1) as a batched JAX program.

Same discipline as `tensorfab/tensor_sim.py` and `twoprod/tensor_sim.py`: exact
forward, integer state, one `lax.scan` over the horizon, proven exact against an
independent reference (`minifab/sim.py`) by `verify_sim.py`. What is new:

1. **Buffers per step, not per machine.** `b[B, S, P]` holds the lots waiting for
   each of the six steps. A machine draws from two buffers, which ones being
   given by `spec.step_of[m, k]`. Routing never points backwards: a lot finishing
   step s always lands in b(s+1), or leaves after step 6. Re-entrancy is carried
   entirely by `step_of`.

2. **The register is tagged by slot.** `reg[B, M, K, P, L]` holds the number of
   lots machine m is running in its slot k (which step), completing in l ticks.
   A machine runs one job at a time, so for each (b, m) at most one (k, l) entry
   is non-zero, and it holds the batch size. "Busy" is a sum, as before.

3. **Batch size is part of the action.** `action[B, M]` is an option index into
   the machine's options (idle, or step x lots), in `minifab.sim.action_space`
   order. A request is refused, and counted, if the machine is busy or the buffer
   holds fewer lots than asked. Batches are never truncated.

4. **Twins share buffers, so machines start in a fixed order.** A, B, C, D, E,
   each seeing the buffers as the previous one left them: five gates in
   sequence instead of one, still an ordinary array computation.

5. **Setups are sized by the last COMPLETED slot** (`last[B, M]`, 0 = never run),
   as in the two-product line. Only E has non-zero setups in Tier 1.

6. **Preventive maintenance, when switched on, is a second register**
   `pmreg[B, M, LP]`. PM is a rule, not a decision: an idle machine whose window
   is open and whose PM for this shift is not done starts PM before any run.

The tick order matches `minifab/sim.py` exactly:
    release -> completions -> PM -> decide -> start (A..E) -> accrue -> shift
Policies decide after completions and PM, on the state a machine will actually
start from.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tensorfab.minifab.spec import MiniFabSpec

NEVER = -(10 ** 6)     # "last PM start" before any PM: far enough back to never bind


class FabState(NamedTuple):
    """Carry for the tick loop. Leading batch dimension B throughout."""

    b: jnp.ndarray              # [B, S, P]  lots waiting for each step
    reg: jnp.ndarray            # [B, M, K, P, L] lots in flight, by slot, completing in l
    pmreg: jnp.ndarray          # [B, M, LP] PM in progress, completing in l
    last: jnp.ndarray           # [B, M] last COMPLETED slot + 1; 0 = never run
    last_pm: jnp.ndarray        # [B, M] tick of the last PM start; NEVER if none
    pm_shift: jnp.ndarray       # [B, M] shift of the last PM; -1 if none
    outs: jnp.ndarray           # [B]  lots out of step 6
    sum_wip: jnp.ndarray        # [B]  sum over ticks of lots in the system
    runs: jnp.ndarray           # [B, M, K] runs started
    lots_in: jnp.ndarray        # [B, M, K] lots started
    setups: jnp.ndarray         # [B, M] runs that needed a setup
    pm_jobs: jnp.ndarray        # [B, M] PMs started
    pm_violations: jnp.ndarray  # [B] PM past end of shift, or a shift without PM
    n_invalid: jnp.ndarray      # [B] requests refused


class FabTrace(NamedTuple):
    """End-of-tick state, recorded before the register shift."""

    b: jnp.ndarray              # [H, B, S] lots waiting per step
    inside: jnp.ndarray         # [H, B, M, K] lots inside each machine, by slot
    run_w: jnp.ndarray          # [H, B, M] sum of lots x ticks to completion
    pm_rem: jnp.ndarray         # [H, B, M] ticks of PM remaining, 0 if none
    outs: jnp.ndarray           # [H, B] cumulative
    n_invalid: jnp.ndarray      # [H, B] cumulative
    action: jnp.ndarray         # [H, B, M] option requested this tick


class TensorMiniFab:
    def __init__(self, spec: MiniFabSpec, dtype: Any = jnp.int32):
        self.spec = spec
        self.dtype = dt = dtype
        M, K, S, L = spec.M, spec.n_slots, spec.n_steps, spec.reg_len
        self.M, self.K, self.S, self.L = M, K, S, L
        self.LP = spec.pm_duration + 1 if spec.pm_on else 1

        step_oh = np.zeros((M, K, S), dtype=np.int32)
        for m in range(M):
            for k in range(K):
                step_oh[m, k, spec.step_of[m, k]] = 1
        opt_slot_oh = np.zeros((M, spec.n_opts, K), dtype=np.int32)
        for m in range(M):
            for o in range(spec.n_opts):
                if spec.opt_slot[m, o] >= 0:
                    opt_slot_oh[m, o, spec.opt_slot[m, o]] = 1

        self._step_oh = jnp.asarray(step_oh, dtype=dt)              # [M, K, S]
        self._occ = jnp.asarray(spec.occ, dtype=dt)                 # [M, K]
        self._setup = jnp.asarray(spec.setup, dtype=dt)             # [M, K+1, K]
        self._opt_slot_oh = jnp.asarray(opt_slot_oh, dtype=dt)      # [M, NOPT, K]
        self._opt_lots = jnp.asarray(spec.opt_lots, dtype=dt)       # [M, NOPT]
        self._releases = jnp.asarray(spec.releases, dtype=dt)       # [H]
        self._slots = jnp.arange(L, dtype=dt)
        self._pm_slots = jnp.arange(self.LP, dtype=dt)
        self._opts = jnp.arange(spec.n_opts, dtype=dt)
        self._last_idx = jnp.arange(K + 1, dtype=dt)
        self._kidx = jnp.arange(1, K + 1, dtype=dt)

        self._rollout = jax.jit(lambda s, xs: self._scan_tape(s, xs, trace=False))
        self._rollout_tr = jax.jit(lambda s, xs: self._scan_tape(s, xs, trace=True))

    # ------------------------------------------------------------------
    def init_state(self, batch: int = 1) -> FabState:
        sp, dt = self.spec, self.dtype
        M, K, S, P, L = self.M, self.K, self.S, sp.n_products, self.L
        zb = jnp.zeros((batch,), dtype=dt)
        zbm = jnp.zeros((batch, M), dtype=dt)
        zbmk = jnp.zeros((batch, M, K), dtype=dt)
        return FabState(
            b=jnp.zeros((batch, S, P), dtype=dt),
            reg=jnp.zeros((batch, M, K, P, L), dtype=dt),
            pmreg=jnp.zeros((batch, M, self.LP), dtype=dt),
            last=zbm,
            last_pm=jnp.full((batch, M), NEVER, dtype=dt),
            pm_shift=jnp.full((batch, M), -1, dtype=dt),
            outs=zb, sum_wip=zb,
            runs=zbmk, lots_in=zbmk, setups=zbm, pm_jobs=zbm,
            pm_violations=zb, n_invalid=zb,
        )

    def busy(self, s: FabState) -> jnp.ndarray:
        """[B, M] — machine occupied by a run or by PM."""
        return (s.reg.sum(axis=(2, 3, 4)) + s.pmreg.sum(axis=-1)) > 0

    # ------------------------------------------------------------------
    # phases 1-3: release, completions, preventive maintenance
    # ------------------------------------------------------------------
    def prepare(self, s: FabState, t: jnp.ndarray, rel: jnp.ndarray) -> FabState:
        dt = self.dtype
        # 1. release into b1, product 0
        b = s.b.at[:, 0, 0].add(rel.astype(dt))

        # 2. completions: register slot 0 empties into the next step's buffer
        done = s.reg[..., 0]                                        # [B, M, K, P]
        reg = s.reg.at[..., 0].set(0)
        per_step = jnp.einsum("bmkp,mks->bsp", done, self._step_oh)  # [B, S, P]
        b = b + jnp.concatenate([jnp.zeros_like(per_step[:, :1]), per_step[:, :-1]], axis=1)
        outs = s.outs + per_step[:, -1].sum(axis=-1)
        fin = done.sum(axis=-1)                                     # [B, M, K]
        newlast = ((fin > 0).astype(dt) * self._kidx).sum(axis=-1)  # [B, M]
        last = jnp.where(fin.sum(axis=-1) > 0, newlast, s.last)
        pmreg = s.pmreg.at[..., 0].set(0)
        s = s._replace(b=b, reg=reg, pmreg=pmreg, last=last, outs=outs)

        # 3. preventive maintenance takes priority over new runs
        if self.spec.pm_on:
            sp = self.spec
            S_ = sp.shift_ticks
            shift = t // S_
            opens = jnp.maximum(shift * S_, s.last_pm + sp.pm_min_gap)
            due = (s.pm_shift != shift) & (t >= opens)
            go = due & ~self.busy(s)                                # [B, M]
            gi = go.astype(dt)
            pm_oh = (self._pm_slots == sp.pm_duration).astype(dt)
            late = go & (t + sp.pm_duration > (shift + 1) * S_)
            s = s._replace(
                pmreg=s.pmreg + gi[..., None] * pm_oh,
                last_pm=jnp.where(go, t, s.last_pm),
                pm_shift=jnp.where(go, shift, s.pm_shift),
                pm_jobs=s.pm_jobs + gi,
                pm_violations=s.pm_violations + late.astype(dt).sum(axis=-1),
            )
        return s

    # ------------------------------------------------------------------
    # phase 5: starts, machine by machine in the fixed order
    # ------------------------------------------------------------------
    def start(self, s: FabState, action: jnp.ndarray) -> FabState:
        """action: [B, M] option indices."""
        dt = self.dtype
        busy = self.busy(s)
        b, reg = s.b, s.reg
        runs, lots_in, setups, invalid = s.runs, s.lots_in, s.setups, s.n_invalid
        for m in range(self.M):
            a_oh = (self._opts[None, :] == action[:, m, None]).astype(dt)    # [B, NOPT]
            lots = a_oh @ self._opt_lots[m]                                  # [B]
            k_oh = a_oh @ self._opt_slot_oh[m]                               # [B, K]
            s_oh = k_oh @ self._step_oh[m]                                   # [B, S]
            have = (s_oh * b[:, :, 0]).sum(axis=-1)                          # [B]
            requested = lots > 0
            ok = requested & ~busy[:, m] & (have >= lots)
            oki = ok.astype(dt)
            n = lots * oki
            b = b.at[:, :, 0].add(-n[:, None] * s_oh)

            last_oh = (self._last_idx[None, :] == s.last[:, m, None]).astype(dt)  # [B, K+1]
            su = ((last_oh @ self._setup[m]) * k_oh).sum(axis=-1)            # [B]
            d = su + (k_oh * self._occ[m]).sum(axis=-1)                      # [B]
            slot_oh = (self._slots[None, :] == d[:, None]).astype(dt)        # [B, L]
            reg = reg.at[:, m, :, 0, :].add(
                n[:, None, None] * k_oh[:, :, None] * slot_oh[:, None, :])

            runs = runs.at[:, m, :].add(oki[:, None] * k_oh)
            lots_in = lots_in.at[:, m, :].add(n[:, None] * k_oh)
            setups = setups.at[:, m].add((ok & (su > 0)).astype(dt))
            invalid = invalid + (requested & ~ok).astype(dt)
        return s._replace(b=b, reg=reg, runs=runs, lots_in=lots_in,
                          setups=setups, n_invalid=invalid)

    # ------------------------------------------------------------------
    # phase 6: end-of-tick accruals; then the register shift
    # ------------------------------------------------------------------
    def accrue(self, s: FabState, t: jnp.ndarray) -> FabState:
        wip = s.b.sum(axis=(1, 2)) + s.reg.sum(axis=(1, 2, 3, 4))
        s = s._replace(sum_wip=s.sum_wip + wip)
        if self.spec.pm_on:
            S_ = self.spec.shift_ticks
            end = ((t + 1) % S_) == 0
            missed = end & (s.pm_shift != t // S_)
            s = s._replace(pm_violations=s.pm_violations
                           + missed.astype(self.dtype).sum(axis=-1))
        return s

    @staticmethod
    def shift(s: FabState) -> FabState:
        reg = jnp.concatenate([s.reg[..., 1:], jnp.zeros_like(s.reg[..., :1])], axis=-1)
        pmreg = jnp.concatenate([s.pmreg[..., 1:], jnp.zeros_like(s.pmreg[..., :1])], axis=-1)
        return s._replace(reg=reg, pmreg=pmreg)

    def _snapshot(self, s: FabState, action: jnp.ndarray) -> FabTrace:
        per_slot = s.reg.sum(axis=(2, 3))                                    # [B, M, L]
        return FabTrace(
            b=s.b.sum(axis=-1),
            inside=s.reg.sum(axis=(3, 4)),
            run_w=(per_slot * self._slots).sum(axis=-1),
            pm_rem=(s.pmreg * self._pm_slots).sum(axis=-1),
            outs=s.outs,
            n_invalid=s.n_invalid,
            action=action,
        )

    def tick(self, s: FabState, t, rel, action) -> Tuple[FabState, FabTrace]:
        s = self.prepare(s, t, rel)
        s = self.start(s, action)
        s = self.accrue(s, t)
        return s, self._snapshot(s, action)

    # ------------------------------------------------------------------
    # rollouts
    # ------------------------------------------------------------------
    def releases_for(self, releases: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """[H, B] per-row release tapes, or [H] shared by every row (broadcast
        inside the tick, so a large batch never materialises an [H, B] copy).
        Defaults to the config's schedule."""
        if releases is None:
            return self._releases
        return jnp.asarray(releases, dtype=self.dtype)

    def _scan_tape(self, s0, xs, trace: bool):
        def body(s, x):
            t, rel, act = x
            s, out = self.tick(s, t, rel, act)
            return self.shift(s), (out if trace else None)
        final, ys = lax.scan(body, s0, xs)
        return (final, ys) if trace else final

    def rollout_tape(self, actions: jnp.ndarray, releases: Optional[jnp.ndarray] = None,
                     trace: bool = False):
        """Open-loop: actions [H, B, M] option indices; releases [H, B] or [H]."""
        H, B = actions.shape[0], actions.shape[1]
        xs = (jnp.arange(H, dtype=self.dtype), self.releases_for(releases)[:H],
              jnp.asarray(actions, dtype=self.dtype))
        fn = self._rollout_tr if trace else self._rollout
        return fn(self.init_state(B), xs)

    def compile_policy_rollout(self, policy, trace: bool = False):
        """Closed-loop: `policy(fab, s, pst) -> (action [B, M], pst)` is called each
        tick on the state after release, completions and PM."""
        def run(s0, p0, xs):
            def body(carry, x):
                s, pst = carry
                t, rel = x
                s = self.prepare(s, t, rel)
                act, pst = policy(self, s, pst)
                s = self.start(s, act)
                s = self.accrue(s, t)
                out = self._snapshot(s, act) if trace else None
                return (self.shift(s), pst), out
            (final, pst), ys = lax.scan(body, (s0, p0), xs)
            return (final, ys) if trace else final
        return jax.jit(run)

    def rollout_policy(self, policy, batch: int = 1, releases: Optional[jnp.ndarray] = None,
                       trace: bool = False):
        rel = self.releases_for(releases)
        xs = (jnp.arange(rel.shape[0], dtype=self.dtype), rel)
        return self.compile_policy_rollout(policy, trace)(
            self.init_state(batch), policy.init(self, batch), xs)


def profit_breakdown(spec: MiniFabSpec, s: FabState) -> dict:
    """Exact profit from the integer accumulators, in float64 on the host."""
    outs = np.asarray(s.outs, dtype=np.int64)
    sum_wip = np.asarray(s.sum_wip, dtype=np.int64)
    revenue = outs * spec.revenue_per_lot
    wip = sum_wip * spec.wip_holding
    return {"revenue": revenue, "wip": wip, "profit": revenue - wip,
            "outs": outs, "sum_wip": sum_wip}
