"""
env_tensor — the Gymnasium wrapper's observation, mask, and reward, tensorized.

Why this exists: measurement, not assumption. On this machine

    bare ProductionLine.run_step     358,277 steps/s
    full FabEnv.step + action_masks   66,553 steps/s   (5.4x slower)

so the *wrapper* costs 4.4x more than the dynamics it wraps. The 18-dim
observation alone contains a `while` loop that scans forward for the next demand
event. Porting only `sim.py` to tensors and leaving the wrapper in Python would
leave the dominant cost in place, so Phase 2a needs this module.

--------------------------------------------------------------------------------
Faithfulness notes (each of these is a place it would be easy to get wrong)
--------------------------------------------------------------------------------
* The mask is computed on the PRE-arrival, PRE-completion state — `FabEnv.step`
  calls `compute_action_mask(self._t)` before `run_step` injects anything.

* The env PROJECTS invalid picks to Idle (`a = a if mask[i,a] else 0`) rather
  than letting the sim refuse them. Because the mask is conservative (it cannot
  see input arriving this tick), the env therefore sometimes idles a machine the
  sim would have allowed to run. That is real env behaviour and is reproduced.

* `invalid_selected` counts against the MASK, not against the sim's internal
  refusal count. Since `mask[:, 0] == 1` always, only a requested Run can be
  invalid.

* Phi_before is evaluated BEFORE `run_step`, so on the pre-injection queues;
  Phi_after is evaluated after. Getting this off by one silently changes the
  shaping term on every tick.

* The observation is emitted at `self._t + 1` — i.e. it describes the state at
  the START of the next tick, and at the final step it is evaluated at t == H.

* The pipeline nudge was removed from env.py in Apr 2026 and is not implemented
  here.

* `invalid_selected` in env.py sums four numpy bools. numpy bool addition is
  logical OR, so the result saturates at 1 rather than counting 0..4. This module
  reproduces that by default (`faithful_invalid=True`) because Phase 2a must be a
  drop-in replacement, and exposes the true count as `StepOut.invalid_count`.
  See README § "A latent bug in env.py".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tensorfab.spec import LineSpec
from tensorfab.tensor_sim import LineState, TensorLine

OBS_DIM = 18
_MAX_UNITS = 64.0
_TIME_CAP = np.array([20.0, 3.0, 20.0, 3.0], dtype=np.float64)
_DEMAND_QTY_SCALE = 200.0


@dataclass(frozen=True)
class Shaping:
    """Mirror of `simplefab_1p.env.ShapingConfig`."""

    enabled: bool = True
    beta: float = 1.0
    gamma: float = 0.99
    w_backlog: float = 1.0
    w_wip: float = 0.1
    w_finished: float = 0.5


class StepOut(NamedTuple):
    obs: jnp.ndarray          # [B, 18]
    reward: jnp.ndarray       # [B]
    base_reward: jnp.ndarray  # [B]  delta-profit alone
    shaping: jnp.ndarray      # [B]
    invalid_selected: jnp.ndarray  # [B]  what env.py actually computes (see below)
    invalid_count: jnp.ndarray     # [B]  the true 0..4 count
    mask: jnp.ndarray         # [B, 4]  the mask the action was projected against


class TensorEnv:
    """
    Batched, jit-friendly equivalent of `FabEnv`.

    Not a Gymnasium Env — there is no per-step Python. It exposes pure functions
    over `(LineState, tick)` so the whole episode can live inside one `lax.scan`.
    """

    def __init__(
        self,
        line: TensorLine,
        shaping: Optional[Shaping] = None,
        invalid_action_penalty: float = 0.0,
        normalize_obs: bool = True,
        faithful_invalid: bool = True,
    ):
        self.line = line
        self.spec = line.spec
        self.shaping = shaping if shaping is not None else Shaping()
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.normalize_obs = bool(normalize_obs)
        # True  -> bug-compatible with env.py (invalid_selected saturates at 1)
        # False -> the true 0..4 count. Changing this changes the reward the
        #          agent sees whenever invalid_action_penalty != 0, so it must be
        #          an explicit opt-in, never a silent "fix".
        self.faithful_invalid = bool(faithful_invalid)

        spec = line.spec
        H, ivl = spec.H, spec.demand_interval
        f = jnp.float32

        # --- observation lookup tables, indexed by tick 0..H inclusive -----
        # dim 16/17 replace get_observation's forward `while` scan.
        dem = np.asarray(spec.demand)
        ticks_until = np.zeros(H + 1, dtype=np.float64)
        next_qty = np.zeros(H + 1, dtype=np.float64)
        nxt = H  # sentinel: no further demand
        for t in range(H - 1, -1, -1):
            if dem[t] > 0:
                nxt = t
            ticks_until[t] = nxt - t
            next_qty[t] = dem[nxt] if nxt < H else 0.0
        ticks_until[H] = 0.0
        next_qty[H] = 0.0

        ts = np.arange(H + 1, dtype=np.float64)
        self._obs_t = jnp.asarray(
            np.stack([
                np.minimum(ts / H, 1.0) if H > 0 else np.zeros(H + 1),   # dim 14
                (ts % ivl) / ivl,                                        # dim 15
                ticks_until / ivl,                                       # dim 16
                next_qty / _DEMAND_QTY_SCALE,                            # dim 17
            ], axis=-1), dtype=f)                                        # [H+1, 4]
        self._obs_t_raw = jnp.asarray(
            np.stack([ts, ts % ivl, ticks_until, next_qty], axis=-1), dtype=f)

        self._time_cap = jnp.asarray(_TIME_CAP, dtype=f)
        self._take = jnp.asarray(spec.take, dtype=line.dtype)

        # --- per-tick profit deltas, as prices on integer increments ------
        self._rev = f(spec.revenue_per_unit)
        self._pcost = jnp.asarray(spec.prod_cost_per_unit * spec.take, dtype=f)
        self._fg = f(spec.fg_holding)
        self._wip = jnp.asarray(spec.wip_holding, dtype=f)
        self._back = f(spec.backorder)

        self._arrivals = jnp.asarray(spec.arrivals, dtype=line.dtype)
        self._demand = jnp.asarray(spec.demand, dtype=line.dtype)
        self._demand_lag = jnp.asarray(spec.demand_lag_cum, dtype=line.dtype)

    # ------------------------------------------------------------------
    def observe(self, s: LineState, t: jnp.ndarray) -> jnp.ndarray:
        """
        18-dim observation, mirroring `sim.get_observation(current_time=t)`.

        `s` must be the carry at the START of tick t (post-shift), which is what
        the reference sees after `self._t += 1`.
        """
        f = jnp.float32
        q = s.q.astype(f)
        qfin = s.qfin.astype(f)[:, None]
        backlog = s.backlog.astype(f)[:, None]

        occupied = s.reg.sum(axis=-1)
        busy = (occupied > 0).astype(f)                      # [B, 4]
        rem = jnp.argmax(s.reg, axis=-1).astype(f) * busy    # [B, 4]

        tvec = self._obs_t[t] if self.normalize_obs else self._obs_t_raw[t]
        tvec = jnp.broadcast_to(tvec, (q.shape[0], 4))

        if self.normalize_obs:
            counts = jnp.concatenate([q, qfin, backlog], axis=-1) / _MAX_UNITS
            rem = rem / self._time_cap
        else:
            counts = jnp.concatenate([q, qfin, backlog], axis=-1)

        return jnp.concatenate([counts, busy, rem, tvec], axis=-1)

    def phi(self, s: LineState) -> jnp.ndarray:
        """Potential Phi(s) = -(w_backlog*backlog + w_wip*ΣWIP + w_finished*qfin)."""
        c = self.shaping
        f = jnp.float32
        return -(
            c.w_backlog * s.backlog.astype(f)
            + c.w_wip * s.q.astype(f).sum(axis=-1)
            + c.w_finished * s.qfin.astype(f)
        )

    def action_mask(self, s: LineState) -> jnp.ndarray:
        """[B, 4] — may machine m be *run* this tick? (Idle is always legal.)"""
        return self.line.action_mask(s)

    # ------------------------------------------------------------------
    def step(self, s: LineState, action: jnp.ndarray, t: int) -> Tuple[LineState, StepOut]:
        """
        One env step. `t` is a static Python int or a traced scalar index.

        action: [B, 4] in {0,1} — the *requested* action, before projection.
        """
        f = jnp.float32
        mask = self.action_mask(s)                          # [B, 4] bool
        maskf = mask.astype(action.dtype)
        bad = action * (1 - maskf)                          # [B, 4] 1 where a Run was refused
        invalid_count = bad.sum(axis=-1).astype(f)          # the true 0..4 count

        # BUG-COMPATIBILITY (see module docstring and README § "A latent bug in
        # env.py"): the reference sums four numpy *bools*, and numpy bool
        # addition is logical OR, so `invalid_selected` saturates at 1 — it is an
        # "any invalid?" flag, not a count. With invalid_action_penalty=-2.0
        # (train_ppo.py:167) an agent requesting four illegal starts is penalised
        # exactly as much as one requesting a single illegal start.
        invalid_selected = (
            jnp.minimum(invalid_count, 1.0) if self.faithful_invalid else invalid_count
        )
        a_proj = action * maskf                             # project invalid -> Idle

        phi_before = self.phi(s)

        s1 = self.line.inject(s, self._arrivals[t], self._demand[t])
        s2 = self.line.advance(s1, a_proj, self._demand_lag[t])

        # Delta-profit for this tick, from integer increments within the tick.
        #
        # The baseline is s1 (POST-injection), not s. `run_step` captures
        # `prev_profit = self.profit_total()` *after* injecting arrivals/demand
        # and after the instant `while queue_fin and demand` match, so revenue
        # earned by that match is folded into the baseline and never appears in
        # the step reward. Using s here instead of s1 inflates the reward by
        # $80 x (instantly matched units) on every demand-arrival tick.
        d_ship = (s2.cum_ship - s1.cum_ship).astype(f)
        d_started = (s2.started - s.started).astype(f)
        late = (s2.sum_late - s.sum_late).astype(f)
        base_reward = (
            self._rev * d_ship
            - (self._pcost * d_started).sum(axis=-1)
            - self._fg * s2.qfin.astype(f)
            - (self._wip * s2.q.astype(f)).sum(axis=-1)
            - self._back * late
        )

        phi_after = self.phi(s2)
        shaping = (
            self.shaping.beta * (self.shaping.gamma * phi_after - phi_before)
            if self.shaping.enabled else jnp.zeros_like(base_reward)
        )
        reward = base_reward + self.invalid_action_penalty * invalid_selected + shaping

        s3 = self.line.shift(s2)
        obs = self.observe(s3, t + 1)
        return s3, StepOut(obs, reward, base_reward, shaping,
                           invalid_selected, invalid_count, mask)

    # ------------------------------------------------------------------
    def rollout(self, actions: jnp.ndarray) -> Tuple[LineState, StepOut]:
        """actions: [H, B, 4] requested actions. Returns (final state, per-tick StepOut)."""
        b = actions.shape[1]

        def body(carry, x):
            s, t = carry
            s, out = self.step(s, x, t)
            return (s, t + 1), out

        (final, _), outs = lax.scan(
            body, (self.line.init_state(b), 0), actions
        )
        return final, outs

    def reset_obs(self, batch: int) -> jnp.ndarray:
        return self.observe(self.line.init_state(batch), 0)


def make_env(
    line: TensorLine,
    shaping: Optional[Shaping] = None,
    invalid_action_penalty: float = 0.0,
    normalize_obs: bool = True,
    faithful_invalid: bool = True,
) -> TensorEnv:
    return TensorEnv(line, shaping, invalid_action_penalty, normalize_obs,
                     faithful_invalid)
