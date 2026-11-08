"""
Why is the tensor Mini-Fab ~23x slower per row-tick than the four-machine line,
and how much faster can it go?

Each variant below is an alternative implementation of the same tick, checked
EXACT against `TensorMiniFab` (itself proven exact against the reference) before
it is timed: every buffer count and the outs at every tick, and every final
accumulator, over 300 shifts at four release rates. PM is off (the default);
the variants do not implement PM, which adds a small [B, M, 7] register and is
not the bottleneck.

Variants, cumulative:
  baseline   TensorMiniFab as built: reg[B, 5, 2, P, 64], shifted every tick
  ring       the register as a ring buffer: completions read slot t mod L, a start
             writes slot (t + d) mod L, nothing shifts. Gradient-safe: both are
             selections, and occupancy is still reg.sum
  split      ring, plus each machine group sized to its own longest job: A/B keep
             64 slots, C/D/E need 18. Gradient-safe
  counters   split, plus busy and lots-in-machines kept as running counters
             instead of re-summing the register. EXACT-ONLY: `free_at > t` is an
             integer comparison, so the linear occupancy gate the differentiable
             line relies on (idle = 1 - reg.sum) is lost
  countdown  no register at all: per machine, when it frees up, how many lots, which
             step. The smallest state. EXACT-ONLY, for the same reason

and the same three ideas with the starts VECTORISED over machines. The baseline
runs five start gates one after another, so every operation appears five times
in the compiled tick. Only twins conflict, so two stages suffice: {A, C, E}, then
{B, D}. The dispatch rule is vectorised the same way (`VecRule`).
  vec            ring register, one group, vectorised. Gradient-safe
  vec-split      ring registers per group, vectorised. Gradient-safe
  vec-countdown  countdown, vectorised. EXACT-ONLY

For each variant it reports fused kernels per tick in the compiled scan body:
each is one pass over the batch, which is what costs time at small batch.
Ablation: the dispatch rule replaced by a near-free data-dependent action, to
price the rule itself.

    python -m tensorfab.minifab.speed_study                      # ~15 min on CPU
    python -m tensorfab.minifab.speed_study --batches 1 1024 10000 --shifts 50 --rounds 5

Timing design: every configuration is compiled and warmed up first, then timed once
per round, with the order rotated each round, and the median is reported. Timing the
variants one after another (the first two logged runs) let machine-state drift over
a 20-minute run masquerade as differences between variants.
"""

from __future__ import annotations

import argparse
import time
from typing import NamedTuple, Tuple

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from minifab.config import constant_rate_schedule, make_minifab_config  # noqa: E402
from minifab.sim import MiniFab  # noqa: E402
from tensorfab.logbook import Run  # noqa: E402
from tensorfab.minifab.policies import make_rule  # noqa: E402
from tensorfab.minifab.spec import MiniFabSpec, spec_from_config  # noqa: E402
from tensorfab.minifab.tensor_sim import TensorMiniFab  # noqa: E402


# ---------------------------------------------------------------------------
# register-based variants: ring buffer, per-group lengths, running counters
# ---------------------------------------------------------------------------
class VState(NamedTuple):
    b: jnp.ndarray          # [B, S, P]
    regs: Tuple             # per machine group: [B, Mg, K, P, Lg]
    last: jnp.ndarray       # [B, M]
    free_at: jnp.ndarray    # [B, M] (counters only)
    inside: jnp.ndarray     # [B]    (counters only)
    t: jnp.ndarray          # []     current tick, for busy = free_at > t
    outs: jnp.ndarray
    sum_wip: jnp.ndarray
    runs: jnp.ndarray
    lots_in: jnp.ndarray
    setups: jnp.ndarray
    n_invalid: jnp.ndarray


class RegisterVariant:
    def __init__(self, spec: MiniFabSpec, ring: bool, split: bool, counters: bool):
        assert not spec.pm_on, "speed-study variants are PM-off only"
        self.spec, self.dtype = spec, jnp.int32
        self.ring, self.counters = ring, counters
        M, K, S = spec.M, spec.n_slots, spec.n_steps
        self.M, self.K, self.S = M, K, S
        need = spec.setup.max(axis=1).max(axis=1) + spec.occ.max(axis=1) + 1   # [M]
        if split:
            batch = [m for m in range(M) if spec.max_batch[m] > 1]
            single = [m for m in range(M) if spec.max_batch[m] == 1]
            self.groups = [g for g in (batch, single) if g]
        else:
            self.groups = [list(range(M))]
        if [m for g in self.groups for m in g] != list(range(M)):
            raise ValueError("groups must preserve machine order")
        self.L = [int(need[g].max()) for g in self.groups]
        self.where = {m: (gi, i) for gi, g in enumerate(self.groups) for i, m in enumerate(g)}

        step_oh = np.zeros((M, K, S), dtype=np.int32)
        for m in range(M):
            for k in range(K):
                step_oh[m, k, spec.step_of[m, k]] = 1
        opt_slot_oh = np.zeros((M, spec.n_opts, K), dtype=np.int32)
        for m in range(M):
            for o in range(spec.n_opts):
                if spec.opt_slot[m, o] >= 0:
                    opt_slot_oh[m, o, spec.opt_slot[m, o]] = 1
        self._step_oh = jnp.asarray(step_oh)
        self._occ = jnp.asarray(spec.occ)
        self._setup = jnp.asarray(spec.setup)
        self._opt_slot_oh = jnp.asarray(opt_slot_oh)
        self._opt_lots = jnp.asarray(spec.opt_lots)
        self._opts = jnp.arange(spec.n_opts, dtype=jnp.int32)
        self._last_idx = jnp.arange(K + 1, dtype=jnp.int32)
        self._kidx = jnp.arange(1, K + 1, dtype=jnp.int32)

    def init_state(self, B: int) -> VState:
        P = self.spec.n_products
        z = jnp.zeros((B,), jnp.int32)
        return VState(
            b=jnp.zeros((B, self.S, P), jnp.int32),
            regs=tuple(jnp.zeros((B, len(g), self.K, P, L), jnp.int32)
                       for g, L in zip(self.groups, self.L)),
            last=jnp.zeros((B, self.M), jnp.int32),
            free_at=jnp.zeros((B, self.M), jnp.int32), inside=z,
            t=jnp.zeros((), jnp.int32), outs=z, sum_wip=z,
            runs=jnp.zeros((B, self.M, self.K), jnp.int32),
            lots_in=jnp.zeros((B, self.M, self.K), jnp.int32),
            setups=jnp.zeros((B, self.M), jnp.int32), n_invalid=z)

    def busy(self, s: VState) -> jnp.ndarray:
        if self.counters:
            return s.free_at > s.t
        return jnp.concatenate([r.sum(axis=(2, 3, 4)) for r in s.regs], axis=1) > 0

    def prepare(self, s: VState, t, rel) -> VState:
        b = s.b.at[:, 0, 0].add(rel)
        per_step = jnp.zeros_like(b)
        regs, last, inside = [], s.last, s.inside
        for g, reg, L in zip(self.groups, s.regs, self.L):
            if self.ring:
                pos = t % L
                done = lax.dynamic_index_in_dim(reg, pos, axis=-1, keepdims=False)
                reg = lax.dynamic_update_index_in_dim(reg, jnp.zeros_like(done), pos, axis=-1)
            else:
                done = reg[..., 0]
                reg = reg.at[..., 0].set(0)
            per_step = per_step + jnp.einsum("bmkp,mks->bsp", done, self._step_oh[jnp.asarray(g)])
            fin = done.sum(axis=-1)
            newlast = ((fin > 0).astype(jnp.int32) * self._kidx).sum(axis=-1)
            gi = jnp.asarray(g)
            last = last.at[:, gi].set(jnp.where(fin.sum(axis=-1) > 0, newlast, last[:, gi]))
            if self.counters:
                inside = inside - done.sum(axis=(1, 2, 3))
            regs.append(reg)
        b = b + jnp.concatenate([jnp.zeros_like(per_step[:, :1]), per_step[:, :-1]], axis=1)
        return s._replace(b=b, regs=tuple(regs), last=last, inside=inside, t=t,
                          outs=s.outs + per_step[:, -1].sum(axis=-1))

    def start(self, s: VState, action) -> VState:
        busy = self.busy(s)
        b, regs = s.b, list(s.regs)
        runs, lots_in, setups, invalid = s.runs, s.lots_in, s.setups, s.n_invalid
        free_at, inside = s.free_at, s.inside
        for m in range(self.M):
            gi, i = self.where[m]
            L = self.L[gi]
            a_oh = (self._opts[None, :] == action[:, m, None]).astype(jnp.int32)
            lots = a_oh @ self._opt_lots[m]
            k_oh = a_oh @ self._opt_slot_oh[m]
            s_oh = k_oh @ self._step_oh[m]
            have = (s_oh * b[:, :, 0]).sum(axis=-1)
            requested = lots > 0
            ok = requested & ~busy[:, m] & (have >= lots)
            oki = ok.astype(jnp.int32)
            n = lots * oki
            b = b.at[:, :, 0].add(-n[:, None] * s_oh)
            last_oh = (self._last_idx[None, :] == s.last[:, m, None]).astype(jnp.int32)
            su = ((last_oh @ self._setup[m]) * k_oh).sum(axis=-1)
            d = su + (k_oh * self._occ[m]).sum(axis=-1)
            pos = (s.t + d) % L if self.ring else d
            slot_oh = (jnp.arange(L, dtype=jnp.int32)[None, :] == pos[:, None]).astype(jnp.int32)
            regs[gi] = regs[gi].at[:, i, :, 0, :].add(
                n[:, None, None] * k_oh[:, :, None] * slot_oh[:, None, :])
            if self.counters:
                free_at = free_at.at[:, m].set(jnp.where(ok, s.t + d, free_at[:, m]))
                inside = inside + n
            runs = runs.at[:, m, :].add(oki[:, None] * k_oh)
            lots_in = lots_in.at[:, m, :].add(n[:, None] * k_oh)
            setups = setups.at[:, m].add((ok & (su > 0)).astype(jnp.int32))
            invalid = invalid + (requested & ~ok).astype(jnp.int32)
        return s._replace(b=b, regs=tuple(regs), free_at=free_at, inside=inside, runs=runs,
                          lots_in=lots_in, setups=setups, n_invalid=invalid)

    def finish(self, s: VState) -> VState:
        inside = s.inside if self.counters else sum(r.sum(axis=(1, 2, 3, 4)) for r in s.regs)
        s = s._replace(sum_wip=s.sum_wip + s.b.sum(axis=(1, 2)) + inside)
        if not self.ring:
            s = s._replace(regs=tuple(jnp.concatenate([r[..., 1:], jnp.zeros_like(r[..., :1])], axis=-1)
                                      for r in s.regs))
        return s

    def compile_policy_rollout(self, policy, trace: bool = False):
        def run(s0, p0, xs):
            def body(carry, x):
                s, pst = carry
                t, rel = x
                s = self.prepare(s, t, rel)
                act, pst = policy(self, s, pst)
                s = self.finish(self.start(s, act))
                return (s, pst), ((s.b.sum(axis=-1), s.outs) if trace else None)
            (final, _), ys = lax.scan(body, (s0, p0), xs)
            return (final, ys) if trace else final
        return jax.jit(run)


# ---------------------------------------------------------------------------
# countdown variant: no register
# ---------------------------------------------------------------------------
class CState(NamedTuple):
    b: jnp.ndarray
    free_at: jnp.ndarray    # [B, M] tick the machine frees up
    jlots: jnp.ndarray      # [B, M] lots in the machine (0 = idle)
    jslot: jnp.ndarray      # [B, M] slot of the job in the machine
    last: jnp.ndarray
    inside: jnp.ndarray
    t: jnp.ndarray
    outs: jnp.ndarray
    sum_wip: jnp.ndarray
    runs: jnp.ndarray
    lots_in: jnp.ndarray
    setups: jnp.ndarray
    n_invalid: jnp.ndarray


class CountdownVariant(RegisterVariant):
    def __init__(self, spec: MiniFabSpec):
        super().__init__(spec, ring=True, split=True, counters=True)
        self._step_of = jnp.asarray(spec.step_of)

    def init_state(self, B: int) -> CState:
        z, zm = jnp.zeros((B,), jnp.int32), jnp.zeros((B, self.M), jnp.int32)
        return CState(b=jnp.zeros((B, self.S, self.spec.n_products), jnp.int32),
                      free_at=zm, jlots=zm, jslot=zm, last=zm, inside=z,
                      t=jnp.zeros((), jnp.int32), outs=z, sum_wip=z,
                      runs=jnp.zeros((B, self.M, self.K), jnp.int32),
                      lots_in=jnp.zeros((B, self.M, self.K), jnp.int32), setups=zm, n_invalid=z)

    def busy(self, s):
        return s.free_at > s.t

    def prepare(self, s: CState, t, rel) -> CState:
        b = s.b.at[:, 0, 0].add(rel)
        done = (s.free_at == t) & (s.jlots > 0)                               # [B, M]
        dl = s.jlots * done.astype(jnp.int32)
        step = jnp.take_along_axis(jnp.broadcast_to(self._step_of, (b.shape[0],) + self._step_of.shape),
                                   s.jslot[..., None], axis=-1)[..., 0]       # [B, M]
        per_step = ((jnp.arange(self.S)[None, None, :] == step[..., None]) * dl[..., None]).sum(axis=1)
        b = b.at[:, 1:, 0].add(per_step[:, :-1])
        return s._replace(b=b, t=t, outs=s.outs + per_step[:, -1], inside=s.inside - dl.sum(axis=-1),
                          last=jnp.where(done, s.jslot + 1, s.last),
                          jlots=jnp.where(done, 0, s.jlots))

    def start(self, s: CState, action) -> CState:
        busy = self.busy(s)
        b, free_at, jlots, jslot = s.b, s.free_at, s.jlots, s.jslot
        runs, lots_in, setups, invalid, inside = s.runs, s.lots_in, s.setups, s.n_invalid, s.inside
        for m in range(self.M):
            a_oh = (self._opts[None, :] == action[:, m, None]).astype(jnp.int32)
            lots = a_oh @ self._opt_lots[m]
            k_oh = a_oh @ self._opt_slot_oh[m]
            s_oh = k_oh @ self._step_oh[m]
            have = (s_oh * b[:, :, 0]).sum(axis=-1)
            requested = lots > 0
            ok = requested & ~busy[:, m] & (have >= lots)
            oki = ok.astype(jnp.int32)
            n = lots * oki
            b = b.at[:, :, 0].add(-n[:, None] * s_oh)
            last_oh = (self._last_idx[None, :] == s.last[:, m, None]).astype(jnp.int32)
            su = ((last_oh @ self._setup[m]) * k_oh).sum(axis=-1)
            d = su + (k_oh * self._occ[m]).sum(axis=-1)
            free_at = free_at.at[:, m].set(jnp.where(ok, s.t + d, free_at[:, m]))
            jlots = jlots.at[:, m].set(jnp.where(ok, n, jlots[:, m]))
            jslot = jslot.at[:, m].set(jnp.where(ok, (k_oh * jnp.arange(self.K)).sum(-1), jslot[:, m]))
            inside = inside + n
            runs = runs.at[:, m, :].add(oki[:, None] * k_oh)
            lots_in = lots_in.at[:, m, :].add(n[:, None] * k_oh)
            setups = setups.at[:, m].add((ok & (su > 0)).astype(jnp.int32))
            invalid = invalid + (requested & ~ok).astype(jnp.int32)
        return s._replace(b=b, free_at=free_at, jlots=jlots, jslot=jslot, inside=inside, runs=runs,
                          lots_in=lots_in, setups=setups, n_invalid=invalid)

    def finish(self, s: CState) -> CState:
        return s._replace(sum_wip=s.sum_wip + s.b.sum(axis=(1, 2)) + s.inside)


# ---------------------------------------------------------------------------
# vectorised across machines: two start stages instead of five
# ---------------------------------------------------------------------------
def stages_from_spec(spec: MiniFabSpec) -> np.ndarray:
    """Start stage of each machine. Machines that share a buffer must start in
    machine order; machines that share none can start together. On the Mini-Fab
    that gives A, C, E in stage 0 and B, D in stage 1: the sequential order A..E,
    exactly, because C's and E's buffers are untouched by A and B."""
    stage = np.zeros(spec.M, dtype=np.int32)
    for m in range(spec.M):
        mine = set(spec.step_of[m].tolist())
        before = [stage[j] + 1 for j in range(m) if mine & set(spec.step_of[j].tolist())]
        stage[m] = max(before, default=0)
    return stage


class VecMixin:
    """Decode every machine's option at once; apply the stages in order."""

    def _vec_setup(self):
        st = stages_from_spec(self.spec)
        self._stage_masks = [jnp.asarray(st == k) for k in range(int(st.max()) + 1)]

    def _decode(self, s, action):
        a_oh = (self._opts[None, None, :] == action[..., None]).astype(jnp.int32)      # [B, M, O]
        lots = jnp.einsum("bmo,mo->bm", a_oh, self._opt_lots)
        k_oh = jnp.einsum("bmo,mok->bmk", a_oh, self._opt_slot_oh)                     # [B, M, K]
        s_oh = jnp.einsum("bmk,mks->bms", k_oh, self._step_oh)                         # [B, M, S]
        last_oh = (self._last_idx[None, None, :] == s.last[..., None]).astype(jnp.int32)
        su = jnp.einsum("bmj,mjk,bmk->bm", last_oh, self._setup, k_oh)
        d = su + jnp.einsum("bmk,mk->bm", k_oh, self._occ)
        return lots, k_oh, s_oh, su, d

    def _staged_take(self, s, lots, s_oh):
        b0 = s.b[:, :, 0]
        free = (lots > 0) & ~self.busy(s)
        n = jnp.zeros_like(lots)
        for mask in self._stage_masks:
            have = jnp.einsum("bms,bs->bm", s_oh, b0)
            n_st = lots * (free & mask & (have >= lots)).astype(jnp.int32)
            b0 = b0 - jnp.einsum("bm,bms->bs", n_st, s_oh)
            n = n + n_st
        return n, b0

    def _count(self, s, lots, k_oh, su, n):
        ok = n > 0
        oki = ok.astype(jnp.int32)
        return dict(runs=s.runs + oki[..., None] * k_oh,
                    lots_in=s.lots_in + n[..., None] * k_oh,
                    setups=s.setups + (ok & (su > 0)).astype(jnp.int32),
                    n_invalid=s.n_invalid + ((lots > 0) & ~ok).astype(jnp.int32).sum(axis=-1))


class VecRegisterVariant(VecMixin, RegisterVariant):
    """Ring-buffer register, one group, starts vectorised over machines. Gradient-safe."""

    def __init__(self, spec: MiniFabSpec):
        RegisterVariant.__init__(self, spec, ring=True, split=False, counters=False)
        self._vec_setup()

    def start(self, s: VState, action) -> VState:
        lots, k_oh, s_oh, su, d = self._decode(s, action)
        n, b0 = self._staged_take(s, lots, s_oh)
        L = self.L[0]
        pos = (s.t + d) % L
        slot_oh = (jnp.arange(L, dtype=jnp.int32)[None, None, :] == pos[..., None]).astype(jnp.int32)
        reg = s.regs[0] + n[:, :, None, None, None] * k_oh[:, :, :, None, None] * slot_oh[:, :, None, None, :]
        return s._replace(b=s.b.at[:, :, 0].set(b0), regs=(reg,), **self._count(s, lots, k_oh, su, n))


class VecSplitVariant(VecMixin, RegisterVariant):
    """Ring-buffer registers sized per machine group (A/B 64, C/D/E 18), starts
    vectorised over machines. Gradient-safe."""

    def __init__(self, spec: MiniFabSpec):
        RegisterVariant.__init__(self, spec, ring=True, split=True, counters=False)
        self._vec_setup()
        self._gidx = [jnp.asarray(g, dtype=jnp.int32) for g in self.groups]

    def start(self, s: VState, action) -> VState:
        lots, k_oh, s_oh, su, d = self._decode(s, action)
        n, b0 = self._staged_take(s, lots, s_oh)
        regs = []
        for gi, reg, L in zip(self._gidx, s.regs, self.L):
            pos = (s.t + d[:, gi]) % L
            slot_oh = (jnp.arange(L, dtype=jnp.int32)[None, None, :] == pos[..., None]).astype(jnp.int32)
            regs.append(reg + n[:, gi][:, :, None, None, None] * k_oh[:, gi][:, :, :, None, None]
                        * slot_oh[:, :, None, None, :])
        return s._replace(b=s.b.at[:, :, 0].set(b0), regs=tuple(regs), **self._count(s, lots, k_oh, su, n))


class VecCountdownVariant(VecMixin, CountdownVariant):
    """No register, starts vectorised over machines. Exact-only."""

    def __init__(self, spec: MiniFabSpec):
        CountdownVariant.__init__(self, spec)
        self._vec_setup()

    def start(self, s: CState, action) -> CState:
        lots, k_oh, s_oh, su, d = self._decode(s, action)
        n, b0 = self._staged_take(s, lots, s_oh)
        ok = n > 0
        return s._replace(
            b=s.b.at[:, :, 0].set(b0),
            free_at=jnp.where(ok, s.t + d, s.free_at),
            jlots=jnp.where(ok, n, s.jlots),
            jslot=jnp.where(ok, (k_oh * jnp.arange(self.K)).sum(-1), s.jslot),
            inside=s.inside + n.sum(axis=-1),
            **self._count(s, lots, k_oh, su, n))


class VecRule:
    """`DispatchRule` vectorised over machines with the same two stages. Same
    decisions, checked by `check_exact` (trajectories identical to the loop rule)."""

    def __init__(self, spec: MiniFabSpec, priority="lbfs", min_batch=3, e_campaign=None):
        assert spec.n_slots == 2
        self.spec, self.e_campaign = spec, e_campaign
        self.e_idx = spec.machines.index("E")
        order = [np.argsort(spec.step_of[m])[::-1] if priority == "lbfs" else np.argsort(spec.step_of[m])
                 for m in range(spec.M)]
        self.p0 = jnp.asarray([int(o[0]) for o in order], dtype=jnp.int32)
        self.p1 = jnp.asarray([int(o[1]) for o in order], dtype=jnp.int32)
        self.mb = jnp.asarray(spec.max_batch, dtype=jnp.int32)
        self.need = jnp.where(self.mb > 1, min_batch, 1)
        self.step_of = jnp.asarray(spec.step_of, dtype=jnp.int32)
        st = stages_from_spec(spec)
        self.masks = [jnp.asarray(st == k) for k in range(int(st.max()) + 1)]

    def init(self, fab, B):
        z = jnp.zeros((B,), jnp.int32)
        return z, z

    def __call__(self, fab, s, pst):
        run_k, run_n = pst
        avail = s.b[:, :, 0]
        B, S = avail.shape
        free = ~fab.busy(s)
        first = jnp.broadcast_to(self.p0, (B, self.spec.M))
        second = jnp.broadcast_to(self.p1, (B, self.spec.M))
        E = self.e_idx
        if self.e_campaign is not None:
            c = jnp.maximum(run_k - 1, 0)
            avail_c = jnp.take_along_axis(avail, self.step_of[E][c][:, None], axis=1)[:, 0]
            swap = (run_k > 0) & (avail_c > 0) & (run_n < self.e_campaign) & (c == self.p1[E])
            first = first.at[:, E].set(jnp.where(swap, self.p1[E], self.p0[E]))
            second = second.at[:, E].set(jnp.where(swap, self.p0[E], self.p1[E]))
        st1 = jnp.take_along_axis(jnp.broadcast_to(self.step_of, (B,) + self.step_of.shape),
                                  first[..., None], axis=2)[..., 0]
        st2 = jnp.take_along_axis(jnp.broadcast_to(self.step_of, (B,) + self.step_of.shape),
                                  second[..., None], axis=2)[..., 0]
        acts = jnp.zeros((B, self.spec.M), jnp.int32)
        for mask in self.masks:
            a1 = jnp.take_along_axis(avail, st1, axis=1)
            a2 = jnp.take_along_axis(avail, st2, axis=1)
            pick1 = free & mask & (a1 >= self.need)
            pick2 = free & mask & ~pick1 & (a2 >= self.need)
            chosen = pick1 | pick2
            slot = jnp.where(pick1, first, second)
            lots = jnp.minimum(self.mb, jnp.where(pick1, a1, a2)) * chosen.astype(jnp.int32)
            acts = jnp.where(chosen, 1 + slot * self.mb + lots - 1, acts)
            st = jnp.where(pick1, st1, st2)
            avail = avail - jnp.einsum("bm,bms->bs", lots,
                                       (jnp.arange(S)[None, None, :] == st[..., None]).astype(jnp.int32))
        chosen_e = acts[:, E] > 0
        slot_e = (acts[:, E] - 1) // self.mb[E]
        same = chosen_e & (slot_e + 1 == run_k)
        run_n = jnp.where(same, run_n + 1, jnp.where(chosen_e, 1, run_n))
        run_k = jnp.where(chosen_e, slot_e + 1, run_k)
        return acts, (run_k, run_n)


class CheapRule:
    """Ablation: a data-dependent but nearly free action, to price the rule itself.
    The tick does the same dense work whatever the action values are."""

    def init(self, fab, B):
        return jnp.zeros((), jnp.int32)

    def __call__(self, fab, s, pst):
        nv = jnp.asarray(fab.spec.n_valid_opts, dtype=jnp.int32)
        return (s.outs[:, None] + s.b[:, 0, :1] + jnp.arange(nv.shape[0])[None, :]) % nv[None, :], pst


def kernels_per_tick(fab, rule, B: int, xs) -> int:
    """Fused kernels in the compiled scan body: each is one pass over the batch."""
    import re
    hlo = fab.compile_policy_rollout(rule).lower(fab.init_state(B), rule.init(fab, B), xs).compile().as_text()
    body = re.search(r"while\([^)]*\),[^\n]*body=%?([\w.\-]+)", hlo).group(1)
    blk = re.search(r"\n%?" + re.escape(body) + r" [^\n]*\{\n(.*?)\n\}", hlo, re.S).group(1)
    return sum(" fusion(" in line for line in blk.split("\n"))


# ---------------------------------------------------------------------------
def check_exact(spec, variant, rates, cfg, rule=None) -> str:
    """Compare a variant (under `rule`, default the loop rule) against TensorMiniFab
    under the loop rule, over the full horizon, closed loop."""
    ref = TensorMiniFab(spec)
    ref_rule = make_rule(spec, "lbfs-full")
    rule = rule or ref_rule
    rel = jnp.asarray(np.stack([constant_rate_schedule(r, cfg["horizon"], cfg["shift_ticks"])
                                for r in rates], axis=1), dtype=jnp.int32)
    xs = (jnp.arange(cfg["horizon"], dtype=jnp.int32), rel)
    fr, tr = ref.compile_policy_rollout(ref_rule, trace=True)(ref.init_state(len(rates)),
                                                              ref_rule.init(ref, len(rates)), xs)
    fv, (vb, vo) = variant.compile_policy_rollout(rule, trace=True)(
        variant.init_state(len(rates)), rule.init(variant, len(rates)), xs)
    if not np.array_equal(np.asarray(tr.b), np.asarray(vb)):
        return "buffers differ"
    if not np.array_equal(np.asarray(tr.outs), np.asarray(vo)):
        return "outs differ"
    for f in ("outs", "sum_wip", "runs", "lots_in", "setups", "n_invalid"):
        if not np.array_equal(np.asarray(getattr(fr, f)), np.asarray(getattr(fv, f))):
            return f"final {f} differs"
    return ""


def four_machine_control(B: int):
    """The existing four-machine tensor line, commander policy, as `tensorfab/bench.py`
    runs it: a same-conditions control for the throughput comparison."""
    from simplefab_1p.config import make_common_config
    from tensorfab import policies as policies4
    from tensorfab.spec import spec_from_common_config
    from tensorfab.tensor_sim import make_line
    spec4 = spec_from_common_config(make_common_config(mode="ALTERNATING_92"))
    line = make_line(spec4)
    fn = line.compile_policy_rollout(policies4.commander(line))
    s0, keys = line.init_state(B), jnp.zeros((spec4.H, 2), dtype=jnp.uint32)
    return (lambda: fn(s0, keys)), spec4.H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 1024, 10000])
    ap.add_argument("--shifts", type=int, default=50, help="timed horizon (throughput is per row-tick)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="interleaved rounds: every variant timed once per round, order rotated")
    args = ap.parse_args()

    params = vars(args) | {"rate": 6.36, "rule": "lbfs-full", "pm": False, "design": "interleaved rounds, median"}
    with Run("minifab_speed_study", params=params) as run:
        cfg_full = make_minifab_config(pm=False)
        spec_full = spec_from_config(cfg_full)
        split_size = 2 * 2 * 64 + 3 * 2 * 18
        # name: (builder, rule kind, gradient-safe, state ints per row)
        variants = {
            "baseline": (TensorMiniFab, "loop", True, 640),
            "ring": (lambda sp: RegisterVariant(sp, ring=True, split=False, counters=False), "loop", True, 640),
            "split": (lambda sp: RegisterVariant(sp, ring=True, split=True, counters=False), "loop", True, split_size),
            "counters": (lambda sp: RegisterVariant(sp, ring=True, split=True, counters=True), "loop", False, split_size),
            "countdown": (CountdownVariant, "loop", False, 15),
            "vec": (VecRegisterVariant, "vec", True, 640),
            "vec-split": (VecSplitVariant, "vec", True, split_size),
            "vec-countdown": (VecCountdownVariant, "vec", False, 15),
        }

        def rule_for(kind, sp):
            return make_rule(sp, "lbfs-full") if kind == "loop" else VecRule(sp, min_batch=3)

        run.log("exactness vs TensorMiniFab + loop rule (300 shifts, rates 3.0 / 6.18 / 6.64 / 8.0, lbfs-full):")
        for name, (build, kind, _, _) in variants.items():
            if name == "baseline":
                continue
            err = check_exact(spec_full, build(spec_full), [3.0, 6.18, 6.64, 8.0], cfg_full,
                              rule=rule_for(kind, spec_full))
            run.log(f"  {name:14s} {'EXACT' if not err else 'MISMATCH: ' + err}")
            if err:
                raise SystemExit(f"{name} is not exact; refusing to time it")

        ref_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            MiniFab(make_minifab_config(release_rate_per_shift=6.36)).run()
            ref_times.append(time.perf_counter() - t0)
        ref_s = float(np.median(ref_times))
        ref_rate = cfg_full["horizon"] / ref_s
        run.log(f"\nreference Python sim: {ref_s:.3f} s per 300-shift run (median of 3, "
                f"range {min(ref_times):.3f}-{max(ref_times):.3f}) = {ref_rate:,.0f} ticks/s")

        cfg = make_minifab_config(horizon_shifts=args.shifts, release_rate_per_shift=6.36)
        spec = spec_from_config(cfg)
        H = cfg["horizon"]
        xs = (jnp.arange(H, dtype=jnp.int32), jnp.asarray(cfg["release_schedule"], dtype=jnp.int32))

        # every timed configuration: variants, two rule ablations, and the four-machine control
        configs = {name: (build(spec), rule_for(kind, spec)) for name, (build, kind, _, _) in variants.items()}
        configs["baseline, no rule"] = (TensorMiniFab(spec), CheapRule())
        configs["vec, no rule"] = (VecRegisterVariant(spec), CheapRule())
        kernels = {name: kernels_per_tick(fab, rule, 1024, xs) for name, (fab, rule) in configs.items()}

        samples = {}        # (name, B) -> list of row-ticks/s
        for B in args.batches:
            calls, ticks = {}, {}
            for name, (fab, rule) in configs.items():
                fn = fab.compile_policy_rollout(rule)
                s0, p0 = fab.init_state(B), rule.init(fab, B)
                calls[name] = (lambda fn=fn, s0=s0, p0=p0: fn(s0, p0, xs))
                ticks[name] = H
            calls["four-machine line"], ticks["four-machine line"] = four_machine_control(B)
            names = list(calls)
            for name in names:                                    # compile + warm up
                jax.block_until_ready(calls[name]())
            for r in range(args.rounds):
                order = names[r % len(names):] + names[:r % len(names)]
                for name in order:
                    t0 = time.perf_counter()
                    jax.block_until_ready(calls[name]())
                    dt = time.perf_counter() - t0
                    samples.setdefault((name, B), []).append(B * ticks[name] / dt)
                    run.record(variant=name, batch=B, round=r, seconds=dt, row_ticks_per_s=B * ticks[name] / dt)
            run.log(f"  B={B}: {args.rounds} interleaved rounds done")

        med = {k: float(np.median(v)) for k, v in samples.items()}
        run.log(f"\nthroughput, closed-loop lbfs-full, {H:,} ticks; median of {args.rounds} interleaved rounds "
                f"(row-ticks/s; spread = min-max as % of median)")
        run.log(f"  {'variant':18s} {'grad-safe':>9s} {'ints/row':>8s} {'kernels':>7s} "
                + " ".join(f"{'B=' + str(B):>20s}" for B in args.batches))
        rows = list(variants) + ["baseline, no rule", "vec, no rule", "four-machine line"]
        for name in rows:
            safe = variants[name][2] if name in variants else None
            size = variants[name][3] if name in variants else ("68" if name == "four-machine line" else "")
            cells = []
            for B in args.batches:
                v = samples[(name, B)]
                spread = (max(v) - min(v)) / med[(name, B)]
                cells.append(f"{med[(name, B)]:>12,.0f} ±{spread / 2:4.0%}")
            k = kernels.get(name, 17 if name == "four-machine line" else "")
            run.log(f"  {name:18s} {('' if safe is None else 'yes' if safe else 'no'):>9s} {str(size):>8s} "
                    f"{str(k):>7s} " + " ".join(f"{c:>20s}" for c in cells))

        rule_share = {}
        for name, cheap in (("baseline", "baseline, no rule"), ("vec", "vec, no rule")):
            for B in args.batches:
                rule_share[f"{name}@{B}"] = 1 - med[(name, B)] / med[(cheap, B)]
        run.log("\ndispatch rule's share of tick time (1 - with/without): "
                + ", ".join(f"{k} {v:.0%}" for k, v in rule_share.items()))

        peak = {n: max(med[(n, b)] for b in args.batches) for n in variants}
        base = peak["baseline"]
        safe_best = max((n for n in variants if variants[n][2]), key=peak.get)
        exact_best = max(variants, key=peak.get)
        four = max(med[("four-machine line", b)] for b in args.batches)
        run.log(f"four-machine line peak {four:,.0f} row-ticks/s in the same rounds "
                f"= {four / base:.1f}x the Mini-Fab baseline, {four / peak[exact_best]:.1f}x the best exact-only")
        run.set_metrics(reference_ticks_per_s=ref_rate, peak=peak, kernels=kernels, rule_share=rule_share,
                        four_machine_peak=four,
                        median={f"{n}@{b}": r for (n, b), r in med.items()},
                        samples={f"{n}@{b}": v for (n, b), v in samples.items()})
        run.headline(f"Mini-Fab tick: baseline {base/1e6:.1f}M row-ticks/s ({base/ref_rate:.0f}x ref); "
                     f"best gradient-safe {peak[safe_best]/1e6:.1f}M ({safe_best}, {peak[safe_best]/ref_rate:.0f}x); "
                     f"best exact-only {peak[exact_best]/1e6:.1f}M ({exact_best}, {peak[exact_best]/ref_rate:.0f}x)")
        run.log("\n" + "-" * 60)
        run.log(f"reference {ref_rate:,.0f} ticks/s · baseline peak {base:,.0f} ({base/ref_rate:.0f}x) · "
                f"best gradient-safe {safe_best} {peak[safe_best]:,.0f} ({peak[safe_best]/ref_rate:.0f}x) · "
                f"best exact-only {exact_best} {peak[exact_best]:,.0f} ({peak[exact_best]/ref_rate:.0f}x)")


if __name__ == "__main__":
    main()
