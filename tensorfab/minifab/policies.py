"""
Dispatching rules for the tensor Mini-Fab, matching `minifab/policies.py`.

A rule turns the state after release, completions and PM into one option per
machine. Machines are considered in the fab's order, and the rule keeps a running
count of the lots it has already handed to an earlier machine this tick, so twins
never claim the same lots — exactly as the reference does.

Supported: priority "lbfs" or "fbfs", a minimum batch on A/B, and E campaigns.
Not supported: priority "fifo". It ranks buffers by the age of their head lot,
which a count-only state does not carry. FIFO runs are still verified exactly, by
recording the reference's decisions and replaying them open-loop.

Campaign state (E's current step and run count) is policy state, carried through
the scan next to the fab state.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from tensorfab.minifab.spec import MiniFabSpec


class DispatchRule:
    def __init__(self, spec: MiniFabSpec, priority: str = "lbfs", min_batch: int = 1,
                 e_campaign: Optional[int] = None):
        if priority == "fifo":
            raise NotImplementedError(
                "fifo ranks buffers by head-lot age, which the count-only state does "
                "not carry; replay a reference fifo run open-loop instead")
        if priority not in ("lbfs", "fbfs"):
            raise ValueError(f"unknown priority {priority!r}")
        self.spec = spec
        self.priority = priority
        self.min_batch = int(min_batch)
        self.e_campaign = e_campaign
        self.e_idx = spec.machines.index("E")
        # static priority order of each machine's two slots
        self.order = []
        for m in range(spec.M):
            steps = spec.step_of[m]
            idx = np.argsort(steps)
            self.order.append(tuple(int(i) for i in (idx[::-1] if priority == "lbfs" else idx)))

    def init(self, fab, batch: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        z = jnp.zeros((batch,), dtype=fab.dtype)
        return z, z          # E's campaign slot + 1 (0 = none), and its run count

    def __call__(self, fab, s, pst):
        sp, dt = self.spec, fab.dtype
        run_k, run_n = pst
        avail = s.b[:, :, 0]                                    # [B, S]
        busy = fab.busy(s)
        B = avail.shape[0]
        steps_all = jnp.asarray(sp.step_of, dtype=dt)           # [M, K]
        acts = []
        for m in range(sp.M):
            mb = int(sp.max_batch[m])
            need = self.min_batch if mb > 1 else 1
            p0, p1 = self.order[m]
            first = jnp.full((B,), p0, dtype=dt)
            second = jnp.full((B,), p1, dtype=dt)
            if m == self.e_idx and self.e_campaign is not None:
                c = jnp.maximum(run_k - 1, 0)
                avail_c = jnp.take_along_axis(avail, steps_all[m][c][:, None], axis=1)[:, 0]
                active = (run_k > 0) & (avail_c > 0) & (run_n < self.e_campaign)
                swap = active & (c == p1)
                first = jnp.where(swap, p1, p0)
                second = jnp.where(swap, p0, p1)
            st1, st2 = steps_all[m][first], steps_all[m][second]
            a1 = jnp.take_along_axis(avail, st1[:, None], axis=1)[:, 0]
            a2 = jnp.take_along_axis(avail, st2[:, None], axis=1)[:, 0]
            free = ~busy[:, m]
            pick1 = free & (a1 >= need)
            pick2 = free & ~pick1 & (a2 >= need)
            chosen = pick1 | pick2
            slot = jnp.where(pick1, first, second)
            lots = jnp.minimum(mb, jnp.where(pick1, a1, a2)) * chosen.astype(dt)
            acts.append(jnp.where(chosen, 1 + slot * mb + lots - 1, 0).astype(dt))
            st = jnp.where(pick1, st1, st2)
            avail = avail - (jnp.arange(sp.n_steps, dtype=dt)[None, :] == st[:, None]).astype(dt) * lots[:, None]
            if m == self.e_idx:
                same = chosen & (slot + 1 == run_k)
                run_n = jnp.where(same, run_n + 1, jnp.where(chosen, 1, run_n))
                run_k = jnp.where(chosen, slot + 1, run_k)
        return jnp.stack(acts, axis=1), (run_k, run_n)


def make_rule(spec: MiniFabSpec, name: str) -> DispatchRule:
    """Names as in `minifab.policies.make_policy`: 'lbfs', 'lbfs-full', 'fbfs-camp4'..."""
    parts = name.split("-")
    min_batch = 3 if "full" in parts else 1
    camp = next((int(p[4:]) for p in parts if p.startswith("camp")), None)
    return DispatchRule(spec, priority=parts[0], min_batch=min_batch, e_campaign=camp)
