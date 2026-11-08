"""
Equivalence harness: tensorfab.minifab.tensor_sim vs the reference `minifab/sim.py`.

Same discipline as `tensorfab/verify.py`: drive both simulators with identical
inputs and require agreement on every integer quantity at every tick. Compared
exactly, per tick:

    lots waiting in each of b1..b6
    lots inside each machine, by the step it is running
    sum over each machine's lots of ticks-to-completion (pins the register slot)
    ticks of PM remaining on each machine
    cumulative outs and cumulative refused requests
    (closed-loop only) the option each machine requested

and at the end: runs and lots started per machine and step, setups per machine,
PM jobs per machine, PM violations, total WIP-ticks, and profit.

The reference side is rebuilt from the reference's own logs (buffer log, cost
log, every machine's event log), never from the tensor code.

Gates:
  T1  spec      the tensor spec and option tables match the config and
                `minifab.sim.action_space` exactly
  T2  tapes     random open-loop action tapes, random and constant releases,
                five configurations (PM off/on, setups, cold start)
  T3  replay    every reference policy, FIFO included, recorded and replayed
                open-loop over 300 shifts, PM off and on
  T4  policies  the tensor dispatch rules run closed-loop against the reference
                rules, decisions compared tick by tick, PM off and on
  T5  anchors   V1-V3 of the publication plan measured on the tensor sim itself:
                845 min, saturation output, pull stability (PM on: [T97]'s
                figures; PM off: their capacity-shifted counterparts)
  T6  mutation  planted errors in the tensor spec must be detected

Usage:
    python -m tensorfab.minifab.verify_sim
    python -m tensorfab.minifab.verify_sim --quick
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from tensorfab import device

device.configure()

import jax.numpy as jnp  # noqa: E402

from minifab.config import (constant_rate_schedule, make_minifab_config,  # noqa: E402
                            single_lot_schedule)
from minifab.policies import make_policy  # noqa: E402
from minifab.sim import MiniFab, action_space  # noqa: E402
from tensorfab.minifab.policies import make_rule  # noqa: E402
from tensorfab.minifab.spec import (MiniFabSpec, option_action, option_index,  # noqa: E402
                                    spec_from_config)
from tensorfab.minifab.tensor_sim import TensorMiniFab, profit_breakdown  # noqa: E402

RESULTS: List[bool] = []
PROFIT_TOL = 1e-6
TRACE_FIELDS = ("b", "inside", "run_w", "pm_rem", "outs", "n_invalid")
FINAL_FIELDS = ("runs", "lots_in", "setups", "pm_jobs", "pm_violations", "sum_wip",
                "outs", "n_invalid")


def report(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# the reference side, rebuilt from the reference's own logs
# ---------------------------------------------------------------------------
class Recorder:
    """Wraps a reference policy and keeps every decision it makes."""

    def __init__(self, inner):
        self.inner, self.tape = inner, []

    def decide(self, state):
        a = self.inner.decide(state)
        self.tape.append(dict(a))
        return a


def reference_trace(ref: MiniFab, spec: MiniFabSpec) -> Dict[str, np.ndarray]:
    H = len(ref.buffer_log)
    M, K = spec.M, spec.n_slots
    inside = np.zeros((H, M, K), dtype=np.int64)
    run_w = np.zeros((H, M), dtype=np.int64)
    pm_rem = np.zeros((H, M), dtype=np.int64)
    runs = np.zeros((M, K), dtype=np.int64)
    lots_in = np.zeros((M, K), dtype=np.int64)
    setups = np.zeros(M, dtype=np.int64)
    pm_jobs = np.zeros(M, dtype=np.int64)
    for i, m in enumerate(spec.machines):
        for j in ref.tools[m].event_log:
            lo, hi = j.start, min(j.finish, H)
            ts = np.arange(lo, hi)
            if j.kind == "PM":
                pm_jobs[i] += 1
                pm_rem[lo:hi, i] += j.finish - ts
                continue
            k = int(np.flatnonzero(spec.step_of[i] == j.step - 1)[0])
            n = len(j.lots)
            runs[i, k] += 1
            lots_in[i, k] += n
            setups[i] += j.setup > 0
            inside[lo:hi, i, k] += n
            run_w[lo:hi, i] += n * (j.finish - ts)
    return {
        "b": np.asarray(ref.buffer_log, dtype=np.int64),
        "inside": inside, "run_w": run_w, "pm_rem": pm_rem,
        "outs": np.array([c["outs"] for c in ref.cost_log], dtype=np.int64),
        "n_invalid": np.cumsum([c["invalid_actions"] for c in ref.cost_log]).astype(np.int64),
        "final": {
            "runs": runs, "lots_in": lots_in, "setups": setups, "pm_jobs": pm_jobs,
            "pm_violations": len(ref.pm_violations),
            "sum_wip": int(sum(ref.wip_total_log())),
            "outs": len(ref.outs), "n_invalid": ref.invalid_total,
        },
        "profit": ref.profit_total(),
    }


def tensor_slice(tr, final, spec: MiniFabSpec, b: int) -> Dict[str, Any]:
    out = {f: np.asarray(getattr(tr, f))[:, b].astype(np.int64) for f in TRACE_FIELDS}
    out["action"] = np.asarray(tr.action)[:, b]
    out["final"] = {f: np.asarray(getattr(final, f))[b].astype(np.int64) for f in FINAL_FIELDS}
    out["profit"] = float(profit_breakdown(spec, final)["profit"][b])
    return out


def compare(ref: Dict, got: Dict) -> Optional[str]:
    """None if identical; otherwise where they first differ."""
    for f in TRACE_FIELDS:
        a, g = ref[f], got[f]
        if a.shape != g.shape:
            return f"{f}: shape {a.shape} vs {g.shape}"
        if not np.array_equal(a, g):
            bad = np.argwhere(a != g)[0]
            t = int(bad[0])
            return f"{f} first differs at tick {t}: reference {a[t].tolist()}, tensor {g[t].tolist()}"
    for f in FINAL_FIELDS:
        if not np.array_equal(np.asarray(ref["final"][f]), got["final"][f]):
            return f"final {f}: reference {ref['final'][f]}, tensor {got['final'][f].tolist()}"
    if abs(ref["profit"] - got["profit"]) > PROFIT_TOL * max(1.0, abs(ref["profit"])):
        return f"profit: reference {ref['profit']:.6f}, tensor {got['profit']:.6f}"
    return None


def actions_to_reference(spec: MiniFabSpec, opts: np.ndarray) -> List[Dict[str, Any]]:
    """[H, M] option indices -> list of reference action dicts."""
    return [{m: option_action(spec, i, int(row[i])) for i, m in enumerate(spec.machines)}
            for row in opts]


def actions_from_reference(spec: MiniFabSpec, tape: List[Dict[str, Any]]) -> np.ndarray:
    return np.array([[option_index(spec, i, a.get(m)) for i, m in enumerate(spec.machines)]
                     for a in tape], dtype=np.int32)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gate_spec() -> None:
    print("\nT1  spec and option tables")
    for pm in (False, True):
        cfg = make_minifab_config(pm=pm)
        sp = spec_from_config(cfg)
        space = action_space(cfg)
        same = all([option_action(sp, i, o) for o in range(int(sp.n_valid_opts[i]))] == space[m]
                   for i, m in enumerate(sp.machines))
        round_trip = all(option_index(sp, i, option_action(sp, i, o)) == o
                         for i in range(sp.M) for o in range(int(sp.n_valid_opts[i])))
        occ = {s + 1: int(sp.occ[i, k]) for i in range(sp.M) for k, s in enumerate(sp.step_of[i])}
        report(same and round_trip and occ == {s: cfg["steps"][s]["occupancy"] for s in cfg["steps"]}
               and sp.reg_len == 64 and sp.pm_on == pm,
               f"options = minifab.sim.action_space, occupancies, register length ({'PM on' if pm else 'PM off'})",
               f"options per machine {sp.n_valid_opts.tolist()}, reg_len {sp.reg_len}, "
               f"PM {'on, ' + str(sp.pm_duration) + ' ticks' if sp.pm_on else 'off'}")


def random_options(spec: MiniFabSpec, H: int, rng: np.random.Generator, p_idle: float) -> np.ndarray:
    opts = np.zeros((H, spec.M), dtype=np.int32)
    for i in range(spec.M):
        draw = rng.integers(1, int(spec.n_valid_opts[i]), size=H)
        opts[:, i] = np.where(rng.random(H) < p_idle, 0, draw)
    return opts


def run_tapes(cfg_kw: Dict[str, Any], n: int, shifts: int, rng: np.random.Generator,
              spec_override: Optional[MiniFabSpec] = None) -> List[Optional[str]]:
    """n random tapes, each with its own release pattern; returns per-tape mismatch.

    `spec_override` replaces the spec on the tensor side only. Everything on the
    reference side (translating options into reference actions, rebuilding its
    trace) uses the spec derived from the config, so a planted error cannot leak
    into the yardstick."""
    base = make_minifab_config(horizon_shifts=shifts, **cfg_kw)
    H = base["horizon"]
    ref_sp = spec_from_config(base)
    sp = spec_override or ref_sp
    fab = TensorMiniFab(sp)
    rel, acts, refs = [], [], []
    for j in range(n):
        if j % 4 == 3:
            r = rng.poisson(0.06, size=H).astype(np.int32)          # bursty releases
        else:
            r = np.asarray(constant_rate_schedule(float(rng.choice([2.0, 4.5, 6.18, 7.5, 10.0])),
                                                  H, base["shift_ticks"]), dtype=np.int32)
        a = random_options(ref_sp, H, rng, float(rng.choice([0.2, 0.5, 0.85])))
        cfg = make_minifab_config(horizon_shifts=shifts, release_schedule=r.tolist(), **cfg_kw)
        ref = MiniFab(cfg).run(actions=actions_to_reference(ref_sp, a))
        refs.append(reference_trace(ref, ref_sp))
        rel.append(r)
        acts.append(a)
    final, tr = fab.rollout_tape(jnp.asarray(np.stack(acts, axis=1)),
                                 jnp.asarray(np.stack(rel, axis=1)), trace=True)
    return [compare(refs[j], tensor_slice(tr, final, sp, j)) for j in range(n)]


CONFIGS = [
    ("PM off, setups, cold start (default)", dict(pm=False)),
    ("PM on, setups, cold start", dict(pm=True)),
    ("PM off, no setups", dict(pm=False, setups=False, cold_start_setup=False)),
    ("PM on, no cold start, holding cost", dict(pm=True, cold_start_setup=False,
                                                wip_holding_per_lot_tick=0.02)),
    ("PM off, no cold start, holding cost", dict(pm=False, cold_start_setup=False,
                                                 wip_holding_per_lot_tick=0.02)),
]


def gate_tapes(n: int, shifts: int) -> None:
    print(f"\nT2  random open-loop tapes: {n} per configuration, {shifts} shifts, random releases")
    rng = np.random.default_rng(20261107)
    for label, kw in CONFIGS:
        res = run_tapes(kw, n, shifts, rng)
        bad = [(j, e) for j, e in enumerate(res) if e]
        report(not bad, label, f"{n} tapes identical at every tick" if not bad else f"tape {bad[0][0]}: {bad[0][1]}")


def unstable_rate(pm: bool) -> float:
    return 6.36 if pm else 6.64


def gate_replay(shifts: int) -> None:
    print(f"\nT3  every reference policy recorded and replayed open-loop ({shifts} shifts)")
    for pm in (False, True):
        rates = [3.0, 6.18, unstable_rate(pm), 8.0]
        tag = "PM on" if pm else "PM off"
        cfg0 = make_minifab_config(horizon_shifts=shifts, pm=pm)
        sp = spec_from_config(cfg0)
        fab = TensorMiniFab(sp)
        for pol in ("lbfs", "fifo", "fbfs", "lbfs-full", "fifo-full", "lbfs-full-camp6"):
            refs, tapes, rels = [], [], []
            for r in rates:
                cfg = make_minifab_config(horizon_shifts=shifts, release_rate_per_shift=r, pm=pm)
                rec = Recorder(make_policy(cfg, pol))
                ref = MiniFab(cfg, policy=rec).run()
                refs.append(reference_trace(ref, sp))
                tapes.append(actions_from_reference(sp, rec.tape))
                rels.append(np.asarray(cfg["release_schedule"], dtype=np.int32))
            final, tr = fab.rollout_tape(jnp.asarray(np.stack(tapes, axis=1)),
                                         jnp.asarray(np.stack(rels, axis=1)), trace=True)
            errs = [compare(refs[j], tensor_slice(tr, final, sp, j)) for j in range(len(rates))]
            bad = [(rates[j], e) for j, e in enumerate(errs) if e]
            report(not bad, f"{tag:6s} {pol:16s} rates {rates}",
                   "" if not bad else f"rate {bad[0][0]}: {bad[0][1]}")


def gate_policies(shifts: int) -> None:
    print(f"\nT4  tensor dispatch rules closed-loop vs the reference rules ({shifts} shifts)")
    for pm in (False, True):
        rates = [3.0, 6.18, unstable_rate(pm), 8.0]
        tag = "PM on" if pm else "PM off"
        cfg0 = make_minifab_config(horizon_shifts=shifts, pm=pm)
        sp = spec_from_config(cfg0)
        fab = TensorMiniFab(sp)
        for pol in ("lbfs", "fbfs", "lbfs-full", "fbfs-full", "lbfs-full-camp6", "lbfs-camp3"):
            refs, ref_acts, rels = [], [], []
            for r in rates:
                cfg = make_minifab_config(horizon_shifts=shifts, release_rate_per_shift=r, pm=pm)
                rec = Recorder(make_policy(cfg, pol))
                ref = MiniFab(cfg, policy=rec).run()
                refs.append(reference_trace(ref, sp))
                ref_acts.append(actions_from_reference(sp, rec.tape))
                rels.append(np.asarray(cfg["release_schedule"], dtype=np.int32))
            final, tr = fab.rollout_policy(make_rule(sp, pol), batch=len(rates),
                                           releases=jnp.asarray(np.stack(rels, axis=1)), trace=True)
            bad = []
            for j, r in enumerate(rates):
                got = tensor_slice(tr, final, sp, j)
                e = compare(refs[j], got)
                if e is None and not np.array_equal(got["action"], ref_acts[j]):
                    t = int(np.argwhere(got["action"] != ref_acts[j])[0][0])
                    e = f"decision differs at tick {t}: reference {ref_acts[j][t].tolist()}, tensor {got['action'][t].tolist()}"
                if e:
                    bad.append((r, e))
            report(not bad, f"{tag:6s} {pol:16s} rates {rates}",
                   "decisions and trajectories identical" if not bad else f"rate {bad[0][0]}: {bad[0][1]}")


# ---------------------------------------------------------------------------
# T5: the plan's validation anchors, measured on the tensor simulator
# ---------------------------------------------------------------------------
def tensor_run(pm: bool, rule: str, rates: List[float], shifts: int = 300):
    cfg = make_minifab_config(horizon_shifts=shifts, pm=pm)
    sp = spec_from_config(cfg)
    fab = TensorMiniFab(sp)
    rel = np.stack([constant_rate_schedule(r, cfg["horizon"], cfg["shift_ticks"]) for r in rates], axis=1)
    final, tr = fab.rollout_policy(make_rule(sp, rule), batch=len(rates),
                                   releases=jnp.asarray(rel, dtype=jnp.int32), trace=True)
    return cfg, final, tr


def wip_series(tr, j: int, S: int):
    wip = np.asarray(tr.b)[:, j].sum(-1) + np.asarray(tr.inside)[:, j].sum(axis=(-1, -2))
    b3 = np.asarray(tr.b)[:, j, 2]
    return wip.reshape(-1, S).mean(axis=1), b3.reshape(-1, S).mean(axis=1)


def trend(series: np.ndarray, start: int = 150) -> float:
    return float(np.polyfit(np.arange(start, len(series)), series[start:], 1)[0])


def gate_anchors() -> None:
    print("\nT5  validation anchors measured on the tensor simulator")
    # V1: one lot through an empty fab
    for setups, cold, pm, want in ((False, False, False, 845), (True, True, False, 865), (True, True, True, 895)):
        cfg = make_minifab_config(horizon_shifts=2, setups=setups, cold_start_setup=cold, pm=pm,
                                  release_schedule=single_lot_schedule(288, 0))
        sp = spec_from_config(cfg)
        _, tr = TensorMiniFab(sp).rollout_policy(make_rule(sp, "lbfs"), batch=1, trace=True)
        outs = np.asarray(tr.outs)[:, 0]
        tpt = int(np.argmax(outs > 0)) * cfg["tick_minutes"] if outs[-1] else None
        report(tpt == want, f"V1  one lot: {want} min ({'setups' if setups else 'no setups'}, "
               f"{'cold start' if cold else 'no cold start'}, {'PM on' if pm else 'PM off'})", f"TPT {tpt}")

    for pm in (True, False):
        tag = "PM on" if pm else "PM off"
        # V2: saturating release; E's ceiling at ~2 setups a shift
        cfg, final, tr = tensor_run(pm, "lbfs-full-camp6", [8.0])
        S, H = cfg["shift_ticks"], cfg["horizon"]
        outs = np.asarray(tr.outs)[:, 0]
        rate = (outs[-1] - outs[50 * S - 1]) / ((H - 50 * S) / S)
        anchor = 6.38 if pm else 700 / 105
        report(abs(rate - anchor) < 0.03,
               f"V2  saturation output ~ {anchor:.2f} lots/shift ({tag}{', [T97]' if pm else ', derived'})",
               f"measured {rate:.3f}")
        # V3: pull stable / unstable, and campaigns restore stability
        stable, unstable = (6.18, 6.36) if pm else (6.36, 6.64)
        _, _, tr = tensor_run(pm, "lbfs-full", [stable, unstable])
        w_st, _ = wip_series(tr, 0, S)
        w_un, b3_un = wip_series(tr, 1, S)
        _, _, trc = tensor_run(pm, "lbfs-full-camp6", [unstable])
        w_c, _ = wip_series(trc, 0, S)
        s_st, s_un, s_b3, s_c = trend(w_st), trend(w_un), trend(b3_un), trend(w_c)
        report(abs(s_st) < 0.02 and s_un > 0.1 and s_b3 > 0.1 and abs(s_c) < 0.02,
               f"V3  pull stable at {stable}, b3 unbounded at {unstable}, campaigns stabilise ({tag})",
               f"WIP trend {s_st:+.3f} / {s_un:+.3f} (b3 {s_b3:+.3f}) / campaigns {s_c:+.3f} lots/shift")


# ---------------------------------------------------------------------------
# T6: the comparison must catch planted errors in the tensor spec
# ---------------------------------------------------------------------------
def gate_mutation() -> None:
    print("\nT6  planted spec errors must be detected")
    base_off = spec_from_config(make_minifab_config(horizon_shifts=8, pm=False))
    base_on = spec_from_config(make_minifab_config(horizon_shifts=8, pm=True))
    E, A, C = (base_off.machines.index(x) for x in ("E", "A", "C"))

    def with_array(sp, name, fn):
        arr = getattr(sp, name).copy()
        fn(arr)
        return dataclasses.replace(sp, **{name: arr})

    k3 = int(np.flatnonzero(base_off.step_of[E] == 2)[0])
    k5 = int(np.flatnonzero(base_off.step_of[A] == 4)[0])
    mutants = [
        ("S3 on E one tick long", dict(pm=False), with_array(base_off, "occ", lambda a: a.__setitem__((E, k3), a[E, k3] + 1))),
        ("E's setups removed", dict(pm=False), with_array(base_off, "setup", lambda a: a.__setitem__(E, 0))),
        ("E's cold start not a changeover", dict(pm=False), with_array(base_off, "setup", lambda a: a.__setitem__((E, 0), 0))),
        ("S5 on A one tick short", dict(pm=False), with_array(base_off, "occ", lambda a: a.__setitem__((A, k5), a[A, k5] - 1))),
        ("C serves S4 in place of S2", dict(pm=False), with_array(base_off, "step_of", lambda a: a.__setitem__((C, 0), 3))),
        ("A batches at most 2", dict(pm=False), with_array(base_off, "opt_lots", lambda a: a.__setitem__((A, slice(None)), np.minimum(a[A], 2)))),
        ("PM lasts 7 ticks", dict(pm=True), dataclasses.replace(base_on, pm_duration=7)),
        ("PM switched off in a PM-on fab", dict(pm=True), dataclasses.replace(base_on, pm_on=False)),
    ]
    rng = np.random.default_rng(7)
    caught = 0
    for name, kw, sp in mutants:
        res = run_tapes(kw, 6, 8, rng, spec_override=sp)
        hit = next((e for e in res if e), None)
        caught += hit is not None
        print(f"        [{'CAUGHT' if hit else 'MISSED'}] {name:32s} {(hit or '')[:90]}")
    clean = sum(e is not None for kw in (dict(pm=False), dict(pm=True)) for e in run_tapes(kw, 6, 8, rng))
    report(caught == len(mutants) and clean == 0,
           f"{caught}/{len(mutants)} planted errors caught; unmutated spec clean",
           f"{clean} discrepancies on unmutated tapes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer tapes and 60-shift policy runs")
    args = ap.parse_args()
    print("Tensor Mini-Fab vs reference simulator — equivalence and validation")
    gate_spec()
    gate_tapes(n=8 if args.quick else 24, shifts=15)
    gate_replay(shifts=60 if args.quick else 300)
    gate_policies(shifts=60 if args.quick else 300)
    gate_anchors()
    gate_mutation()
    print("\n" + "=" * 64)
    ok = all(RESULTS)
    print(f"TENSOR MINIFAB VERIFICATION {'PASSED' if ok else 'FAILED'}  ({sum(RESULTS)}/{len(RESULTS)} gates)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
