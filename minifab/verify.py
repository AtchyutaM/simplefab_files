"""
Verification gates for the Mini-Fab reference simulator (Tier 1).

The checker in this file is deliberately independent of the simulator's
internals. It takes the spec values straight from `config.STEP_SPEC` in minutes,
and rebuilds what must have happened from the lots' recorded histories and the
machines' event logs: every duration, every setup, every hand-off, the contents
of every buffer at every tick, and the work in process. If the simulator and
the reconstruction disagree anywhere, a check fails and says where.

Gates:
  C1  config     the spec transcribes correctly into ticks; 845 min and 6.38/shift
                 follow from it; pm=False removes PM and nothing else
  C2  one lot    exact tick-by-tick path of a single lot, four ways (setups and PM
                 on/off), including the 845-minute anchor
  C4  fuzz       the checker on random action tapes, most of them infeasible, with
                 setups, PM and the cold-start setup toggled per run

and then once with preventive maintenance off (the default) and once with it on:

  C3  invariants the full independent checker on every policy at several release
                 rates over 300 shifts
  C5  identity   Little's-law accounting: total WIP-ticks equal total time in system
  C6  determinism two identical runs are identical
  C7  capacity   saturating release: E is the bottleneck and output matches its
                 time budget (the [T97] 6.38 lots/shift derivation with PM, 6.67
                 without); greedy batches make A/B the bottleneck instead; PM time
                 is exactly 6 ticks per machine per shift, or zero
  C8  pull       [T97]'s pull experiment at 6.18 and 6.36 lots/shift; without PM the
                 same experiment one capacity step up, at 6.36 and 6.64

Usage:  python -m minifab.verify              (all gates, PM off and on)
        python -m minifab.verify --pm off     (per-mode gates with PM off only)
        python -m minifab.verify --quick      (shorter fuzz and policy runs)
"""
from __future__ import annotations

import argparse
import random
import sys
from typing import Any, Dict, List

import numpy as np

from minifab.config import (STEP_SPEC, MACHINE_SPEC, SHIFT_MINUTES, E_SETUP_STEP_MINUTES,
                            PM_MINUTES_PER_SHIFT_TIER1, PM_MIN_GAP_MINUTES,
                            MIN_TPT_MINUTES, BOTTLENECK_LOTS_PER_SHIFT,
                            make_minifab_config, single_lot_schedule, constant_rate_schedule)
from minifab.policies import make_policy
from minifab.sim import MiniFab, action_space

STEPS = tuple(sorted(STEP_SPEC))
RESULTS: List[bool] = []


def report(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# the independent checker
# ---------------------------------------------------------------------------
def spec_occupancy(step: int, tick: int) -> int:
    _, _, load, proc, unload = STEP_SPEC[step]
    return (load + proc + unload) // tick


def spec_setup(machine: str, prev: Any, nxt: int, cfg: Dict[str, Any]) -> int:
    if machine != "E" or not cfg_setups_on(cfg):
        return 0
    tick = cfg["tick_minutes"]
    if prev is None:
        return E_SETUP_STEP_MINUTES // tick if cold_start_on(cfg) else 0
    return 0 if prev == nxt else E_SETUP_STEP_MINUTES // tick


def cfg_setups_on(cfg) -> bool:
    return cfg["setup_times"]["E"][3][6] > 0


def cold_start_on(cfg) -> bool:
    return cfg["setup_times"]["E"][None][3] > 0


def e_ceiling(pm: bool, setups_per_shift: float = 2.0) -> float:
    """E's lots per shift from the spec minutes: what is left of a shift after PM
    and setups, divided by E's minutes per lot (S3 + S6). With PM and two setups
    this is [T97]'s (720 - 30 - 20) / 105 = 6.38; without PM, 700 / 105 = 6.67."""
    pm_min = PM_MINUTES_PER_SHIFT_TIER1 if pm else 0
    e_lot = sum(sum(STEP_SPEC[s][2:]) for s in (3, 6))
    return (SHIFT_MINUTES - pm_min - setups_per_shift * E_SETUP_STEP_MINUTES) / e_lot


def mode(pm: bool) -> str:
    return "PM on" if pm else "PM off"


def check_run(fab: MiniFab) -> List[str]:
    """Return a list of every discrepancy found. Empty list means the run is valid."""
    cfg = fab.cfg
    tick = cfg["tick_minutes"]
    S = SHIFT_MINUTES // tick
    T = len(fab.buffer_log)
    errs: List[str] = []

    def err(msg: str) -> None:
        if len(errs) < 25:
            errs.append(msg)

    # releases --------------------------------------------------------------
    expected_released = int(sum(cfg["release_schedule"][:T]))
    if len(fab.lots) != expected_released:
        err(f"released {len(fab.lots)} lots, schedule says {expected_released}")

    # every lot's history ---------------------------------------------------
    for lot in fab.lots.values():
        if lot.entries.get(1, (None,))[0] != lot.release_tick:
            err(f"lot {lot.lot_id}: entered b1 at {lot.entries.get(1)}, released at {lot.release_tick}")
        done = sorted(lot.runs)
        if done != list(STEPS[:len(done)]):
            err(f"lot {lot.lot_id}: ran steps {done}, not a prefix of 1..6")
            continue
        for s in done:
            m, start, setup, finish = lot.runs[s]
            if m not in STEP_SPEC[s][1]:
                err(f"lot {lot.lot_id}: S{s} on {m}, not eligible")
            entered = lot.entries[s][0]
            if start < entered:
                err(f"lot {lot.lot_id}: S{s} started {start} before entering b{s} at {entered}")
            if finish - start != setup + spec_occupancy(s, tick):
                err(f"lot {lot.lot_id}: S{s} took {finish-start} ticks, spec "
                    f"{setup}+{spec_occupancy(s, tick)}")
            completed = finish < T
            if s < STEPS[-1]:
                nxt = lot.entries.get(s + 1)
                if completed and (nxt is None or nxt[0] != finish):
                    err(f"lot {lot.lot_id}: S{s} finished {finish}, entered b{s+1} at {nxt}")
                if not completed and nxt is not None:
                    err(f"lot {lot.lot_id}: in b{s+1} before S{s} completed")
            else:
                if completed and lot.out_tick != finish:
                    err(f"lot {lot.lot_id}: S6 finished {finish}, out at {lot.out_tick}")
                if not completed and lot.out_tick is not None:
                    err(f"lot {lot.lot_id}: out before S6 completed")
        # a lot that waits must be waiting for the step after its last run
        waiting_for = [s for s in lot.entries if s not in lot.runs]
        if len(waiting_for) > 1:
            err(f"lot {lot.lot_id}: waiting in several buffers {waiting_for}")

    # machines --------------------------------------------------------------
    pm_cfg = cfg["pm"]
    for m, tool in fab.tools.items():
        prev_finish, prev_step = -1, None
        runs_by_lot = 0
        pm_starts: List[int] = []
        for j in tool.event_log:
            if j.start < prev_finish:
                err(f"{m}: job at {j.start} overlaps previous ending {prev_finish}")
            if j.kind == "PM":
                pm_starts.append(j.start)
                if j.finish - j.start != PM_MINUTES_PER_SHIFT_TIER1 // tick:
                    err(f"{m}: PM at {j.start} lasted {j.finish-j.start} ticks")
            else:
                if j.step not in STEPS or m not in STEP_SPEC[j.step][1]:
                    err(f"{m}: ran ineligible step S{j.step}")
                if not 1 <= len(j.lots) <= MACHINE_SPEC[m][1]:
                    err(f"{m}: batch of {len(j.lots)} at {j.start}")
                want = spec_setup(m, prev_step, j.step, cfg)
                if j.setup != want:
                    err(f"{m}: setup {j.setup} at {j.start} after S{prev_step}->S{j.step}, spec {want}")
                for lid in j.lots:
                    r = fab.lots[lid].runs.get(j.step)
                    if r is None or r[0] != m or r[1] != j.start:
                        err(f"{m}: job at {j.start} lists lot {lid}, lot disagrees")
                runs_by_lot += len(j.lots)
                # the next setup is sized by this run only once it has completed,
                # and every run here completes before the next job starts
                prev_step = j.step
            prev_finish = j.finish
        n_lot_runs = sum(1 for lot in fab.lots.values() for r in lot.runs.values() if r[0] == m)
        if n_lot_runs != runs_by_lot:
            err(f"{m}: event log has {runs_by_lot} lot-runs, lots record {n_lot_runs}")

        # preventive maintenance: once per complete shift, inside the shift,
        # window rule respected, and no run starts while PM is due
        if pm_cfg is not None:
            gap = PM_MIN_GAP_MINUTES // tick
            last = None
            for k in range(T // S):
                in_shift = [p for p in pm_starts if k * S <= p < (k + 1) * S]
                if len(in_shift) != 1:
                    err(f"{m}: {len(in_shift)} PMs in shift {k}")
                    continue
                p = in_shift[0]
                opens = k * S if last is None else max(k * S, last + gap)
                if p < opens:
                    err(f"{m}: PM at {p} before its window opened at {opens}")
                if p + PM_MINUTES_PER_SHIFT_TIER1 // tick > (k + 1) * S:
                    err(f"{m}: PM at {p} runs past end of shift {k}")
                for j in tool.event_log:
                    if j.kind == "RUN" and opens <= j.start < p:
                        err(f"{m}: run started at {j.start} while PM was due (window {opens}, PM {p})")
                last = p
        elif pm_starts:
            err(f"{m}: PM ran with PM disabled")

    # FIFO within each step's buffer ------------------------------------------
    for s in STEPS:
        entered = sorted((lot.entries[s][1], lot.lot_id) for lot in fab.lots.values() if s in lot.entries)
        started = [fab.lots[lid].runs[s][1] if s in fab.lots[lid].runs else None for _, lid in entered]
        seen_waiting = False
        last_start = -1
        for st in started:
            if st is None:
                seen_waiting = True
            else:
                if seen_waiting:
                    err(f"b{s}: a later arrival started while an earlier one still waits")
                    break
                if st < last_start:
                    err(f"b{s}: starts out of FIFO order")
                    break
                last_start = st

    # buffer contents and work in process, rebuilt tick by tick ---------------
    buf = np.zeros((T + 1, len(STEPS)), dtype=np.int64)
    inproc = np.zeros(T + 1, dtype=np.int64)
    for lot in fab.lots.values():
        for s, (t_in, _) in lot.entries.items():
            t_out = lot.runs[s][1] if s in lot.runs else T
            if t_in < T:
                buf[t_in, s - 1] += 1
                buf[min(t_out, T), s - 1] -= 1
        for s, (_, start, _, finish) in lot.runs.items():
            if start < T:
                inproc[start] += 1
                inproc[min(finish, T)] -= 1
    buf = np.cumsum(buf, axis=0)[:T]
    inproc = np.cumsum(inproc)[:T]
    logged = np.asarray(fab.buffer_log, dtype=np.int64)
    if not np.array_equal(buf, logged):
        t_bad = int(np.argmax(np.any(buf != logged, axis=1)))
        err(f"buffer contents disagree first at tick {t_bad}: rebuilt {buf[t_bad].tolist()}, "
            f"logged {logged[t_bad].tolist()}")
    if not np.array_equal(inproc, np.asarray(fab.inproc_log)):
        err("lots inside machines disagree with the rebuilt count")

    # conservation: released = waiting + inside machines + out -------------------
    rel = np.cumsum(np.asarray(cfg["release_schedule"][:T], dtype=np.int64))
    outs_by = np.zeros(T + 1, dtype=np.int64)
    for lot in fab.lots.values():
        if lot.out_tick is not None:
            outs_by[lot.out_tick] += 1
    outs_by = np.cumsum(outs_by)[:T]
    if not np.array_equal(logged.sum(axis=1) + np.asarray(fab.inproc_log) + outs_by, rel):
        err("conservation broken: waiting + inside + out != released")
    if len(fab.outs) != int(outs_by[-1] if T else 0):
        err("outs list disagrees with lots' out ticks")
    return errs


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gate_config() -> None:
    print("\nC1  configuration")
    cfg = make_minifab_config(pm=True)
    occ = {s: cfg["steps"][s]["occupancy"] for s in STEPS}
    report(occ == {1: 57, 2: 12, 3: 15, 4: 16, 5: 63, 6: 6},
           "occupancy per step in 5-min ticks", str(occ))
    total = sum(l + p + u for _, _, l, p, u in STEP_SPEC.values())
    report(total == MIN_TPT_MINUTES, "load + process + unload sums to the 845-min anchor", f"{total} min")
    rate = e_ceiling(pm=True)
    report(abs(rate - BOTTLENECK_LOTS_PER_SHIFT) < 0.005,
           "E bottleneck formula reproduces 6.38 lots/shift", f"{rate:.4f}")
    elig = {m: cfg["machines"][m]["steps"] for m in cfg["machine_order"]}
    report(elig == {"A": (1, 5), "B": (1, 5), "C": (2, 4), "D": (2, 4), "E": (3, 6)},
           "eligibility: A,B->S1,S5  C,D->S2,S4  E->S3,S6", str(elig))
    try:
        make_minifab_config(tick_minutes=15)
        report(False, "a tick that does not divide the spec times is refused")
    except ValueError:
        report(True, "a tick that does not divide the spec times is refused")
    sched = constant_rate_schedule(6.18, 43200, 144)
    report(sum(sched) == 1854, "constant 6.18 lots/shift releases exactly 1,854 lots in 300 shifts",
           f"{sum(sched)}")
    off = make_minifab_config(pm=False)
    rest = lambda c: {k: v for k, v in c.items() if k != "pm"}
    report(off["pm"] is None and cfg["pm"] == {"duration": 6, "min_gap": 72} and rest(off) == rest(cfg),
           "pm=False removes PM and changes nothing else in the config",
           f"PM on {cfg['pm']}, PM off {off['pm']}; default: "
           f"{'PM off' if make_minifab_config()['pm'] is None else 'PM on'}")


def gate_single_lot() -> None:
    print("\nC2  one lot through an empty fab")
    cases = [
        (False, False, False, 845, [(1, 0, 57), (2, 57, 69), (3, 69, 84), (4, 84, 100), (5, 100, 163), (6, 163, 169)]),
        (True, False, False, 855, [(1, 0, 57), (2, 57, 69), (3, 69, 84), (4, 84, 100), (5, 100, 163), (6, 163, 171)]),
        (True, True, False, 865, [(1, 0, 57), (2, 57, 69), (3, 69, 86), (4, 86, 102), (5, 102, 165), (6, 165, 173)]),
        (True, True, True, 895, [(1, 6, 63), (2, 63, 75), (3, 75, 92), (4, 92, 108), (5, 108, 171), (6, 171, 179)]),
    ]
    for setups, cold, pm, minutes, path in cases:
        cfg = make_minifab_config(horizon_shifts=2, setups=setups, cold_start_setup=cold, pm=pm,
                                  release_schedule=single_lot_schedule(288, 0))
        # greedy batches: a lone lot must not wait for two more that never come
        fab = MiniFab(cfg, policy=make_policy(cfg, "lbfs")).run()
        lot = fab.lots[0]
        got = [(s, r[1], r[3]) for s, r in sorted(lot.runs.items())]
        tpt = (lot.out_tick - lot.release_tick) * 5 if lot.out_tick is not None else None
        why = {845: "no setups, no PM: the [T97] anchor",
               855: "+10 min E step change S3->S6",
               865: "+10 min E cold start",
               895: "+30 min PM at shift start"}[minutes]
        report(tpt == minutes and got == path and not check_run(fab),
               f"{minutes} min  ({why})", f"TPT {tpt}")


def gate_policies(rates, policies, shifts=300, pm=True) -> None:
    print(f"\nC3  full checker on every policy x release rate ({shifts} shifts each, {mode(pm)})")
    for pol in policies:
        bad = []
        for r in rates:
            cfg = make_minifab_config(horizon_shifts=shifts, release_rate_per_shift=r, pm=pm)
            fab = MiniFab(cfg, policy=make_policy(cfg, pol)).run()
            e = check_run(fab)
            if e or fab.invalid_total or fab.pm_violations:
                bad.append((r, e[:2], fab.invalid_total, fab.pm_violations[:1]))
        report(not bad, f"{pol:18s} rates {rates}", "" if not bad else str(bad[:1]))


def random_tape(cfg, rng: random.Random, p_idle: float) -> List[Dict[str, Any]]:
    space = action_space(cfg)
    tape = []
    for _ in range(cfg["horizon"]):
        a = {}
        for m, opts in space.items():
            a[m] = None if rng.random() < p_idle else rng.choice(opts[1:])
        # occasionally something malformed or ineligible, which must be refused
        if rng.random() < 0.02:
            a[rng.choice(list(space))] = rng.choice([(9, 1), (1, 0), (2, 3), (3, 2), ("x", 1)])
        tape.append(a)
    return tape


def gate_fuzz(n: int) -> None:
    print(f"\nC4  fuzz: {n} random action tapes, setups, PM and cold start toggled")
    rng = random.Random(20261107)
    fails, invalid_seen, n_off = [], 0, 0
    for i in range(n):
        setups, pm, cold = rng.random() < 0.8, rng.random() < 0.5, rng.random() < 0.5
        n_off += not pm
        rate = rng.choice([2.0, 4.5, 6.18, 7.5, 10.0])
        cfg = make_minifab_config(horizon_shifts=rng.choice([3, 8, 15]), release_rate_per_shift=rate,
                                  setups=setups, pm=pm, cold_start_setup=cold)
        tape = random_tape(cfg, rng, p_idle=rng.choice([0.2, 0.5, 0.85]))
        fab = MiniFab(cfg).run(actions=tape)
        invalid_seen += fab.invalid_total
        e = check_run(fab)
        if e:
            fails.append((i, e[:2]))
    report(not fails, "independent checker finds no discrepancy",
           f"{n} runs ({n_off} with PM off), {invalid_seen:,} infeasible requests correctly refused"
           if not fails else str(fails[:2]))


def gate_identity(pm: bool = True) -> None:
    print(f"\nC5  Little's-law accounting ({mode(pm)})")
    # Little's law needs a stable run. The exact identity below holds for any run;
    # the L = lambda*W comparison only means something when WIP is not growing.
    cfg = make_minifab_config(release_rate_per_shift=5.5, pm=pm)
    fab = MiniFab(cfg, policy=make_policy(cfg, "lbfs-full-camp6")).run()
    T = len(fab.buffer_log)
    wip_ticks = int(sum(fab.wip_total_log()))
    in_system = sum((lot.out_tick if lot.out_tick is not None else T) - lot.release_tick
                    for lot in fab.lots.values())
    report(wip_ticks == in_system, "sum of WIP over ticks == sum of lots' time in system (exact)",
           f"{wip_ticks:,} == {in_system:,}")
    m = fab.metrics()
    lam = m["outs"] / T
    w = m["avg_tpt_minutes"] / 5
    report(abs(m["avg_wip_lots"] - lam * w) / m["avg_wip_lots"] < 0.02,
           "average WIP ~ throughput x average TPT (finite-horizon edges < 2%)",
           f"L={m['avg_wip_lots']:.2f}  lambda*W={lam*w:.2f}")


def gate_determinism(pm: bool = True) -> None:
    print(f"\nC6  determinism ({mode(pm)})")
    a = MiniFab(make_minifab_config(release_rate_per_shift=6.18, pm=pm)).run()
    b = MiniFab(make_minifab_config(release_rate_per_shift=6.18, pm=pm)).run()
    same = (a.buffer_log == b.buffer_log and a.outs == b.outs
            and [j for t in a.tools.values() for j in t.event_log]
            == [j for t in b.tools.values() for j in t.event_log])
    report(same, "two identical runs produce identical histories")


def busy_in(tool, t0: int, t1: int, kind: str = None) -> int:
    """Ticks of [t0, t1) a machine spends occupied, clipping jobs at the edges."""
    return sum(max(0, min(j.finish, t1) - max(j.start, t0))
               for j in tool.event_log if kind is None or j.kind == kind)


def gate_capacity(pm: bool = True) -> Dict[str, Any]:
    print(f"\nC7  capacity under saturating release (8 lots/shift, shifts 50-300, {mode(pm)})")
    cfg = make_minifab_config(release_rate_per_shift=8.0, pm=pm)
    S, H, warm, tick = cfg["shift_ticks"], cfg["horizon"], 50, cfg["tick_minutes"]
    pm_t = PM_MINUTES_PER_SHIFT_TIER1 // tick if pm else 0
    e_lot = spec_occupancy(3, tick) + spec_occupancy(6, tick)     # E: 21 ticks per lot
    t0 = warm * S
    shifts_ss = (H - t0) / S
    out, fabs = {}, {}
    for pol in ("lbfs-full-camp6", "lbfs-full", "fifo-full-camp6", "lbfs"):
        fab = MiniFab(cfg, policy=make_policy(cfg, pol)).run()
        fabs[pol] = fab
        outs_ss = sum(1 for i in fab.outs if fab.lots[i].out_tick >= t0)
        rate = outs_ss / shifts_ss
        e = fab.tools["E"]
        e_setups = sum(1 for j in e.event_log if j.kind == "RUN" and j.setup > 0 and j.start >= t0) / shifts_ss
        e_idle = S - busy_in(e, t0, H) / shifts_ss
        formula = (S - pm_t - 2 * e_setups) / e_lot
        bufs = np.asarray(fab.buffer_log[t0:])
        growth = (bufs[-S:].mean(axis=0) - bufs[:S].mean(axis=0)) / (shifts_ss - 1)
        out[pol] = (rate, e_setups, e_idle, formula, growth)
        print(f"        {pol:16s} outs/shift {rate:5.3f}  E setups/shift {e_setups:4.2f}  "
              f"E idle {e_idle*5:5.1f} min/shift  E budget {formula:5.3f}  "
              f"buffer growth/shift b1..b6 [{' '.join(f'{g:+.2f}' for g in growth)}]")
    rate, setups, idle, formula, growth = out["lbfs-full-camp6"]
    report(idle * 5 < 5 and abs(rate - formula) < 0.03,
           "E saturated and output matches E's time budget (lbfs-full-camp6)",
           f"{rate:.3f} vs {formula:.3f}, E idle {idle*5:.1f} min/shift")
    anchor = e_ceiling(pm)
    report(abs(setups - 2) < 0.25 and abs(rate - anchor) < 0.03,
           "at ~2 setups per shift, output ~ the [T97] 6.38 lots/shift" if pm else
           "at ~2 setups per shift, output ~ (720 - 20)/105 = 6.67 lots/shift (derived; [T97] ran with PM)",
           f"{setups:.2f} setups -> {rate:.3f}")
    # Excess release must pile up in front of the binding step. If E binds, b5 and b6
    # (the steps after E's first pass) stay bounded while upstream buffers grow.
    report(abs(growth[4]) < 0.05 and abs(growth[5]) < 0.05 and growth[:3].sum() > 1.0,
           "excess work queues upstream of E, never in b5/b6: E is the binding machine",
           f"growth/shift b5 {growth[4]:+.3f}, b6 {growth[5]:+.3f}, b1-b3 {growth[:3].sum():+.2f}")
    # Greedy LBFS starts S5 with whatever is waiting, so A and B spend their time on
    # small batches. They are then saturated, E is not, and output equals A/B's time
    # budget at the batch sizes actually used: A/B ticks free of PM per shift divided
    # by A/B ticks per lot (S1 occupancy / mean S1 batch + S5 occupancy / mean S5 batch).
    g = fabs["lbfs"]
    ab = [g.tools[m] for m in ("A", "B")]
    runs = [j for tool in ab for j in tool.event_log if j.kind == "RUN" and j.start >= t0]
    k1 = float(np.mean([len(j.lots) for j in runs if j.step == 1]))
    k5 = float(np.mean([len(j.lots) for j in runs if j.step == 5]))
    ab_free = sum(H - t0 - busy_in(tool, t0, H, "PM") for tool in ab) / shifts_ss
    ab_idle = sum(H - t0 - busy_in(tool, t0, H) for tool in ab) / shifts_ss
    ab_budget = ab_free / (spec_occupancy(1, tick) / k1 + spec_occupancy(5, tick) / k5)
    rate_g, _, e_idle_g, _, _ = out["lbfs"]
    report(ab_idle < 1 and e_idle_g > 5 and abs(rate_g - ab_budget) < 0.03,
           "greedy partial batches make A/B the bottleneck; output matches A/B's time budget",
           f"{rate_g:.3f} vs {ab_budget:.3f} at mean batch S1 {k1:.2f}, S5 {k5:.2f}; "
           f"A/B idle {ab_idle*5:.0f} min/shift, E idle {e_idle_g*5:.0f}")
    # PM accounting: exactly one 6-tick PM per machine per shift, or none at all.
    want = pm_t * (H // S)
    got = {m: tt["pm"] for fab in fabs.values() for m, tt in fab.metrics()["tool_time"].items()}
    n_pm_jobs = sum(j.kind == "PM" for fab in fabs.values() for t in fab.tools.values() for j in t.event_log)
    ok = all(v == want for v in got.values()) and (pm or n_pm_jobs == 0)
    report(ok, f"every machine spends exactly {pm_t} ticks per shift in PM ({want:,} over {H // S} shifts)"
           if pm else "no machine spends any time in PM, and no PM job exists",
           f"PM ticks per machine {sorted(set(got.values()))}, PM jobs {n_pm_jobs:,}")
    return out


def pull_run(policy: str, rate: float, pm: bool = True):
    cfg = make_minifab_config(release_rate_per_shift=rate, pm=pm)
    fab = MiniFab(cfg, policy=make_policy(cfg, policy)).run()
    S = cfg["shift_ticks"]
    b3 = np.array([b[2] for b in fab.buffer_log]).reshape(-1, S).mean(axis=1)
    wip = np.array(fab.wip_total_log()).reshape(-1, S).mean(axis=1)
    slope = float(np.polyfit(np.arange(150, 300), wip[150:], 1)[0])
    b3_slope = float(np.polyfit(np.arange(150, 300), b3[150:], 1)[0])
    e_setups = sum(1 for j in fab.tools["E"].event_log if j.kind == "RUN" and j.setup > 0) / 300
    return fab.metrics(), b3, slope, b3_slope, e_setups


def gate_pull(pm: bool = True) -> None:
    # With PM these are [T97]'s rates. Without PM, E gains 30 min a shift, so the
    # same experiment sits one capacity step up: [T97]'s unstable 6.36 must now be
    # stable, and its 6.36 scaled by the capacity ratio 6.667/6.381 (= 6.64) is the
    # rate pull cannot carry.
    stable, unstable = (6.18, 6.36) if pm else (6.36, 6.64)
    print(f"\nC8  [T97] simple-pull experiment, 300 shifts from empty buffers ({mode(pm)})")
    print("        [T97] does not define its pull rule exactly; LBFS with full batches (the")
    print("        built-in Commander) is the standard pull rule for re-entrant lines.")
    if not pm:
        print("        [T97] ran with PM. Without it the test moves up one capacity step:")
        print("        6.36 must now be stable, and 6.64 (6.36 x 6.667/6.381) unstable.")
    res = {}
    for pol in ("lbfs-full", "fifo-full", "lbfs-full-camp6", "lbfs"):
        row = []
        for r in (stable, unstable):
            m, b3, slope, b3s, su = pull_run(pol, r, pm)
            res[(pol, r)] = (m, b3, slope, b3s, su)
            row.append(f"{r}: outs {m['outs_per_shift']:.2f}  E setups {su:.2f}/shift  "
                       f"b3 end {b3[-10:].mean():5.1f}  WIP trend {slope:+.3f}/shift")
        print(f"        {pol:16s} " + "  |  ".join(row))
    _, _, s_st, _, _ = res[("lbfs-full", stable)]
    _, _, s_un, b3s_un, su_un = res[("lbfs-full", unstable)]
    report(abs(s_st) < 0.02, f"pull at {stable} lots/shift is stable", f"WIP trend {s_st:+.3f} lots/shift")
    report(s_un > 0.1 and b3s_un > 0.1,
           f"pull at {unstable} lots/shift is unstable, buffer 3 growing" + (" ([T97] Fig. 6)" if pm else ""),
           f"WIP trend {s_un:+.3f}, b3 trend {b3s_un:+.3f} lots/shift")
    cap = e_ceiling(pm, su_un)
    report(cap < unstable <= e_ceiling(pm) + 0.02,
           f"mechanism: setups cut E below {unstable} (T97's explanation)",
           f"{su_un:.2f} setups/shift -> E capacity {cap:.2f} lots/shift")
    _, _, sc, _, _ = res[("lbfs-full-camp6", unstable)]
    report(abs(sc) < 0.02, f"capping setups with E campaigns makes {unstable} stable again",
           f"WIP trend {sc:+.3f} lots/shift")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pm", choices=("both", "on", "off"), default="both",
                    help="run the per-mode gates with preventive maintenance on, off, or both")
    args = ap.parse_args()
    print("Mini-Fab reference simulator, Tier 1 — verification")
    gate_config()
    gate_single_lot()
    gate_fuzz(40 if args.quick else 200)
    for pm in {"both": (False, True), "on": (True,), "off": (False,)}[args.pm]:
        print("\n" + "-" * 64 + f"\nPREVENTIVE MAINTENANCE {'ON' if pm else 'OFF'}")
        gate_policies(rates=[3.0, 5.0, 6.18, 6.36, 8.0],
                      policies=["lbfs", "fifo", "fbfs", "lbfs-full", "fifo-full", "lbfs-full-camp6"],
                      shifts=60 if args.quick else 300, pm=pm)
        gate_identity(pm)
        gate_determinism(pm)
        gate_capacity(pm)
        gate_pull(pm)
    print("\n" + "=" * 64)
    ok = all(RESULTS)
    print(f"MINIFAB VERIFICATION {'PASSED' if ok else 'FAILED'}  ({sum(RESULTS)}/{len(RESULTS)} gates)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
