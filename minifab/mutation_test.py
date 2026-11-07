"""
Mutation test for the Mini-Fab verification checker.

A checker that never fails proves nothing, so this plants one bug at a time in
the simulator (by patching a method in memory, never on disk), runs a mix of
policies and random action tapes, and confirms `verify.check_run` reports a
discrepancy. Every bug is tried with preventive maintenance on and with it off,
so the checker is shown to be sharp in both modes. It then confirms the
unmodified simulator is clean in both.

Usage:  python -m minifab.mutation_test
"""
import random, copy
import minifab.sim as sim
from minifab.config import make_minifab_config
from minifab.policies import make_policy
from minifab.verify import check_run, random_tape

def runs_for_check(pm):
    out = []
    for pol, rate in (("lbfs-full", 6.18), ("lbfs", 5.0), ("fifo-full-camp6", 7.0)):
        cfg = make_minifab_config(horizon_shifts=20, release_rate_per_shift=rate, pm=pm)
        out.append(sim.MiniFab(cfg, policy=make_policy(cfg, pol)).run())
    rng = random.Random(7)
    for _ in range(4):
        cfg = make_minifab_config(horizon_shifts=8, release_rate_per_shift=6.0, pm=pm)
        out.append(sim.MiniFab(cfg).run(actions=random_tape(cfg, rng, 0.4)))
    return out

orig_process, orig_update = sim.Tool.process, sim.Tool.update
orig_try, orig_complete, orig_pm = sim.MiniFab._try_start, sim.MiniFab._complete, sim.MiniFab._maybe_start_pm

def m_dur(self, step, product, lots, t):          # S3 takes one tick too long
    job = orig_process(self, step, product, lots, t)
    if step == 3:
        job2 = sim.Job(job.kind, job.machine, job.step, job.product, job.lots, job.start, job.setup, job.finish + 1)
        self.job = job2; self.finish_tick = job2.finish; self.event_log[-1] = job2
        return job2
    return job
def m_nosetup(self, step):                         # setup never applied
    return 0
def m_lifo(self, m, action, t):                    # take lots from the back of the buffer
    try:
        step, k, prod = self._parse(action)
    except (TypeError, ValueError):
        return orig_try(self, m, action, t)
    buf = self.buffers[step][prod] if step in self.buffers and 0 <= prod < 1 else None
    if buf is not None and len(buf) >= 2: buf.reverse()
    r = orig_try(self, m, action, t)
    if buf is not None and len(buf) >= 2: buf.reverse()
    return r
def m_batch4(self, m, action, t):                  # A may take 4 lots
    tool = self.tools[m]
    if m == "A": tool.max_batch = 4
    a = (action[0], 4) if (m == "A" and action and action[0] == 1 and len(self.buffers[1][0]) >= 4) else action
    return orig_try(self, m, a, t)
def m_route(self, job, t):                         # S2 output sent to b4 instead of b3
    if job.step == 2:
        for lid in job.lots: self._enter(lid, 4, t)
        return
    return orig_complete(self, job, t)
def m_skip_pm(self, tool, t):                      # PM skipped on every 5th shift
    if (t // self.shift_ticks) % 5 == 4: return
    return orig_pm(self, tool, t)
def m_late_done(self, t):                          # completion seen one tick late
    if self.busy and t >= self.finish_tick + 1:
        return orig_update(self, t)
    return None
def m_elig(self, m, action, t):                    # E may run step 2
    if m == "E" and self.buffers[2][0] and not self.tools["E"].busy:
        self.tools["E"].steps = (2, 3, 6); self.tools["E"].occupancy[2] = 12
        self.tools["E"].setup_times = {**self.tools["E"].setup_times}
        for k in list(self.tools["E"].setup_times): self.tools["E"].setup_times[k] = {**self.tools["E"].setup_times[k], 2: 0}
        self.tools["E"].setup_times.setdefault(2, {2: 0, 3: 0, 6: 0})
        return orig_try(self, m, (2, 1), t)
    return orig_try(self, m, action, t)
def m_lose(self, job, t):                          # one lot vanishes at S4 completion
    if job.step == 4 and job.lots and job.lots[0] % 7 == 0:
        job = sim.Job(job.kind, job.machine, job.step, job.product, job.lots[1:], job.start, job.setup, job.finish)
    return orig_complete(self, job, t)
def m_pm_when_off(self, tool, t):                  # PM still runs after it is switched off
    if self.pm is None:
        self.pm = {"duration": 6, "min_gap": 72}
        try:
            return orig_pm(self, tool, t)
        finally:
            self.pm = None
    return orig_pm(self, tool, t)

ON, OFF, BOTH = (True,), (False,), (True, False)
mutations = [                                      # name, patch, PM modes it can occur in
    ("S3 runs one tick long", ("Tool", "process", m_dur), BOTH),
    ("E setup never applied", ("Tool", "setup_for", m_nosetup), BOTH),
    ("buffers served last-in-first-out", ("MiniFab", "_try_start", m_lifo), BOTH),
    ("A batches 4 lots", ("MiniFab", "_try_start", m_batch4), BOTH),
    ("S2 output routed to b4", ("MiniFab", "_complete", m_route), BOTH),
    ("PM skipped every 5th shift", ("MiniFab", "_maybe_start_pm", m_skip_pm), ON),
    ("completions seen a tick late", ("Tool", "update", m_late_done), BOTH),
    ("E runs ineligible step 2", ("MiniFab", "_try_start", m_elig), BOTH),
    ("a lot disappears at S4", ("MiniFab", "_complete", m_lose), BOTH),
    ("PM runs although switched off", ("MiniFab", "_maybe_start_pm", m_pm_when_off), OFF),
]
caught = tried = 0
print(f"  {'planted bug':34s} {'PM on':8s} {'PM off':8s} first discrepancy reported")
for name, (cls, attr, fn), modes in mutations:
    target = getattr(sim, cls)
    saved = getattr(target, attr)
    cells, first = [], ""
    for pm in (True, False):
        if pm not in modes:
            cells.append("n/a")
            continue
        setattr(target, attr, fn)
        try:
            errs = []
            for fab in runs_for_check(pm):
                errs += check_run(fab)
            ok = bool(errs)
        except Exception as e:
            ok, errs = True, [f"crashed: {e!r}"]
        finally:
            setattr(target, attr, saved)
        tried += 1
        caught += ok
        cells.append("CAUGHT" if ok else "MISSED")
        first = first or (errs[0][:70] if errs else "")
    print(f"  {name:34s} {cells[0]:8s} {cells[1]:8s} {first}")
# and the unmutated simulator must still be clean, in both modes
clean = {pm: sum(len(check_run(f)) for f in runs_for_check(pm)) for pm in (True, False)}
print(f"\n  {caught}/{tried} planted bugs caught (10 bugs, each in every PM mode it can occur in);"
      f"  unmutated simulator: {clean[True]} discrepancies with PM on, {clean[False]} with PM off")
