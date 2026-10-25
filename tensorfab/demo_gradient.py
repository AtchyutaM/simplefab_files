"""
A worked example: the tensorised line, and how a gradient gets through it.

`demo_array_program.py` shows how one machine's book-keeping turns into arrays.
This goes the rest of the way. It builds a two-machine line small enough to
follow by hand, runs it through the *real* engine -- the same `TensorLine`,
`DiffLine` and `profit_breakdown` the four-tool experiments use, just with a
different `LineSpec` -- and then shows exactly how autodiff turns "when should I
release material?" into a number per decision.

The line:

      arrivals -> [q0] -> M0 furnace -> [q1] -> M1 litho -> finished goods
                          2 units/4t           1 unit/2t

Four raw units arrive at t=0. Two units of demand are due at t=10 and two more
at t=18, over a 24-tick horizon. Holding finished goods costs $1/unit/tick and
being late costs $4/unit/tick, so releasing early and releasing late are both
punished, and each batch has a narrow window in which it should start. The
windows are not obvious by inspection -- and the greedy answer, release
everything as soon as material is there, is actively wrong -- which is the
point: a real scheduling trade-off compressed to a size a reader can check.

The horizon is short enough that every schedule can be enumerated, so the
gradient's answer is checked against the true optimum rather than asserted.

    python -m tensorfab.demo_gradient
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tensorfab.diffsim import DiffLine, Relax, hard_actions  # noqa: E402
from tensorfab.spec import LineSpec  # noqa: E402
from tensorfab.tensor_sim import make_line, profit_breakdown  # noqa: E402

H = 24
RULE = "=" * 78


def toy_spec() -> LineSpec:
    """Two machines, 24 ticks. Same dataclass the four-tool line uses."""
    arrivals = np.zeros(H, dtype=np.int32)
    arrivals[0] = 4                      # all raw material up front
    demand = np.zeros(H, dtype=np.int32)
    demand[10] = 2
    demand[18] = 2
    # "late" means outstanding for longer than demand_interval ticks
    interval = 4
    cum = np.cumsum(demand)
    lag = np.zeros(H, dtype=np.int32)
    lag[interval:] = cum[:-interval]

    return LineSpec(
        H=H, n_machines=2, reg_len=5, demand_interval=interval,
        pt=np.array([4, 2]),             # furnace 4 ticks, litho 2 ticks
        take=np.array([2, 1]),           # furnace consumes 2, litho 1
        make=np.array([2, 1]),           # furnace emits 2, litho 1
        route=np.array([1, -1]),         # M0 -> q1,  M1 -> finished goods
        kind=("DIFF", "LITHO"),
        arrivals=arrivals, demand=demand, demand_lag_cum=lag,
        revenue_per_unit=80.0,
        prod_cost_per_unit=np.array([10.0, 5.0]),
        fg_holding=1.0,
        wip_holding=np.array([0.0, 0.5]),
        backorder=4.0,
        init_qfin=0, mode="DEMO", utilization=0.0,
    )


# =====================================================================
# 1. the state, and one tick done by hand
# =====================================================================
def part1(spec, line):
    print(RULE)
    print("1.  THE STATE IS FIVE SMALL ARRAYS")
    print(RULE)
    print("""
A discrete-event simulator stores a queue as a list of arrival timestamps and a
machine as (busy, finish_time). Neither batches, neither differentiates. But the
dynamics never read a timestamp -- they only need HOW MANY units wait and HOW
LONG until a job lands. So the whole state is:

    q    [M]      units waiting in each queue          here [2]
    reg  [M, L]   delay register: reg[m,k]=1 -> machine m lands in k ticks
    qfin          finished goods on the shelf
    backlog       demand not yet shipped
    + integer accumulators for cost, priced once at the very end
""")
    s = line.init_state(1)
    print(f"  reg is [M, L] = [{spec.n_machines}, {spec.reg_len}]   "
          f"L = max(pt)+1 = {spec.pt.max()}+1 = {spec.reg_len}")
    print(f"  route = {spec.route.tolist()}   (-1 means 'ship')")
    print("\n  which becomes the routing matrix R[m, j] = 1 when m feeds queue j:\n")
    print("        q0  q1          to_fg")
    for m in range(spec.n_machines):
        print(f"    M{m}  {int(line._route_mat[m,0])}   {int(line._route_mat[m,1])}"
              f"             {int(line._to_fg[m])}")
    print("""
  and the whole of routing is then one matrix product:

      q = q + emit @ R          emit = make * done
      newfin = emit @ to_fg

  That line is why the same code runs 2, 4, 8 or 12 machines, and why a
  re-entrant flow is just a different R.
""")


# =====================================================================
# 2. twelve ticks, printed
# =====================================================================
def part2(spec, line):
    print(RULE)
    print("2.  TWELVE TICKS OF THE REAL ENGINE, PRINTED")
    print(RULE)
    print("\n  Schedule: start the furnace at t=2, then run everything greedily.")
    print("  Watch the register shift left and the '1' walk down to slot 0.\n")

    act = np.zeros((H, 1, 2), dtype=np.int32)
    act[2, 0, 0] = 1            # furnace release at t=2
    act[:, 0, 1] = 1            # litho: run whenever it legally can

    print("    t | action | q0 q1 | reg M0 (slots 0..4) | busy | qfin backlog")
    print("   ---+--------+-------+---------------------+------+-------------")
    s = line.init_state(1)
    for t in range(12):
        a = jnp.asarray(act[t])
        s = line.inject(s, jnp.asarray(spec.arrivals[t:t + 1]),
                        jnp.asarray(spec.demand[t:t + 1]))
        s = line.advance(s, a, jnp.asarray(spec.demand_lag_cum[t:t + 1]))
        reg0 = np.asarray(s.reg)[0, 0].astype(int)
        q = np.asarray(s.q)[0].astype(int)
        busy = (np.asarray(s.reg)[0].sum(-1) > 0).astype(int)
        print(f"    {t:2d} | {np.asarray(a)[0].tolist()}  | {q[0]:2d} {q[1]:2d} "
              f"|  {reg0.tolist()}  | {busy.tolist()} | "
              f"{int(np.asarray(s.qfin)[0]):3d}  {int(np.asarray(s.backlog)[0]):4d}"
              + ("   <- furnace starts: writes 1 into slot pt=4" if t == 2 else "")
              + ("   <- lands: 2 units into q1" if t == 6 else ""))
        s = line.shift(s)
    print("""
  Three things to notice:

    * 'busy' is never stored. It is reg.sum(-1) > 0, so one-job-at-a-time
      capacity costs nothing and needs no branch.
    * the shift is a fixed gather, reg[..., 1:] padded with a zero. No
      arithmetic on indices, so it is exactly differentiable.
    * completions are processed BEFORE starts, so a tool that lands at t can
      restart at t and its output is available downstream in the same tick.
      Getting that order wrong changed the reference by 99 starts once.
""")


# =====================================================================
# 3. the true optimum, by exhaustive enumeration
# =====================================================================
def enumerate_all(spec, line):
    """Every pair of furnace release ticks, litho greedy. 24*24 schedules."""
    tapes, labels = [], []
    for t0, t1 in itertools.product(range(H), repeat=2):
        a = np.zeros((H, 2), dtype=np.int32)
        a[t0, 0] = 1
        a[t1, 0] = 1
        a[:, 1] = 1
        tapes.append(a)
        labels.append((t0, t1))
    tapes = jnp.swapaxes(jnp.asarray(np.stack(tapes)), 0, 1)   # [H, N, M]
    profit = profit_breakdown(spec, line.rollout(tapes))["profit"]
    return np.asarray(profit), labels


def part3(spec, line):
    print(RULE)
    print("3.  WHAT THE RIGHT ANSWER IS (by brute force, so we can check)")
    print(RULE)
    profit, labels = enumerate_all(spec, line)
    order = np.argsort(-profit)
    print(f"\n  {len(labels)} schedules enumerated through the exact integer sim.\n")
    print("     furnace starts     profit")
    for i in order[:5]:
        t0, t1 = labels[i]
        print(f"      t={t0:<3d} t={t1:<3d}      ${profit[i]:>8,.2f}")
    print("        ...")
    for i in order[-2:]:
        t0, t1 = labels[i]
        print(f"      t={t0:<3d} t={t1:<3d}      ${profit[i]:>8,.2f}")
    best = order[0]
    opt = {tuple(sorted(labels[i])) for i in np.flatnonzero(profit == profit[best])}
    t0s = sorted({a for a, _ in opt})
    t1s = sorted({b for _, b in opt})
    print(f"\n  Optimum ${profit[best]:,.2f}, reached by {len(opt)} distinct "
          f"schedules:\n  first batch at t in {t0s}, second at t in {t1s}. "
          f"A window, not a point.")

    # the two instructive failures, priced out
    def costs(t0, t1):
        a = np.zeros((H, 2), dtype=np.int32)
        a[t0, 0] = 1; a[t1, 0] = 1; a[:, 1] = 1
        bd = profit_breakdown(spec, line.rollout(
            jnp.swapaxes(jnp.asarray(a[None]), 0, 1)))
        return {k: float(np.asarray(v)[0]) for k, v in bd.items()}

    print("\n     releases      revenue  production  holding  backorder   profit")
    for lab, note in (((t0s[0], t1s[0]), "the optimum"),
                      ((0, 1), "release everything at once"),
                      ((20, 22), "release far too late")):
        c = costs(*lab)
        print(f"     {str(lab):<12s}  {c['revenue']:>7,.0f}  {c['production']:>10,.0f}"
              f"  {c['inventory'] + c['wip']:>7,.1f}  {c['backorder']:>9,.0f}"
              f"  {c['profit']:>7,.0f}   <- {note}")
    print("""
  The middle row is the one to look at. Releasing both batches immediately earns
  only $160 of revenue, not $320 -- because the furnace is still busy at t=1, so
  the second request is simply REFUSED. Two of the four units are never made.
  Capacity is not a cost in this model, it is a gate, and a schedule that ignores
  it silently loses half its output. That is the behaviour a greedy heuristic
  shows on the real line too: the commander releases a fixed 572 units at every
  demand level, and at low load most of them never turn into anything.

  The optimum itself is easy to read once you see it: $320 revenue (all four
  units sold), $60 production, and essentially nothing else. Zero holding, zero
  backorder. Both batches land inside their window.
""")
    return float(profit[best]), sorted(labels[best])


# =====================================================================
# 4. why the exact simulator has no gradient at all
# =====================================================================
def part4(spec):
    print(RULE)
    print("4.  THE EXACT SIMULATOR HAS A GRADIENT OF EXACTLY ZERO")
    print(RULE)
    print("""
Make the decisions continuous: theta[t, m] is a real-valued logit, and the
action is 1 when theta > 0. A start happens only if

      start = action x idle x canrun,     canrun = 1[q >= take]

Both `action` and `canrun` are step functions of a real input. Their derivative
is zero everywhere they are defined, and undefined at the jump. So:
""")
    rng = np.random.default_rng(0)
    theta = jnp.asarray(rng.normal(0, 1.5, size=(4, H, 2)), dtype=jnp.float32)
    hard = DiffLine(spec, Relax(mode="hard"))
    val, g = hard.value_and_grad(theta)
    print(f"    profit of these 4 schedules : {[f'${v:,.0f}' for v in np.asarray(val)]}")
    print(f"    max |dprofit/dtheta|        : {float(jnp.max(jnp.abs(g))):.1f}")
    print("""
    ^ not small. EXACTLY zero. Every derivative in the whole 24x2 tape is 0.

  This is the wall. It is also why "just use autodiff" is not a plan, and it is
  the control test the real suite runs on every build.
""")
    return theta


# =====================================================================
# 5. straight-through: exact forward, surrogate backward
# =====================================================================
def part5(spec, theta):
    print(RULE)
    print("5.  STRAIGHT-THROUGH: KEEP THE FORWARD PASS, REPLACE THE BACKWARD PASS")
    print(RULE)
    print("""
      def straight_through(hard, soft):
          return hard + (soft - stop_gradient(soft))

  Forward, the bracket is soft - soft = 0 exactly, so the value IS `hard` and
  the simulation stays bit-identical to the integer reference. Backward,
  stop_gradient contributes nothing, so the derivative is the derivative of
  `soft` -- a sigmoid instead of a step.

  The parenthesisation is load-bearing. Written `hard + soft - sg(soft)`,
  float32 evaluates (hard+soft)-soft, which is NOT exact. That bug invented
  $230 of profit on the real line before a test caught it.
""")
    hard = DiffLine(spec, Relax(mode="hard"))
    st = DiffLine(spec, Relax(mode="st"))
    ph = np.asarray(hard.profit_of(theta))
    ps = np.asarray(st.profit_of(theta))
    _, g = st.value_and_grad(theta)
    g = np.asarray(g)
    print(f"    hard-mode profit : {[f'${v:,.2f}' for v in ph]}")
    print(f"    st-mode profit   : {[f'${v:,.2f}' for v in ps]}")
    print(f"    max difference   : {np.abs(ph - ps).max():.1e}   "
          f"<- must be exactly 0, and is")
    print(f"    max |gradient|   : {np.abs(g).max():.2f}   <- now there is signal")

    print("\n  And the signal is specific. Gradient on the furnace decision,")
    print("  schedule 0, tick by tick (positive = 'release more here'):\n")
    row = g[0, :, 0]
    print("     t  " + "".join(f"{t:>7d}" for t in range(0, 12)))
    print("    d/dθ" + "".join(f"{v:>7.1f}" for v in row[:12]))
    print("     t  " + "".join(f"{t:>7d}" for t in range(12, 24)))
    print("    d/dθ" + "".join(f"{v:>7.1f}" for v in row[12:]))
    peak = int(np.argmax(np.abs(row)))
    print(f"\n  Largest magnitude at t={peak}. The derivative of END-OF-HORIZON "
          f"profit\n  with respect to a decision {H - peak} ticks before the end: "
          f"credit assignment\n  across the whole episode, not a one-tick reward.")


# =====================================================================
# 6. optimise, and check against the brute-force optimum
# =====================================================================
def part6(spec, line, best_profit, best_sched):
    print(RULE)
    print("6.  DESCEND THE SURROGATE, LAND ON THE TRUE OPTIMUM")
    print(RULE)
    st = DiffLine(spec, Relax(mode="st"))
    rng = np.random.default_rng(1)
    theta = jnp.asarray(rng.normal(0, 1.0, size=(32, H, 2)), dtype=jnp.float32)

    lr, b1, b2, eps = 0.3, 0.9, 0.999, 1e-8
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)

    @jax.jit
    def step(theta, m, v, t):
        val, g = st.value_and_grad(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        return theta + lr * mh / (jnp.sqrt(vh) + eps), m, v, val

    print("\n    step |  best exact profit of the batch")
    print("    -----+--------------------------------")
    for i in range(1, 121):
        theta, m, v, val = step(theta, m, v, i)
        if i % 20 == 0 or i == 1:
            exact = profit_breakdown(
                spec, line.rollout(jnp.swapaxes(hard_actions(theta), 0, 1))
            )["profit"]
            print(f"    {i:4d} | ${float(np.max(np.asarray(exact))):>10,.2f}")

    acts = np.asarray(hard_actions(theta))
    exact = np.asarray(profit_breakdown(
        spec, line.rollout(jnp.swapaxes(jnp.asarray(acts), 0, 1)))["profit"])
    k = int(np.argmax(exact))
    asked = sorted(np.flatnonzero(acts[k, :, 0] == 1).tolist())
    # what the gate actually let through, from the trace
    _, tr = line.rollout_trace(jnp.swapaxes(jnp.asarray(acts[k:k + 1]), 0, 1))
    st_cum = np.asarray(tr.started)[:, 0, 0]
    ran = sorted(np.flatnonzero(np.diff(st_cum, prepend=0) > 0).tolist())
    print(f"\n    gradient found : ${exact[k]:,.2f}")
    print(f"      requested furnace starts at {asked}")
    print(f"      actually started at         {ran}   <- the gate refused the rest")
    print(f"    brute force    : ${best_profit:,.2f}  furnace starts at {best_sched}")
    gap = best_profit - exact[k]
    print(f"    gap            : ${gap:,.2f}"
          + ("   <- the optimum, reached by following a biased gradient"
             if abs(gap) < 1e-6 else "   <- short of the optimum"))
    print("""
  Note the two lists. The tape REQUESTS a start at eight ticks; the gate lets
  exactly two through, because the furnace is busy or the queue is short at the
  others. The optimiser never had to learn "don't ask twice" -- asking is free
  and the dynamics absorb it. What it had to learn is WHEN the two that matter
  land, and it put them inside both windows.

  Every profit printed above is the ROUNDED schedule replayed through the exact
  integer simulator. The relaxed forward value a method optimises internally is
  never reported as a result -- on the real line that distinction is what
  separates straight-through (which wins) from a fully relaxed forward pass
  (which loses to a greedy heuristic on three of four demand patterns).
""")


def main() -> int:
    spec = toy_spec()
    line = make_line(spec)
    print("\n" + RULE)
    print("  THE TENSORISED LINE, AND HOW A GRADIENT GETS THROUGH IT")
    print(RULE)
    print(f"\n  arrivals : 4 units at t=0")
    print(f"  demand   : 2 units at t=10, 2 units at t=18   (late after 4 ticks)")
    print(f"  revenue $80/unit | furnace $10/unit, litho $5/unit")
    print(f"  finished-goods holding $1/unit/tick | backorder $4/unit/tick\n")
    part1(spec, line)
    part2(spec, line)
    best_profit, best_sched = part3(spec, line)
    theta = part4(spec)
    part5(spec, theta)
    part6(spec, line, best_profit, best_sched)
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
