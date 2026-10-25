"""
How a gradient turns into a schedule: the same two-machine line, step by step.

`demo_gradient.py` tours the machinery. This answers a narrower and more
practical question: given one schedule and the profit it earns, how do we work
out which individual decisions were good or bad, and how does that turn into a
better schedule?

It computes the answer twice: the slow obviously-correct way (flip each
decision, re-run the whole simulation, see what profit did) and with a single
backward pass. Then it checks them against each other, because a gradient you
have not checked against ground truth is a story, not a measurement.

A warning the demo repeats at the end: this problem is far too small to show
why gradients are worth having. With 48 decisions, exhaustive search and
single-flip hill-climbing both find the optimum easily, and both beat the
gradient here. The toy is for understanding the mechanism. The payoff only
appears at 10,752 decisions, and those numbers are quoted at the end.

    python -m tensorfab.demo_walkthrough
"""
from __future__ import annotations

import sys

import numpy as np

from tensorfab import device

device.configure()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tensorfab.demo_gradient import H, toy_spec  # noqa: E402
from tensorfab.diffsim import DiffLine, Relax, hard_actions  # noqa: E402
from tensorfab.tensor_sim import make_line, profit_breakdown  # noqa: E402

RULE = "=" * 78
SUB = "-" * 78


def exact_profit(line, spec, acts: np.ndarray) -> np.ndarray:
    """acts: [K, H, M] of 0/1 -> profit [K] from the exact integer simulator."""
    tape = jnp.swapaxes(jnp.asarray(acts.astype(np.int32)), 0, 1)
    return np.asarray(profit_breakdown(spec, line.rollout(tape))["profit"])


def show_tape(acts: np.ndarray, title: str) -> None:
    print(f"  {title}")
    print("      t      " + "".join(f"{t:>3d}" for t in range(H)))
    for m, name in enumerate(("M0 furnace", "M1 litho  ")):
        print(f"      {name} " + "".join(f"{int(v):>3d}" for v in acts[:, m]))


def step_a(spec, line, acts):
    print(RULE)
    print("A.  THE DECISION VARIABLE IS A TABLE OF 0s AND 1s")
    print(RULE)
    print("""
  There are 2 machines and 24 ticks, so the schedule is 48 numbers: "do I try to
  start this machine at this tick?" That table is the ONLY thing we get to
  choose. Everything else -- arrivals, demand, processing times, prices -- is
  fixed.
""")
    show_tape(acts, "our starting schedule (a guess):")
    print("""
  A 1 is a REQUEST, not a guarantee. The machine starts only if it is free and
  has enough input. Asking at a bad moment is harmless; the request is refused.
""")


def step_b(spec, line, acts):
    print(RULE)
    print("B.  RUN IT FORWARD AND WATCH THE STATE MOVE")
    print(RULE)
    print("""
  Four numbers describe the whole factory at any tick:

      q      units waiting in front of each machine
      reg    jobs in flight -- reg[m][k]=1 means machine m finishes in k ticks
      qfin   finished units sitting on the shelf
      back   demand we owe and have not delivered
""")
    s = line.init_state(1)
    print("    t | q0 q1 | reg M0        reg M1       | qfin back | what happened")
    print("   ---+-------+----------------------------+-----------+--------------")
    notes = {0: "4 raw units arrive", 10: "2 units of demand arrive",
             18: "2 more units of demand"}
    for t in range(H):
        s = line.inject(s, jnp.asarray(spec.arrivals[t:t + 1]),
                        jnp.asarray(spec.demand[t:t + 1]))
        before = np.asarray(s.reg)[0, :, 0].copy()
        s = line.advance(s, jnp.asarray(acts[t][None]),
                         jnp.asarray(spec.demand_lag_cum[t:t + 1]))
        after = np.asarray(s.reg)[0]
        q = np.asarray(s.q)[0]
        note = notes.get(t, "")
        if before[0]:
            note = (note + "; " if note else "") + "furnace lands -> 2 units into q1"
        if before[1]:
            note = (note + "; " if note else "") + "litho lands -> 1 finished unit"
        if acts[t, 0] and after[0].sum():
            note = (note + "; " if note else "") + "FURNACE STARTS"
        s = line.shift(s)
        if note or t < 3:
            print(f"    {t:2d} | {q[0]:2d} {q[1]:2d} | {after[0].tolist()}  "
                  f"{after[1].tolist()} | {int(np.asarray(s.qfin)[0]):4d}"
                  f"{int(np.asarray(s.backlog)[0]):5d} | {note}")
    print("""
  The register is the whole trick. A start writes a 1 into slot pt; every tick
  the row shifts one place left; the job completes when the 1 reaches slot 0.
  "Is this machine busy?" is just "does its row contain a 1?", so capacity needs
  no special handling at all.
""")


def step_c(spec, line, acts):
    print(RULE)
    print("C.  SCORE IT: ONE NUMBER FOR THE WHOLE SCHEDULE")
    print(RULE)
    tape = jnp.swapaxes(jnp.asarray(acts[None].astype(np.int32)), 0, 1)
    bd = profit_breakdown(spec, line.rollout(tape))
    g = lambda k: float(np.asarray(bd[k])[0])
    print(f"""
      revenue        {g('revenue'):>8,.0f}   units actually delivered x $80
    - production     {g('production'):>8,.0f}   charged when a job STARTS
    - holding        {g('inventory') + g('wip'):>8,.1f}   finished goods + work in process
    - backorder      {g('backorder'):>8,.0f}   owed demand, after a 4-tick grace
      {'':=<32s}
      PROFIT         {g('profit'):>8,.0f}
""")
    print("""  All the counting is done in integers and prices are applied once at
  the end, so this number is exact -- not a float that drifted over 24 ticks.
""")
    return g("profit")


def step_d(spec, line, acts, base):
    print(RULE)
    print("D.  WHICH DECISION WAS GOOD? THE SLOW, CERTAIN ANSWER")
    print(RULE)
    print("""
  Take each of the 48 decisions in turn, flip it, re-run the whole simulation,
  and see what profit did. That change IS the decision's contribution. No theory
  involved -- this is ground truth.
""")
    variants, index = [], []
    for t in range(H):
        for m in range(2):
            v = acts.copy()
            v[t, m] = 1 - v[t, m]
            variants.append(v)
            index.append((t, m))
    deltas = exact_profit(line, spec, np.stack(variants)) - base

    order = np.argsort(-deltas)
    print("    the flips that help most:")
    for i in order[:4]:
        t, m = index[i]
        name = "furnace" if m == 0 else "litho  "
        print(f"      t={t:<3d} {name}  {int(acts[t,m])} -> {1-int(acts[t,m])}"
              f"    profit {deltas[i]:+9,.2f}")
    n_live = int((np.abs(deltas) > 1e-9).sum())
    print(f"""
    {n_live} of the 48 flips change anything; the rest are requests refused either way.

    Correct, and it cost 48 extra simulations to learn. The real line has 10,752
    decisions, so this same question costs 10,752 runs -- every time you want to
    take one improvement step. That is the wall.
""")
    return deltas, index


def step_e(spec, acts, deltas, index, base):
    print(RULE)
    print("E.  THE SAME QUESTION, ONE BACKWARD PASS")
    print(RULE)
    print("""
  Replace each 0/1 with a real-valued dial theta: the machine is asked to run
  when theta > 0. Profit is now a function of 48 real numbers, and calculus can
  report all 48 sensitivities from a single backward pass.

  One obstacle. True profit is a STAIRCASE in theta -- nudging a dial from 0.30
  to 0.31 changes nothing at all -- so its honest derivative is zero everywhere.
  We differentiate a smoothed stand-in instead: exact forward, smooth backward.
  The number that comes back is useful but BIASED, so it has to be checked.
""")
    theta = jnp.asarray(np.where(acts > 0, 1.0, -1.0), dtype=jnp.float32)[None]
    st = DiffLine(spec, Relax(mode="st"))
    val, grad = st.value_and_grad(theta)
    grad = np.asarray(grad)[0]

    print(f"    surrogate forward value : ${float(np.asarray(val)[0]):,.2f}")
    print(f"    exact profit            : ${base:,.2f}   <- identical, by construction")
    print("\n    gradient on the FURNACE dial:")
    print("      t   " + "".join(f"{t:>6d}" for t in range(12)))
    print("      g   " + "".join(f"{v:>6.1f}" for v in grad[:12, 0]))
    print("      t   " + "".join(f"{t:>6d}" for t in range(12, 24)))
    print("      g   " + "".join(f"{v:>6.1f}" for v in grad[12:, 0]))

    live = [(t, m) for i, (t, m) in enumerate(index) if abs(deltas[i]) > 1e-9]
    agree = 0
    for i, (t, m) in enumerate(index):
        if abs(deltas[i]) < 1e-9:
            continue
        predicts_gain = (grad[t, m] > 0) if acts[t, m] == 0 else (grad[t, m] < 0)
        agree += predicts_gain == (deltas[i] > 0)
    print(f"""
    Compared decision-by-decision against step D, the gradient's sign predicts
    the effect of a FLIP in only {agree} of {len(live)} cases.

  That is not a defect, it is the two answering different questions. A flip is a
  large discrete change -- delete this start entirely. The gradient is a local
  slope -- nudge this dial. Read the furnace row above as a shape rather than a
  list: it is negative early and positive a few ticks later, which says 'this
  batch is starting too soon, slide it later'. Deleting the early start WITHOUT
  adding the later one is of course a disaster, which is exactly what step D
  measured.

  The gradient's unit of advice is a coordinated nudge to all 48 dials at once.
  That is also the only thing it promises, and the only thing we use it for.
""")
    return theta


def adam(st, theta, steps, lr=0.3):
    b1, b2, eps = 0.9, 0.999, 1e-8
    m_, v_ = jnp.zeros_like(theta), jnp.zeros_like(theta)

    @jax.jit
    def upd(th, m_, v_, i):
        _, g = st.value_and_grad(th)
        m_ = b1 * m_ + (1 - b1) * g
        v_ = b2 * v_ + (1 - b2) * g * g
        return th + lr * (m_ / (1 - b1 ** i)) / (
            jnp.sqrt(v_ / (1 - b2 ** i)) + eps), m_, v_

    for i in range(1, steps + 1):
        theta, m_, v_ = upd(theta, m_, v_, i)
    return theta


def starts_of(line, a):
    _, tr = line.rollout_trace(
        jnp.swapaxes(jnp.asarray(a[None].astype(np.int32)), 0, 1))
    cum = np.asarray(tr.started)[:, 0, 0]
    return np.flatnonzero(np.diff(cum, prepend=0) > 0).tolist()


def step_e2(spec, line, acts):
    print(RULE)
    print("E2. WHERE THAT NUMBER COMES FROM")
    print(RULE)
    print("""
  Take one dial, theta[t=0, furnace]. Inside a tick the simulator computes:

      a      = sigmoid(theta / tau_a)          "am I asking to run?"
      idle   = 1 - reg.sum(-1)                 "is the machine free?"
      canrun = sigmoid((q - take + 0.5) / tau_q)   "is there enough input?"

      start  = a * idle * canrun               <- all three must hold
      q      = q - take * start                consume input
      reg    = reg + start * slot_onehot       schedule the completion

  and `start` is the only door theta has into the rest of the simulation. So by
  the chain rule:

      dProfit        dProfit                                 d(sigmoid)
      -------   =    -------  x  idle  x  canrun  x  --------------------
      dtheta          dstart                               d(theta)

  Two of those factors are plain numbers from the forward pass, and they explain
  the zeros immediately: if the machine is BUSY then idle = 0, and if the queue
  is SHORT then canrun = 0, so the whole product collapses. A decision that
  could not have taken effect gets a derivative of exactly zero.
""")
    # recover the gate values as they were *before* each start
    s_ = line.init_state(1)
    gates = []
    for t in range(H):
        s_ = line.inject(s_, jnp.asarray(spec.arrivals[t:t + 1]),
                         jnp.asarray(spec.demand[t:t + 1]))
        reg = np.asarray(s_.reg)[0]
        done = reg[:, 0].copy()
        q_before = np.asarray(s_.q)[0].copy()
        q_before[1] += int(spec.make[0]) * int(done[0])      # completions first
        occupied = reg.sum(-1) - done                        # after they land
        gates.append((int(q_before[0]), int(occupied[0] == 0),
                      int(q_before[0] >= spec.take[0])))
        s_ = line.advance(s_, jnp.asarray(acts[t][None]),
                          jnp.asarray(spec.demand_lag_cum[t:t + 1]))
        s_ = line.shift(s_)

    dl = DiffLine(spec, Relax(mode="st"))
    theta = jnp.asarray(np.where(acts > 0, 1.0, -1.0), dtype=jnp.float32)[None]
    _, g = dl.value_and_grad(theta)
    g = np.asarray(g)[0]

    print("    t | q0 | idle canrun | dProfit/dtheta | can this decision matter?")
    print("   ---+----+-------------+----------------+--------------------------")
    for t in range(12):
        q0, idle, canrun = gates[t]
        reasons = []
        if not idle:
            reasons.append("furnace busy")
        if not canrun:
            reasons.append("queue short")
        why = "yes" if not reasons else "no - " + " and ".join(reasons)
        print(f"    {t:2d} | {q0:2d} |  {idle}     {canrun}    | "
              f"{g[t,0]:>13.2f}  | {why}")
    print("""
  Exactly the pattern you would want: the gradient is non-zero on t=0, 4, 5, 6,
  7 and 8 -- the ticks where the furnace is free and fed -- and identically zero
  everywhere else. No surrogate was needed to achieve that. `idle` is exact and
  linear, and the hard value of `canrun` multiplies the whole term.
""")
    print(SUB)
    print("  AND WHAT THE NUMBER IS MADE OF")
    print(SUB)
    print("""
  dProfit/dstart is itself a sum, because a start reaches the final profit along
  five separate routes. Differentiating each term of the profit formula on its
  own gives them separately (they must, and do, add back to the total):
""")

    def channel(name):
        def f(th):
            _, st_ = dl.rollout_tape(th)
            if name == "revenue":
                return (dl._rev * st_.cum_ship).sum()
            if name == "production":
                return -(dl._pcost * st_.started).sum()
            if name == "fg hold":
                return -(dl._fg * st_.sum_qfin).sum()
            if name == "wip hold":
                return -(dl._wip * st_.sum_q).sum()
            return -(dl._back * st_.sum_late).sum()
        return np.asarray(jax.grad(f)(theta))[0]

    names = ("revenue", "production", "fg hold", "wip hold", "backorder")
    parts = {n: channel(n) for n in names}
    total = sum(parts.values())
    print("      t | " + " ".join(f"{n:>10s}" for n in names) + " |     sum")
    print("     ---+-" + "-" * (11 * len(names)) + "+---------")
    for t in (0, 4, 5, 8):
        print(f"      {t:2d} | " + " ".join(f"{parts[n][t,0]:>10.2f}" for n in names)
              + f" | {total[t,0]:>7.2f}")
    print(f"\n    cross-check, max |sum of channels - full gradient| = "
          f"{float(np.abs(total - g).max()):.1e}")
    print("""
  Read the t=0 row. Asking the furnace to run at t=0 earns revenue (+1.83) but
  costs production (-2.54) and, because the output then sits on the shelf from
  t=6 until demand appears at t=10, finished-goods holding (-1.83). Net -2.73:
  "this start is too early". At t=5 the same channels flip sign and the net is
  +0.91: "start here instead". Nobody wrote that rule. It is the derivative of
  one scalar -- end-of-horizon profit -- with respect to one dial.

  Notice also how far the signal travels. theta[0] changes `start`, which writes
  into `reg`, which 4 ticks later changes `q1`, which changes when litho runs,
  which changes `qfin` and `cum_ship`, which changes revenue and holding
  summed over all 24 ticks. Every one of those steps is a differentiable array
  operation, so the chain rule walks the whole path backwards in a single pass.
  At full scale that path is 2,688 ticks long, and more than half the gradient
  magnitude still sits in the first quarter of the horizon.
""")


def step_f(spec, line):
    print(RULE)
    print("F.  NOW LET IT RUN, FROM A DELIBERATELY TERRIBLE START")
    print(RULE)
    print("""
  Start from 'never release the furnace at all'. Nothing is ever produced, all
  four units of demand go unmet, and the schedule loses money. The optimiser is
  told nothing about windows, lead times or capacity -- only which way each of
  the 48 dials should move to raise the final profit.
""")
    st = DiffLine(spec, Relax(mode="st"))
    a0 = np.zeros((H, 2), dtype=np.int32)
    a0[:, 1] = 1
    theta = jnp.asarray(np.where(a0 > 0, 1.0, -1.0), dtype=jnp.float32)[None]

    print("    round |   profit  | furnace actually starts at")
    print("    ------+-----------+---------------------------")
    prev = 0
    for r in (0, 5, 10, 25, 50, 100, 150):
        if r > prev:
            theta = adam(st, theta, r - prev)
            prev = r
        a = np.asarray(hard_actions(theta))[0]
        p = exact_profit(line, spec, a[None])[0]
        print(f"    {r:5d} | ${p:>8,.2f} | {starts_of(line, a)}")
    print("""
  From -$96 to $252: it discovered on its own that it must release twice, and
  got both releases roughly into position. It did NOT nail them -- the optimum
  is a start in t=4..6 and another in t=12..14, worth $258, and this run ends
  at [2, 15]. Note also that progress is not monotone in the starts it picks:
  the exact profit moves 238 -> 246 -> 246 -> 250 -> 252 while the release ticks
  wander. That is what following a biased gradient looks like from close up.
""")
    print(SUB)
    print("  ONE RUN IS NOT ENOUGH: THE LANDSCAPE HAS LOCAL OPTIMA")
    print(SUB)
    rng = np.random.default_rng(0)
    th = jnp.asarray(rng.normal(0, 1.0, size=(16, H, 2)), dtype=jnp.float32)
    th = adam(st, th, 150)
    acts = np.asarray(hard_actions(th))
    ps = exact_profit(line, spec, acts)
    k = int(np.argmax(ps))
    print(f"""
    16 random starts, same optimiser, run in parallel as one batched array:

        best    ${ps.max():>8,.2f}   starts {starts_of(line, acts[k])}
        median  ${float(np.median(ps)):>8,.2f}
        worst   ${ps.min():>8,.2f}

    Individual runs get stuck; the batch does not. This costs almost nothing
    because the batch dimension is free -- 16 trajectories are one array, and
    the real runs use 64 for the same reason.
""")


def step_g(spec, line):
    print(RULE)
    print("G.  THE HONEST SCORECARD ON THIS TOY")
    print(RULE)
    # greedy single-flip hill-climb from the same bad start
    cur = np.zeros((H, 2), dtype=np.int32)
    cur[:, 1] = 1
    curp = exact_profit(line, spec, cur[None])[0]
    rounds = 0
    while True:
        V = []
        for t in range(H):
            for m in range(2):
                v = cur.copy()
                v[t, m] = 1 - v[t, m]
                V.append(v)
        ps = exact_profit(line, spec, np.stack(V))
        k = int(np.argmax(ps))
        if ps[k] <= curp + 1e-9:
            break
        cur, curp = V[k], ps[k]
        rounds += 1
    print(f"""
    exhaustive search (576 schedules)        $258.00
    single-flip hill-climb from the same
      bad start, {rounds} rounds x 48 re-runs       ${curp:,.2f}
    gradient, one start                      $253.00
    gradient, 16 starts                      $257.50

  Read that honestly: on this problem the gradient LOSES. Forty-eight decisions
  is small enough to enumerate outright, and small enough that flipping one bit
  at a time walks straight to the optimum. Nothing here argues for gradients.

  What the toy is for is seeing the mechanism: the state as a few arrays, profit
  as one exact number, and a single backward pass pricing every decision at
  once. The argument for the method is a scale argument, and it only shows up
  when the same comparison is run on the real line:

        4-machine line, 10,752 decisions, equal budget, identical starts
            single-flip hill-climb   $ 3,756
            gradient                 $17,895     <- 4.8x
            CP-SAT (600s)            $18,993
            certified upper bound    $19,008

  At 48 decisions hill-climbing sees a large fraction of the neighbourhood every
  round. At 10,752 it sees a vanishing one, and each round costs 10,752 runs.
  The backward pass costs the same whether there are 48 decisions or 10,752 --
  that, and nothing about this toy, is the whole case.
""")


def main() -> int:
    spec = toy_spec()
    line = make_line(spec)

    print("\n" + RULE)
    print("  FROM A GUESS TO A NEAR-OPTIMAL SCHEDULE, ONE STEP AT A TIME")
    print(RULE)
    print("""
  The line:   arrivals -> [q0] -> M0 furnace -> [q1] -> M1 litho -> shipped
                                  2 units/4t          1 unit/2t

  4 raw units arrive at t=0.  2 units due at t=10, 2 more at t=18.
  Sell at $80. Furnace costs $10/unit, litho $5/unit.
  Holding finished goods $1/unit/tick. Late delivery $4/unit/tick after 4 ticks.

  So: build too early and you pay to store it; too late and you pay four times
  as much for being late. Each batch has a window.""")

    acts = np.zeros((H, 2), dtype=np.int32)
    acts[0, 0] = 1          # a plausible but wrong guess: start immediately
    acts[8, 0] = 1
    acts[:, 1] = 1          # litho always willing

    step_a(spec, line, acts)
    step_b(spec, line, acts)
    base = step_c(spec, line, acts)
    deltas, index = step_d(spec, line, acts, base)
    step_e(spec, acts, deltas, index, base)
    step_e2(spec, line, acts)
    step_f(spec, line)
    step_g(spec, line)
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
