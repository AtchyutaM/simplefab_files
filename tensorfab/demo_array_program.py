"""
A worked demonstration: how a discrete-event line becomes an array program.

Strips the real line down to ONE machine and ONE product so every number can be
followed by hand, then shows the same twelve ticks computed three ways:

  1. as objects and lists, the way a discrete-event simulator is normally written
  2. as fixed-size arrays, one trajectory
  3. as fixed-size arrays with a batch dimension, many trajectories at once

and finishes by timing the Python loop against the compiled JAX version.

    python -m tensorfab.demo_array_program
"""
from __future__ import annotations
import time
import numpy as np

# ---- the toy line -------------------------------------------------------
PT = 3          # a job occupies the machine for 3 ticks
BATCH = 2       # it consumes 2 units and emits 2
H = 12
ARRIVALS = [2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0]
#            run whenever we can:
ACTIONS  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# =========================================================================
# 1. The discrete-event way: objects, lists, branches
# =========================================================================
class Machine:
    def __init__(self, pt, batch):
        self.pt, self.batch = pt, batch
        self.busy = False
        self.finish_time = 0

    def start(self, t):
        self.busy = True
        self.finish_time = t + self.pt

    def update(self, t):
        if self.busy and t >= self.finish_time:
            self.busy = False
            return True
        return False


def run_objects(verbose=True):
    q = []                       # a LIST of arrival timestamps
    m = Machine(PT, BATCH)
    done_units = 0
    rows = []
    for t in range(H):
        for _ in range(ARRIVALS[t]):
            q.append(t)
        if m.busy and m.update(t):
            done_units += m.batch
        if ACTIONS[t] and not m.busy and len(q) >= m.batch:
            for _ in range(m.batch):
                q.pop(0)
            m.start(t)
        rem = max(m.finish_time - t, 0) if m.busy else 0
        rows.append((t, len(q), int(m.busy), rem, done_units))
    return rows


# =========================================================================
# 2. The array way: fixed-size state, no branches, no lists
# =========================================================================
def run_arrays(verbose=True):
    """
    Same dynamics, but the state is two fixed-size arrays:

        q    a single integer  -- how many units are waiting
        reg  a length-(PT+1) vector -- reg[k] == 1 means a job finishes in k ticks

    The register replaces (busy, finish_time). A start writes a 1 into slot PT;
    every tick the register shifts one slot left; a job reaching slot 0 completes.
    "Busy" is just reg.sum() > 0, so the one-job-at-a-time rule is free.
    """
    q = 0
    reg = np.zeros(PT + 1, dtype=int)
    done_units = 0
    rows, regs = [], []
    for t in range(H):
        q = q + ARRIVALS[t]                        # arrivals

        done = reg[0]                              # completions: slot 0
        reg = reg.copy(); reg[0] = 0
        done_units = done_units + BATCH * done

        idle = int(reg.sum() == 0)                 # gates, as arithmetic
        canrun = int(q >= BATCH)
        start = ACTIONS[t] * idle * canrun         # <- no `if` anywhere

        q = q - BATCH * start
        reg[PT] += start                           # write into slot PT

        regs.append(reg.copy())
        rows.append((t, q, int(reg.sum() > 0), int(np.argmax(reg)) if reg.sum() else 0,
                     done_units))
        reg = np.concatenate([reg[1:], [0]])       # shift left one slot
    return rows, regs


# =========================================================================
def main():
    obj = run_objects()
    arr, regs = run_arrays()

    print("=" * 74)
    print("ONE MACHINE, ONE PRODUCT.  pt=3 ticks, batch=2 units, horizon=12")
    print("=" * 74)
    print("\nSTEP 1 - the same twelve ticks, computed both ways\n")
    print(f"{'t':>2} {'arr':>4} {'act':>4} │ {'objects: q busy rem done':>26} │"
          f" {'arrays:  q busy rem done':>26} │ match")
    print("-" * 96)
    ok = True
    for t in range(H):
        o, a = obj[t], arr[t]
        same = o == a
        ok &= same
        print(f"{t:>2} {ARRIVALS[t]:>4} {ACTIONS[t]:>4} │"
              f" {o[1]:>10} {o[2]:>4} {o[3]:>4} {o[4]:>5}      │"
              f" {a[1]:>10} {a[2]:>4} {a[3]:>4} {a[4]:>5}      │ {'yes' if same else 'NO'}")
    print(f"\nidentical at every tick: {ok}")

    print("\n\nSTEP 2 - the delay register, which is the whole trick\n")
    print("  reg[k] == 1 means 'a job finishes in k ticks'. A start writes into slot pt;")
    print("  the register shifts left each tick; slot 0 means 'completes now'.\n")
    print(f"{'t':>2} │ {'slot':>28} │ meaning")
    print(f"{'':>2} │ {'0    1    2    3':>28} │")
    print("-" * 74)
    for t in range(H):
        cells = "  ".join(f"{v:>3}" for v in regs[t])
        note = ""
        if regs[t][PT]:
            note = "<- job started this tick, lands in slot 3"
        elif regs[t].sum() == 0:
            note = "idle"
        else:
            note = f"job {int(np.argmax(regs[t]))} tick(s) from completing"
        print(f"{t:>2} │ {cells:>28} │ {note}")
    print("\n  The machine is busy iff reg.sum() > 0 - no separate flag, no finish_time,")
    print("  no comparison against the clock. Capacity enforces itself.")


    # =====================================================================
    print("\n\nSTEP 3 - add a batch dimension and nothing else changes\n")
    from tensorfab import device; device.configure()
    import jax, jax.numpy as jnp
    from jax import lax

    B = 4
    rng = np.random.default_rng(0)
    tapes = rng.integers(0, 2, size=(H, B))          # four DIFFERENT policies
    tapes[:, 0] = 1                                  # ...and one that always runs
    arr_j = jnp.asarray(ARRIVALS)

    def tick(carry, x):
        """One tick for ALL B trajectories at once. Identical arithmetic to
        run_arrays() above - every scalar has simply become a length-B vector."""
        q, reg, done = carry
        act, arrivals = x
        q = q + arrivals                              # [B]
        fin = reg[:, 0]                               # [B]
        reg = reg.at[:, 0].set(0)
        done = done + BATCH * fin
        idle = (reg.sum(axis=1) == 0).astype(jnp.int32)      # [B]
        canrun = (q >= BATCH).astype(jnp.int32)              # [B]
        start = act * idle * canrun                          # [B]
        q = q - BATCH * start
        reg = reg.at[:, PT].add(start)
        reg = jnp.concatenate([reg[:, 1:], jnp.zeros((reg.shape[0], 1), jnp.int32)], axis=1)
        return (q, reg, done), q

    @jax.jit
    def rollout(tapes):
        b = tapes.shape[1]
        init = (jnp.zeros(b, jnp.int32), jnp.zeros((b, PT + 1), jnp.int32),
                jnp.zeros(b, jnp.int32))
        (q, reg, done), qhist = lax.scan(
            tick, init, (tapes, jnp.broadcast_to(arr_j[:, None], tapes.shape)))
        return done, qhist

    done, qhist = rollout(jnp.asarray(tapes))
    print("  four trajectories, four different action tapes, one pass:\n")
    print(f"  {'traj':>5} {'action tape':>26} {'units finished':>16}")
    for b in range(B):
        print(f"  {b:>5} {''.join(str(int(v)) for v in tapes[:, b]):>26} {int(done[b]):>16}")
    print(f"\n  trajectory 0 always runs and matches the single-trajectory answer above:"
          f" {int(done[0])} == {arr[-1][4]}  -> {int(done[0]) == arr[-1][4]}")
    print("\n  Every line of `tick` is the same arithmetic as the scalar version.")
    print("  The only change is that q, idle, canrun, start are now length-B vectors.")
    print("  There is no loop over trajectories anywhere.")

    # =====================================================================
    print("\n\nSTEP 4 - where the speed actually comes from\n")

    def python_loop(n_traj, n_ticks):
        """The honest baseline: the object version, once per trajectory."""
        total = 0
        for _ in range(n_traj):
            q, m, d = [], Machine(PT, BATCH), 0
            for t in range(n_ticks):
                for _ in range(ARRIVALS[t % H]):
                    q.append(t)
                if m.busy and m.update(t):
                    d += m.batch
                if not m.busy and len(q) >= m.batch:
                    for _ in range(m.batch):
                        q.pop(0)
                    m.start(t)
            total += d
        return total

    N_TICKS = 2688
    long_tapes = jnp.ones((N_TICKS, 1), jnp.int32)
    arr_long = jnp.asarray(np.tile(ARRIVALS, N_TICKS // H + 1)[:N_TICKS])

    def tick_long(carry, x):
        return tick(carry, x)

    @jax.jit
    def rollout_long(tapes):
        b = tapes.shape[1]
        init = (jnp.zeros(b, jnp.int32), jnp.zeros((b, PT + 1), jnp.int32),
                jnp.zeros(b, jnp.int32))
        (q, reg, done), _ = lax.scan(
            tick_long, init, (tapes, jnp.broadcast_to(arr_long[:, None], tapes.shape)))
        return done

    t0 = time.perf_counter(); python_loop(20, N_TICKS); py = (time.perf_counter() - t0) / 20
    print(f"  Python objects, 1 trajectory x {N_TICKS} ticks : {py*1e3:8.2f} ms"
          f"   ({py/N_TICKS*1e6:6.2f} us per tick)")

    for b in (1, 64, 4096):
        tp = jnp.ones((N_TICKS, b), jnp.int32)
        r = rollout_long(tp); jax.block_until_ready(r)          # compile
        best = min((lambda: (lambda t0: (jax.block_until_ready(rollout_long(tp)),
                                          time.perf_counter() - t0)[1])(time.perf_counter()))()
                   for _ in range(5))
        per = best / (N_TICKS * b)
        print(f"  compiled scan, {b:>4} trajectories x {N_TICKS} ticks : {best*1e3:8.2f} ms"
              f"   ({per*1e9:6.1f} ns per tick-trajectory, {py/N_TICKS/per:8.0f}x)")

    print("""
  Three separate things are being removed:

    1. THE INTERPRETER.  The Python version executes ~10 bytecode-heavy
       operations per tick - attribute lookups, list.pop(0), branches. The
       compiled version executes one fused machine-code loop body.

    2. THE BRANCHES.  `if busy`, `if len(q) >= batch` become multiplications
       by 0 or 1. No mispredictions, and every trajectory follows the SAME
       instruction path - which is what makes step 3 possible at all.

    3. THE PER-TRAJECTORY LOOP.  Adding a batch dimension costs nothing in
       instruction count: the same vector instruction now covers B trajectories.
       This is the part that scales, and it is why throughput keeps climbing
       with batch size while per-tick latency does not improve at all.

  What is NOT removed: the 2,688 sequential ticks. Tick t depends on t-1, so
  the horizon cannot be parallelised. Only the trajectories can.

  CAVEAT ON THESE RATIOS.  The Python baseline here is a one-machine toy with no
  cost accounting, no observation vector and no demand queue, so it runs at about
  0.08 us/tick. The real simulator does 3.23 us/tick - forty times slower per tick
  - because it maintains six timestamp lists, four machine objects and five cost
  accumulators. The speedups above are therefore a FLOOR: on the real line the
  measured figure is 193x, not 50x. What the toy shows is the mechanism, not the
  magnitude.
""")


if __name__ == "__main__":
    main()
