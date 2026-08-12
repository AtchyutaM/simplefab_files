# `tensorfab` — the SimpleFab line as a batched, differentiable JAX program

**Status:** Phase 1 complete and verified. Phase 2a env wrapper complete and
verified. Phase 2b de-risking spike run — mechanism confirmed, competitive
advantage not yet demonstrated (§ 4B). Built 2026-08-02 on macOS (Apple Silicon,
XLA CPU backend); no GPU numbers yet.

This folder is the implementation of Phase 1 from [`docs/tensorized_line_report.md`](../docs/tensorized_line_report.md).
It is self-contained: nothing outside this folder was modified, and nothing here
imports `torch`, `stable_baselines3`, or `sb3_contrib`. The existing PPO/BC/MILP
pipeline is untouched and still runs exactly as before.

---

## 1. Why this exists (context for a cold start)

Read this section if you are picking the project up without the preceding
conversation in front of you.

The project schedules a 4-machine serial production line (M0 batch/16 ticks →
M1 single/2 → M2 batch/16 → M3 single/2, batch size 4, horizon 2688 ticks).
Three approaches exist in the repo already:

| Approach | Where | Result on ALTERNATING_92 |
|---|---|---|
| CP-SAT MILP (exact, full-information, open-loop) | [`simplefab_1p/optim.py`](../simplefab_1p/optim.py) | **$33,333** (LP bound $33,987, ~2% gap) |
| Behavioral cloning from MILP trajectories | [`scripts_1p/pretrain_bc.py`](../scripts_1p/pretrain_bc.py) | $31,103 (93.3%) |
| Maskable PPO | [`scripts_1p/train_ppo.py`](../scripts_1p/train_ppo.py) | $31,493 best, but *regresses* to ~$30K with more steps |
| Commander greedy heuristic | [`simplefab_1p/sim.py`](../simplefab_1p/sim.py) | $31,300.91 |

Two problems block further progress, both documented in
[`docs/why_bc_isnt_optimal.md`](../docs/why_bc_isnt_optimal.md):

1. **Reward misalignment.** The env's per-tick reward is local (`Δprofit + shaping`).
   Production decisions carry immediate visible costs while their revenue arrives
   much later, so with γ=0.999 the local gradient says "be cautious about
   producing" while the global optimum says "pre-build aggressively for the demand
   spikes." PPO follows the local signal into an under-production equilibrium and
   *erodes* the BC prior. This is structural, not a tuning bug.
2. **Sample cost.** 8M–16M env steps across 16 Python subprocess environments,
   with score-function gradients that are high-variance.

The tensor sim attacks both: it makes sampling 1–2 orders of magnitude cheaper,
and it makes the dynamics differentiable so that a policy can be trained by
backpropagating from *true end-of-episode profit* through the real dynamics
rather than from a local, misaligned proxy reward.

**What it is not.** Not a neural network. Nothing is learned, there are no weights.
It is `simplefab_1p/sim.py` re-expressed as array arithmetic. It does not make the
simulation more accurate (the reference is already exact), does not reduce
single-rollout latency, and does not provide optimality guarantees.

---

## 2. What is in this folder

**Phase 1 — the exact simulator**

| File | Purpose |
|---|---|
| `spec.py` | `LineSpec` — the static description of the line (processing times, schedules, prices), derived from the existing `simplefab_1p.config.make_common_config` so parameters have exactly one source of truth. Plain numpy, no JAX. |
| `tensor_sim.py` | **The core.** `LineState` (the pytree carry), `TensorLine` (the compiled simulator), the one-tick contract, `lax.scan` rollouts, and profit accounting. Start reading here. |
| `policies.py` | Action sources: the greedy `commander` (matching `sim.commander_decide` exactly), plus unmasked and masked Bernoulli for benchmarking and stress-testing. |
| `verify.py` | **The gate.** Drives the tensor sim and the reference `ProductionLine` with identical actions and asserts agreement on every integer quantity at every tick. |
| `bench.py` | Throughput benchmark of the bare dynamics. Re-measures the Python baseline on whatever machine it runs on. |
| `device.py` | The only platform-aware module. Backend selection, x64 toggle, macOS/CUDA warnings. Porting touches this file and nothing else. |

**Phase 2a — the env wrapper** (obs + mask + shaping + reward)

| File | Purpose |
|---|---|
| `env_tensor.py` | `TensorEnv` — the 18-dim observation, action mask, potential-based shaping, and per-tick reward as batched ops. Not a Gymnasium Env: pure functions over `(LineState, tick)` so a whole episode fits in one `lax.scan`. |
| `verify_env.py` | Equivalence gate vs `FabEnv` on obs, mask, reward, base reward, shaping, and invalid counts. |
| `bench_env.py` | Throughput of the full env path vs `FabEnv`. |

**Phase 2b — the differentiable line**

| File | Purpose |
|---|---|
| `diffsim.py` | `DiffLine` — the same dynamics with configurable gates: `hard` (exact, zero gradient), `st` (straight-through: exact forward, surrogate backward), `soft` (fully relaxed forward). Where all the relaxation decisions live. |
| `verify_diffsim.py` | Asserts hard-forward == exact sim, st-forward == hard-forward *bit-for-bit*, and hard-gradient == 0. |
| `spike_openloop.py` | **The de-risking experiment.** Gradient ascent vs hill-climbing vs random search vs commander vs MILP, at equal budget, across BPTT depths. |

**Infrastructure**

| File | Purpose |
|---|---|
| `logbook.py` | Run logging. Every experiment writes a timestamped directory under `runs/` (env capture, params, full transcript, metrics, series) and appends a row to `EXPERIMENTS.md`. |
| `EXPERIMENTS.md` | The auto-appended run index. One row per run; narrative interpretation lives here in the README. |
| `runs/` | Per-run artifacts. |
| `requirements.txt` | Dependencies, with the CPU and CUDA install lines. |

### Commands

```bash
# Confirm what JAX will run on  (do this FIRST on a new machine)
python -m tensorfab.device

# Equivalence gates — run all three after any change to the dynamics
python -m tensorfab.verify --all-modes      # tensor sim   vs sim.py
python -m tensorfab.verify_env              # tensor env   vs env.py
python -m tensorfab.verify_diffsim          # relaxed line vs tensor sim

# Throughput
python -m tensorfab.bench --batches 1 64 1024 10000
python -m tensorfab.bench_env

# The Phase 2b spike
python -m tensorfab.spike_openloop --milp                       # depth sweep
python -m tensorfab.spike_openloop --problem ALTERNATING_92 \
    --horizons 2688 --steps 600 --milp --milp-seconds 180       # hard problem
```

---

## 3. Design decisions worth understanding before editing

### 3.1 The one-tick contract

`run_step(t)` in the reference does, in this exact order:

1. inject arrivals into q0 and demand into the demand queue;
2. instantly match finished-goods inventory against outstanding demand (the
   `while queue_fin and demand` loop) → revenue;
3. completions, machine by machine: M0→q1 (×4), M1→q2, M2→q3 (×4), M3 → ship if
   demand outstanding else store;
4. starts, machine by machine: if requested but busy/starved, count an invalid
   and do nothing; else consume input, charge production, set `finish_time`;
5. accrue FG holding, WIP holding, and backorder cost on the **end-of-tick** state.

Steps 3 and 4 are separate, which has two consequences the tensor version
preserves: a machine completing at tick *t* can be restarted at tick *t*, and a
unit deposited into q1 by M0 at tick *t* is available to M1 at tick *t*.

`step` is split into `inject` (1–2) and `advance` (3–5) because a policy observes
the state *between* them — that is where `sim.commander_decide` is called.

### 3.2 The delay register

The reference stores `(busy, finish_time)` per machine and each queue as a Python
list of arrival timestamps. Only counts and countdowns affect dynamics, so state
collapses to:

```
q        [B, 4]      queue counts q0..q3
reg      [B, 4, 17]  reg[b,m,k] == 1  ⟺  machine m completes a job in k ticks
qfin     [B]         finished-goods inventory
backlog  [B]         outstanding demand
```

A start at tick *t* writes a 1 into slot `pt`; the register shifts left one slot at
the end of every tick; a job reaching slot 0 completes. "Busy" is just
`reg.sum(-1) > 0`, so the one-job-at-a-time capacity constraint comes for free.
Register length is `max(pt) + 1 = 17`.

**Why a register rather than an integer countdown.** `done = reg[..., 0]` is a pure
*selection*, so in Phase 2b the gradient flows from a start decision straight
through to its completion event. An integer countdown would need `rem == 1`, a
comparison whose derivative is zero almost everywhere. The shift is a fixed
gather, also exactly differentiable. This costs `B × 4 × 17` int32s — nothing.

### 3.3 Backorder aging without timestamps

The demand queue is strictly FIFO, so the units still outstanding are always the
most recently arrived ones. Therefore

```
late(t) = max(0, D(t − demand_interval) − cumulative_shipped(t))
```

where `D` is cumulative arrived demand, precomputed in `spec.demand_lag_cum`. This
is the same accounting the CP-SAT model in `optim.py` uses, and it removes the last
reason to store per-unit data.

### 3.4 Integer accumulators, prices applied once

We do **not** accumulate dollars per tick. We accumulate the underlying integer
quantities — `cum_ship`, `started` per machine, `Σqfin`, `Σq_i`, `Σlate` — and
multiply by prices once at the end (`profit_breakdown`). Equivalent by
construction, and it buys three things: no 2688-step float rounding to argue about
in the verification, an all-integer carry, and profit as a *linear* function of the
accumulators, which is convenient for the pathwise gradient work.

### 3.5 Two "can this machine start?" predicates, same test, different instants

Both are faithful to the reference. They apply the *same* availability test —
`reg.sum(-1) == 0 and q >= take` — at different points in the tick, and
conflating the instants will silently break equivalence.

- `TensorLine.action_mask(state)` mirrors `sim.compute_action_mask`, which the
  Gymnasium env calls **before** `run_step`. It sees the pre-arrival,
  pre-completion state.
- `policies.commander` mirrors `sim.commander_decide`, called **inside**
  `run_step` after injection but before completions. It sees post-arrival `q0`.

The subtlety that matters, and that `verify_env.py` caught: the reference
predicate is `not machine.busy and t >= finish_time`, and **`not busy`
dominates**. `busy` stays True until `update()` fires inside `run_step`, so a
machine completing at tick *t* is still busy when the mask is taken and is *not*
offered as runnable — even though the sim would happily restart it at tick *t*.
Hence `reg.sum(-1)`, including slot 0, not `reg[..., 1:]`. (`t >= finish_time` is
vacuous: for an idle machine `finish_time` is always in the past.)

The mask is therefore conservative in two independent ways — a machine finishing
this tick is hidden, and input arriving this tick is invisible — and the env
compounds this by *projecting* refused actions to Idle. Real env behaviour,
reproduced exactly.

---

## 4. Phase 1 results (macOS, XLA CPU backend, 2026-08-02)

### 4.1 Equivalence — PASS

`python -m tensorfab.verify --all-modes` compares 7 action regimes × 4 demand
modes = 28 full 2688-tick trajectories. Regimes: all-run (maximal invalid-action
load), all-idle (pure backorder accounting), Bernoulli at p ∈ {0.9, 0.5, 0.25,
0.1}, and the state-driven commander heuristic. Modes: UNIFORM, ALTERNATING,
ALTERNATING_50, ALTERNATING_92.

Compared **exactly** (`==`), at every one of the 2688 ticks: `q0..q3`, `qfin`,
`backlog`, cumulative shipments, machine busy flags, machine time-remaining,
cumulative starts per machine, invalid-action counts, and all three cost
accumulators. Profit is compared in float64 to a $1e-6 tolerance and agrees to
**~1e-9 or better** — the residual is the reference's own incremental float
accumulation, not a dynamics difference.

Independent cross-check: the tensor commander scores **$31,300.905**, matching the
reference commander and the "~$31K in milliseconds" warm-start figure in
`why_bc_isnt_optimal.md`. All 10,000 trajectories in a batch produce bit-identical
profit.

### 4.2 Throughput — the projection holds, on CPU alone

`python -m tensorfab.bench --batches 1 64 1024 10000 100000 --reps 5`,
mode ALTERNATING_92, commander policy:

| batch | steps/sec | vs Python | ms/batch | µs/episode | compile s |
|---:|---:|---:|---:|---:|---:|
| — (reference Python sim) | 309,485 | 1.0× | — | 8,690 | — |
| 1 | 2,778,654 | 9.0× | 1.0 | 967 | 0.04 |
| 64 | 36,830,132 | 119× | 4.7 | 73 | 0.05 |
| 1,024 | 30,409,261 | 98× | 90.5 | 88 | 0.14 |
| 10,000 | **59,893,949** | **193×** | 449 | 45 | 0.51 |
| 100,000 | 46,325,063 | 150× | 5,803 | 58 | 5.83 |

Reading these honestly:

- **The 1–2 order-of-magnitude projection is confirmed, and beaten, without a GPU.**
  Peak 59.9M steps/sec is 193× the re-measured Python baseline. (The report cited
  432K steps/sec from an earlier measurement; this machine measures 309K today.
  The speedup ratio is what matters and it is computed against the fresh number.)
- **Single-trajectory latency is not the win — exactly as predicted.** B=1 gives
  only 9×, and that 9× is removal of Python interpreter overhead, not parallelism.
- **Scaling is non-monotonic on CPU** (B=64 beats B=1024). That is cache-residency
  and thread-scheduling behaviour on an 18-core CPU, not a bug. Expect a
  *different* and much cleaner curve on GPU, where the knee should sit far higher.
- **Compile time is a fixed one-off of well under a second** at useful batch sizes.
  `lax.scan` compiles the loop *body* once rather than unrolling 2688 copies —
  this is precisely the property that makes JAX the right choice here, and the
  reason a `torch.compile` port would need CUDA Graphs to compete.
- **B=100,000 is past the knee on CPU** (memory bandwidth). Worth re-testing on GPU.

### 4.3 What this does *not* establish

Faster simulation is a samples-per-second lever, not a how-hard-is-the-search
lever. Nothing here has improved any policy yet. The gradient pathway (Phase 2b)
is untested and carries real risk — see § 6.

---

## 4A. Phase 2a partial — the env wrapper, tensorized and verified

**Motivation was a measurement, not a guess.** Profiling the existing stack found
that the wrapper costs more than the dynamics it wraps:

| | steps/sec |
|---|---:|
| bare `ProductionLine.run_step` | 358,277 |
| full `FabEnv.step` + `action_masks` | 56,980–66,553 |
| actual PPO throughput (8M runs, 16 envs, from `no_nudge_experiments.log`) | 13,650 |

The 18-dim observation alone contains a forward `while` scan for the next demand
event. Porting only `sim.py` and leaving the wrapper in Python would have left
the dominant cost in place. Hence `env_tensor.py`.

**Verified** (`python -m tensorfab.verify_env`): 3 shaping configurations ×
3 action regimes × 2688 ticks. All 18 observation dims, the action mask, the
invalid counts, delta-profit, the shaping term, and the total reward agree with
`FabEnv` to float32 round-off or exactly.

**Throughput** (`python -m tensorfab.bench_env`, ALTERNATING_92, XLA CPU):

| batch | steps/sec | vs `FabEnv` |
|---:|---:|---:|
| 1 | 500,555 | 8.8× |
| 64 | 8,927,239 | 157× |
| 1,024 | 12,266,406 | 215× |
| 4,096 | **17,552,338** | **308×** |

Note this is ~3× slower than the bare dynamics (59.9M) — the observation and
shaping really do cost more than the physics, in tensor form as in Python.

### A latent bug in `env.py` (found by the verification, still live)

`FabEnv.step` computes

```python
invalid_selected = int(
    (mask[0, a0] == 0) + (mask[1, a1] == 0) + (mask[2, a2] == 0) + (mask[3, a3] == 0)
)
```

Each term is a numpy `bool_`, and **numpy bool addition is logical OR, not
integer addition** — so `np.True_ + np.True_` is `np.True_`. The result saturates
at 1. `invalid_selected` is an "was anything invalid?" flag, not a 0–4 count.

This is not cosmetic: [`train_ppo.py:167`](../scripts_1p/train_ppo.py#L167) passes
`invalid_action_penalty=-2.0`, so an agent requesting four illegal starts is
penalised exactly as much as one requesting a single illegal start. The gradient
signal telling the policy to stop requesting illegal actions is three-quarters
missing at its worst.

`env_tensor.py` **reproduces this behaviour by default** (`faithful_invalid=True`)
because Phase 2a must be a drop-in replacement that can reproduce existing
results. The true count is always available as `StepOut.invalid_count`, and
`faithful_invalid=False` switches the reward to use it. Fixing it in `env.py`
would change the reward function and invalidate comparison against every logged
run, so that is a deliberate decision to take separately, not a drive-by patch.

---

## 4B. Phase 2b spike — does the surrogate gradient earn its keep?

This is the experiment that tests the strategic thesis. Full rationale in
`spike_openloop.py`'s docstring; runs are in `EXPERIMENTS.md`.

**Design.** Optimise an open-loop action tape `theta[B, H, 4]` — no policy
network, no RL, no value function — so the gradient is isolated. Every method
starts from the *same* random tapes and gets the *same* budget in
forward-rollout-equivalents (one gradient step ≈ 3 forwards). Every reported
number is the rounded tape run through the **exact integer simulator**; relaxed
profit is never reported as a result.

### Result 1: the mechanism works — gradients survive 2688 steps of thresholds

This is the most important finding, and it is positive.

| H | initial ‖g‖ | non-finite grads | grad mass in first quarter | 90% mass spans |
|---:|---:|---:|---:|---:|
| 336 | 7.4e4 | 0 / 400 | 37.2% | 90% of horizon |
| 672 | 8.3e5 | 0 / 400 | 41.9% | 93% |
| 1,344 | 1.1e6 | 0 / 400 | 57.9% | 96% |
| 2,688 | 1.2e7 | 0 / 400 | 60.4% | 98% |
| 2,688 (ALTERNATING_92) | 3.3e7 | 0 / 600 | 51.3% | 96% |

Credit assignment does **not** vanish. At H=2688, *more* than half the gradient
mass sits in the first quarter of the tape — the gradient is telling the
optimiser about decisions 2,000+ ticks before their revenue arrives. That is
precisely the long-range credit assignment PPO's local reward gets wrong, and it
is the entire mechanistic premise of the approach. Suh et al.'s predicted failure
mode (pathwise gradients degenerating through stiff/discontinuous dynamics) **did
not materialise here**. Zero non-finite gradients at any depth.

The gradients do *explode* with depth — four orders of magnitude from H=336 to
H=2688. Adam's per-coordinate normalisation absorbs it here, but any real
training loop needs global-norm clipping.

### Result 2: straight-through beats the fully-relaxed forward, decisively

`grad-soft` (relaxed forward) is consistently **worse than the greedy commander**
— by $565 at H=336 and by $2,275 at H=2688. Optimising the relaxed system
optimises the wrong system, and rounding back to integers throws the gains away.
The report's exact-forward/approximate-backward design choice (§ 4.4) was correct,
and the cheap-looking alternative is a trap.

### Result 3: the gradient does NOT decisively beat zeroth-order search

Exact-sim profit, best of batch, equal budget:

| problem | commander | random | hillclimb | grad-soft | **grad-st** | MILP |
|---|---:|---:|---:|---:|---:|---:|
| self-similar H=336 | 3,923 | 4,101 | **4,155** | 3,359 | 4,153 | 4,221 |
| self-similar H=672 | 8,111 | 8,311 | 8,405 | 7,356 | **8,417** | 8,473 |
| self-similar H=1344 | 16,415 | 16,825 | **16,936** | 15,288 | 16,901 | 16,976 |
| self-similar H=2688 | 33,046 | 33,790 | 33,854 | 30,770 | **33,868** | 33,711 † |
| **ALTERNATING_92** | 31,301 | 32,220 | 32,754 | 30,501 | **32,846** | 33,290 |

† the 90 s CP-SAT run had not converged and is below random search; not a valid
ceiling at that row.

On the easy self-similar family the gradient **ties** hill-climbing: it wins 2 of
4 horizons, loses 2, and every margin is under 0.3%. On the hard bursty problem it
wins by $92 (0.28%) and reaches 98.7% of the MILP optimum versus the commander's
94.0%.

**A methodological finding worth keeping:** the self-similar UNIFORM family cannot
discriminate between optimisers — random search lands within 0.3% of every method.
Method comparisons must use the bursty ALTERNATING problems, where the commander
sits ~6% below the MILP and there is real room to demonstrate skill.

### Verdict: thesis not falsified, but not yet demonstrated

Honest reading, separating what was shown from what was not:

- ✅ **Shown:** the differentiable line produces usable, finite, long-reaching
  gradients through 2,688 discontinuous steps. The headline risk from the
  literature did not bite.
- ✅ **Shown:** exact-forward straight-through is the right design; the relaxed
  forward is actively harmful.
- ❌ **Not shown:** that the gradient is *better* than a good zeroth-order method.
  It ties on easy problems and wins narrowly on hard ones.
- ❌ **Not addressed:** anything about closed-loop policies. Both methods here
  produce an open-loop tape, which is the MILP's game — and the MILP still wins it.

**The most important caveat cuts in the gradient's favour.** Open-loop tape
optimisation is the setting *most favourable to zeroth-order search*: 10,752
independent parameters whose effects are local and nearly separable, where
flipping a random bit is a sensible move. The actual Phase 2b proposition is a
closed-loop policy with ~30K *entangled* parameters, each affecting all 2,688
ticks. There, a zeroth-order estimator needs samples scaling with dimension while
backprop gets the full gradient in one backward pass. So this spike ran the
gradient's *worst* case and it still tied or narrowly won.

That is a "proceed, with tempered expectations and the ES baseline carried
forward as a permanent control" result. It is not a green light, and it is not a
kill.

---

## 5. Running on the Windows/NVIDIA machine

### 5.1 The macOS caveat (why the numbers above are CPU numbers)

Development happened on macOS/Apple Silicon, where **JAX has no CUDA backend**.
Everything above ran on the XLA *CPU* backend. That still gets the main structural
win — the tick loop compiles into a single fused program instead of 27,000 Python
iterations — but it cannot reach the 10⁶–10⁸ steps/sec regime that Brax/Isaac
Gym/EnvPool report, which needs real data-parallel hardware.

`jax-metal` (Apple's Metal plugin) exists but is **experimental and routinely hits
unimplemented ops** on exactly the `scan`/scatter patterns this simulator is built
from. Do not build the GPU story on it. Treat macOS as the CPU reference point and
re-run `bench.py` unchanged on the NVIDIA box for the GPU claim.

### 5.2 The Windows caveat — read this before installing anything

**JAX does not ship official CUDA wheels for native Windows.** `pip install
"jax[cuda12]"` on Windows Python will fail or silently give you a CPU build. The
supported path is **WSL2** (Windows Subsystem for Linux), where JAX uses the normal
Linux CUDA wheels and talks to the Windows NVIDIA driver through WSL's GPU
passthrough. Unofficial community Windows builds exist; they lag badly and are not
worth the debugging time.

### 5.3 Setup steps

```powershell
# 1. In Windows PowerShell (admin), install WSL2 + Ubuntu:
wsl --install -d Ubuntu
# Reboot if prompted. Make sure the Windows-side NVIDIA driver is current —
# that is the ONLY NVIDIA driver you install. Do NOT install a driver inside WSL.
```

```bash
# 2. Inside the WSL2 Ubuntu shell:
nvidia-smi                       # must show the GPU; if not, stop and fix the driver

git clone <this repo> && cd simplefab_files
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install numpy
pip install -U "jax[cuda12]"     # pulls the CUDA runtime as pip packages; no
                                 # system CUDA toolkit install is required

# 3. Confirm the backend BEFORE trusting any number:
python -m tensorfab.device       # must print  backend: gpu   devices: [CudaDevice(id=0)]

# 4. Re-run the gate, then the benchmark:
python -m tensorfab.verify --all-modes
python -m tensorfab.bench --batches 1 64 1024 10000 100000 1000000
```

### 5.4 What to expect and watch for

- **`device.py` warns you** if you are on a non-macOS machine and still land on the
  CPU backend — that means the CUDA wheel did not take. Believe the warning.
- **Verification must still pass on GPU.** It is dtype- and backend-independent, so
  a failure means something real. Run it before benchmarking, every time.
- **Push the batch sizes much higher.** State is only ~`B × 80` int32s
  (`4 + 4×17 + scalars`), so B = 10⁶ is roughly 320 MB — comfortably in range on a
  modern card. The CPU knee at B≈10⁴ is not where the GPU knee will be.
- **Compile time grows with batch size** on this workload; budget a few seconds at
  very large B. It is a one-off per shape.
- **A/B against CPU on the same box** with `device.configure(platform_name="cpu")`
  to separate "GPU hardware" from "XLA compilation" in the speedup, which is a
  distinction the report explicitly asks to keep honest.
- **Nothing else needs to change.** `spec.py`, `tensor_sim.py`, `policies.py`,
  `verify.py`, and `bench.py` contain no platform-conditional code.

---

## 6. Next steps

Updated 2026-08-02 after the env port and the Phase 2b spike.

**Next up — the closed-loop gradient test. This is the real question.**
The open-loop spike ran the gradient's worst case and it tied. The actual
proposition is a reactive policy trained by backpropagating end-of-episode profit
through the dynamics, where the parameterisation shifts decisively away from
zeroth-order methods. Concretely:

1. Attach a small MLP policy inside the `lax.scan`, consuming
   `env_tensor.TensorEnv.observe` and emitting masked action logits.
2. Straight-through only — the spike showed `soft` is a trap.
3. **Global-norm gradient clipping is mandatory**, not optional: ‖g‖ spans four
   orders of magnitude across the depth sweep and hits 3.3e7 at H=2688.
4. `jax.checkpoint` on the scan for BPTT memory once B and the network grow.
5. Carry the hill-climb/ES baseline forward as a permanent control at equal
   budget. If the gradient cannot beat it here either, the thesis is falsified and
   the honest move is to stop.
6. Compare against PPO's $31,493, BC's $31,103, and the MILP's $33,333 — but note
   the MILP and the open-loop tape have full information, so only the *closed-loop*
   comparison against PPO/BC is apples-to-apples.

Expect to fight BPTT stability. SHAC-style short horizons and truncated BPTT are
the mitigations of record.

**Phase 2a — finish the PPO swap (infrastructure, no algorithmic risk).**
The env wrapper is done and verified (§ 4A); what remains is the training loop.
Replace the 16 `SubprocVecEnv` Python environments with the batched sim and measure
wall-clock to reach current profit numbers. The honest complication, already
identified: **`MaskablePPO` is PyTorch.** A JAX sim driving a PyTorch policy crosses
a framework boundary every tick, which destroys the fused-loop advantage — the
rollout can no longer live inside one XLA program. Two coherent options:

- *Full JAX* — port PPO (PureJaxRL's single-file PPO plus `jnp.where(mask, logits,
  -inf)` for masking). Cost: reimplementing training code that is already debugged,
  and losing the sb3 checkpoint format and the existing BC pipeline. Payoff: rollout
  *and* update compile into one program, which is where the large end-to-end wins
  come from.
- *Chunked hand-off* — keep sb3, but have the sim advance N ticks per boundary
  crossing to amortise the transfer. Cheaper to build, caps the upside.

This decision is not yet made and should be made with GPU benchmark numbers in
hand, not before.

**Phase 2b — DONE as a de-risking spike; see § 4B for results.** The surrogate
design (`diffsim.py`) and the equivalence gates are built and passing. Gradients
propagate through 2,688 discontinuous steps with zero non-finite values and
long-range credit assignment intact. The open-loop comparison against zeroth-order
search was a tie, which is why the closed-loop test above is now the priority.

**Phase 3 — scale study.** Extend to multi-machine, multi-product, reentrant
routing. In the tensor formulation, reentrancy is just a richer routing map
indexing which queue a completion deposits into (possibly the same machine's input
again) plus per-product delay registers — the dynamics stay elementwise. In MILP the
same reentrancy couples many binary start variables on shared tools and is a
principal driver of the combinatorial explosion. Chart where CP-SAT solve time
explodes against where the tensor-sim policy still trains, to locate the crossover.

**Phase 4 — hybridize.** Keep the MILP as DAgger oracle, warm start, and accuracy
benchmark where it is tractable; let the tensor-sim policy carry the regimes it
cannot reach.

---

## 7. Gotchas for whoever picks this up

- **Run `verify.py` after any change to `tensor_sim.py`.** It is fast and it is the
  only thing standing between you and a fast simulator of the wrong line.
- **`compile_policy_rollout` captures the policy in a closure**, so each call
  returns a function JAX must compile from scratch. Hoist it out of loops.
  `rollout_policy` is a convenience wrapper that recompiles every call — fine for
  verification, wrong for benchmarks and training.
- **`rollout_trace` allocates `[H, B, ...]` buffers.** Verification only. Never use
  it at large B.
- **Changing the tick ordering in `advance` will break equivalence.** The
  completions-before-starts split is load-bearing (§ 3.1).
- **WIP cost is charged on queue *counts*, not on units inside machines.** Units in
  flight are in `reg`, not in `q`, and are correctly not charged — matching the
  reference. Also note `wip_cost_per_unit[0] = 0.0`, so raw material in q0 is free.
- **`config.py` currently holds the halved WIP costs** (`[0, 0.0025, 0.005, 0.0075]`)
  from the halved-WIP experiment, which are half the values quoted in `CLAUDE.md`.
  `spec.py` reads from `config.py`, so the tensor sim tracks whatever is actually
  configured. Do not hardcode prices anywhere in this folder.
- **`spec.py` is the only place that reads `make_common_config`.** Keep it that way
  so parameters have one source of truth.
