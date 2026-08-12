# A Tensorized, Differentiable Production Line: Scaling RL-Based Production Planning Beyond MILP

*SimpleFab RL Project — technical report*
*Draft v1 — 2026-07-12*

---

## Executive summary

We have been learning production-release policies for a small semiconductor-style
line (1 product, 4 machines) using Maskable PPO, with a CP-SAT MILP as a
provably-optimal reference. PPO reaches ~87–94% of the MILP optimum but is
sample-hungry and, more importantly, is driven by a *local* reward signal that
disagrees with the *global* optimum — so fine-tuning often erodes good behavior
rather than improving it.

This report argues for a specific piece of infrastructure: **re-expressing the
line's dynamics as a batched, differentiable tensor program** (a "tensor sim")
rather than the current object-oriented Python simulator. This is *not* a neural
network and nothing is learned — it is the same dynamics, forward-exact,
rewritten so that (a) thousands of trajectories run in parallel on a GPU and
(b) automatic differentiation can trace decisions → profit.

The honest value proposition has three parts:

1. **Throughput** (reliable win): 1–2 orders of magnitude more environment
   steps per second, which accelerates *every* sampling-based method we already
   run.
2. **A new gradient pathway** (promising, not guaranteed): analytic
   "pathwise" policy gradients through the real dynamics to true profit, which
   directly attack the reward-misalignment problem PPO suffers from.
3. **Scalability** (the strategic thesis): as the target problem grows to many
   machines, multiple products, and **reentrant flows**, exact MILP scales
   poorly (combinatorial explosion), while the tensor-sim + learning approach
   scales roughly *linearly in state size* for simulation and *degrades
   gracefully* for optimization. The approach looks worse than MILP at toy
   scale and becomes the only tractable option at realistic scale.

What it does **not** do: it does not make the simulation more accurate (it is
already exact), it does not reduce single-rollout latency, and it does not
provide optimality guarantees. Simulation speed and optimization difficulty are
orthogonal — a faster simulator changes the "samples per second" lever, not the
"how hard is the search" lever.

---

## 1. Introduction and problem statement

### 1.1 The immediate system

`simplefab_1p` simulates a discrete-time, single-product, 4-machine serial line:

```
q0 → M0 (batch, 16 ticks) → q1 → M1 (single, 2 ticks)
   → q2 → M2 (batch, 16 ticks) → q3 → M3 (single, 2 ticks) → ship / inventory
```

Batch machines (M0, M2) consume 4 units and emit 4 units per job; single
machines (M1, M3) process one unit. Demand arrives in periodic bursts
(interval = 672 ticks, horizon = 2688 ticks = 4 periods). The controller
decides, each tick, which machines to start (`MultiDiscrete([2,2,2,2])`), subject
to an action mask that forbids starting a busy machine or one with insufficient
input. The objective is **profit**:

```
profit = revenue − production − FG-holding − WIP-holding − backorder
       = $80·shipped − $24·produced − $0.04·Σqfin
         − Σ(0.005·q1 + 0.01·q2 + 0.015·q3) − $0.15·Σ(late demand)
```

This is a *release/dispatch scheduling* problem: how much raw material to admit
and when to run each stage, so that production tracks bursty demand without
over-building (holding cost) or under-building (backorder cost).

### 1.2 The real target

The small line is a testbed. The problem we actually care about has:

- **Many machines** (tens to hundreds of tools).
- **Multiple products** sharing tools, with product-dependent processing and
  **setup/changeover** costs and times.
- **Reentrant flows**: the defining feature of semiconductor fabs — a wafer
  visits the same photolithography/etch tools many times across its route, so
  the line is not a simple flow shop. Reentrancy couples early and late
  production decisions on the *same* machine and is the primary source of
  scheduling difficulty (Kumar, 1993).

At that scale, exact optimization (MILP/CP-SAT) faces combinatorial explosion:
job-shop scheduling is NP-hard (Garey, Johnson & Sethi, 1976), and reentrancy
plus product mix multiplies the decision-variable count and the branch-and-bound
tree. **The strategic question of this report is whether a simulation/learning
approach can scale where exact optimization cannot.**

---

## 2. Initial approach and the difficulties with PPO

### 2.1 What we built

- **Simulator** (`sim.py`): deterministic discrete-time line dynamics.
- **Environment** (`env.py`): Gymnasium wrapper, 18-dim observation, action
  masking, potential-based reward shaping, and a "pipeline nudge" penalty for
  idling M0 during the critical lead time.
- **Policy**: Maskable PPO (sb3-contrib), a stock `[128,128]` MLP.
- **Reference optimum** (`optim.py`): a CP-SAT MILP that solves the full
  2688-tick horizon to provable optimality (10,752 binary variables,
  Commander-heuristic warm start). For the 92%-utilization case it reports
  **$33,333** (LP upper bound $33,987, ~2% gap).
- **Imitation pipeline**: behavioral cloning (BC) from MILP-optimal
  trajectories, with an `ExpertOracle`/`MIPExpert` seam for future DAgger.

### 2.2 The results and the core problem

| Setting | Agent profit | % of MILP optimum |
|---|---|---|
| BC v2 alone | $31,103 | 93.3% |
| BC v2 + PPO (no nudge, 8M) | $31,493 | 94.5% |
| BC v2 + PPO (no nudge, 16M) | $30,053 | ↓ regressed |
| BC v2 + PPO (with nudge) | $28,678 | ↓ destroyed |
| Halved WIP cost + PPO | $32,200 | 96.6% (best) |

Two difficulties emerged, documented in `docs/why_bc_isnt_optimal.md`:

1. **Reward misalignment.** The env reward at tick *t* is local
   (`Δprofit + shaping`). Most *production* decisions carry an immediate visible
   cost (production cost, WIP/holding), while their *revenue* arrives much later
   when the unit ships. With discounting (γ=0.999), the local gradient says "be
   cautious about producing," while the global optimum says "pre-build
   aggressively to meet the demand spikes." PPO follows the local signal into an
   **under-production equilibrium** (~$30K), eroding the BC prior. This is not a
   tuning bug; it is a structural disagreement between the local reward and the
   global objective.

2. **Sample cost.** Runs use 8M–16M environment steps across 16 parallel Python
   subprocess environments. Data collection is a real wall-clock bottleneck, and
   the score-function (REINFORCE-style) gradient PPO uses is high-variance,
   requiring many samples per unit of progress.

Both difficulties motivate the same infrastructure: a way to (a) generate far
more experience per second and (b) obtain a *lower-variance* gradient that is
aimed at *true* profit rather than a local proxy.

---

## 3. Literature review and state of the art

### 3.1 RL for production scheduling

Deep RL for scheduling has grown quickly, especially for the job-shop scheduling
problem (JSSP) and semiconductor manufacturing:

- **Semiconductor dispatching.** Waschneck et al. (2018) applied deep Q-networks
  as cooperative dispatching agents for global fab scheduling — an early
  demonstration on realistic fab structure. Fabs remain a canonical hard case
  because of reentrancy and tool contention.
- **Learning to dispatch (GNN + RL).** Zhang et al. (2020, NeurIPS), "Learning
  to Dispatch for Job Shop Scheduling via Deep RL" (L2D), represent the
  disjunctive graph of a JSSP with a graph neural network and learn dispatching
  priorities with PPO, generalizing across problem sizes. Park et al. (2021,
  IJPR) develop a similar GNN + RL scheduler. GNN encoders are now the dominant
  representation for size-agnostic scheduling policies.
- **Surveys.** Mazyavkina et al. (2021) survey RL for combinatorial
  optimization; multiple recent surveys cover RL specifically for machine/
  production scheduling. The consistent finding: learned dispatchers match or
  beat hand-crafted priority rules and generalize across instances, but rarely
  *prove* optimality and typically trail exact solvers on small instances where
  those solvers are tractable.
- **State of the art, honestly stated.** For small/medium instances, exact
  MILP/CP solvers remain the accuracy gold standard. RL's value is (i)
  *amortized* inference (a trained policy produces a schedule in one forward
  pass vs. a solver's search) and (ii) *scaling* to instances where exact
  methods time out. This is precisely the regime our target problem lives in.

### 3.2 Differentiable and GPU-accelerated simulation

A parallel line of work makes *simulators* fast and differentiable:

- **GPU-vectorized environments.** Isaac Gym (Makoviychuk et al., 2021), Brax
  (Freeman et al., 2021), EnvPool (Weng et al., 2022), and JAX ecosystems
  (Gymnax, PureJaxRL; Lange, 2022; Lu et al., 2022) run thousands–millions of
  environment copies on a single accelerator, reporting **orders-of-magnitude**
  throughput gains and end-to-end training times measured in minutes rather than
  hours/days. The mechanism is data-parallelism: identical dynamics, batched
  over a leading dimension, with no per-environment Python or inter-process
  overhead.
- **Differentiable physics.** DiffTaichi (Hu et al., 2020) and Brax expose
  autograd through the dynamics, enabling gradient-based control. Degrave et al.
  (2019) built a differentiable engine specifically for robotics learning.
- **Analytic policy gradients.** Stochastic Value Gradients (Heess et al., 2015)
  and Short-Horizon Actor-Critic (SHAC; Xu et al., 2022) backpropagate through
  known/differentiable dynamics to train policies with *pathwise* gradients,
  which are lower-variance than score-function gradients when the dynamics are
  smooth.
- **The crucial caveat.** Suh et al. (2022, ICML), "Do Differentiable Simulators
  Give Better Policy Gradients?", show that through **stiff or discontinuous
  dynamics**, pathwise gradients can be *higher-variance or more biased* than
  plain REINFORCE. Contacts in physics are the analog of our discrete events
  (batch completions, threshold gates, ship-vs-store branches). This is the
  single most important piece of prior art tempering expectations: a
  differentiable simulator is not automatically a better optimizer.

### 3.3 Model-based RL and world models

Ha & Schmidhuber (2018, "World Models") and the Dreamer line (Hafner et al.,
through DreamerV3, 2023) *learn* neural dynamics models and plan/train inside
them. These are approximate by construction. Our situation is different and
easier: **we already know the exact dynamics**, so we do not need to learn a
model — we only need to *re-express the known model* in a differentiable,
batched form. This sidesteps the model-accuracy problem that world-model methods
must manage.

### 3.4 Where this project sits

Our toolbox already spans the two poles: an **exact MILP** (optimal, open-loop,
non-scaling) and **model-free PPO** (scalable, reactive, reward-misaligned,
sample-hungry). The gap is a middle path: a fast, differentiable, *known* model
that supports both massive sampling and low-variance analytic gradients. That is
exactly the tensor-sim proposed here.

---

## 4. Tensorized reimplementation of the line

### 4.1 The idea in one sentence

Replace the per-object, one-trajectory-at-a-time Python loop with **fixed-size
state tensors updated by pure array arithmetic**, carrying a leading batch
dimension `B` so that `B` trajectories advance in lockstep, and using only
operations that automatic differentiation can trace.

### 4.2 State representation

The current sim stores each queue as a Python list of arrival timestamps. For
the *dynamics*, only counts and countdowns matter, so the entire state of `B`
trajectories becomes a handful of tensors:

| Tensor | Shape | Meaning |
|---|---|---|
| `q` | `[B, 4]` | queue counts q0..q3 |
| `delay` | `[B, 4, pt_max]` | per-machine shift register: units emitting in *k* ticks |
| `qfin` | `[B]` | finished-goods inventory |
| `backlog` | `[B]` | unmet demand |
| `costs` | `[B, 5]` | running revenue / production / FG-holding / WIP / backorder |

The `delay` shift register is the elegant part: a machine that starts at tick *t*
emits its batch exactly `pt` ticks later. Modeling this as a length-`pt` register
that shifts by one each tick captures the processing delay *and* the "one job at
a time" capacity constraint, and a shift is a fixed linear map — fully
differentiable. Backorder aging (a demand unit becomes "late" after
`demand_interval` ticks) is handled exactly as the MILP already does it: track
cumulative demand and cumulative shipments and difference them against the
schedule, rather than storing per-unit timestamps.

### 4.3 The transition as ~10 elementwise ops

One tick, for all `B` trajectories at once (schematic):

```python
# 1. arrivals (constants for tick t, broadcast over the batch)
q[:, 0] += arrivals[t]
backlog += demand[t]

# 2. advance delay lines; units leaving slot 0 are completions
done   = delay[:, :, 0]                 # [B, 4]
delay  = shift_left(delay)              # fixed matmul → differentiable
q[:, 1] += 4 * done[:, 0]               # M0 batch → q1
q[:, 2] +=     done[:, 1]               # M1 → q2
q[:, 3] += 4 * done[:, 2]               # M2 batch → q3
newfin  =      done[:, 3]               # M3 → finished

# 3. ship = min(available, backlog)
avail   = qfin + newfin
shipped = minimum(avail, backlog)       # differentiable (subgradient)
revenue += 80 * shipped; backlog -= shipped; qfin = avail - shipped

# 4. starts: valid iff action AND idle AND enough input
idle   = (delay.sum(-1) == 0)           # [B, 4]
canrun = (q >= thresh)                  # thresh = [4, 1, 4, 1]
start  = action * idle * canrun         # [B, 4] — the hard gate
q     -= thresh * start                 # consume inputs
delay  = load_batch(delay, start)       # insert into slot pt-1
production += cost_vec @ (size * start)

# 5. holding / WIP / backorder costs (elementwise on counts)
costs += 0.04 * qfin + (wvec * q).sum(-1) + 0.15 * late(backlog, t)
```

Every line is a batched tensor op. The only remaining loop is over the 2688
ticks — unavoidable, because tick *t* depends on *t−1* and the thresholds make
the recurrence nonlinear (no parallel-scan shortcut across time).

**This is not a neural network.** Nothing is learned; there are no weights and no
matmuls in the dynamics themselves (they are elementwise). It is `sim.py`
re-expressed as a differentiable, batched program.

### 4.4 Exactness: forward vs. backward

- **Forward pass: bit-exact.** Keeping integer counts and hard thresholds, the
  tensor sim reproduces `sim.py` exactly. (A `verify()` test asserting equality
  of profit and per-tick state against the reference sim keeps this honest — the
  same discipline the MILP already uses.)
- **Backward pass: surrogate.** The true derivative of a step function
  `1[q ≥ thresh]` is zero almost everywhere and undefined at the jump — useless
  for optimization. So the *backward* pass replaces those hard gates with smooth
  surrogates (sigmoid-shaped straight-through estimators) and the discrete action
  with a relaxation (e.g. straight-through / Gumbel-softmax). Gradients are
  therefore **biased but usable**. Exact-forward, approximate-backward is
  inherent, not a shortcoming to be engineered away: you cannot have exact
  gradients of a discontinuous function.

### 4.5 How much advantage over the existing simulator?

**Measured baseline.** The current Python sim runs at **432,000 sim-steps/sec**
single-threaded (2.31 µs/step, 6.22 ms per 2688-tick episode) on this machine.
PPO training currently multiplies this across 16 subprocess environments but
pays inter-process serialization overhead on every step.

**Projected tensor-sim throughput.** Two regimes, and it is important not to
conflate them:

- *Single trajectory:* **no improvement, possibly worse.** One rollout is still
  2688 sequential steps; on a GPU the per-step kernel-launch overhead can make a
  single trajectory *slower* than the Python sim. Latency is not the win.
- *Batched (the win):* with `B = 10^3–10^4` trajectories advancing together,
  aggregate throughput enters the regime GPU-vectorized simulators report —
  **10^6–10^8 steps/sec**, i.e. roughly **1–2+ orders of magnitude** over the
  current setup. Because the dynamics are elementwise (no matmul until a policy
  network is attached), the per-step cost is tiny and batching is near-ideal.
  (These figures are the reference class from Brax/Isaac Gym/EnvPool, to be
  validated by our own benchmark, not a number we have measured for this line.)

**Answer to "is the speedup because of GPU hardware?"** Partly. The GPU is what
makes *data-parallelism* (batching many trajectories) cheap, so yes, the hardware
enables the throughput. But two clarifications matter:

1. A large part of the current cost is Python interpreter and IPC overhead;
   moving to a compiled tensor program removes that *even on CPU*.
2. Crucially, **faster simulation does not by itself make optimization faster.**
   Simulation speed (samples/sec) and optimization difficulty (how hard the
   search is) are orthogonal. More samples/sec accelerates sample-limited
   methods, but the scheduling search is non-convex and combinatorial; beyond a
   point you hit diminishing returns and local optima regardless of how fast the
   simulator runs. GPU speed moves the "samples per second" lever, not the "how
   hard is the search" lever.

---

## 5. What this buys us — now and at scale

### 5.1 Immediate wins (small line)

1. **Cheaper PPO.** Drop-in replacement of the 16 Python envs with one batched
   GPU env accelerates data collection by 1–2 orders of magnitude, shrinking
   8M–16M-step runs from hours toward minutes. This benefits every
   sampling-based method we already run, with zero change to accuracy.
2. **A gradient aimed at true profit.** With a differentiable line we can train a
   *reactive policy* by backpropagating from end-of-episode profit through the
   real dynamics (pathwise / analytic policy gradient), instead of PPO's
   high-variance score-function gradient over a *local, misaligned* reward. This
   directly targets the reward-misalignment problem in
   `docs/why_bc_isnt_optimal.md`: credit assignment comes from the dynamics
   themselves, so no reward shaping is needed and the "be cautious about
   producing" local bias disappears.

### 5.2 On finding *optimal* solutions — honest scope

Two distinct uses, one weak and one strong:

- **Open-loop gradient planning (weak here).** Parameterize the release/action
  schedule as continuous variables, forward-sim to profit, gradient-ascend. This
  finds *local* optima through a non-convex, surrogate-gradient landscape — **no
  certificate.** For the small known-demand problem the MILP already gives the
  *global* optimum with a gap bound, so gradient planning is strictly weaker at
  the MILP's own game. It is insurance for when the MILP stops scaling, not a way
  to beat it now.
- **Closed-loop pathwise policy gradient (the real prize, with caveats).** A
  reactive policy trained through the differentiable line is robust to demand
  deviations (unlike the MILP's fixed open-loop schedule) and can be far more
  sample-efficient than PPO. **Caveat (Suh et al., 2022):** through our
  discontinuous dynamics the pathwise gradient may be biased or high-variance,
  and backprop-through-time over 2688 steps invites vanishing/exploding
  gradients. Mitigations exist (short-horizon windows à la SHAC, truncated BPTT,
  blending pathwise and score-function gradients), but success is empirical and
  problem-dependent, not guaranteed.

**Bottom line for optimality:** the tensor sim does not manufacture an optimality
guarantee. It offers a faster environment and a cleaner gradient toward a
*reactive* policy trained on *true* profit — a better route to a good policy,
never a proof of the best one.

### 5.3 Why this inverts in our favor at scale (the strategic thesis)

The value proposition flips as the problem grows to many machines, multiple
products, and reentrant flows:

| | Exact MILP/CP | Tensor-sim + learning |
|---|---|---|
| Small instance | **Optimal, tractable** | Worse (local, approximate) |
| Simulation cost vs. size | n/a | ~**linear** in state dimension |
| Optimization cost vs. size | **combinatorial explosion** (NP-hard; reentrancy + product mix blow up variables and the B&B tree) | **degrades gracefully** — still returns a policy, just not provably optimal |
| Reentrant flows | couple decisions on shared tools → very hard | just more entries in `q`, `delay`, and the route map |
| Multi-product / setups | more integer variables, weaker relaxations | more state/action dimensions; `MultiDiscrete` handles it |
| Demand uncertainty | per-instance open-loop solve | one reactive policy generalizes across realizations |
| Inference at deploy time | re-solve (seconds–minutes) | one forward pass (milliseconds) |

The key asymmetry: adding machines, products, and reentrancy grows the tensor
sim's cost **additively/linearly** (bigger tensors, a route map instead of a
fixed chain) while growing the exact solver's cost **combinatorially**. At
realistic fab scale the MILP is expected to become intractable for full-horizon
optimality, at which point a fast differentiable simulator that trains a reactive
policy is not the *weaker* option — it may be the *only* tractable one. The small
line is where we validate the machinery against a solver that still works; the
large line is where the machinery earns its keep.

Reentrancy specifically is where this matters most. In a reentrant line a wafer
revisits the same tool many times; in the tensor formulation this is just a
richer routing map indexing which queue a completion deposits into (possibly the
same machine's input again), and per-product `delay` registers. The dynamics stay
elementwise and differentiable. In MILP, the same reentrancy couples many binary
start variables on shared tools and is a principal driver of the explosion.

---

## 6. Future work

1. **Build and verify the tensor sim (Phase 1).** Reimplement the 4-machine line
   as a batched differentiable program (JAX or PyTorch). Gate it behind a
   `verify()` test asserting bit-exact agreement with `sim.py` across random
   action sequences. *Deliverable: banked throughput win, measured.*
2. **Benchmark honestly.** Report real steps/sec vs. the 432K baseline, at
   `B ∈ {1, 64, 1k, 10k}`, on CPU and GPU. Confirm or correct the 1–2
   order-of-magnitude projection.
3. **Swap it into PPO (Phase 2a).** Replace the `SubprocVecEnv` envs with the
   batched sim; measure wall-clock to reach current profit numbers. Pure
   infrastructure win, no algorithmic risk.
4. **Pathwise policy gradient (Phase 2b, the experiment).** Train a reactive
   policy by backprop-through-time to true profit. Compare against PPO and the
   MILP. Expect to fight gradient-stability issues; try SHAC-style short
   horizons and truncated BPTT. *This is the test of whether the gradient
   pathway beats model-free RL on this problem.*
5. **Scale study (Phase 3).** Extend the tensor sim to multi-machine,
   multi-product, reentrant routing. Chart where CP-SAT solve time explodes vs.
   where the tensor-sim policy still trains, to locate the crossover at which the
   learning approach dominates.
6. **Hybridize with the MILP (Phase 4).** Keep using the MILP where it is
   tractable — as an oracle for DAgger, for warm starts, and as the accuracy
   benchmark — while the tensor-sim policy carries the regimes the MILP cannot
   reach.

---

## 7. Conclusions

Expressing the line as fixed-size state tensors updated by pure array arithmetic
makes the *simulation* dramatically faster in aggregate (via batching that GPUs
make cheap) and makes it *differentiable*. It does **not**, by itself, make
finding optimal policies faster or guaranteed — simulation speed and optimization
difficulty are orthogonal, and the surrogate gradients through discontinuous
dynamics come with real, documented risks.

The reason to build it anyway is not the small line, where the MILP already wins.
It is the **trajectory toward the real problem**: many machines, multiple
products, reentrant flows — exactly the regime where exact optimization is
expected to collapse under combinatorial explosion, and where a fast,
differentiable, *known-exact* simulator that trains a reactive policy on true
profit is the most credible path to good, deployable schedules. The tensor sim is
infrastructure that (1) accelerates everything we already do, (2) opens a
principled gradient pathway aimed at the objective we actually care about, and
(3) scales the way the exact solver cannot. We propose building it in stages so
the guaranteed throughput win lands first, before we invest in the higher-risk,
higher-reward gradient and scaling work.

---

## References

- Degrave, J., Hermans, M., Dambre, J., wyffels, F. (2019). *A Differentiable
  Physics Engine for Deep Learning in Robotics.* Frontiers in Neurorobotics.
- Freeman, C. D., Frey, E., Raichuk, A., Girgin, S., Mordatch, I., Bachem, O.
  (2021). *Brax — A Differentiable Physics Engine for Large Scale Rigid Body
  Simulation.* NeurIPS Datasets & Benchmarks.
- Garey, M. R., Johnson, D. S., Sethi, R. (1976). *The Complexity of Flowshop and
  Jobshop Scheduling.* Mathematics of Operations Research.
- Ha, D., Schmidhuber, J. (2018). *World Models.* NeurIPS.
- Hafner, D., et al. (2023). *Mastering Diverse Domains through World Models
  (DreamerV3).*
- Heess, N., Wayne, G., Silver, D., Lillicrap, T., Tassa, Y., Erez, T. (2015).
  *Learning Continuous Control Policies by Stochastic Value Gradients.* NeurIPS.
- Hu, Y., et al. (2020). *DiffTaichi: Differentiable Programming for Physical
  Simulation.* ICLR.
- Kumar, P. R. (1993). *Re-entrant lines.* Queueing Systems.
- Lange, R. T. (2022). *Gymnax: A JAX-based Reinforcement Learning Environment
  Library.*
- Lu, C., et al. (2022). *Discovered Policy Optimisation* / PureJaxRL
  (end-to-end GPU RL in JAX).
- Makoviychuk, V., et al. (2021). *Isaac Gym: High Performance GPU-Based Physics
  Simulation for Robot Learning.* NeurIPS Datasets & Benchmarks.
- Mazyavkina, N., Sviridov, S., Ivanov, S., Burnaev, E. (2021). *Reinforcement
  Learning for Combinatorial Optimization: A Survey.* Computers & Operations
  Research.
- Park, J., et al. (2021). *Learning to schedule job-shop problems: representation
  and policy learning using GNN and RL.* International Journal of Production
  Research.
- Suh, H. J. T., Simchowitz, M., Zhang, K., Tedrake, R. (2022). *Do Differentiable
  Simulators Give Better Policy Gradients?* ICML.
- Waschneck, B., et al. (2018). *Optimization of global production scheduling with
  deep reinforcement learning.* Procedia CIRP.
- Weng, J., et al. (2022). *EnvPool: A Highly Parallel Reinforcement Learning
  Environment Execution Engine.* NeurIPS.
- Xu, J., et al. (2022). *Accelerated Policy Learning with Parallel Differentiable
  Simulation (SHAC).* ICLR.
- Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., Xu, C. (2020). *Learning to
  Dispatch for Job Shop Scheduling via Deep Reinforcement Learning.* NeurIPS.
</content>
</invoke>
