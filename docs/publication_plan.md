# From working instrument to publishable paper — a three-phase plan

**Status:** draft for editing · written 2026-10-18
**Scope:** what has to happen for the tensorfab differentiable-scheduling work to stand up as a paper.

---

## 0. Where we stand, and the claim we can actually defend

### 0.1 Current results

| Result | Evidence | Status |
|---|---|---|
| Surrogate gradients survive 2,688 steps of hard gates; zero non-finite values; >50% of gradient mass in the first quarter of the horizon | 4 demand patterns, verified | Confirmed |
| Gradient beats the best zeroth-order search by ~10× at 50% utilisation, by $213–$398 at ~90%, and **loses** at saturation | 1-product, 4 patterns | Confirmed |
| In 2 products the same mechanism appears as a monotone load sweep: +$17,216 / +$12,397 / +$2,803 / −$726 at 50/60/80/92% | utilisation sweep | Confirmed |
| Advantage grows with pipeline depth | depth sweep M=4/8/12 | **Falsified** (our own routing bug) |
| Reaches 91% of a *certified dual bound* in 36 s; CP-SAT needs ~110 s to overtake, 1,800 s to plateau | anytime curve | Confirmed |
| Low-load weakness is surrogate bias | τ_q sweep 4.0→0.05 | **Falsified** — sharpening made it worse |
| Low-load weakness is an initialisation basin | idle-logit sweep; 86.3%→99.5% of CP-SAT at 50% load | Confirmed |
| **A learned method beats the exact solver** | all configurations to date | **0 / 10** † |

† 10 is the deck's count: 4 one-product demand patterns + 3 two-product mixes @ 80% + 3 further
points on the utilisation sweep. The depth configurations (M = 8, 12) add 2–3 more depending on
how the M=4 overlap is counted, and CP-SAT wins those too — so the direction is unaffected. The
exact denominator should be fixed and stated once before submission.

### 0.2 The defensible claim

> On a deterministic, full-information, open-loop production-scheduling problem with batching,
> sequence-dependent changeovers and inventory/backorder economics, a differentiable simulator
> with straight-through gradients reaches 91–96% of a certified optimality bound within seconds,
> beats every zeroth-order search baseline wherever the schedule contains real slack, and **does
> not beat CP-SAT at any budget we tested**. The margin over search is governed by available
> slack, not by horizon length.

That is a *calibration* result, and the field currently has demand for it (§4.1). It is not a
methods contribution.

### 0.3 What is NOT ours

The straight-through-through-a-discrete-event-simulator idea is **Che & Dong (2024)**,
[arXiv:2409.03740](https://arxiv.org/abs/2409.03740). Differences that are genuinely ours:

| | Che & Dong 2024 | This work |
|---|---|---|
| System | Multi-class queuing networks | Production line: batching, sequence-dependent setups, inventory/backorder economics |
| Where straight-through sits | `argmin` selecting the **next event** | Capacity and input **gates** (`q ≥ take`, binary action); fixed tick, so no event-ordering argmin exists |
| Extra relaxation | Fractional capacity sharing | None — `idle` is exact and linear |
| Policy class | Closed-loop neural policy | Open-loop schedule |
| Baselines | RL only (REINFORCE, PPO) | Greedy heuristic, random search, hill-climb, **CP-SAT with certified dual bounds** |
| Headline | 50–1000× sample efficiency over RL | 0/10 against the exact solver; slack-governed margin over search |

**The exact-solver comparison with certified bounds is the distinctive asset.** It is precisely
what that paper, and most of the differentiable-simulation literature, omits.

---

## Phase 1 — Statistical rigour (seeds and confidence intervals)

**Why first:** cheapest credibility available, and without it nothing else is publishable.
[Liu et al. (2025)](https://arxiv.org/abs/2512.22642) list "omission of confidence intervals,
variance analysis and significance testing across multiple runs" as a core failure of the ML-for-
scheduling literature. We currently commit it: **34 of 37 logged runs use `seed = 0`.** Every
headline number in this project is a single sample.

### 1.1 Experiment inventory to be re-run with seeds

| ID | Experiment | Configurations | Methods |
|---|---|---|---|
| E1 | 1-product, demand patterns, M=4 | UNIFORM 50%, ALT_50 50%, ALT 89%, ALT_92 92% | commander, random, hill-climb, grad-soft, grad-ST, CP-SAT + bound |
| E2 | Pipeline depth @ 50% util | M = 4, 8, 12 | as E1 |
| E3 | Pipeline depth @ 92% util | M = 4, 8, 12 | as E1 |
| E4 | 2-product mixes @ 80% util | α = 0.5, 0.9, 0.1 | as E1 |
| E5 | 2-product utilisation sweep, α=0.5 | 50, 60, 80, 92% | commander, grad-ST, CP-SAT + bound |
| E6 | Anytime curve | α=0.5, 80% | grad-ST vs CP-SAT incumbent over wall time |
| E7 | Initialisation-bias sweep | 4 utilisations × 4 biases | grad-ST |
| E8 | Gate-temperature sweep | τ_q ∈ {4, 1, 0.25, 0.1, 0.05} | grad-ST |
| E9 | Optimisation-budget sweep | 400 / 1,200 / 3,000 steps | grad-ST |

### 1.2 Sources of randomness — including one we have been ignoring

1. **θ initialisation** — `jax.random.PRNGKey(seed)`, affects grad-ST and grad-soft.
2. **Random search** sampling.
3. **Hill-climb** start (shares the grad-ST start by design — this pairing is useful, see §1.4).
4. **CP-SAT itself.** It runs an 8-worker portfolio whose result depends on thread timing.
   *This is not hypothetical:* we observed the same α=0.1 model return **\$28,992** and
   **\$30,295** on two runs at the same budget — a **4.3% swing**, large enough to have flipped
   a headline conclusion. CP-SAT must be treated as a stochastic method and seeded via
   `solver.parameters.random_seed`.

### 1.3 Design

- **n = 10 seeds** per (method × configuration). Pilot with n = 3 on E5 to estimate σ, then
  confirm n via a power calculation for the smallest effect we want to resolve (the −$726 at
  92% util is the hardest; large effects like +$12,397 need far fewer).
- **Estimator:** report **mean ± 95% CI**. With n = 10 and no normality guarantee, use a
  **BCa bootstrap** CI rather than a t-interval.
- **Report CIs, not p-values.** With ~10 configurations × 6 methods, null-hypothesis testing
  invites p-value soup. If a test is required, use the paired form below and control FDR with
  Benjamini–Hochberg.
- **Exploit the pairing.** grad-ST and hill-climb start from *identical* random tapes. The
  correct comparison is therefore the **paired difference per seed**, whose CI is far tighter
  than comparing two independent means. Report the paired mean difference and its CI; use
  **Wilcoxon signed-rank** if a test is wanted.
- **Report two scales** for every result: absolute dollars (with CI) and % of certified dual
  bound (with CI). The utilisation study showed these can move in opposite directions.
- **Fix and log** everything else: commit hash, JAX version, backend, thread count, wall time.
  Already done by `logbook.Run`; extend to record the seed explicitly.

### 1.4 Compute budget (measured, sequential — solves must run alone)

| Component | Budget |
|---|---|
| grad-ST, n=10, 10 configs (~35 s each) | ~1.0 h |
| Zeroth-order baselines, n=10 | ~1.9 h |
| CP-SAT @ 600 s, n=10, 10 configs | **16.7 h** |
| CP-SAT @ 1,800 s, n=10 | 50 h — too much |

**Recommendation:** seed study at **600 s** CP-SAT budget (16.7 h, one overnight run), with the
1,800 s runs retained as single-seed headline references and labelled as such. Everything must be
sequential — concurrent load silently degrades CP-SAT incumbents (§1.2).

### 1.5 Deliverable

A results table where every number is `mean ± CI (n=10)`, plus a paired-difference table for
grad-ST vs best zeroth-order. Re-check whether the surviving claims hold: the **−$726 at 92%**
and the **+$606 / −$180 / +$801 depth sequence** are the two most likely to dissolve into noise.

---

## Phase 2 — A recognised benchmark

**Why:** a reviewer dismisses a 4-tool line in one sentence. The field's open benchmarks are
Minifab and SMT2020.

### 2.1 Minifab (Intel / ASU, Kempf)

- **5 machines, 3 workstations, 6 process steps.** Two implant tools with **sequence-dependent
  setups**; one litho tool; two diffusion tools that **batch 3 lots**.
- **Re-entrant:** 6 steps over 5 machines.
- **3 products** — Pa, Pb and test wafers — at 51 / 30 / 3 lots per week.
- Includes machine failures, preventive maintenance, operator assignment, load/unload.
- Short cycle time (days), constant loading.

**Fit to our stack: very close.** Batch tools, single tools and sequence-dependent setups are
already in `tensorfab/twoprod/`. Re-entrancy is a different routing matrix `R` — the design
already anticipates it ("a route pointing backwards is just a different R"), though it has never
been exercised. Operators are a genuinely new constraint (a shared resource across tools) and are
the main modelling work.

### 2.2 SMT2020 (Kopp, Hassoun, Kalir & Mönch, *IEEE T-SM*)

- **~106 tool groups, 913–1,071 machines**, 12 families.
- **10 products**, routes of **242–583 process steps**.
- **~10,000 wafer starts/week**; cycle times of weeks.
- Two variants: high-volume/low-mix and low-volume/high-mix, plus engineering-lot variants.

### 2.3 The decisive finding: no certified bounds exist for either

A thorough search found **no published MILP/CP certified optimality bounds for Minifab or
SMT2020**. The Minifab literature is dispatching rules, approximate dynamic programming, RL and
graph neural networks; reported results are *relative* improvements (e.g. ~1.6% more completed
lots, ~2.7% lower cycle time vs heuristics; double-digit tardiness gains on Minifab and SMT2020
in [arXiv:2505.11135](https://arxiv.org/abs/2505.11135)). SMT2020 is far beyond exact methods.

Two consequences:

1. **SMT2020 cannot serve our thesis.** It can demonstrate that the tensor simulator scales, but
   it cannot produce optimality gaps — and optimality gaps are our contribution. Treat it as an
   optional scalability appendix, not a main result.
2. **Minifab is the target, and the first certified dual bounds for it would be a contribution in
   their own right**, independent of the gradient method.

### 2.4 Work plan

| Step | Work | Risk |
|---|---|---|
| 2a | Port Minifab into `LineSpec`: 5 tools, re-entrant `R`, batch-3 diffusion, sequence-dependent implant setups, 3 products | Medium — re-entrancy untested |
| 2b | Add the operator resource (shared, finite) to the tick | Medium — new constraint class |
| 2c | Verify against a published Minifab implementation (e.g. the open-source SimPy/AnyLogic models) on throughput and cycle time | **High — this is the credibility gate** |
| 2d | Extend the CP-SAT model to re-entrancy + operators; run long solves for incumbents **and dual bounds** | High — may not close |
| 2e | Run the Phase-1 protocol (n=10, CIs) on Minifab | Low |

**Gate:** if 2c does not reproduce published Minifab numbers, stop and fix before any claim.

---

## Phase 3 — The regime where learning should win

**This is the scientific core.** Everything to date hands CP-SAT its best case: deterministic,
full-information, open-loop, single-shot. The literature is explicit about where that flips —
[NCO for stochastic FJSP](https://arxiv.org/abs/2412.14052) beats CP-SAT by ~6% on larger
instances **specifically under stochasticity**, and [FrontierCO](https://arxiv.org/abs/2505.16952)
finds ML competitive when speed matters and instances are not solver-friendly.

### 3.1 The hypothesis to test

> Under stochastic processing times and equipment outages, the exact solver must either solve a
> much larger scenario-based program or re-optimise continually, while a differentiable simulator
> optimises expected profit over sampled scenarios **at no extra cost, because scenarios occupy a
> batch axis it already has**. There exists a variability level above which the gradient method
> wins on profit per unit of compute.

**Why the architecture favours us:** the batch dimension currently holds 64 *random restarts*.
Under stochasticity the same axis holds 64 *scenarios*, and one gradient step optimises a
sample-average approximation of expected profit. A scenario-based CP-SAT model grows roughly
linearly in scenarios and becomes intractable. This is the strongest structural argument the
method has, and it is currently untested.

### 3.2 Modelling the randomness — Factory Physics

Hopp & Spearman separate two outage classes, and the distinction matters for both the simulator
and the formulation.

**Non-preemptive outages** (setups, operator breaks, PM) — occur *between* jobs, detachable:

```
t_e   = t_0 + t_s / N_s
σ_e²  = σ_0² + σ_s²/N_s + ((N_s − 1)/N_s²) · t_s²
c_e²  = σ_e² / t_e²
```

**Preemptive outages** (random failures) — occur *during* processing:

```
A     = m_f / (m_f + m_r)                       availability
t_e   = t_0 / A
c_e²  = c_0² + (1 + c_r²) · A · (1 − A) · m_r / t_0
```

and the Kingman VUT queue-time approximation that motivates them:

```
CT_q = ( (c_a² + c_e²) / 2 ) · ( u / (1 − u) ) · t_e
```

> ⚠ **Verify these coefficients against Factory Physics Ch. 8 before publication.** They are
> reproduced from memory; the primary source could not be retrieved during this search, and the
> Kingman form above is the only one confirmed from a live source.

**What we inject, in increasing order of difficulty:**

| Level | Randomness | Simulator work |
|---|---|---|
| S1 | Random **setup durations** (non-preemptive) — we already have setups, make them stochastic | Small |
| S2 | Random **processing times** | Small — the delay register takes a sampled `pt`, but `reg_len` must cover the tail |
| S3 | **Preemptive failures** — MTTF/MTTR, tool down mid-job | Medium — needs a repair state and a decision about whether the in-flight job is lost or resumed |
| S4 | Stochastic **demand** | Small |

**Recommendation:** S1 + S3. S1 is the Factory Physics non-preemptive case the user specified and
is nearly free given our existing setup machinery; S3 is where the real variability lives in a fab
and is what makes re-planning valuable. Treat the variability level (`c_e²`) as the **independent
variable of the whole study** — sweep it, and find the crossover.

### 3.3 The fairness protocol — the part that must not be got wrong

Under randomness, *who knows what and when* decides the result. Four distinct comparators:

| Comparator | Information | Role |
|---|---|---|
| **PI (perfect information)** | CP-SAT solves each realised scenario with hindsight | **Upper bound.** Not achievable by anyone; the gap to it is the EVPI. |
| **Det-EPT** | CP-SAT solves once on deterministic *effective* times `t_e`, schedule then executed open-loop against random realisations | The Factory Physics industrial default |
| **SAA-CP** | CP-SAT on an N-scenario stochastic program, executed open-loop | The honest exact-solver attempt; watch it blow up in N |
| **Rolling CP-SAT** | Re-optimise every k ticks on realised state, **charged its solve time** | The strong practical baseline |
| **Grad-SAA** | Our method: one open-loop schedule optimising mean profit over N sampled scenarios | Phase 3a |
| **Grad-policy** | Closed-loop policy trained by pathwise gradients | Phase 3b — the Che & Dong setting |

**Non-negotiables:**
- Evaluation scenarios are **held out** from training scenarios.
- Rolling CP-SAT is charged wall-clock for every re-solve; if it cannot re-solve within the
  replanning interval, that is a result, not an excuse.
- Every profit figure is the realised schedule replayed through the **exact integer simulator**,
  as now.
- Report **expected profit with CIs over evaluation scenarios**, plus a risk measure (CVaR at
  5%) — robustness under variability is the selling point, so measure it.

### 3.4 Phase 3a vs 3b

- **3a (open-loop SAA)** is a small step from where we are: swap the batch axis from restarts to
  scenarios, add a held-out evaluation set. Days of work. Tests the core hypothesis.
- **3b (closed-loop policy)** is the stronger paper and the fair fight against PPO — but it is
  the Che & Dong setting, so novelty rests entirely on the different system and the solver
  comparison. Weeks of work.

**Recommendation:** do 3a, publish on it, and treat 3b as the follow-on. 3a alone converts the
paper from "calibration note" to "we identified the crossover."

---

## 4. Paper

### 4.1 Framing and venue

Working title: **"Differentiable simulation versus exact solvers for production scheduling:
a certified-bound evaluation."**

The hook is that the differentiable-simulation literature reports large wins over RL and
essentially never reports optimality gaps, while FrontierCO and Liu et al. are simultaneously
calling for compute-matched comparisons against classical solvers and honest scoping. We supply
exactly that, for a system class (production lines with batching, setups and inventory economics)
that has no published differentiable treatment.

| Venue | Fit |
|---|---|
| Winter Simulation Conference | Strong — simulation + semiconductor + Minifab/SMT2020 are native |
| ML-for-OR workshop (NeurIPS/ICML) | Strong for the calibration framing |
| *Computers & Industrial Engineering* / *IISE Transactions* | Good with Phase 2 + 3 complete |
| NeurIPS/ICML main track | **No** — no new estimator, no theory |

### 4.2 Structure

1. Problem class and why it is not a queuing network
2. The tensor formulation (route matrix, delay register, integer accumulators, exactness proof)
3. Straight-through placement and the three gates; the exact-forward verification gate
4. Deterministic results with certified bounds — the 0/N result and the slack-governed regime map
5. The stochastic study — the crossover
6. Negative results and corrections: the depth retraction, the τ_q refutation, the initialisation basin
7. Limitations

### 4.3 Keep the corrections in

Three claims were falsified in development: depth, surrogate-bias, and the Adam second-moment
hypothesis. Reporting them is cheap and is exactly the practice both critical reviews ask for.

---

## 5. Timeline and risks

| Phase | Work | Calendar |
|---|---|---|
| 1 — seeds & CIs | 1 day coding, ~17 h compute | 1 week |
| 3a — stochastic, our line | S1+S3 simulator, SAA harness, rolling CP-SAT, variability sweep | 2–3 weeks |
| 2 — Minifab port | Port, operators, validation, CP-SAT extension | 3–4 weeks |
| 3b — closed-loop | optional | +4 weeks |

**Doing Phase 3a before Phase 2 is deliberate**: it tests the central hypothesis on a system we
fully control and already trust. If the crossover does not exist, that is decision-relevant before
spending a month on a benchmark port.

| Risk | Mitigation |
|---|---|
| Seeds dissolve the marginal results | Expected for 2 of them; report honestly, the big effects will survive |
| No crossover exists even at high variability | That is a publishable negative result, and a sharper one than what we have |
| Minifab validation (2c) fails | Gate — stop and fix before claiming anything |
| CP-SAT cannot produce useful bounds on Minifab | Fall back to best-known incumbents, drop "certified" from Minifab claims |
| Rolling CP-SAT is accidentally advantaged/handicapped by the replan interval | Sweep the interval; report the curve, not one point |

---

## 6. Open decisions

1. **Venue and deadline** — sets how much of Phase 2/3b is in scope.
2. **Phase 3a or 3b** — open-loop SAA (days, tests the hypothesis) vs closed-loop policy (weeks,
   stronger paper, more overlap with Che & Dong).
3. **Randomness set** — S1 setups only, or S1 + S3 failures? Recommendation is both.
4. **Failure semantics** — when a tool fails mid-job, is the in-flight job lost, resumed, or
   restarted? This changes the delay register and the CP-SAT model materially.
5. **Minifab or our line for Phase 3?** Recommendation: our line first.
6. **Operators in Minifab** — model them, or use the common simplification of ignoring them and
   say so?

---

## References

- Che & Dong, *Differentiable Discrete Event Simulation for Queuing Network Control*, [arXiv:2409.03740](https://arxiv.org/abs/2409.03740)
- *FrontierCO: Real-World and Large-Scale Evaluation of ML Solvers for Combinatorial Optimization*, [arXiv:2505.16952](https://arxiv.org/abs/2505.16952)
- Liu, Lin, Chen, Wu & Shen, *Machine Learning for Scheduling Decision Systems: A Critical Review*, [arXiv:2512.22642](https://arxiv.org/abs/2512.22642)
- *Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling*, [arXiv:2412.14052](https://arxiv.org/abs/2412.14052) (AAAI)
- *Scalability of RL Methods for Dispatching in Semiconductor Frontend Fabs*, [arXiv:2505.11135](https://arxiv.org/abs/2505.11135)
- *Semiconductor Fab Scheduling with Self-Supervised and Reinforcement Learning*, [arXiv:2302.07162](https://arxiv.org/abs/2302.07162)
- Kopp, Hassoun, Kalir & Mönch, *SMT2020 — A Semiconductor Manufacturing Testbed*, IEEE Transactions on Semiconductor Manufacturing
- Hopp & Spearman, *Factory Physics*, Ch. 8 (effective process time, outages, VUT)
- Suh et al., *Do Differentiable Simulators Give Better Policy Gradients?*, ICML 2022
