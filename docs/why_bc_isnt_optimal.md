# Why BC isn't optimal even though MIP is, and what PPO is supposed to add

A common confusion: if the MIP finds the provably-optimal schedule, and BC just imitates the MIP, why doesn't BC reproduce $33,333 instead of $31,103? And then what is PPO actually adding on top?

This doc explains:
1. The MIP formulation in plain terms (what it does, what it knows)
2. Why BC can't fully recover the MIP's profit
3. What PPO is supposed to fix that BC doesn't
4. What we actually observed in this project

---

## 1. The MIP — what it solves

The MIP is a deterministic, full-information optimization. Implemented in
`simplefab_1p/optim.py` using OR-Tools CP-SAT.

### Decision variables

For each machine `m ∈ {0, 1, 2, 3}` and tick `t ∈ [0, H)`:
- `x[m][t] ∈ {0, 1}` — machine `m` starts a job at tick `t`

That's it. With `H = 2,688`, we have **10,752 binary decision variables**. The MIP picks the schedule (which ticks each machine starts) that maximizes profit.

### Constraints — straight from `sim.py`'s mechanics

Encoded as linear constraints on the binary `x` variables:

- **Capacity**: at most one start per machine in any `pt[m]`-tick window (M0/M2 batch machines have `pt=16`, M1/M3 single-unit have `pt=2`). Otherwise the same machine would be running two jobs at once.
- **Queue dynamics** (recurrent — O(1) terms each):
  - `q0[t] = q0[t-1] + arrivals[t] - 4 · x[0][t]`  (raw materials in/out)
  - `q1[t] = q1[t-1] + 4 · x[0][t-16] - x[1][t]`  (M0 completes 16 ticks after start, deposits 4 units in q1)
  - …same shape for q2, q3
- **Queues ≥ 0**: bound on each `q*[t]` ensures we never start a machine without enough input.
- **Shipping**: `cum_ship[t]` is monotone, bounded by `qfin0 + (M3 done by tick t)` (can't ship what we haven't produced) and by `cum_demand[t+1]` (can't ship what hasn't been ordered).
- **Backorder accounting**: `late[t] ≥ cum_demand_arrived_672_ticks_ago[t] - cum_ship[t]`. The solver minimizes `late` because backorder is a cost.

### Objective

Linear in the decision variables (after scaling all costs by 1000 to keep integer arithmetic):

```
maximize  $80 · cum_ship[H-1]
        - $24 · (units produced)
        - Σ (0.005 · q1[t] + 0.01 · q2[t] + 0.015 · q3[t])
        - $0.04 · Σ qfin[t]
        - $0.15 · Σ late[t]
```

### How CP-SAT actually solves it

- 10,752 binary variables × thousands of linear constraints. Pure brute force would never finish.
- We provide a **Commander heuristic warm-start hint** (`AddHint`): "start by trying the greedy 'always run when allowed' schedule." This gives the solver a feasible solution worth ~$31K in milliseconds, then it refines.
- LP relaxation gives an upper bound the solver tightens against the integer solution. For ALTERNATING_92, LP bound = $33,987, integer solution at 180s = $33,333 (gap = 1.96%).
- Result: a specific list of start times per machine, plus the simulator's profit when the schedule is replayed.

### Critical property: full information

**The MIP sees the entire 2,688-tick horizon at once.** It knows when every demand event arrives, when every raw material delivery happens, and computes a globally-optimal schedule conditioned on all of it.

This is the key asymmetry with the RL agent.

---

## 2. Why BC can't fully recover the MIP's profit

BC ("behavioral cloning") trains a neural network policy `π_θ(a | obs)` to imitate the MIP's actions: minimize cross-entropy between the policy's action distribution and the MIP's action at each tick.

### The asymmetries that prevent perfect imitation

**(a) The agent operates on a 18-dim observation, not the full schedule.**

The MIP knows everything. The policy sees only:
- Current queue lengths (q0, q1, q2, q3, qfin, demand backlog)
- Each machine's busy flag and time-remaining
- Global progress `t / H`
- Period progress `(t mod 672) / 672`
- Ticks until next demand event + upcoming demand quantity

Several MIP decisions depend on info the policy *can't see*. Example: if there's a 224-unit demand spike at tick 1,344 *and* a 224-unit spike at tick 2,016, the optimal schedule starts pre-building during week 2 to absorb both. The policy sees one upcoming demand event, not the second one. Even a perfect imitator of the policy's input is making decisions with strictly less information than the MIP.

**(b) Function-approximator information loss.**

A 128×128 MLP has ~30K parameters. The MIP's solution is a 10,752-bit vector. The MLP must compress the *function* from observation to action into 30K weights. Some compression is information-lossy on this state space.

**(c) Compounding error / distribution shift.**

BC trains on `(state, action)` pairs *the expert visited*. When deployed, the policy makes small errors → visits states the expert never showed → makes larger errors → fails further off-distribution. This is the classic "BC fails on its own state distribution" problem (Ross & Bagnell 2010).

We observed this directly: BC v1 (sparse dataset, almost all states near the optimal trajectory) topped out at $30,730. BC v2 (richer dataset, with deliberate perturbations and re-solves at off-distribution states) topped out at $31,103. More state coverage → more recovery → ~$370 better. Diminishing returns set in because pure BC dataset enlargement is bounded by what the perturbation strategy explores.

**(d) Stochastic evaluation.**

We evaluate with temperature 0.7 sampling, best of 20 episodes. Even a perfectly-imitating policy has variance from sampling. The MIP is deterministic; the eval is not.

### Net effect on this project

BC alone (v2): **$31,103 best, 93.3% of MIP optimum.**

That ~$2,200 gap is the sum of all the asymmetries above.

---

## 3. What PPO is *supposed* to add

The hope is that PPO addresses BC's failure modes:

**(a) Distribution shift correction.** PPO trains on *the policy's own rollouts*. It sees the off-distribution states the agent visits in deployment, and adjusts the policy based on actual outcomes there. In principle, this fixes the compounding-error problem BC has.

**(b) Refinement via the reward signal.** BC labels tell you "the expert took action A" but not "action A was 0.3 better than action B". PPO's value function and advantage estimate provide gradient information about *how much better* one action is than another, allowing fine-tuning beyond the expert.

**(c) Sharper policy where BC is uncertain.** If the BC dataset is ambiguous (multiple expert actions at similar states), the BC policy ends up with high entropy. PPO can sharpen the distribution toward whichever action actually performs better in the env.

In a well-aligned env, the standard recipe is: **BC for warm-starting near-optimal behavior, PPO for the last mile.**

---

## 4. What we actually observed

In our env, **PPO mostly hurt rather than helped** because the env's reward signal is *misaligned* with the MIP-optimal trajectory:

- BC v2 alone: $31,103 best
- BC v2 + PPO no_nudge 8M: $31,493 best (+$390 — small win)
- BC v2 + PPO no_nudge 16M: $30,053 best (-$1,050 — *regressed*)
- BC v2 fixed + PPO with nudge: $28,678 best (-$2,425 — destroyed)

What's happening: PPO finds gradient signal pointing toward an under-production equilibrium (~$30K). With the pipeline nudge, this attractor was steep — PPO destroyed BC priors quickly. Without the nudge, the attractor is shallower but still pulls — over enough updates, PPO drags the policy back to ~$30K regardless of where BC starts it.

### Why the env reward and the MIP optimum disagree

The MIP optimum tells us: *over the full horizon*, ship every unit, time production tightly to demand spikes, accept ~$987 of unavoidable WIP transit cost, get $33,333.

The env reward at tick `t` is local: `Δprofit + invalid_penalty + shaping_term`. This local signal tells PPO things like:
- "Idling M1 right now saves $0.005/tick of WIP cost on the unit in q1." → reward
- "Starting M3 right now incurs a $4 production cost." → punishment
- "Producing more inventory now grows future FG holding cost." → punishment

PPO sees an immediate cost for *most production decisions* and only sees the revenue for shipped units much later (when the unit eventually clears the pipeline + ships). With temporal discounting at γ=0.999, future rewards are damped. The local gradient signal says "be cautious about producing"; the global optimum says "produce aggressively to meet the spikes." These two signals point in different directions on average — that's the misalignment.

### The asymmetric bottom line

- **The MIP exploits global information** the policy never has access to.
- **BC inherits the *outputs* of that global reasoning** (what action to take where), but loses some quality through observation compression and function approximation.
- **PPO is supposed to refine** but only does if the env's *local* reward signal points to the same optimum the MIP found globally.
- **In our env, it doesn't** — the local reward signal points to a sub-optimum. So PPO erodes BC priors over time.

This is why the next experiments target the env reward (halve WIP costs to soften the local "be cautious" signal) and the broader path forward is full DAgger (use MIP, not env reward, to correct PPO drift).
