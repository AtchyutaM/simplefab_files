# Expert-Guided Training (BC + DAgger)

This document explains the expert-guided training pipeline that sits alongside
the standard PPO training in `scripts_1p/train_ppo.py`. It covers what the
system does, how the pieces fit together, how to use it, and how to extend it
to multi-product systems.

## Why we built this

Vanilla PPO trained from scratch regressed at high utilization (92%
alternating demand): the agent under-produced, shipping 558/616 vs the greedy
Commander's 616/616. The CP-SAT MIP showed the true optimum is $33,333 — so
the agent left $3.7K (11%) on the table.

Behavioral Cloning (BC) on a single optimal trajectory gets us close but is
fragile: the policy only sees states the expert visited. Once it deploys and
makes a small mistake, it drifts off-distribution and has no idea how to
recover. **DAgger** (Dataset Aggregation) fixes this: train, roll out, ask
the expert what it would do at *the visited states*, add those (state,
expert_action) pairs to the dataset, retrain.

For our problem the expert is the MIP — given any state, the MIP can solve
for the optimal remaining schedule and tell us the next action.

## Architecture

```
simplefab_1p/
├── sim.py                # unchanged — the deterministic line dynamics
├── env.py                # unchanged — Gym wrapper
├── config.py             # unchanged — demand modes
├── optim.py              # CP-SAT model. Now supports partial-state solving:
│                         # solve_optimal_schedule(cfg, t_start, initial_state, lookahead)
└── expert.py             # ExpertOracle abstraction + MIPExpert implementation.
                          # The BC/DAgger code talks to this interface only.

scripts_1p/
├── train_ppo.py          # unchanged — vanilla PPO; --pretrained still works
├── pretrain_bc.py        # unchanged — supervised BC trainer
├── generate_expert_data.py   # rewritten — uses MIPExpert to produce
│                         # (obs, action, mask) datasets for BC
├── run_optim.py          # CLI for one-shot MIP solves (full or partial)
└── dagger_train.py       # (Phase 3, deferred) full DAgger loop

docs/
└── expert_training.md    # this file

output_data_1p/
├── optim/                # saved MIP solutions per mode
└── expert_data/          # generated BC/DAgger datasets (.npz)
```

### Layer responsibilities

| Layer | What it knows | What it doesn't |
|---|---|---|
| **MIP (`optim.py`)** | the line's dynamics, costs, demand schedule, CP-SAT API | RL, datasets, observations |
| **ExpertOracle (`expert.py`)** | how to query an expert and convert sim state ↔ MIP state ↔ FabEnv obs | which optimizer is being used (could be MIP, LP, heuristic) |
| **BC trainer (`pretrain_bc.py`)** | how to do supervised learning on (obs, action) pairs | where the data came from |
| **PPO trainer (`train_ppo.py`)** | environment dynamics via FabEnv | whether the policy was BC-pretrained |

The clean cut is at `ExpertOracle`: anything that implements
`query_action(state) -> action` qualifies as an expert. This is the seam where
multi-product extension happens.

## How to use

### Generate an expert dataset

```bash
# Default: 6 trajectories (1 optimal + 5 perturbed at ~5 deviations each)
.venv/bin/python scripts_1p/generate_expert_data.py --mode ALTERNATING_92

# Quick check: just the optimal trajectory, no perturbations
.venv/bin/python scripts_1p/generate_expert_data.py \
    --mode ALTERNATING_92 --num-trajectories 1 --target-perturbations 0
```

Outputs `output_data_1p/expert_data/<mode>_<n>traj_<p>pert.npz` with arrays
`obs (N, 18) float32`, `actions (N, 4) int64`, `masks (N, 8) int64`.

### BC pretraining (existing script, unchanged)

```bash
.venv/bin/python scripts_1p/pretrain_bc.py \
    --data output_data_1p/expert_data/alternating_92_6traj_5pert.npz \
    --out output_data_1p/models/bc_alternating_92 \
    --epochs 30
```

### PPO fine-tune (existing script, unchanged)

```bash
.venv/bin/python scripts_1p/train_ppo.py \
    --mode ALTERNATING_92 \
    --pretrained output_data_1p/models/bc_alternating_92.zip
```

### One-shot optimization (no training)

```bash
.venv/bin/python scripts_1p/run_optim.py --mode ALTERNATING_92 --time-limit 300
```

## How to disable / fall back to plain PPO

The expert pipeline is fully optional. Skip steps 1 and 2 — `train_ppo.py`
runs from scratch with no MIP dependency:

```bash
.venv/bin/python scripts_1p/train_ppo.py --mode ALTERNATING_92
```

If you want the codebase to never touch OR-Tools at all, you can `pip
uninstall ortools` and remove the imports in `optim.py` / `expert.py` /
`run_optim.py` / `generate_expert_data.py`. The rest of the system
(simulation, env, PPO trainer, BC trainer) does not import these.

## Extending to multi-product

The framework is designed so that adding multi-product support means
**writing one new file** and the rest of the pipeline reuses unchanged.

Steps:

1. **Write `simplefab_2p/optim.py`** — the multi-product MIP. New decision
   variables need to encode product mix on each batch (`x[m][t][p]` for
   product p), setup times, possibly setup costs.

2. **Write `simplefab_2p/expert.py`** — a `MultiProductMIPExpert` class
   implementing the same `ExpertOracle` interface as `MIPExpert`. The
   `query_action(state, t)` signature is the same; only the underlying call
   into the MIP changes.

3. **Reuse `generate_expert_data.py`** — it talks only to
   `MIPExpert` / `ExpertOracle`. Pass the multi-product expert and the
   pipeline just works.

4. **Reuse `pretrain_bc.py` and `train_ppo.py`** — they work on
   `(obs, action)` arrays without caring about the action semantics, as long
   as the env reports the same observation/action spaces it always does. If
   the multi-product action space has more dimensions, those scripts still
   work (sb3 handles arbitrary `MultiDiscrete`).

The pipeline is product-agnostic *because* the expert is the only layer that
encodes product semantics. Everything else operates on standard RL types.

## How the pieces fit at runtime (full DAgger flow)

```
[1] Generate dataset (current Phase 2)
    ┌────────────────┐
    │ MIPExpert      │  full-horizon solve once
    │  .generate_    │  → optimal trajectory (2,688 samples)
    │   trajectory() │
    └────────┬───────┘
             │ for k in (1..N-1):
             │   roll out optimal-with-noise
             │   re-solve MIP from each post-deviation state
             │   record (visited_obs, expert_action) per tick
             ▼
    output_data_1p/expert_data/<...>.npz

[2] BC pretrain
    .npz  ──>  pretrain_bc.py (NLL loss on action distribution)
            ──>  output_data_1p/models/bc_<mode>.zip

[3] PPO fine-tune
    bc_<mode>.zip  ──>  train_ppo.py --pretrained <...>
                    ──>  output_data_1p/models/ppo_fab_1p_policy.zip

[4] (Phase 3, future) DAgger inner loop
    Repeat:
      roll out current policy in env, capture every visited (state, t)
      query MIPExpert.query_action(state, t) for each
      append to dataset, retrain BC, optionally PPO fine-tune
```

## Performance considerations

- **Full-horizon MIP solve**: ~30-180s with the Commander warm-start. Without
  the warm-start it stalls (we tested: 180s of search reached -$26K from a
  cold start).
- **Partial-state MIP solve** (DAgger queries): proportional to the
  `lookahead_ticks`. Default = full remaining horizon. For fast oracle
  queries during DAgger, reduce lookahead to e.g. 1344 (2 weeks).
- **Sample size**: each trajectory contributes 2,688 samples (one per tick).
  6 trajectories ≈ 16K samples; that's likely enough for BC to converge on a
  small `[128, 128]` MLP.
- **Cost scaling**: CP-SAT works in integers; we scale all costs by 1000
  internally. Monetary outputs are rescaled.

## Pitfalls

1. **The "theoretical max" formula in `generate_dashboards_compare.py` is an
   overestimate.** It assumes only $0.36 of WIP transit cost per batch and no
   FG holding. Real schedules incur more because alternating demand forces
   pre-building. Use the LP bound from `optim.py` (`objective_bound`) as the
   tight upper limit. For `ALTERNATING_92` that's $33,987, not $34,441.

2. **Sim/MIP must agree exactly.** When you change `sim.py` (e.g., add new
   cost term, change processing time), you MUST update `optim.py` to match.
   The `verify_solution()` helper checks that `sim.py`'s replay profit equals
   the MIP's reported profit; if it diverges, the formulation has drifted.

3. **Action mask vs MIP optimality.** The MIP enforces capacity and queue
   constraints by construction, so its actions are always mask-valid by
   FabEnv's standards. If you ever see a mask violation in expert data, it
   means the recurrence has a bug — re-derive the queue dynamics carefully.
