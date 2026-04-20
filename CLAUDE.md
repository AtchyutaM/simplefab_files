# SimpleFab RL Project

## Overview
RL agent for scheduling a single-product, 4-machine serial production line (semiconductor fab).
Uses Masked PPO (sb3-contrib) to learn production scheduling that maximizes profit under inventory, WIP, and backorder costs.

## Architecture
- **Simulation**: `simplefab_1p/sim.py` — Discrete-time ProductionLine with 4 machines: M0 (batch/16 ticks) → M1 (single/2 ticks) → M2 (batch/16 ticks) → M3 (single/2 ticks). Batch size = 4.
- **Environment**: `simplefab_1p/env.py` — Gymnasium wrapper. 18-dim obs, MultiDiscrete([2,2,2,2]) actions, action masking, potential-based reward shaping, pipeline nudge penalty.
- **Config**: `simplefab_1p/config.py` — Central config generator. Modes: UNIFORM, ALTERNATING, ALTERNATING_50.
- **Training**: `scripts_1p/train_ppo.py` — PPO training with `--mode` and `--pretrained` CLI args. Auto-launches TensorBoard.
- **Reports**: `scripts_1p/generate_dashboards_compare.py` — Generates dashboards and HTML comparison reports with `--exp-name`, `--mode`, `--prev-model` args.

## Current Parameters (as of latest commit)
- **WIP costs**: Q0=$0.00, Q1=$0.005, Q2=$0.01, Q3=$0.015 per unit per tick (progressive)
- **FG holding**: $0.04/unit/tick
- **Backorder**: $0.15/unit/tick
- **Revenue**: $80/unit, Production cost: $24/unit ($8+$4+$8+$4)
- **Shaping**: w_backlog=1.0, w_wip=0.0, w_finished=0.5, beta=0.5, gamma=0.999
- **Pipeline nudge**: -5.0 (in env.py, applied when M0 idles during critical lead time)
- **Training**: 8M steps, 8 envs, n_steps=2688, batch_size=672, ent_coef=0.05, lr=3e-4
- **Evaluation**: Temperature-scaled sampling T=0.7, best of 20 episodes

## Running Experiments

```bash
# Train (mode: UNIFORM, ALTERNATING, ALTERNATING_50)
python scripts_1p/train_ppo.py --mode UNIFORM

# Generate comparison report
python scripts_1p/generate_dashboards_compare.py --exp-name <name> --mode <mode> [--prev-model <path>]

# TensorBoard
python -m tensorboard.main --logdir ./tb_fab_1p --reload_interval 2
```

## Experiment Results Summary

| Experiment | Demand Pattern | Util | Agent Profit | Commander | Theoretical Max | % of Max |
|------------|---------------|------|-------------|-----------|-----------------|----------|
| Uniform 50% (with WIP) | [84,84,84,84]=336 | 50% | $17,607 | -$159 | $18,786 | 93.7% |
| Alternating 50% | [56,56,112,112]=336 | 50% | $16,337 | -$2,813 | $18,786 | 87.0% |
| Alternating 89% | [80,80,220,220]=600 | 89% | $29,965 | $29,480 | $33,546 | 89.3% |

## Key Findings
1. Agent's advantage grows with excess capacity (low utilization) — at 50% util agent wins by $17K+, at 89% by only ~$500
2. Temperature-scaled evaluation (T=0.7) recovers 5-8% over deterministic argmax
3. Progressive WIP costs prevent WIP parking behavior but need careful nudge tuning
4. Nudge=-5.0 works for high-util, nudge=-3.5 was too weak at 89% causing under-production

## Next Steps (Planned)
1. **LP/MIP optimal solution**: Formulate the scheduling problem as MIP using OR-Tools, solve for provably optimal schedules
2. **Behavioral Cloning from MIP**: Use optimal trajectories as BC pretraining, then fine-tune with PPO (--pretrained pipeline exists)
3. This is expected to close the remaining 7-13% gap to theoretical maximum

## Saved Models
- `output_data_1p/models/ppo_fab_1p_policy.zip` — Latest trained model (Alternating 89%)
- `output_data_1p/models/ppo_fab_1p_policy_no_wip_cost.zip` — Uniform 50% model without WIP costs
- Checkpoints in `output_data_1p/models/ppo_fab_1p_checkpoint_*_steps.zip` (from latest run only)

## Reports
All HTML reports in `output_data_1p/reports/`:
- `uniform_50pct_comparison_report.html` — Original uniform experiment (no WIP costs)
- `uniform_50pct_wip_comparison_report.html` — Uniform with WIP costs
- `alternating_50pct_wip_comparison_report.html` — Alternating 50% util
- `alternating_89pct_wip_comparison_report.html` — Alternating 89% util

## Setup
```bash
# Python 3.12, dependencies in .venv
python -m venv .venv
source .venv/bin/activate
pip install sb3-contrib stable-baselines3 gymnasium numpy pandas matplotlib tensorboard
```
