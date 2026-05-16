#!/bin/bash
# Halved WIP costs experiment: BC v2 + PPO no_nudge 8M
# WIP costs in config.py: Q1=$0.0025, Q2=$0.005, Q3=$0.0075 (was 2x each)
set -e
cd "$(dirname "$0")"

V2_BC="output_data_1p/models/bc_alternating_92_v2.zip"
PRIOR_BC_V2_NO_NUDGE="output_data_1p/models/ppo_fab_1p_bc_v2_no_nudge.zip"
HALFWIP_MODEL="output_data_1p/models/ppo_fab_1p_bc_v2_no_nudge_halfwip.zip"
HALFWIP_OPTIM="output_data_1p/optim/alternating_92_halfwip_optim.pkl"

# ------------------------------------------------------------------
echo "===MIP_RECOMPUTE_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/run_optim.py \
    --mode ALTERNATING_92 \
    --time-limit 180 \
    --workers 12 \
    --quiet \
    --save "$HALFWIP_OPTIM"
echo "===MIP_RECOMPUTE_DONE $(date +%H:%M:%S)==="

# ------------------------------------------------------------------
echo "===MIP_DASHBOARD_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/render_optim_dashboard.py \
    --solution "$HALFWIP_OPTIM" \
    --exp-name alternating_92pct_halfwip
echo "===MIP_DASHBOARD_DONE $(date +%H:%M:%S)==="

# ------------------------------------------------------------------
echo "===PPO_TRAIN_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/train_ppo.py \
    --mode ALTERNATING_92 \
    --pretrained "$V2_BC" \
    --total-timesteps 8000000
echo "===PPO_TRAIN_DONE $(date +%H:%M:%S)==="
cp output_data_1p/models/ppo_fab_1p_policy.zip "$HALFWIP_MODEL"

# ------------------------------------------------------------------
echo "===REPORT_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_bc_v2_no_nudge_halfwip \
    --mode ALTERNATING_92 \
    --prev-model "$PRIOR_BC_V2_NO_NUDGE"
echo "===REPORT_DONE $(date +%H:%M:%S)==="

echo "===PIPELINE_COMPLETE $(date +%H:%M:%S)==="
