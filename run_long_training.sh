#!/bin/bash
# 16M-step training (2x prior 8M) with no nudge — both vanilla and BC v2+PPO.
# Goal: see if the remaining ~5.5% gap to MIP optimum closes with more training.
set -e
cd "$(dirname "$0")"

V2_BC="output_data_1p/models/bc_alternating_92_v2.zip"
PRIOR_VANILLA_NO_NUDGE="output_data_1p/models/ppo_fab_1p_vanilla_no_nudge.zip"
PRIOR_BC_V2_NO_NUDGE="output_data_1p/models/ppo_fab_1p_bc_v2_no_nudge.zip"
VANILLA_LONG="output_data_1p/models/ppo_fab_1p_vanilla_no_nudge_16M.zip"
BC_V2_LONG="output_data_1p/models/ppo_fab_1p_bc_v2_no_nudge_16M.zip"

TOTAL_STEPS=16000000

for f in "$V2_BC" "$PRIOR_VANILLA_NO_NUDGE" "$PRIOR_BC_V2_NO_NUDGE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

# ------------------------------------------------------------------
echo "===EXP1_VANILLA_LONG_PPO_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/train_ppo.py --mode ALTERNATING_92 --total-timesteps $TOTAL_STEPS
echo "===EXP1_VANILLA_LONG_PPO_DONE $(date +%H:%M:%S)==="
cp output_data_1p/models/ppo_fab_1p_policy.zip "$VANILLA_LONG"
echo "Saved $VANILLA_LONG"

echo "===EXP1_REPORT_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_vanilla_no_nudge_16M \
    --mode ALTERNATING_92 \
    --prev-model "$PRIOR_VANILLA_NO_NUDGE"
echo "===EXP1_REPORT_DONE $(date +%H:%M:%S)==="

# ------------------------------------------------------------------
echo "===EXP2_BC_V2_LONG_PPO_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/train_ppo.py --mode ALTERNATING_92 --pretrained "$V2_BC" --total-timesteps $TOTAL_STEPS
echo "===EXP2_BC_V2_LONG_PPO_DONE $(date +%H:%M:%S)==="
cp output_data_1p/models/ppo_fab_1p_policy.zip "$BC_V2_LONG"
echo "Saved $BC_V2_LONG"

echo "===EXP2_REPORT_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_bc_v2_no_nudge_16M \
    --mode ALTERNATING_92 \
    --prev-model "$PRIOR_BC_V2_NO_NUDGE"
echo "===EXP2_REPORT_DONE $(date +%H:%M:%S)==="

# ------------------------------------------------------------------
# Cross comparison: vanilla_long (prev) vs bc_v2_long (current)
echo "===CROSS_REPORT_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_no_nudge_16M_cross \
    --mode ALTERNATING_92 \
    --prev-model "$VANILLA_LONG"
echo "===CROSS_REPORT_DONE $(date +%H:%M:%S)==="

echo "===PIPELINE_COMPLETE $(date +%H:%M:%S)==="
