#!/bin/bash
# Two-experiment run with the pipeline nudge REMOVED from env.py:
#   1. Vanilla PPO (no BC, no nudge)
#   2. BC v2 (rich) + PPO (no nudge)
# Each gets its own report (vs original vanilla as prev), plus a cross report.
set -e
cd "$(dirname "$0")"

V2_BC="output_data_1p/models/bc_alternating_92_v2.zip"
ORIGINAL_VANILLA="output_data_1p/models/ppo_fab_1p_vanilla.zip"
VANILLA_NO_NUDGE="output_data_1p/models/ppo_fab_1p_vanilla_no_nudge.zip"
BC_V2_NO_NUDGE="output_data_1p/models/ppo_fab_1p_bc_v2_no_nudge.zip"

for f in "$V2_BC" "$ORIGINAL_VANILLA"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

# ------------------------------------------------------------------
# Experiment 1: Vanilla PPO (no BC, no nudge)
echo "===EXP1_VANILLA_NO_NUDGE_PPO_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/train_ppo.py --mode ALTERNATING_92
echo "===EXP1_VANILLA_NO_NUDGE_PPO_DONE $(date +%H:%M:%S)==="
cp output_data_1p/models/ppo_fab_1p_policy.zip "$VANILLA_NO_NUDGE"
echo "Saved $VANILLA_NO_NUDGE"

echo "===EXP1_REPORT_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_vanilla_no_nudge \
    --mode ALTERNATING_92 \
    --prev-model "$ORIGINAL_VANILLA"
echo "===EXP1_REPORT_DONE $(date +%H:%M:%S)==="

# ------------------------------------------------------------------
# Experiment 2: BC v2 + PPO (no nudge, with hyperparam fix from earlier)
echo "===EXP2_BC_V2_NO_NUDGE_PPO_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/train_ppo.py --mode ALTERNATING_92 --pretrained "$V2_BC"
echo "===EXP2_BC_V2_NO_NUDGE_PPO_DONE $(date +%H:%M:%S)==="
cp output_data_1p/models/ppo_fab_1p_policy.zip "$BC_V2_NO_NUDGE"
echo "Saved $BC_V2_NO_NUDGE"

echo "===EXP2_REPORT_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_bc_v2_no_nudge \
    --mode ALTERNATING_92 \
    --prev-model "$ORIGINAL_VANILLA"
echo "===EXP2_REPORT_DONE $(date +%H:%M:%S)==="

# ------------------------------------------------------------------
# Cross comparison: vanilla_no_nudge (prev) vs bc_v2_no_nudge (current).
# ppo_fab_1p_policy.zip is currently bc_v2_no_nudge; pass --prev-model to override prev.
echo "===CROSS_REPORT_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_no_nudge_cross \
    --mode ALTERNATING_92 \
    --prev-model "$VANILLA_NO_NUDGE"
echo "===CROSS_REPORT_DONE $(date +%H:%M:%S)==="

echo "===PIPELINE_COMPLETE $(date +%H:%M:%S)==="
