#!/bin/bash
# Clean comparison run: BC v1 (sparse dataset) and BC v2 (rich dataset)
# WITH the hyperparameter inheritance bug fixed.
# Produces three reports:
#   - alternating_92pct_bc_v1_fixed: v1_fixed vs vanilla
#   - alternating_92pct_bc_v2_fixed: v2_fixed vs vanilla
#   - alternating_92pct_v1_vs_v2:    v2_fixed vs v1_fixed
set -e
cd "$(dirname "$0")"

V1_BC="output_data_1p/models/bc_alternating_92.zip"
V2_BC="output_data_1p/models/bc_alternating_92_v2.zip"
VANILLA="output_data_1p/models/ppo_fab_1p_vanilla.zip"
V1_FIXED="output_data_1p/models/ppo_fab_1p_bc_v1_fixed.zip"
V2_FIXED="output_data_1p/models/ppo_fab_1p_bc_v2_fixed.zip"

# Sanity check inputs
for f in "$V1_BC" "$V2_BC" "$VANILLA"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

# ------------------------------------------------------------------
echo "===V1_PPO_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/train_ppo.py \
    --mode ALTERNATING_92 \
    --pretrained "$V1_BC"
echo "===V1_PPO_DONE $(date +%H:%M:%S)==="
cp output_data_1p/models/ppo_fab_1p_policy.zip "$V1_FIXED"
echo "Saved v1_fixed model to $V1_FIXED"

echo "===V1_REPORT_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_bc_v1_fixed \
    --mode ALTERNATING_92 \
    --prev-model "$VANILLA"
echo "===V1_REPORT_DONE $(date +%H:%M:%S)==="

# ------------------------------------------------------------------
echo "===V2_PPO_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/train_ppo.py \
    --mode ALTERNATING_92 \
    --pretrained "$V2_BC"
echo "===V2_PPO_DONE $(date +%H:%M:%S)==="
cp output_data_1p/models/ppo_fab_1p_policy.zip "$V2_FIXED"
echo "Saved v2_fixed model to $V2_FIXED"

echo "===V2_REPORT_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_bc_v2_fixed \
    --mode ALTERNATING_92 \
    --prev-model "$VANILLA"
echo "===V2_REPORT_DONE $(date +%H:%M:%S)==="

# ------------------------------------------------------------------
# Cross comparison: ppo_fab_1p_policy.zip is currently v2_fixed
echo "===CROSS_REPORT_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_v1_vs_v2 \
    --mode ALTERNATING_92 \
    --prev-model "$V1_FIXED"
echo "===CROSS_REPORT_DONE $(date +%H:%M:%S)==="

echo "===PIPELINE_COMPLETE $(date +%H:%M:%S)==="
