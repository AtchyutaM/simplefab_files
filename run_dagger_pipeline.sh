#!/bin/bash
# Chained launch: dataset gen -> BC pretrain -> PPO fine-tune -> report.
# All output goes to dagger_pipeline.log. Step boundaries are tagged so we can
# easily grep progress.
set -e
cd "$(dirname "$0")"

NUM_TRAJ=8
TARGET_PERT=15
DATA="output_data_1p/expert_data/alternating_92_${NUM_TRAJ}traj_${TARGET_PERT}pert.npz"
BC_OUT="output_data_1p/models/bc_alternating_92_v2"

echo "===STEP_1_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_expert_data.py \
    --mode ALTERNATING_92 \
    --num-trajectories $NUM_TRAJ \
    --target-perturbations $TARGET_PERT \
    --time-limit 30 \
    --workers 12
echo "===STEP_1_DONE $(date +%H:%M:%S)==="

echo "===STEP_2_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/pretrain_bc.py \
    --data "$DATA" \
    --out "$BC_OUT" \
    --epochs 30
echo "===STEP_2_DONE $(date +%H:%M:%S)==="

echo "===STEP_3_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/train_ppo.py \
    --mode ALTERNATING_92 \
    --pretrained "${BC_OUT}.zip"
echo "===STEP_3_DONE $(date +%H:%M:%S)==="

echo "===STEP_4_START $(date +%H:%M:%S)==="
.venv/bin/python -u scripts_1p/generate_dashboards_compare.py \
    --exp-name alternating_92pct_bc_v2 \
    --mode ALTERNATING_92 \
    --prev-model output_data_1p/models/ppo_fab_1p_bc_v1.zip
echo "===STEP_4_DONE $(date +%H:%M:%S)==="

echo "===PIPELINE_COMPLETE $(date +%H:%M:%S)==="
