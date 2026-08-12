#!/usr/bin/env bash
# Retrain Maskable PPO on the three demand patterns that have historical results
# in CLAUDE.md but whose models were overwritten by the ALTERNATING_92 work.
#
# Uses scripts_1p/train_ppo.py UNCHANGED so the hyperparameters are identical to
# the historical runs (8M steps, 16 envs, n_steps=2688, batch=1344, ent_coef=0.05,
# gamma=0.999, shaping beta=0.5/w_wip=0.0, invalid_action_penalty=-2.0).
# Everything runs under the CURRENT (halved) WIP costs, so these numbers are
# directly comparable to the tensorfab method comparison.
#
# train_ppo.py saves to fixed filenames, so this script moves each result aside
# after the run and restores the pre-existing ALTERNATING_92 model at the end.
# A full backup was taken to output_data_1p/models_backup_20260802/ first.

set -uo pipefail
cd "$(dirname "$0")/.."

MODELS=output_data_1p/models
OUT=tensorfab/runs/ppo_multipattern
mkdir -p "$OUT"

MODES=("UNIFORM" "ALTERNATING_50" "ALTERNATING")

for MODE in "${MODES[@]}"; do
    echo "=============================================================="
    echo "PPO retrain: $MODE   ($(date +%H:%M:%S))"
    echo "=============================================================="
    python scripts_1p/train_ppo.py --mode "$MODE" --total-timesteps 8000000 \
        > "$OUT/train_${MODE}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAILED (rc=$rc) — see $OUT/train_${MODE}.log"
        tail -20 "$OUT/train_${MODE}.log"
        continue
    fi
    cp "$MODELS/ppo_fab_1p_policy.zip"   "$MODELS/ppo_mp_${MODE}.zip"
    cp "$MODELS/vecnorm_stats_1p.pkl"    "$MODELS/vecnorm_mp_${MODE}.pkl"
    echo "  saved -> $MODELS/ppo_mp_${MODE}.zip"
    grep -E "fps|time_elapsed" "$OUT/train_${MODE}.log" | tail -2
done

# Restore the pre-existing ALTERNATING_92 artifacts that train_ppo.py clobbered.
cp output_data_1p/models_backup_20260802/ppo_fab_1p_policy.zip "$MODELS/" 2>/dev/null
cp output_data_1p/models_backup_20260802/vecnorm_stats_1p.pkl  "$MODELS/" 2>/dev/null
echo "restored original ppo_fab_1p_policy.zip / vecnorm_stats_1p.pkl"
echo "ALL PPO RETRAINS DONE ($(date +%H:%M:%S))"
