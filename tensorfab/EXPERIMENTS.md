# tensorfab experiment log

Auto-appended by `tensorfab/logbook.py`. One row per run; full detail in
`tensorfab/runs/<run-id>/` (env capture, params, transcript, metrics, series).
Newest rows at the bottom. Interpretation lives in `tensorfab/README.md` §4-4B.

## Reading this log

- `20260802-082534_openloop_spike` — smoke test, H=336 only, tiny budget. Ignore.
- `20260802-082653_openloop_spike` — **the BPTT depth sweep.** H in
  {336, 672, 1344, 2688} on the self-similar UNIFORM family. Established that
  gradients survive 2688 steps with zero non-finite values and that credit
  assignment reaches the full horizon. Also established that this problem family
  is too easy to discriminate between optimisers (random search lands within 0.3%
  of everything), which is why the next run switched problems.
- `20260802-083703_openloop_spike` — **the method comparison**, on the hard bursty
  ALTERNATING_92 problem where the commander sits 6% below the MILP. Gradient
  reached 98.7% of the MILP optimum and beat hill-climbing by 0.28%.
- `20260802-085128_bench_env` — throughput of the tensorized env wrapper.

## Reproducing

```bash
python -m tensorfab.spike_openloop --milp --milp-seconds 90
python -m tensorfab.spike_openloop --problem ALTERNATING_92 --horizons 2688 \
    --steps 600 --batch 64 --milp --milp-seconds 180
python -m tensorfab.bench_env
```

Runs are seeded (`--seed`, default 0) and the seed is recorded in `params.json`.
`meta.json` records the git commit and whether the tree was dirty — every run so
far is `1413848+dirty`, since `tensorfab/` is not yet committed.

| run id | experiment | headline | wall s | backend | commit |
|---|---|---|---|---|---|
| `20260802-082534_openloop_spike` | openloop_spike | gradient WINS vs zeroth-order (1/1 horizons); beats commander 1/1; credit reach 89% of horizon | 2 | cpu | 1413848+dirty |
| `20260802-082653_openloop_spike` | openloop_spike | gradient LOSES vs zeroth-order (2/4 horizons); beats commander 4/4; credit reach 94% of horizon | 440 | cpu | 1413848+dirty |
| `20260802-083703_openloop_spike` | openloop_spike | gradient WINS vs zeroth-order (1/1 horizons); beats commander 1/1; credit reach 96% of horizon | 241 | cpu | 1413848+dirty |
| `20260802-085128_bench_env` | bench_env | tensor env peak 17,552,338 steps/s at B=4096 (308x FabEnv's 56,980) | 5 | cpu | 1413848+dirty |
| `20260802-092504_compare_patterns` | compare_patterns | FAILED: TypeError | 990 | cpu | 1413848+dirty |
| `20260802-100336_eval_ppo_patterns` | eval_ppo_patterns | PPO retrained on 3 patterns; temperature sampling adds $1,060 on average over argmax | 62 | cpu | 1413848+dirty |
| `20260802-094439_compare_patterns` | compare_patterns | grad-st beats hillclimb on 4/4 patterns (margin $213 to $15,727); mean 95.3% of CP-SAT dual bound; CP-SAT failed to converge on 0/4 | 2686 | cpu | 1413848+dirty |
