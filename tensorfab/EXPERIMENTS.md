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
| `20260812-114513_compare_patterns` | compare_patterns | grad-st beats hillclimb on 1/1 patterns (margin $14,139 to $14,139); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 41 | cpu | 0da7577+dirty |
| `20260812-114555_compare_patterns` | compare_patterns | FAILED: TypeError | 0 | cpu | 0da7577+dirty |
| `20260812-114556_compare_patterns` | compare_patterns | FAILED: TypeError | 0 | cpu | 0da7577+dirty |
| `20260812-114634_compare_patterns` | compare_patterns | grad-st beats hillclimb on 1/1 patterns (margin $14,139 to $14,139); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 41 | cpu | 0da7577+dirty |
| `20260812-114716_compare_patterns` | compare_patterns | grad-st beats hillclimb on 1/1 patterns (margin $12,877 to $12,877); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 83 | cpu | 0da7577+dirty |
| `20260812-114840_compare_patterns` | compare_patterns | grad-st beats hillclimb on 1/1 patterns (margin $12,019 to $12,019); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 128 | cpu | 0da7577+dirty |
| `20260812-123613_compare_patterns` | compare_patterns | grad-st beats hillclimb on 1/1 patterns (margin $70 to $70); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 3 | cpu | 0da7577+dirty |
| `20260812-123541_compare_patterns` | compare_patterns | grad-st beats hillclimb on 0/1 patterns (margin $-18 to $-18); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 58 | cpu | 0da7577+dirty |
| `20260812-123637_compare_patterns` | compare_patterns | grad-st beats hillclimb on 0/1 patterns (margin $-18 to $-18); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 60 | cpu | 0da7577+dirty |
| `20260812-123640_compare_patterns` | compare_patterns | grad-st beats hillclimb on 0/1 patterns (margin $-751 to $-751); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 107 | cpu | 0da7577+dirty |
| `20260812-123737_compare_patterns` | compare_patterns | grad-st beats hillclimb on 0/1 patterns (margin $-751 to $-751); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 108 | cpu | 0da7577+dirty |
| `20260812-123828_compare_patterns` | compare_patterns | grad-st beats hillclimb on 0/1 patterns (margin $-1,078 to $-1,078); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 160 | cpu | 0da7577+dirty |
| `20260812-123926_compare_patterns` | compare_patterns | grad-st beats hillclimb on 0/1 patterns (margin $-1,078 to $-1,078); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 159 | cpu | 0da7577+dirty |
| `20260812-125719_compare_patterns` | compare_patterns | grad-st beats hillclimb on 0/1 patterns (margin $-1,106 to $-1,106); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 154 | cpu | 0da7577+dirty |
| `20260812-125954_compare_patterns` | compare_patterns | grad-st beats hillclimb on 0/1 patterns (margin $-1,076 to $-1,076); mean nan% of CP-SAT dual bound; CP-SAT failed to converge on 0/1 | 152 | cpu | 0da7577+dirty |
| `20260812-120047_long_milp_depth` | long_milp_depth | CP-SAT vs depth: M=4 inc $18,993 bound $19,003 gap 0.1% \| M=8 inc $18,840 bound $19,003 gap 0.9% \| M=12 inc $18,934 bound $19,008 gap 0.4% | 5402 | cpu | 0da7577+dirty |
| `20260906-180010_twoprod_compare` | twoprod_compare | grad-st beats zeroth-order on 0/1 mixes (margin $-631 to $-631) | 3 | cpu | fda76ea+dirty |
| `20260906-180113_twoprod_compare` | twoprod_compare | grad-st beats zeroth-order on 3/3 mixes (margin $666 to $2,125) | 242 | cpu | fb63529+dirty |
| `20260906-180516_twoprod_compare` | twoprod_compare | grad-st beats zeroth-order on 3/3 mixes (margin $967 to $2,664) | 243 | cpu | fb63529+dirty |
| `20260906-190453_twoprod_long_milp` | twoprod_long_milp | 2-product CP-SAT at 1800s: gaps 3.27% / 1.76% / 1.44%; proven optimal 0/3 | 4343 | cpu | 88ead65+dirty |
