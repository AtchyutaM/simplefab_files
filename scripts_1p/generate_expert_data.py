"""
Generate (obs, action, mask) datasets from the MIP expert oracle.

Replaces the prior Commander-heuristic generator. Uses simplefab_1p.expert.MIPExpert
to produce one optimal trajectory plus N perturbed trajectories (DAgger-style:
optimal action labels at off-optimal states).

Output:
  .npz with arrays
    obs:     (N, 18) float32  — FabEnv-normalized observations
    actions: (N, 4)  int64    — MultiDiscrete action labels (the expert's choice)
    masks:   (N, 8)  int64    — flattened action masks (for masked BC training)

Compatible with scripts_1p/pretrain_bc.py without changes.

Examples:
  # ALTERNATING_92, default: 1 optimal + 5 perturbed trajectories, target 5 perturbations each
  python scripts_1p/generate_expert_data.py --mode ALTERNATING_92

  # Quick single-trajectory dataset (no perturbations) — fastest
  python scripts_1p/generate_expert_data.py --mode ALTERNATING_92 --num-trajectories 1 --target-perturbations 0

  # Larger dataset for harder problems
  python scripts_1p/generate_expert_data.py --mode ALTERNATING_92 --num-trajectories 20 --target-perturbations 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplefab_1p.config import make_common_config
from simplefab_1p.expert import MIPExpert


def generate_dataset(
    mode: str,
    num_trajectories: int,
    target_perturbations: int,
    time_limit_per_solve: float,
    num_workers: int,
    seed: int,
    lookahead_ticks: int = 0,
    verbose: bool = True,
) -> dict:
    """Build the dataset by aggregating one optimal trajectory + (num_trajectories - 1)
    perturbed trajectories.

    `target_perturbations` sets the EXPECTED number of deviations per perturbed
    trajectory. We translate this to a per-tick perturb_prob = target / H.
    """
    cfg = make_common_config(mode=mode)
    H = int(cfg["time_horizon"])

    expert = MIPExpert(
        cfg,
        time_limit_sec=time_limit_per_solve,
        lookahead_ticks=lookahead_ticks if lookahead_ticks > 0 else None,
        num_workers=num_workers,
        verbose=False,
    )

    if verbose:
        print(f"Mode: {mode}, H={H}")
        print(f"Building {num_trajectories} trajectories ({num_trajectories - 1} perturbed at ~{target_perturbations} deviations each)")

    # Trajectory 0: pure optimal (also caches the MIP solution for reuse)
    t0 = time.time()
    full = expert.generate_full_trajectory()
    if verbose:
        print(f"[traj 0/{num_trajectories - 1}] optimal: {len(full)} samples in {time.time() - t0:.1f}s")

    obs_list = [q.obs for q in full]
    act_list = [q.action for q in full]
    mask_list = [q.mask for q in full]

    perturb_prob = (target_perturbations / H) if H > 0 else 0.0
    rng = np.random.default_rng(seed)

    for i in range(1, num_trajectories):
        t0 = time.time()
        traj_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        perturbed = expert.generate_perturbed_trajectory(perturb_prob=perturb_prob, rng=traj_rng)
        if verbose:
            print(f"[traj {i}/{num_trajectories - 1}] perturbed: {len(perturbed)} samples in {time.time() - t0:.1f}s")
        obs_list.extend(q.obs for q in perturbed)
        act_list.extend(q.action for q in perturbed)
        mask_list.extend(q.mask for q in perturbed)

    obs = np.asarray(obs_list, dtype=np.float32)
    actions = np.asarray(act_list, dtype=np.int64)
    # pretrain_bc.py expects masks as int64 in the file (they're 0/1 anyway)
    masks = np.asarray(mask_list, dtype=np.int64)

    return {
        "obs": obs,
        "actions": actions,
        "masks": masks,
        "metadata": {
            "mode": mode,
            "num_trajectories": num_trajectories,
            "target_perturbations": target_perturbations,
            "perturb_prob": perturb_prob,
            "horizon": H,
            "lookahead_ticks": lookahead_ticks if lookahead_ticks > 0 else H,
            "time_limit_per_solve": time_limit_per_solve,
            "seed": seed,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ALTERNATING_92",
                        help="Demand mode: UNIFORM, ALTERNATING, ALTERNATING_50, ALTERNATING_92")
    parser.add_argument("--num-trajectories", type=int, default=6,
                        help="Total trajectories (1 optimal + N-1 perturbed). Default 6.")
    parser.add_argument("--target-perturbations", type=int, default=5,
                        help="Expected deviations per perturbed trajectory. Default 5.")
    parser.add_argument("--time-limit", type=float, default=30.0,
                        help="Solver wall-time per MIP solve in seconds. Default 30.")
    parser.add_argument("--workers", type=int, default=12,
                        help="CP-SAT parallel workers. Default 12.")
    parser.add_argument("--lookahead-ticks", type=int, default=0,
                        help="Lookahead horizon for re-solves (0 = full horizon). Smaller = faster oracle queries but more myopic.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for perturbations.")
    parser.add_argument("--out", type=str, default="",
                        help="Output .npz path (default: output_data_1p/expert_data/<mode>_<n>traj_<p>pert.npz)")
    args = parser.parse_args()

    out_path = args.out
    if not out_path:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "output_data_1p", "expert_data")
        fname = f"{args.mode.lower()}_{args.num_trajectories}traj_{args.target_perturbations}pert.npz"
        out_path = os.path.join(out_dir, fname)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f"=== MIP Expert Data Generation ===")
    print(f"  Mode:                 {args.mode}")
    print(f"  Num trajectories:     {args.num_trajectories}  ({args.num_trajectories - 1} perturbed)")
    print(f"  Target perturbations: {args.target_perturbations} per trajectory")
    print(f"  Time limit / solve:   {args.time_limit}s")
    print(f"  Workers:              {args.workers}")
    print(f"  Lookahead:            {'full horizon' if args.lookahead_ticks <= 0 else str(args.lookahead_ticks) + ' ticks'}")
    print(f"  Output:               {out_path}")
    print()

    t_total = time.time()
    data = generate_dataset(
        mode=args.mode,
        num_trajectories=args.num_trajectories,
        target_perturbations=args.target_perturbations,
        time_limit_per_solve=args.time_limit,
        num_workers=args.workers,
        seed=args.seed,
        lookahead_ticks=args.lookahead_ticks,
        verbose=True,
    )
    elapsed = time.time() - t_total

    print(f"\n=== Dataset Summary ===")
    print(f"  Total samples: {len(data['obs'])}")
    print(f"  obs shape:     {data['obs'].shape}")
    print(f"  actions shape: {data['actions'].shape}")
    print(f"  masks shape:   {data['masks'].shape}")
    print(f"  Action means (M0..M3 Run-rate): {data['actions'].mean(axis=0).round(3)}")
    print(f"  Total wall time: {elapsed:.1f}s")

    np.savez_compressed(
        out_path,
        obs=data["obs"],
        actions=data["actions"],
        masks=data["masks"],
    )
    print(f"\n=> Saved to {out_path}")


if __name__ == "__main__":
    main()
