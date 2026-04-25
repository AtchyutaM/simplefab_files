"""
Run the CP-SAT optimization to find the profit-maximizing schedule.

Examples:
  python scripts_1p/run_optim.py --mode ALTERNATING_92
  python scripts_1p/run_optim.py --mode UNIFORM --time-limit 60 --workers 12

Saves the solution to output_data_1p/optim/<mode>_optim.pkl for later
behavioral cloning use.
"""
import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplefab_1p.config import make_common_config
from simplefab_1p.optim import solve_optimal_schedule, verify_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ALTERNATING_92",
                        help="Demand mode: UNIFORM, ALTERNATING, ALTERNATING_50, ALTERNATING_92")
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="Solver wall-time limit in seconds (default 300)")
    parser.add_argument("--workers", type=int, default=8,
                        help="CP-SAT parallel workers (default 8)")
    parser.add_argument("--save", type=str, default="",
                        help="Custom pickle output path (default output_data_1p/optim/<mode>_optim.pkl)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress solver search log")
    args = parser.parse_args()

    cfg = make_common_config(mode=args.mode)
    print(f"=== SimpleFab MIP Optimization ===")
    print(f"  Mode:           {args.mode}")
    print(f"  Horizon:        {cfg['time_horizon']} ticks")
    print(f"  Total arrivals: {cfg['arrivals'][0]}")
    print(f"  Total demand:   {cfg['demand'][0]}")
    print(f"  Initial qfin:   {cfg['initial_finished_inventory'][0]}")
    print(f"  Util:           {cfg['utilization']*100:.1f}%")
    print()

    result = solve_optimal_schedule(
        cfg,
        time_limit_sec=args.time_limit,
        num_workers=args.workers,
        verbose=not args.quiet,
    )

    print(f"\n=== Solver Result ===")
    print(f"  Status:        {result['status']}")
    print(f"  Optimal:       {result.get('is_optimal', False)}")
    print(f"  Wall time:     {result['solver_wall_time']:.1f}s")

    if result["profit"] is None:
        print(f"  Best bound:    ${result['objective_bound']:,.2f}")
        print("  No feasible solution found.")
        return

    print(f"  Profit:        ${result['profit']:,.2f}")
    print(f"  Best bound:    ${result['objective_bound']:,.2f}")
    gap = result['objective_bound'] - result['profit']
    if abs(result['profit']) > 1e-6:
        gap_pct = 100 * gap / abs(result['profit'])
        print(f"  Gap to bound:  ${gap:,.2f}  ({gap_pct:.3f}%)")
    print(f"  Throughput:    {result['throughput']} / {cfg['demand'][0]}")

    print(f"\n=== Breakdown (from solver) ===")
    b = result["breakdown"]
    print(f"  Revenue:       ${b['revenue']:>10,.2f}")
    print(f"  Production:    ${b['production']:>10,.2f}")
    print(f"  WIP:           ${b['wip']:>10,.2f}")
    print(f"  FG inventory:  ${b['inventory']:>10,.2f}")
    print(f"  Backorder:     ${b['backorder']:>10,.2f}")
    net = b['revenue'] - b['production'] - b['wip'] - b['inventory'] - b['backorder']
    print(f"  Net (check):   ${net:>10,.2f}")

    print(f"\n=== Sim Replay (sanity check) ===")
    sim_stats = verify_solution(cfg, result["starts"])
    print(f"  Profit:        ${sim_stats['profit']:>10,.2f}")
    print(f"  Revenue:       ${sim_stats['revenue']:>10,.2f}")
    print(f"  Production:    ${sim_stats['production']:>10,.2f}")
    print(f"  WIP:           ${sim_stats['wip']:>10,.2f}")
    print(f"  FG inventory:  ${sim_stats['inventory']:>10,.2f}")
    print(f"  Backorder:     ${sim_stats['backorder']:>10,.2f}")
    print(f"  Throughput:    {sim_stats['throughput']}")

    diff = sim_stats['profit'] - result['profit']
    if abs(diff) < 1.0:
        print(f"  -> MIP and sim agree (delta=${diff:+.4f})")
    else:
        print(f"  !! MISMATCH (delta=${diff:+,.2f}) - formulation may have drifted from sim semantics")

    save_path = args.save
    if not save_path:
        save_dir = os.path.join(os.path.dirname(__file__), "..", "output_data_1p", "optim")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{args.mode.lower()}_optim.pkl")

    payload = {
        "mode": args.mode,
        "config": cfg,
        "result": result,
        "sim_stats": sim_stats,
    }
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n=> Saved solution to {save_path}")


if __name__ == "__main__":
    main()
