from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

from simplefab_1p import make_common_config
from simplefab_1p.sim import ProductionLine
from simplefab_1p.gantt import extract_gantt_data, plot_gantt_chart


def main():
    common = make_common_config()

    line = ProductionLine(common_cfg=common)
    line.logging_enabled = True

    for t in range(common["time_horizon"]):
        line.run_step(current_time=t, actions_override=None)

    totals = line.costs
    profit = line.profit_total()

    print("\n=== ECONOMIC SUMMARY (1-Product Simulation) ===")
    print(f"Revenue:            {totals['revenue']:.2f}")
    print(f"Production costs:   {totals['production']:.2f}")
    print(f"Inventory costs:    {totals['inventory']:.2f}")
    print(f"Backorder costs:    {totals['backorder']:.2f}")
    print(f"-------------------------------")
    print(f"Total Profit:       {profit:.2f}")
    print(f"\nThroughput:         {len(line.demand_met_log)} units shipped")
    print(f"Total demand:       {line.total_demand} units")

    # Gantt
    gantt_df = extract_gantt_data(line)
    plot_gantt_chart(gantt_df)

    # Save event log to CSV
    out_dir = os.path.join(os.getcwd(), "output_data_1p")
    os.makedirs(out_dir, exist_ok=True)

    event_log_df = pd.DataFrame(line.event_log)
    csv_path = os.path.join(out_dir, "final_event_log.csv")
    event_log_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot queues over time
    if not event_log_df.empty:
        fig, axs = plt.subplots(6, 1, figsize=(12, 24), sharex=True)
        queue_names = ["Queue 0", "Queue 1", "Queue 2", "Queue 3", "Queue Fin", "Demand Queue"]

        for i, q in enumerate(queue_names):
            col = q
            if col in event_log_df.columns:
                axs[i].plot(event_log_df["Time"], event_log_df[col], label="Product 0", color="tab:blue")
            axs[i].set_title(f"{q} Over Time")
            axs[i].set_ylabel("Quantity")
            axs[i].legend()

        axs[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
