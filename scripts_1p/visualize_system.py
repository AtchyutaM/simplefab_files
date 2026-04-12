"""
Comprehensive single-product system visualization.
Shows material flow, machine activity, queue levels, demand, costs — all in one chart.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplefab_1p import make_common_config
from simplefab_1p.sim import ProductionLine

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output_data_1p')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────
C = {
    'bg':       '#0d1117',
    'card':     '#161b22',
    'text':     '#c9d1d9',
    'grid':     '#30363d',
    'blue':     '#58a6ff',
    'orange':   '#f78166',
    'green':    '#3fb950',
    'red':      '#f85149',
    'purple':   '#d2a8ff',
    'yellow':   '#e3b341',
    'cyan':     '#56d4dd',
    'pink':     '#f778ba',
}


def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(C['card'])
    ax.set_title(title, color=C['text'], fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color=C['text'], fontsize=9)
    ax.set_ylabel(ylabel, color=C['text'], fontsize=9)
    ax.tick_params(colors=C['text'], labelsize=8)
    for s in ax.spines.values():
        s.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.3, linestyle='--')


def main():
    cfg = make_common_config()
    H = cfg["time_horizon"]

    # ── Run the simulation with full logging ──────────────────────────────
    line = ProductionLine(common_cfg=cfg)
    line.logging_enabled = True

    # Track per-tick data
    tick_data = []

    for t in range(H):
        line.run_step(current_time=t, actions_override=None)

        cl = line.cost_log[-1]
        tick_data.append({
            "t": t,
            "day": t / 96.0,
            "week": t / 672.0,
            # Queue levels
            "q0": len(line.queues["queue0"]),
            "q1": len(line.queues["queue1"]),
            "q2": len(line.queues["queue2"]),
            "q3": len(line.queues["queue3"]),
            "q_fin": len(line.queues["queue_fin"]),
            "demand_q": len(line.queues["demand"]),
            # Machine busy
            "m0_busy": int(line.machine0.busy),
            "m1_busy": int(line.machine1.busy),
            "m2_busy": int(line.machine2.busy),
            "m3_busy": int(line.machine3.busy),
            # Cumulative costs
            "revenue": cl["revenue"],
            "production": cl["production"],
            "inventory": cl["inventory"],
            "backorder": cl["backorder"],
            "profit": cl["profit"],
            # Per-step
            "step_reward": cl["step_reward"],
            # Arrivals and demand at this tick
            "arrivals_this_tick": cfg["arrivals_schedule"][0][t] if t < H else 0,
            "demand_this_tick": cfg["demand_schedule"][0][t] if t < H else 0,
        })

    df = pd.DataFrame(tick_data)

    # Compute incremental costs per tick
    df["inv_cost_tick"] = df["inventory"].diff().fillna(0)
    df["bo_cost_tick"] = df["backorder"].diff().fillna(0)
    df["prod_cost_tick"] = df["production"].diff().fillna(0)
    df["rev_tick"] = df["revenue"].diff().fillna(0)

    # Cumulative throughput
    cum_shipped = np.zeros(H)
    for ship_time in line.demand_met_log:
        if ship_time < H:
            cum_shipped[ship_time:] += 1
    df["cum_shipped"] = cum_shipped

    cum_demand = np.cumsum(cfg["demand_schedule"][0])
    df["cum_demand"] = cum_demand[:H]

    cum_arrivals = np.cumsum(cfg["arrivals_schedule"][0])
    df["cum_arrivals"] = cum_arrivals[:H]

    # ── Build the mega chart ──────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 28), facecolor=C['bg'])
    gs = gridspec.GridSpec(7, 1, hspace=0.35, height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0, 1.2, 1.0])

    days = df["day"]

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Panel 1: Material Flow — Cumulative Arrivals, Throughput, Demand  │
    # └─────────────────────────────────────────────────────────────────────┘
    ax1 = fig.add_subplot(gs[0])
    style_ax(ax1, "① Material Flow: Arrivals → Factory → Shipped vs Demand", "Day", "Cumulative Units")

    ax1.plot(days, df["cum_arrivals"], color=C['cyan'], linewidth=2, label="Raw Material Arrived")
    ax1.plot(days, df["cum_shipped"], color=C['green'], linewidth=2.5, label="Units Shipped (demand met)")
    ax1.step(days, df["cum_demand"], where='post', color=C['red'], linewidth=2, linestyle='--', label="Cumulative Demand")

    # Mark demand arrival moments
    for t in range(H):
        if cfg["demand_schedule"][0][t] > 0:
            ax1.axvline(x=t/96, color=C['red'], alpha=0.3, linestyle=':')
            ax1.annotate(f"+{cfg['demand_schedule'][0][t]} demand",
                        xy=(t/96, df["cum_demand"].iloc[t]), fontsize=8, color=C['red'],
                        xytext=(t/96 + 0.5, df["cum_demand"].iloc[t] + 20),
                        arrowprops=dict(arrowstyle='->', color=C['red'], lw=0.8))

    # Mark arrival moments
    for t in range(H):
        if cfg["arrivals_schedule"][0][t] > 0:
            ax1.axvline(x=t/96, color=C['cyan'], alpha=0.2, linestyle=':')

    # Shade the backorder gap
    ax1.fill_between(days, df["cum_shipped"], df["cum_demand"],
                     where=df["cum_demand"] > df["cum_shipped"],
                     alpha=0.15, color=C['red'], label="Backorder gap")

    ax1.legend(loc='upper left', facecolor=C['card'], edgecolor=C['grid'],
              labelcolor=C['text'], fontsize=9)

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Panel 2: Machine Activity (Gantt-style)                           │
    # └─────────────────────────────────────────────────────────────────────┘
    ax2 = fig.add_subplot(gs[1])
    style_ax(ax2, "② Machine Activity (Gantt)", "Day", "")

    machine_labels = ["M3 (Single)", "M2 (Batch)", "M1 (Single)", "M0 (Batch)"]
    y_positions = {0: 3, 1: 2, 2: 1, 3: 0}
    machine_colors = [C['blue'], C['purple'], C['blue'], C['purple']]

    for m_idx, m_name in enumerate(["machine0", "machine1", "machine2", "machine3"]):
        m = getattr(line, m_name)
        y = y_positions[m_idx]
        bars = []
        for e in m.event_log:
            st, fin = e[0], e[1]
            bars.append((st / 96.0, (fin - st) / 96.0))
        if bars:
            ax2.broken_barh(bars, (y - 0.3, 0.6), facecolors=machine_colors[m_idx], alpha=0.8)

    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(machine_labels[::-1])
    ax2.set_ylim(-0.5, 3.5)

    # demand arrival lines
    for t in range(H):
        if cfg["demand_schedule"][0][t] > 0:
            ax2.axvline(x=t/96, color=C['red'], alpha=0.3, linestyle=':', linewidth=1)

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Panel 3: WIP Queues (Queue 0–3)                                   │
    # └─────────────────────────────────────────────────────────────────────┘
    ax3 = fig.add_subplot(gs[2])
    style_ax(ax3, "③ Work-In-Process Queues (Material Inside Factory)", "Day", "Units in Queue")

    ax3.plot(days, df["q0"], color=C['cyan'], linewidth=1.5, label="Q0 (before M0)", alpha=0.9)
    ax3.plot(days, df["q1"], color=C['blue'], linewidth=1.5, label="Q1 (M0→M1)", alpha=0.9)
    ax3.plot(days, df["q2"], color=C['purple'], linewidth=1.5, label="Q2 (M1→M2)", alpha=0.9)
    ax3.plot(days, df["q3"], color=C['pink'], linewidth=1.5, label="Q3 (M2→M3)", alpha=0.9)

    ax3.legend(loc='upper right', facecolor=C['card'], edgecolor=C['grid'],
              labelcolor=C['text'], fontsize=9, ncol=2)

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Panel 4: Finished Inventory & Backorder Queues                    │
    # └─────────────────────────────────────────────────────────────────────┘
    ax4 = fig.add_subplot(gs[3])
    style_ax(ax4, "④ Finished Inventory vs Backorders (Unmet Demand)", "Day", "Units")

    ax4.fill_between(days, 0, df["q_fin"], alpha=0.4, color=C['green'], label="Finished Inventory")
    ax4.plot(days, df["q_fin"], color=C['green'], linewidth=1.5)

    ax4.fill_between(days, 0, df["demand_q"], alpha=0.4, color=C['red'], label="Backorders (unmet demand)")
    ax4.plot(days, df["demand_q"], color=C['red'], linewidth=1.5)

    # annotate peak backorder
    peak_bo_idx = df["demand_q"].idxmax()
    peak_bo = df["demand_q"].iloc[peak_bo_idx]
    peak_day = df["day"].iloc[peak_bo_idx]
    ax4.annotate(f"Peak: {peak_bo} units",
                xy=(peak_day, peak_bo), fontsize=9, color=C['red'],
                xytext=(peak_day + 2, peak_bo + 5),
                arrowprops=dict(arrowstyle='->', color=C['red'], lw=1))

    ax4.legend(loc='upper right', facecolor=C['card'], edgecolor=C['grid'],
              labelcolor=C['text'], fontsize=9)

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Panel 5: Per-Tick Cost Breakdown (inventory vs backorder)         │
    # └─────────────────────────────────────────────────────────────────────┘
    ax5 = fig.add_subplot(gs[4])
    style_ax(ax5, "⑤ Per-Tick Costs: Where Money Is Being Lost", "Day", "Cost per Tick ($)")

    # smooth for readability
    window = 48  # half day
    bo_smooth = df["bo_cost_tick"].rolling(window, center=True, min_periods=1).mean()
    inv_smooth = df["inv_cost_tick"].rolling(window, center=True, min_periods=1).mean()

    ax5.fill_between(days, 0, bo_smooth, alpha=0.5, color=C['red'], label="Backorder cost/tick")
    ax5.fill_between(days, 0, inv_smooth, alpha=0.5, color=C['yellow'], label="Inventory holding cost/tick")

    ax5.legend(loc='upper right', facecolor=C['card'], edgecolor=C['grid'],
              labelcolor=C['text'], fontsize=9)

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Panel 6: Cumulative Economics                                     │
    # └─────────────────────────────────────────────────────────────────────┘
    ax6 = fig.add_subplot(gs[5])
    style_ax(ax6, "⑥ Cumulative Economics Over Time", "Day", "Cumulative ($)")

    ax6.plot(days, df["revenue"], color=C['green'], linewidth=2, label="Revenue")
    ax6.plot(days, df["production"], color=C['orange'], linewidth=1.5, label="Production Cost")
    ax6.plot(days, df["backorder"], color=C['red'], linewidth=2, label="Backorder Cost")
    ax6.plot(days, df["inventory"], color=C['yellow'], linewidth=1.5, label="Inventory Cost")
    ax6.plot(days, df["profit"], color=C['purple'], linewidth=2.5, label="Net Profit", linestyle='-')

    # Mark where profit goes negative or flat
    ax6.axhline(y=0, color=C['text'], alpha=0.3, linewidth=0.5)

    ax6.legend(loc='upper left', facecolor=C['card'], edgecolor=C['grid'],
              labelcolor=C['text'], fontsize=9)

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  Panel 7: Snapshot at key moments                                   │
    # └─────────────────────────────────────────────────────────────────────┘
    ax7 = fig.add_subplot(gs[6])
    style_ax(ax7, "⑦ System Snapshot at Key Moments", "", "")
    ax7.axis('off')

    # Key moment analysis
    snapshots = []
    key_ticks = [0, 95, 96, 200, 672, 768, 1344, 2688-1]
    for t_snap in key_ticks:
        if t_snap >= H:
            t_snap = H - 1
        row = df.iloc[t_snap]
        shipped = int(row["cum_shipped"])
        demand = int(row["cum_demand"])
        snapshots.append({
            "tick": int(row["t"]),
            "day": f"{row['day']:.1f}",
            "q0": int(row["q0"]), "q1": int(row["q1"]),
            "q2": int(row["q2"]), "q3": int(row["q3"]),
            "fin_inv": int(row["q_fin"]),
            "backorders": int(row["demand_q"]),
            "shipped": shipped,
            "cum_demand": demand,
            "profit": f"${row['profit']:,.0f}",
            "bo_cost": f"${row['backorder']:,.0f}",
        })

    # Build table
    col_labels = ["Tick", "Day", "Q0", "Q1", "Q2", "Q3", "Fin.Inv", "Backlog", "Shipped", "Cum.Dem", "Profit", "BO Cost"]
    cell_text = []
    for s in snapshots:
        cell_text.append([
            str(s["tick"]), s["day"],
            str(s["q0"]), str(s["q1"]), str(s["q2"]), str(s["q3"]),
            str(s["fin_inv"]), str(s["backorders"]),
            str(s["shipped"]), str(s["cum_demand"]),
            s["profit"], s["bo_cost"],
        ])

    table = ax7.table(cellText=cell_text, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style table
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(C['grid'])
        if key[0] == 0:  # header
            cell.set_facecolor(C['blue'])
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(C['card'])
            cell.set_text_props(color=C['text'])

    # ── Save ──────────────────────────────────────────────────────────────
    path = os.path.join(OUT_DIR, "system_visualization.png")
    fig.savefig(path, dpi=150, facecolor=C['bg'], bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ── Print key stats ───────────────────────────────────────────────────
    print("\n=== KEY INSIGHTS ===")
    print(f"At t=96 (day 1, first demand arrives):")
    r96 = df.iloc[96]
    print(f"  Units shipped so far:      {int(r96['cum_shipped'])}")
    print(f"  Finished inventory:        {int(r96['q_fin'])}")
    print(f"  Demand just arrived:       {cfg['demand_schedule'][0][96]}")
    print(f"  Immediate backorders:      {int(r96['demand_q'])}")
    print(f"  WIP in factory:  Q0={int(r96['q0'])}, Q1={int(r96['q1'])}, Q2={int(r96['q2'])}, Q3={int(r96['q3'])}")

    print(f"\nAt t=672 (end of week 1, before second arrival):")
    r672_before = df.iloc[671]
    print(f"  Total shipped:             {int(r672_before['cum_shipped'])}")
    print(f"  Remaining backorders:      {int(r672_before['demand_q'])}")
    print(f"  Finished inventory:        {int(r672_before['q_fin'])}")
    print(f"  Cumulative backorder cost: ${r672_before['backorder']:,.2f}")

    print(f"\nAt t=672 (start of week 2, new arrivals + demand):")
    r672 = df.iloc[672]
    print(f"  New raw material arrives:  {cfg['arrivals_schedule'][0][672]}")

    print(f"\nAt t=768 (day 8, second demand arrives):")
    r768 = df.iloc[768]
    print(f"  Total shipped:             {int(r768['cum_shipped'])}")
    print(f"  Backorders after 2nd dem:  {int(r768['demand_q'])}")

    print(f"\nFinal (t={H-1}):")
    rfin = df.iloc[-1]
    print(f"  Total shipped:             {int(rfin['cum_shipped'])}")
    print(f"  Total demand:              {cfg['demand'][0]}")
    print(f"  Unmet demand:              {int(rfin['demand_q'])}")
    print(f"  Excess inventory:          {int(rfin['q_fin'])}")
    print(f"  Revenue:                   ${rfin['revenue']:,.2f}")
    print(f"  Production cost:           ${rfin['production']:,.2f}")
    print(f"  Backorder cost:            ${rfin['backorder']:,.2f}")
    print(f"  Inventory cost:            ${rfin['inventory']:,.2f}")
    print(f"  Net profit:                ${rfin['profit']:,.2f}")
    print(f"  Theoretical max profit:    $34,496 (if zero inv/BO)")
    print(f"  Lost to backorders:        ${rfin['backorder']:,.2f}")
    print(f"  Lost to inventory:         ${rfin['inventory']:,.2f}")


if __name__ == "__main__":
    main()
