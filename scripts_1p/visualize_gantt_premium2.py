"""
Stunning 3-Panel Premium Visualization: Gantt, Units (Material Flow), and Costs.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplefab_1p import make_common_config
from simplefab_1p.sim import ProductionLine

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output_data_1p')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Premium Dark Palette ────────────────────────────────────────────────
C = {
    'bg':       '#0b0f19',  # Deep dark blue background
    'card':     '#131a28',  # Slightly lighter blue-grey for panels
    'text':     '#e2e8f0',  # Crisp off-white text
    'text_dim': '#94a3b8',  # Dimmed text for labels
    'grid':     '#1e293b',  # Subtle grid lines
    
    # Gantt
    'gantt_b':  '#8b5cf6',  # Vibrant Purple (Batch)
    'gantt_s':  '#3b82f6',  # Bright Blue (Single)
    'arrival':  '#06b6d4',  # Cyan (Arrival)
    'demand':   '#ef4444',  # Red (Demand)
    'shipped':  '#10b981',  # Emerald Green (Shipment)
    
    # Units
    'wip':      '#38bdf8',  # Sky blue for WIP
    'fin_inv':  '#10b981',  # Emerald for Finished Inventory
    'backorder':'#ef4444',  # Red for Backorders
    
    # Costs
    'profit':   '#fbbf24',  # Amber/Gold for Profit
    'cost_inv': '#f59e0b',  # Orange for Inventory Cost
    'cost_bo':  '#ef4444',  # Red for Backorder Cost
}

def style_axes(ax, title=None, ylabel=None):
    ax.set_facecolor(C['card'])
    if title:
        ax.set_title(title, color=C['text'], fontsize=16, fontweight='bold', pad=15, loc='left')
    if ylabel:
        ax.set_ylabel(ylabel, color=C['text_dim'], fontsize=12, labelpad=10, fontweight='bold')
    ax.tick_params(colors=C['text_dim'], labelsize=10, bottom=True, left=True)
    for s in ax.spines.values():
        s.set_color(C['grid'])
        s.set_linewidth(1.5)
    ax.grid(True, color=C['grid'], alpha=0.6, linestyle='-', linewidth=1)

def draw_week_markers(ax, max_w, text_y, text_color):
    for w in range(0, max_w+1, 672):
        ax.axvline(w, color=C['grid'], linestyle='-', linewidth=2, zorder=0)
        ax.text(w + 15, text_y, f"Wk {w//672 + 1}", color=text_color, alpha=0.5, fontsize=10, fontweight='bold')

def main():
    cfg = make_common_config()
    H = cfg["time_horizon"]

    # 1) RUN SIMULATION
    line = ProductionLine(common_cfg=cfg)
    line.logging_enabled = True

    tick_data = []
    for t in range(H):
        line.run_step(current_time=t, actions_override=None)
        cl = line.cost_log[-1]
        
        tick_data.append({
            "t": t,
            "wip": len(line.queues["queue0"]) + len(line.queues["queue1"]) + len(line.queues["queue2"]) + len(line.queues["queue3"]),
            "fin_inv": len(line.queues["queue_fin"]),
            "backorder_qty": len(line.queues["demand"]),
            "revenue": cl["revenue"],
            "production": cl["production"],
            "inventory": cl["inventory"],
            "backorder": cl["backorder"],
            "profit": cl["profit"],
            "m0_busy": line.machine0.busy,
            "m1_busy": line.machine1.busy,
            "m2_busy": line.machine2.busy,
            "m3_busy": line.machine3.busy,
        })
    df = pd.DataFrame(tick_data)

    shipped_counts = np.zeros(H)
    for t_ship in line.demand_met_log:
        if t_ship < H:
            shipped_counts[t_ship] += 1

    # 2) BUILD FIGURE (3 Panels)
    fig = plt.figure(figsize=(20, 18), facecolor=C['bg'])
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1.2, 1.2], hspace=0.15)
    
    x_arr = df['t'].values
    
    # -------------------------------------------------------------------------
    # PANEL 1: GANTT CHART (Machine Scheduling & Events)
    # -------------------------------------------------------------------------
    ax_gantt = fig.add_subplot(gs[0])
    style_axes(ax_gantt, title="1. Machine Activity & Material Events", ylabel="Machine Stage")
    
    machines = ["M3 (Single)", "M2 (Batch)", "M1 (Single)", "M0 (Batch)"]
    y_pos = [30, 20, 10, 0]
    
    ax_gantt.set_yticks(y_pos)
    ax_gantt.set_yticklabels(machines, fontweight='bold', color=C['text'])
    ax_gantt.set_ylim(-10, 45)
    ax_gantt.set_xlim(-20, H + 20)
    
    machine_cols = [C['gantt_s'], C['gantt_b'], C['gantt_s'], C['gantt_b']]
    machine_logs = [line.machine3.event_log, line.machine2.event_log, line.machine1.event_log, line.machine0.event_log]
    
    for m_log, y, col in zip(machine_logs, y_pos, machine_cols):
        bars = [(st, fin - st) for (st, fin, *rest) in m_log]
        if bars:
            ax_gantt.broken_barh(bars, (y - 3, 6), facecolors=col, edgecolor='#000000', linewidth=0.5, alpha=0.85)

    arrivals_dict = {t: qty for t, qty in enumerate(cfg["arrivals_schedule"][0]) if qty > 0}
    demand_dict = {t: qty for t, qty in enumerate(cfg["demand_schedule"][0]) if qty > 0}
    
    draw_week_markers(ax_gantt, H, 40, C['text_dim'])

    for t_arr, qty in arrivals_dict.items():
        score = ax_gantt.scatter(t_arr, -6, color=C['arrival'], marker='^', s=200, zorder=5, edgecolor=C['bg'], linewidth=1.5)
        ax_gantt.text(t_arr + 20, -6, f"+{qty}", color=C['arrival'], fontsize=11, fontweight='bold', va='center')
        ax_gantt.axvline(t_arr, color=C['arrival'], linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)
        
    for t_dem, qty in demand_dict.items():
        ax_gantt.scatter(t_dem, 40, color=C['demand'], marker='v', s=200, zorder=5, edgecolor=C['bg'], linewidth=1.5)
        ax_gantt.text(t_dem - 20, 40, f"-{qty}", color=C['demand'], fontsize=11, fontweight='bold', ha='right', va='center')
        ax_gantt.axvline(t_dem, color=C['demand'], linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)

    # Shipped items spray
    shipped_x = []
    shipped_y = []
    for t_ship, qty in enumerate(shipped_counts):
        if qty > 0:
            shipped_x.extend([t_ship] * int(qty))
            shipped_y.extend(np.random.uniform(28.5, 31.5, int(qty)))
            
    if shipped_x:
        ax_gantt.scatter(shipped_x, shipped_y, color=C['shipped'], s=12, alpha=0.6, zorder=4, edgecolor='none')

    custom_lines = [
        Patch(facecolor=C['gantt_b'], label='Batch Machine'),
        Patch(facecolor=C['gantt_s'], label='Single Machine'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=C['arrival'], markersize=12, label='Raw Arrival'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=C['demand'], markersize=12, label='Demand Hit'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C['shipped'], markersize=8, label='Goods Shipped')
    ]
    ax_gantt.legend(handles=custom_lines, loc='upper left', bbox_to_anchor=(0.01, 1.15),
                    ncol=5, facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'], framealpha=0.9, fontsize=11)
    ax_gantt.set_xticklabels([])

    # -------------------------------------------------------------------------
    # PANEL 2: UNITS (Material Flow & Storage)
    # -------------------------------------------------------------------------
    ax_units = fig.add_subplot(gs[1], sharex=ax_gantt)
    style_axes(ax_units, title="2. Factory State: Where is the material?", ylabel="Volume (Units)")
    
    # We will plot overlapping area charts
    ax_units.plot(x_arr, df['wip'], color=C['wip'], linewidth=2.5, label='Work-in-Process (WIP)')
    ax_units.fill_between(x_arr, 0, df['wip'], color=C['wip'], alpha=0.2)
    
    ax_units.plot(x_arr, df['fin_inv'], color=C['fin_inv'], linewidth=2.5, label='Finished Goods on Shelf')
    ax_units.fill_between(x_arr, 0, df['fin_inv'], color=C['fin_inv'], alpha=0.2)
    
    if df['backorder_qty'].max() > 0:
        ax_units.plot(x_arr, df['backorder_qty'], color=C['backorder'], linewidth=2.5, linestyle='--', label='Backordered Demand')
        ax_units.fill_between(x_arr, 0, df['backorder_qty'], color=C['backorder'], alpha=0.1)

    max_vol = max(df['wip'].max(), df['fin_inv'].max(), df['backorder_qty'].max())
    ax_units.set_ylim(-2, max_vol * 1.15)
    
    draw_week_markers(ax_units, H, max_vol * 1.05, C['text_dim'])

    ax_units.legend(loc='upper right', facecolor=C['card'], edgecolor=C['grid'], 
                  labelcolor=C['text'], framealpha=0.9, fontsize=11)
    ax_units.set_xticklabels([])

    # -------------------------------------------------------------------------
    # PANEL 3: ECONOMICS (Costs vs Profit)
    # -------------------------------------------------------------------------
    ax_fin = fig.add_subplot(gs[2], sharex=ax_gantt)
    style_axes(ax_fin, title="3. Financial Performance: Profit & Cumulative Costs", ylabel="Value ($)")
    
    # Fill cumulative inventory and backorder costs at the BOTTOM to keep them visible but separate from profit
    ax_fin.fill_between(x_arr, 0, df['inventory'], color=C['cost_inv'], alpha=0.4)
    ax_fin.plot(x_arr, df['inventory'], color=C['cost_inv'], linewidth=2, label=f"Cumul. Inventory Hold Cost (${df['inventory'].iloc[-1]:,.0f})")
    
    if df['backorder'].max() > 0:
        ax_fin.plot(x_arr, df['backorder'], color=C['cost_bo'], linewidth=2, label=f"Cumul. Backorder Penalty (${df['backorder'].iloc[-1]:,.0f})")
    
    # Net Profit
    ax_fin.plot(x_arr, df['profit'], color=C['profit'], linewidth=3.5, label=f"NET PROFIT (${df['profit'].iloc[-1]:,.0f})", zorder=10)
    
    # Shade profit cleanly: Gold if positive, Red if negative
    ax_fin.fill_between(x_arr, 0, df['profit'], where=(df['profit'] >= 0), color=C['profit'], alpha=0.15, zorder=5)
    ax_fin.fill_between(x_arr, 0, df['profit'], where=(df['profit'] < 0), color=C['cost_bo'], alpha=0.2, zorder=5)
    
    ax_fin.axhline(0, color=C['text_dim'], linewidth=1.5, alpha=0.5, zorder=1)

    draw_week_markers(ax_fin, H, df['profit'].max() * 0.9, C['text_dim'])

    # Format X Axis
    ax_fin.set_xlabel("Time Horizon (Ticks & Weeks)", color=C['text_dim'], fontsize=12, fontweight='bold', labelpad=12)
    def format_x(x, pos):
        if x < 0 or x > H: return ""
        days = x / 96
        return f"Tick {int(x)}\n(Day {days:.1f})"
    ax_fin.xaxis.set_major_locator(ticker.MultipleLocator(336))
    ax_fin.xaxis.set_major_formatter(ticker.FuncFormatter(format_x))

    ax_fin.legend(loc='upper right', facecolor=C['card'], edgecolor=C['grid'], 
                  labelcolor=C['text'], framealpha=0.9, fontsize=11)
                  
    # Overlay Summary KPI Box
    kpi_text = (
        f"TOTAL SHIPPED: {int(shipped_counts.sum())} / {cfg['demand'][0]}\n"
        f"TOTAL REVENUE: ${df['revenue'].iloc[-1]:,.0f}\n"
        f"TOTAL PROD COST: ${df['production'].iloc[-1]:,.0f}\n"
        f"---------------------------\n"
        f"FINAL PROFIT:  ${df['profit'].iloc[-1]:,.0f}"
    )
    props = dict(boxstyle='round,pad=0.8', facecolor=C['bg'], edgecolor=C['profit'], alpha=0.95, linewidth=2)
    ax_fin.text(0.02, 0.92, kpi_text, transform=ax_fin.transAxes, fontsize=13, fontweight='bold',
                color=C['text'], verticalalignment='top', horizontalalignment='left', bbox=props, zorder=15)

    # FINAL SAVEOUT
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "stunning_system_overview.png")
    fig.savefig(path, dpi=250, facecolor=C['bg'], bbox_inches='tight')
    plt.close()
    print(f"Stunning Premium Visualization saved to: {path}")

if __name__ == "__main__":
    main()
