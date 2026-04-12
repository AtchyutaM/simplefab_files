"""
Premium Gantt-based visualization of the 1-Product SimpleFab system.
Combines factory activity (Gantt, arrivals, demand, shipped) with real-time economics.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplefab_1p import make_common_config
from simplefab_1p.sim import ProductionLine

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output_data_1p')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Styling Constants ───────────────────────────────────────────────────
C = {
    'bg':       '#0d1117',
    'card':     '#161b22',
    'text':     '#c9d1d9',
    'grid':     '#30363d',
    'gantt_b':  '#ba82fa',  # Batch machine color
    'gantt_s':  '#58a6ff',  # Single machine color
    'arrival':  '#56d4dd',  # Cyan for raw materials
    'demand':   '#f85149',  # Red for demand drop
    'shipped':  '#3fb950',  # Green for shipped goods
    'profit':   '#f0e575',  # Yellow/Gold for profit
    'cost_inv': '#e3b341',  # Inventory holding cost
    'cost_bo':  '#f85149',  # Backorder cost
    'cost_pr':  '#f78166',  # Production cost
    'rev':      '#3fb950',  # Revenue
}

def style_axes(ax, title=None, ylabel=None):
    ax.set_facecolor(C['card'])
    if title:
        ax.set_title(title, color=C['text'], fontsize=14, fontweight='bold', pad=15)
    if ylabel:
        ax.set_ylabel(ylabel, color=C['text'], fontsize=11, labelpad=10)
    ax.tick_params(colors=C['text'], labelsize=10, bottom=True, left=True)
    for s in ax.spines.values():
        s.set_color(C['grid'])
        s.set_linewidth(1.5)
    ax.grid(True, color=C['grid'], alpha=0.5, linestyle='--', linewidth=0.8)

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
            "revenue": cl["revenue"],
            "production": cl["production"],
            "inventory": cl["inventory"],
            "backorder": cl["backorder"],
            "profit": cl["profit"],
        })
    df = pd.DataFrame(tick_data)

    # Shipped items array
    shipped_counts = np.zeros(H)
    for t_ship in line.demand_met_log:
        if t_ship < H:
            shipped_counts[t_ship] += 1

    # 2) BUILD FIGURE
    fig = plt.figure(figsize=(18, 12), facecolor=C['bg'])
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1.2], hspace=0.15)
    
    # --- PANEL 1: GANTT CHART & LOGISTICS ---
    ax_gantt = fig.add_subplot(gs[0])
    style_axes(ax_gantt, title="Production Timeline: Machine Scheduling & Material Flow", ylabel="Machine")
    
    machines = ["M3 (Single)", "M2 (Batch)", "M1 (Single)", "M0 (Batch)"]
    y_pos = [30, 20, 10, 0]
    
    ax_gantt.set_yticks(y_pos)
    ax_gantt.set_yticklabels(machines, fontweight='bold')
    ax_gantt.set_ylim(-10, 42)
    ax_gantt.set_xlim(-50, H + 50)
    
    # Plot Gantt Bars
    machine_logs = [line.machine3.event_log, line.machine2.event_log, line.machine1.event_log, line.machine0.event_log]
    machine_colors = [C['gantt_s'], C['gantt_b'], C['gantt_s'], C['gantt_b']]
    
    for idx, (m_log, y, col) in enumerate(zip(machine_logs, y_pos, machine_colors)):
        bars = [(st, fin - st) for (st, fin, *rest) in m_log]
        if bars:
            ax_gantt.broken_barh(bars, (y - 3, 6), facecolors=col, edgecolor='black', linewidth=0.5, alpha=0.9)
            
    # Overlays: Arrivals and Demand
    arrivals_dict = {t: qty for t, qty in enumerate(cfg["arrivals_schedule"][0]) if qty > 0}
    demand_dict = {t: qty for t, qty in enumerate(cfg["demand_schedule"][0]) if qty > 0}
    
    # Draw vertical demarcations for weeks
    for w in range(0, H+1, 672):
        ax_gantt.axvline(w, color=C['grid'], linestyle='-', linewidth=2, alpha=0.8, zorder=0)
        ax_gantt.text(w + 10, 39, f"Week {w//672}", color=C['text'], alpha=0.6, fontsize=10, fontweight='bold')

    for t_arr, qty in arrivals_dict.items():
        ax_gantt.scatter(t_arr, -7, color=C['arrival'], marker='^', s=150, zorder=5, edgecolor='black')
        ax_gantt.text(t_arr + 15, -8, f" Arrivals\n +{qty}", color=C['arrival'], fontsize=9, fontweight='bold', va='center')
        ax_gantt.axvline(t_arr, color=C['arrival'], linestyle=':', linewidth=1.5, alpha=0.4, zorder=0)
        
    for t_dem, qty in demand_dict.items():
        ax_gantt.scatter(t_dem, 37, color=C['demand'], marker='v', s=150, zorder=5, edgecolor='black')
        ax_gantt.text(t_dem - 15, 38, f"Demand \n+{qty} ", color=C['demand'], fontsize=9, fontweight='bold', ha='right', va='center')
        ax_gantt.axvline(t_dem, color=C['demand'], linestyle=':', linewidth=1.5, alpha=0.4, zorder=0)

    # Shipped Items Overlay (Dots on M3)
    shipped_x = []
    shipped_y = []
    for t_ship, qty in enumerate(shipped_counts):
        if qty > 0:
            shipped_x.extend([t_ship] * int(qty))
            shipped_y.extend(np.random.uniform(29, 31, int(qty))) # Jitter for visibility on M3
            
    if shipped_x:
        ax_gantt.scatter(shipped_x, shipped_y, color=C['shipped'], s=10, alpha=0.8, zorder=4, label='Unit Shipped')

    # Gantt Legend
    custom_lines = [
        Patch(facecolor=C['gantt_b'], label='Batch Process'),
        Patch(facecolor=C['gantt_s'], label='Single Unit Process'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=C['arrival'], markersize=10, label='Raw Material Arrival'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=C['demand'], markersize=10, label='Customer Demand'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C['shipped'], markersize=6, label='Unit Shipped')
    ]
    ax_gantt.legend(handles=custom_lines, loc='upper left', bbox_to_anchor=(0.01, 1.15),
                    ncol=5, facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'], framealpha=0.9, fontsize=10)

    # Remove X-ticks from Gantt (shared with bottom)
    ax_gantt.set_xticklabels([])

    # --- PANEL 2: FINANCIALS ---
    ax_fin = fig.add_subplot(gs[1], sharex=ax_gantt)
    style_axes(ax_fin, title="Financial Impact: Profit & Costs Over Time", ylabel="Cumulative ($)")
    
    t_arr = df['t'].values
    
    # Plot areas for costs (negative stack conceptually, but plotted as positive lines for scale)
    ax_fin.fill_between(t_arr, 0, df['revenue'], color=C['rev'], alpha=0.15)
    ax_fin.plot(t_arr, df['revenue'], color=C['rev'], linewidth=2.5, label=f"Revenue (${df['revenue'].iloc[-1]:,.0f})")
    
    ax_fin.plot(t_arr, df['production'], color=C['cost_pr'], linewidth=2, linestyle='--', label=f"Prod Cost (${df['production'].iloc[-1]:,.0f})")
    
    # Inventory and Backorder
    ax_fin.fill_between(t_arr, 0, df['inventory'], color=C['cost_inv'], alpha=0.3)
    ax_fin.plot(t_arr, df['inventory'], color=C['cost_inv'], linewidth=2, label=f"Inv Cost (${df['inventory'].iloc[-1]:,.0f})")
    
    # Note currently backorder is 0, but we'll show it if it exists
    if df['backorder'].max() > 0:
        ax_fin.fill_between(t_arr, 0, df['backorder'], color=C['cost_bo'], alpha=0.3)
        ax_fin.plot(t_arr, df['backorder'], color=C['cost_bo'], linewidth=2, label=f"Backorder (${df['backorder'].iloc[-1]:,.0f})")

    # Profit Line (Thick & Glowing)
    ax_fin.plot(t_arr, df['profit'], color=C['profit'], linewidth=3, zorder=10, label=f"NET PROFIT (${df['profit'].iloc[-1]:,.0f})")
    ax_fin.fill_between(t_arr, 0, df['profit'], where=df['profit']>0, color=C['profit'], alpha=0.1)
    ax_fin.fill_between(t_arr, 0, df['profit'], where=df['profit']<0, color=C['cost_bo'], alpha=0.15)
    
    # X Axis Setup
    ax_fin.set_xlabel("Time (Ticks)", color=C['text'], fontsize=12, fontweight='bold', labelpad=10)
    
    def format_x_ticks(x, pos):
        if x < 0 or x > H: return ""
        days = x / 96
        week = x / 672
        if x % 672 == 0:
            return f"Tick {int(x)}\n(Wk {int(week)})"
        return f"Tick {int(x)}\n(D {days:.1f})"

    ax_fin.xaxis.set_major_locator(ticker.MultipleLocator(336)) # Half week
    ax_fin.xaxis.set_major_formatter(ticker.FuncFormatter(format_x_ticks))
    
    ax_fin.axhline(0, color=C['text'], linewidth=1.5, alpha=0.7)

    # Sync week gridlines
    for w in range(0, H+1, 672):
        ax_fin.axvline(w, color=C['grid'], linestyle='-', linewidth=2, alpha=0.8, zorder=0)

    # Financial Legend
    ax_fin.legend(loc='upper left', ncol=2, facecolor=C['card'], edgecolor=C['grid'], 
                  labelcolor=C['text'], framealpha=0.9, fontsize=11)
    
    # Add summary box
    summary_text = (
        f"FINAL PERFORMANCE\n"
        f"------------------\n"
        f"Total Shipped: {int(shipped_counts.sum())} / {cfg['demand'][0]}\n"
        f"Total Revenue: ${df['revenue'].iloc[-1]:,.0f}\n"
        f"Net Profit:    ${df['profit'].iloc[-1]:,.0f}"
    )
    props = dict(boxstyle='round', facecolor=C['card'], edgecolor=C['grid'], alpha=0.9)
    ax_fin.text(0.98, 0.05, summary_text, transform=ax_fin.transAxes, fontsize=11, fontweight='bold',
                color=C['text'], verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=15)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "premium_gantt_analysis.png")
    fig.savefig(path, dpi=200, facecolor=C['bg'], bbox_inches='tight')
    plt.close()
    print(f"Saved Premium Visualization to: {path}")

if __name__ == "__main__":
    main()
