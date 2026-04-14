"""
Ultimate SimpleFab Dashboard - Version 3
Includes discrete color tracing in Gantt, zero Queue 0, decoupled Finished Goods, and SVG output for infinite zoom.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import matplotlib.cm as cm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simplefab_1p import make_common_config
from simplefab_1p.sim import ProductionLine

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output_data_1p')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color Palette ────────────────────────────────────────────────────────
C = {
    'bg':       '#0b0f19',
    'card':     '#131a28',
    'text':     '#f8fafc',
    'text_dim': '#94a3b8',
    'grid':     '#1e293b',
    
    'arrival':  '#2dd4bf', 
    'demand':   '#f43f5e', 
    'shipped':  '#10b981', 
    
    # Queues WIP
    'q1':       '#0ea5e9',
    'q2':       '#3b82f6',
    'q3':       '#8b5cf6',
    
    # Fin Goods
    'q_fin':    '#10b981', 
    'backlog':  '#f43f5e', 
    
    # Financials
    'rev':      '#10b981',
    'c_prod':   '#d946ef', 
    'c_inv':    '#fb923c', 
    'c_bo':     '#f43f5e', 
    'profit':   '#fbbf24', 
}

def style_ax(ax, title=None, ylabel=None):
    ax.set_facecolor(C['card'])
    for s in ax.spines.values():
        s.set_color(C['grid'])
        s.set_linewidth(1.5)
    ax.grid(True, color=C['grid'], alpha=0.8, linestyle='--')
    ax.tick_params(colors=C['text_dim'], labelsize=10, bottom=True, left=True)
    if title: ax.set_title(title, color=C['text'], fontsize=16, fontweight='bold', pad=15, loc='left')
    if ylabel: ax.set_ylabel(ylabel, color=C['text_dim'], fontsize=12, fontweight='bold', labelpad=10)

def draw_time_grid(ax, H, text_y, text_color, draw_text=True):
    """Draw vertical lines for weeks (solid) and days (dotted)"""
    for d in range(0, H+1, 96):
        if d % 672 == 0:
            ax.axvline(d, color=C['grid'], linestyle='-', linewidth=2.5, zorder=0)
            if draw_text:
                ax.text(d + 15, text_y, f"Wk {d//672 + 1}", color=text_color, alpha=0.7, fontsize=11, fontweight='bold')
        else:
            ax.axvline(d, color=C['grid'], linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)

def main():
    cfg = make_common_config()
    H = cfg["time_horizon"]

    line = ProductionLine(common_cfg=cfg)
    line.logging_enabled = True

    data = []
    
    for t in range(H):
        line.run_step(current_time=t, actions_override=None)
        cl = line.cost_log[-1]
        
        data.append({
            "t": t,
            "q0": len(line.queues["queue0"]),
            "q1": len(line.queues["queue1"]),
            "q2": len(line.queues["queue2"]),
            "q3": len(line.queues["queue3"]),
            "q_fin": len(line.queues["queue_fin"]),
            "backlog": len(line.queues["demand"]),
            "cum_rev": cl["revenue"],
            "cum_prod": cl["production"],
            "cum_inv": cl["inventory"],
            "cum_bo": cl["backorder"],
            "profit": cl["profit"],
        })
    df = pd.DataFrame(data)

    shipped_counts = np.zeros(H)
    for t_ship in line.demand_met_log:
        if t_ship < H: shipped_counts[t_ship] += 1

    # 5 ROW LAYOUT
    fig = plt.figure(figsize=(22, 28), facecolor=C['bg'])
    # Reduced hspace to make plots feel like cohesive subplots sharing the same x-axis
    gs = gridspec.GridSpec(5, 1, height_ratios=[0.5, 1.8, 1.2, 1.2, 1.8], hspace=0.3)
    
    x_arr = df['t'].values
    
    # -------------------------------------------------------------------------
    # ROW 1: KPI Dashboard
    # -------------------------------------------------------------------------
    ax_kpi = fig.add_subplot(gs[0])
    ax_kpi.axis('off')
    
    total_shipped = int(shipped_counts.sum())
    total_demand = cfg['demand'][0]
    final_rev = df['cum_rev'].iloc[-1]
    final_cost = df['cum_prod'].iloc[-1] + df['cum_inv'].iloc[-1] + df['cum_bo'].iloc[-1]
    final_profit = df['profit'].iloc[-1]
    
    kpis = [
        ("TOTAL DEMAND", f"{total_demand}", C['text']),
        ("UNITS SHIPPED", f"{total_shipped}", C['arrival']),
        ("REVENUE", f"${final_rev:,.0f}", C['rev']),
        ("TOTAL COSTS", f"${final_cost:,.0f}", C['c_bo']),
        ("NET PROFIT", f"${final_profit:,.0f}", C['profit']),
    ]
    
    box_width = 1.0 / len(kpis)
    for i, (title, val, color) in enumerate(kpis):
        x = i * box_width + (box_width / 2)
        ax_kpi.text(x, 0.6, title, fontsize=14, color=C['text_dim'], ha='center', va='center', fontweight='bold', transform=ax_kpi.transAxes)
        ax_kpi.text(x, 0.2, val, fontsize=28, color=color, ha='center', va='center', fontweight='heavy', transform=ax_kpi.transAxes)
        if i < len(kpis) - 1: ax_kpi.axvline(x=x + (box_width / 2), ymin=0.2, ymax=0.8, color=C['grid'], linewidth=2)
            
    # -------------------------------------------------------------------------
    # ROW 2: Gantt Chart (Color tracing)
    # -------------------------------------------------------------------------
    ax_gantt = fig.add_subplot(gs[1])
    style_ax(ax_gantt, title="Machine Scheduling & Item Tracing (Categorical colors track distinct individual batches)", ylabel="Machine")
    
    machines = ["M3 (Single)", "M2 (Batch)", "M1 (Single)", "M0 (Batch)"]
    y_pos = [30, 20, 10, 0]
    ax_gantt.set_yticks(y_pos)
    ax_gantt.set_yticklabels(machines, fontweight='bold', color=C['text'])
    ax_gantt.set_ylim(-15, 45)
    ax_gantt.set_xlim(-20, H + 20)
    
    machine_logs = [line.machine3.event_log, line.machine2.event_log, line.machine1.event_log, line.machine0.event_log]
    
    # Use categorical colors to make distinct units stand out
    cmap = cm.get_cmap('tab20')
    
    for m_log, y in zip(machine_logs, y_pos):
        for i, (st, fin, *rest) in enumerate(m_log):
            # Index-based color cycles every 20 units. Since it's FIFO, unit i on M0 is unit i on M1, M2, M3
            color = cmap(i % 20)
            ax_gantt.broken_barh([(st, fin - st)], (y-3, 6), facecolors=color, edgecolor='#000000', linewidth=0.5)

    draw_time_grid(ax_gantt, H, 40, C['text_dim'])

    # Overlays
    arrivals_dict = {t: qty for t, qty in enumerate(cfg["arrivals_schedule"][0]) if qty > 0}
    demand_dict = {t: qty for t, qty in enumerate(cfg["demand_schedule"][0]) if qty > 0}
    for t_arr, qty in arrivals_dict.items():
        ax_gantt.scatter(t_arr, -10, color=C['arrival'], marker='^', s=200, zorder=5, edgecolor=C['bg'])
        ax_gantt.text(t_arr + 20, -10, f"+{qty} Raw", color=C['arrival'], fontsize=10, fontweight='bold', va='center')
        ax_gantt.axvline(t_arr, color=C['arrival'], linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)
    for t_dem, qty in demand_dict.items():
        ax_gantt.scatter(t_dem, 40, color=C['demand'], marker='v', s=200, zorder=5, edgecolor=C['bg'])
        ax_gantt.text(t_dem - 20, 40, f"-{qty} Dem", color=C['demand'], fontsize=10, fontweight='bold', ha='right', va='center')
        ax_gantt.axvline(t_dem, color=C['demand'], linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)
    shipped_x = []; shipped_y = []
    for t_ship, qty in enumerate(shipped_counts):
        if qty > 0:
            shipped_x.extend([t_ship]*int(qty))
            shipped_y.extend(np.random.uniform(28.5, 31.5, int(qty)))
    if shipped_x: ax_gantt.scatter(shipped_x, shipped_y, color=C['shipped'], s=15, alpha=0.7, zorder=10, label='Goods Shipped')
    
    ax_gantt.set_xticklabels([])

    # -------------------------------------------------------------------------
    # ROW 3: Active Work-In-Process (WIP: Queues 1, 2, 3)
    # -------------------------------------------------------------------------
    gs_wip = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2], hspace=0.0)
    
    ax_q1 = fig.add_subplot(gs_wip[0], sharex=ax_gantt)
    ax_q2 = fig.add_subplot(gs_wip[1], sharex=ax_gantt)
    ax_q3 = fig.add_subplot(gs_wip[2], sharex=ax_gantt)

    style_ax(ax_q1, title="Active Work-In-Process (Raw Queue Levels)", ylabel="Wait M1")
    style_ax(ax_q2, ylabel="Wait M2")
    style_ax(ax_q3, ylabel="Wait M3")

    # Discrete step plots with translucent fill
    ax_q1.plot(x_arr, df['q1'], drawstyle='steps-post', color=C['q1'], linewidth=2, alpha=0.9)
    ax_q1.fill_between(x_arr, 0, df['q1'], step='post', color=C['q1'], alpha=0.3)
    
    ax_q2.plot(x_arr, df['q2'], drawstyle='steps-post', color=C['q2'], linewidth=2, alpha=0.9)
    ax_q2.fill_between(x_arr, 0, df['q2'], step='post', color=C['q2'], alpha=0.3)
    
    ax_q3.plot(x_arr, df['q3'], drawstyle='steps-post', color=C['q3'], linewidth=2, alpha=0.9)
    ax_q3.fill_between(x_arr, 0, df['q3'], step='post', color=C['q3'], alpha=0.3)

    for i, (ax, col) in enumerate(zip([ax_q1, ax_q2, ax_q3], ['q1', 'q2', 'q3'])):
        max_q = df[col].max()
        max_q = 5 if max_q == 0 else max_q
        ax.set_ylim(0, max_q * 1.2)
        ax.set_yticks([0, int(max_q/2), int(max_q)] if max_q >= 4 else list(range(int(max_q)+1)))
        
        # Only draw the "Wk 1" text on the top plot to avoid overlap
        draw_time_grid(ax, H, max_q * 1.05, C['text_dim'], draw_text=(i == 0))
        
        # Hide the physical x-axis ticks and labels on all, since they share x with ax_fininv anyway
        ax.set_xticklabels([])
        if i < 2:
            ax.tick_params(labelbottom=False, bottom=False)

    # -------------------------------------------------------------------------
    # ROW 4: Deliverables (Finished Goods vs Backlog vs Demand)
    # -------------------------------------------------------------------------
    ax_fininv = fig.add_subplot(gs[3], sharex=ax_gantt)
    style_ax(ax_fininv, title="Distribution: Finished Goods & Demand Backlog", ylabel="Units")
    
    ax_fininv.fill_between(x_arr, 0, df['q_fin'], color=C['q_fin'], alpha=0.2)
    ax_fininv.plot(x_arr, df['q_fin'], color=C['q_fin'], linewidth=3, label='Finished Goods on Shelf')
    
    if df['backlog'].max() > 0:
        ax_fininv.fill_between(x_arr, 0, df['backlog'], color=C['backlog'], alpha=0.2)
        ax_fininv.plot(x_arr, df['backlog'], color=C['backlog'], linewidth=3, label='Backordered Demand')

    max_fin = max(df['q_fin'].max(), df['backlog'].max(), 10)
    
    # Add clear demand indicators
    for t_dem, qty in demand_dict.items():
        ax_fininv.axvline(t_dem, color=C['demand'], linestyle='--', linewidth=2, alpha=0.8, zorder=0)
        ax_fininv.scatter(t_dem, max_fin * 1.05, color=C['demand'], marker='v', s=150, zorder=5, edgecolor=C['bg'])
        ax_fininv.text(t_dem - 20, max_fin * 1.05, f"-{qty} Demand Arr", color=C['demand'], fontsize=12, fontweight='bold', ha='right', va='center')

    ax_fininv.set_ylim(-5, max_fin * 1.3)
    draw_time_grid(ax_fininv, H, max_fin * 1.2, C['text_dim'])
    ax_fininv.legend(loc='upper left', facecolor=C['bg'], edgecolor=C['grid'], labelcolor=C['text'], ncol=2)
    ax_fininv.set_xticklabels([])

    # -------------------------------------------------------------------------
    # ROW 5: Financials
    # -------------------------------------------------------------------------
    ax_fin = fig.add_subplot(gs[4], sharex=ax_gantt)
    style_ax(ax_fin, title="Financial Breakdown: Revenues vs Costs", ylabel="Cumulative ($)")
    
    ax_fin.fill_between(x_arr, 0, df['cum_rev'], color=C['rev'], alpha=0.2)
    ax_fin.plot(x_arr, df['cum_rev'], color=C['rev'], linewidth=2, label='Cumulative Revenue')
    
    ax_fin.stackplot(x_arr, -df['cum_prod'], -df['cum_inv'], -df['cum_bo'],
                     labels=['Production Cost (Machine Operation)', 'Inventory Holding Cost ($0.02 / unit / tick)', 'Backorder Penalty ($0.10 / unit / tick)'],
                     colors=[C['c_prod'], C['c_inv'], C['c_bo']], alpha=0.7)
                     
    ax_fin.plot(x_arr, df['profit'], color=C['profit'], linewidth=3.5, label='NET PROFIT', zorder=10)
    ax_fin.axhline(0, color=C['text'], linewidth=1.5, alpha=0.8, zorder=5)

    ax_fin.set_xlabel("Time Horizon (Ticks / Days)", color=C['text'], fontsize=16, fontweight='bold', labelpad=15)
    def format_x(x, pos):
        if x < 0 or x > H: return ""
        return f"T{int(x)}\nD{x/96:.0f}"
    
    # Tick every day (96 ticks)
    ax_fin.xaxis.set_major_locator(ticker.MultipleLocator(96))
    ax_fin.xaxis.set_major_formatter(ticker.FuncFormatter(format_x))
    
    # Vertically stretch X tick labels to fit if necessary (or just leave them smaller)
    ax_fin.tick_params(axis='x', rotation=45, labelsize=10)
    
    ax_fin.legend(loc='upper right', facecolor=C['bg'], edgecolor=C['grid'], labelcolor=C['text'], ncol=2)
    y_max = max(df['cum_rev'].max(), df['profit'].max(), 1000) * 1.15
    y_min = min(df['profit'].min(), (-df['cum_prod'] - df['cum_inv'] - df['cum_bo']).min(), -1000) * 1.15
    ax_fin.set_ylim(y_min, y_max)
    draw_time_grid(ax_fin, H, y_max * 0.85, C['text_dim'])

    # Save as both PNG and SVG (SVG enables infinite zoom without quality loss in a browser)
    png_path = os.path.join(OUT_DIR, "ultimate_dashboard_v3.png")
    svg_path = os.path.join(OUT_DIR, "ultimate_dashboard_v3.svg")
    
    # --- RL HYPERPARAMETER SUMMARY FOOTER ---
    footer_text = (
        "RL Configuration Settings | "
        "ent_coef: 0.05 (Precise Exploitation) | "
        "gamma: 0.999 (Long Horizon 693-tick half-life) | "
        "Shaping: w_backlog=1.0, w_wip=0.1, w_finished=0.5 | "
        "Lead-Time Idle Nudge: -5.0 / tick"
    )
    plt.figtext(0.5, 0.015, footer_text, ha='center', va='bottom', fontsize=12, color='#A5D6FF', fontweight='bold', 
                bbox=dict(facecolor='#161B22', edgecolor='#30363D', boxstyle='round,pad=0.5', alpha=0.9))

    fig.savefig(png_path, dpi=250, facecolor=C['bg'], bbox_inches='tight')
    fig.savefig(svg_path, format='svg', facecolor=C['bg'], bbox_inches='tight')
    plt.close()
    
    print(f"Ultimate Dashboard generated.")
    print(f"PNG File: {png_path}")
    print(f"SVG (Infinite Zoom) File: {svg_path}")

if __name__ == "__main__":
    main()
