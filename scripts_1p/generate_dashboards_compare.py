import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from sb3_contrib import MaskablePPO

# Setup pathing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simplefab_1p.config import make_common_config
from simplefab_1p.env import FabEnv
from simplefab_1p.sim import commander_decide

# Colors styling (Ultimate Dashboard V3)
C = {
    'bg': '#1e1e2e', 'card': '#181825', 'text': '#cdd6f4', 'text_dim': '#a6adc8',
    'grid': '#313244',
    'q_raw': '#89b4fa',  # Raw material / Queue 0
    'q1': '#04a5e5',     # Wait M1
    'q2': '#1e66f5',     # Wait M2
    'q3': '#7287fd',     # Wait M3
    'q_fin': '#40a02b',  # Finished goods array
    'backlog': '#e64553', # Demand backlog / Late units
    'rev': '#40a02b', 'c_prod': '#8839ef', 'c_inv': '#df8e1d', 'c_bo': '#e64553',
    'profit': '#f9e2af',
    'demand': '#e64553'
}

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output_data_1p"))
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Reused functions from visualize_dashboard
# -----------------------------------------------------------------------------
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
    for d in range(0, H+1, 96):
        if d % 672 == 0:
            ax.axvline(d, color=C['grid'], linestyle='-', linewidth=2.5, zorder=0)
            if draw_text:
                ax.text(d + 15, text_y, f"Wk {d//672 + 1}", color=text_color, alpha=0.7, fontsize=11, fontweight='bold')
        else:
            ax.axvline(d, color=C['grid'], linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)

# -----------------------------------------------------------------------------
# Generate Single Dashboard Method
# -----------------------------------------------------------------------------
def generate_dashboard_for_sim(env: FabEnv, H: int, output_path: str):
    line = env.line
    data = []
    
    # Generate Gantt Logs
    m_logs = {}
    for m_idx, m_obj in enumerate([line.machine0, line.machine1, line.machine2, line.machine3]):
        m_logs[m_idx] = []
        for (st, end, *rest) in m_obj.event_log:
            m_logs[m_idx].append({'start': st, 'duration': end - st})

    # Prepare DataFrame
    for log in line.cost_log:
        t = log['Time']
        state = line.event_log[t] if t < len(line.event_log) else line.event_log[-1]
        data.append({
            'tick': t,
            'q0': state.get('queue0', 0),
            'q1': state.get('queue1', 0),
            'q2': state.get('queue2', 0),
            'q3': state.get('queue3', 0),
            'q_fin': state.get('queue_fin', 0),
            'backlog': state.get('demand', 0),
            'cum_rev': log['revenue'],
            'cum_prod': log['production'],
            'cum_inv': log['inventory'],
            'cum_bo': log['backorder'],
            'profit': log['profit']
        })
    df = pd.DataFrame(data)
    x_arr = df['tick'].values

    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20')

    demand_dict = {}
    for t in range(len(line.demand_schedule)):
        if line.demand_schedule[t] > 0:
            demand_dict[t] = line.demand_schedule[t]
    
    raw_dict = {}
    for t in range(len(line.arrivals_schedule)):
        if line.arrivals_schedule[t] > 0:
            raw_dict[t] = line.arrivals_schedule[t]

    fig = plt.figure(figsize=(24, 30), facecolor=C['bg'])
    gs = gridspec.GridSpec(5, 1, height_ratios=[0.5, 3, 3, 2.5, 3], hspace=0.3)

    # ROW 1: HEADER
    ax_header = fig.add_subplot(gs[0])
    ax_header.set_facecolor(C['bg'])
    ax_header.axis('off')
    
    def add_kpi(x, y, label, val_text, color):
        ax_header.text(x, y+0.4, label, color=C['text_dim'], fontsize=16, fontweight='bold', ha='center', va='center')
        ax_header.text(x, y, val_text, color=color, fontsize=32, fontweight='bold', ha='center', va='center')
        if x < 0.8:
            ax_header.plot([x+0.12, x+0.12], [y, y+0.4], color=C['grid'], linewidth=2)

    total_dem = line.total_demand
    shipped = len(line.demand_met_log)
    profit = df['profit'].iloc[-1]
    rev = df['cum_rev'].iloc[-1]
    costs = df['cum_prod'].iloc[-1] + df['cum_inv'].iloc[-1] + df['cum_bo'].iloc[-1]
    
    add_kpi(0.1, 0.5, "TOTAL DEMAND", f"{total_dem:,}", C['text'])
    add_kpi(0.35, 0.5, "UNITS SHIPPED", f"{shipped:,}", C['q_fin'])
    add_kpi(0.6, 0.5, "REVENUE", f"${rev:,.0f}", C['rev'])
    add_kpi(0.85, 0.5, "NET PROFIT", f"${profit:,.0f}", C['profit'])
    add_kpi(0.72, 0.5, "TOTAL COSTS", f"${costs:,.0f}", C['c_bo'])

    # ROW 2: Gantt Chart
    ax_gantt = fig.add_subplot(gs[1])
    style_ax(ax_gantt, title="Machine Scheduling & Item Tracing (Categorical colors track distinct individual batches)", ylabel="Machine")
    
    y_pos = {0: 3, 1: 2, 2: 1, 3: 4}
    m_labels = {0: "M0 (Batch)", 1: "M1 (Single)", 2: "M2 (Batch)", 3: "M3 (Single)"}

    ax_gantt.set_yticks(list(y_pos.values()))
    ax_gantt.set_yticklabels([m_labels[k] for k in y_pos.keys()], fontsize=12, fontweight='bold', color=C['text'])
    ax_gantt.set_xlim(0, H)

    for m in range(4):
        h = 0.6
        y = y_pos[m]
        for i, op in enumerate(m_logs[m]):
            c = cmap(i % 20)
            ax_gantt.add_patch(plt.Rectangle((op['start'], y - h/2), op['duration'], h, 
                                            facecolor=c, edgecolor='black', linewidth=0.5, alpha=0.9))

    for t_arr, qty in raw_dict.items():
        ax_gantt.axvline(t_arr, color=C['q1'], linestyle=':', linewidth=1.5, alpha=0.7, zorder=0)
        ax_gantt.scatter(t_arr, 0.2, color=C['q1'], marker='^', s=100)
        ax_gantt.text(t_arr + 15, 0.2, f"+{qty} Raw", color=C['q1'], fontsize=11, fontweight='bold', va='center')

    for t_dem, qty in demand_dict.items():
        ax_gantt.axvline(t_dem, color=C['demand'], linestyle=':', linewidth=1.5, alpha=0.7, zorder=0)
        ax_gantt.scatter(t_dem, 4.8, color=C['demand'], marker='v', s=100)
        ax_gantt.text(t_dem - 15, 4.8, f"-{qty} Dem", color=C['demand'], fontsize=11, fontweight='bold', ha='right', va='center')

    draw_time_grid(ax_gantt, H, 4.5, C['text_dim'])
    ax_gantt.set_ylim(0, 5)
    ax_gantt.set_xticklabels([])

    # ROW 3: Active WIP
    gs_wip = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2], hspace=0.0)
    ax_q1 = fig.add_subplot(gs_wip[0], sharex=ax_gantt)
    ax_q2 = fig.add_subplot(gs_wip[1], sharex=ax_gantt)
    ax_q3 = fig.add_subplot(gs_wip[2], sharex=ax_gantt)

    style_ax(ax_q1, title="Active Work-In-Process (Raw Queue Levels)", ylabel="Wait M1")
    style_ax(ax_q2, ylabel="Wait M2")
    style_ax(ax_q3, ylabel="Wait M3")

    ax_q1.plot(x_arr, df['q1'], drawstyle='steps-post', color=C['q1'], linewidth=2, alpha=0.9)
    ax_q1.fill_between(x_arr, 0, df['q1'], step='post', color=C['q1'], alpha=0.3)
    ax_q2.plot(x_arr, df['q2'], drawstyle='steps-post', color=C['q2'], linewidth=2, alpha=0.9)
    ax_q2.fill_between(x_arr, 0, df['q2'], step='post', color=C['q2'], alpha=0.3)
    ax_q3.plot(x_arr, df['q3'], drawstyle='steps-post', color=C['q3'], linewidth=2, alpha=0.9)
    ax_q3.fill_between(x_arr, 0, df['q3'], step='post', color=C['q3'], alpha=0.3)

    for i, (ax, col) in enumerate(zip([ax_q1, ax_q2, ax_q3], ['q1', 'q2', 'q3'])):
        max_q = max(df['q1'].max(), df['q2'].max(), df['q3'].max()) # Shared scale for fairness
        max_q = 5 if max_q == 0 else max_q
        ax.set_ylim(0, max_q * 1.2)
        ax.set_yticks([0, int(max_q/2), int(max_q)] if max_q >= 4 else list(range(int(max_q)+1)))
        draw_time_grid(ax, H, max_q * 1.05, C['text_dim'], draw_text=(i == 0))
        ax.set_xticklabels([])
        if i < 2: ax.tick_params(labelbottom=False, bottom=False)

    # ROW 4: Deliverables
    ax_fininv = fig.add_subplot(gs[3], sharex=ax_gantt)
    style_ax(ax_fininv, title="Distribution: Finished Goods & Demand Backlog", ylabel="Units")
    ax_fininv.fill_between(x_arr, 0, df['q_fin'], color=C['q_fin'], alpha=0.2)
    ax_fininv.plot(x_arr, df['q_fin'], color=C['q_fin'], linewidth=3, label='Finished Goods on Shelf')
    if df['backlog'].max() > 0:
        ax_fininv.fill_between(x_arr, 0, df['backlog'], color=C['backlog'], alpha=0.2)
        ax_fininv.plot(x_arr, df['backlog'], color=C['backlog'], linewidth=3, label='Backordered Demand')
    max_fin = max(df['q_fin'].max(), df['backlog'].max(), 10)
    for t_dem, qty in demand_dict.items():
        ax_fininv.axvline(t_dem, color=C['demand'], linestyle='--', linewidth=2, alpha=0.8, zorder=0)
        ax_fininv.scatter(t_dem, max_fin * 1.05, color=C['demand'], marker='v', s=150, zorder=5, edgecolor=C['bg'])
        ax_fininv.text(t_dem - 20, max_fin * 1.05, f"-{qty} Demand Arr", color=C['demand'], fontsize=12, fontweight='bold', ha='right', va='center')
    ax_fininv.set_ylim(-5, max_fin * 1.3)
    draw_time_grid(ax_fininv, H, max_fin * 1.2, C['text_dim'])
    ax_fininv.legend(loc='upper left', facecolor=C['bg'], edgecolor=C['grid'], labelcolor=C['text'], ncol=2)
    ax_fininv.set_xticklabels([])

    # ROW 5: Financials
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
    ax_fin.xaxis.set_major_locator(ticker.MultipleLocator(96))
    ax_fin.xaxis.set_major_formatter(ticker.FuncFormatter(format_x))
    ax_fin.tick_params(axis='x', rotation=45, labelsize=10)
    ax_fin.legend(loc='upper right', facecolor=C['bg'], edgecolor=C['grid'], labelcolor=C['text'], ncol=2)
    
    # Static Y scale to make comparative analysis fair (Hardcoded based on SimpleFab typical limits)
    ax_fin.set_ylim(-35000, 50000)
    draw_time_grid(ax_fin, H, 40000, C['text_dim'])

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

    base_path = os.path.splitext(output_path)[0]
    fig.savefig(f"{base_path}.png", dpi=250, facecolor=C['bg'], bbox_inches='tight')
    fig.savefig(f"{base_path}.svg", format='svg', facecolor=C['bg'], bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Main Execution Strategy
# -----------------------------------------------------------------------------
def run_and_plot(model=None, output_filename="dashboard.png"):
    cfg = make_common_config()
    H = cfg["time_horizon"]
    env = FabEnv(common_cfg=cfg, normalize_obs=True, invalid_action_penalty=0.0)

    obs, _ = env.reset()
    env.line.logging_enabled = True
    done = False
    
    while not done:
        if model is None:
            # Baseline Heuristic
            actions_dict = commander_decide(
                env.line.machine0, env.line.machine1,
                env.line.machine2, env.line.machine3,
                env.line.queues)
            action = np.array([int(actions_dict[f"machine{m}"]) for m in range(4)])
        else:
            # RL Policy
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)

        if model is None:
            env.line.run_step(current_time=env.line._t, actions_override=None)
            done = env.line._t >= env.line.H
        else:
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

    generate_dashboard_for_sim(env, H, os.path.join(OUT_DIR, output_filename))
    
    print(f"[{'Agent' if model else 'Heuristic'}]: Profit: {env.line.profit_total():.0f} | InvCost: {env.line.costs['inventory']:.0f}")

def main():
    print("Generating Commander Baseline Dashboard...")
    run_and_plot(model=None, output_filename="commander_dashboard.png")
    
    model_path = os.path.join(OUT_DIR, "models", "ppo_fab_1p_policy.zip")
    if os.path.exists(model_path):
        print("Loading trained RL Agent...")
        model = MaskablePPO.load(model_path)
        print("Generating RL Agent Dashboard...")
        run_and_plot(model=model, output_filename="agent_dashboard.png")
    else:
        print("Waiting for training to complete to generate Agent Dashboard...")

if __name__ == "__main__":
    main()
