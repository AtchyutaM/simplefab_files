import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import argparse
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from simplefab import make_common_config
from simplefab.env import FabEnv
from simplefab.eval import ACTIONS

# Ensure output directory exists
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output_data', 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# Colors styling
C = {
    'bg': '#0d1117', 'card': '#161b22', 'text': '#c9d1d9',
    'agent': '#58a6ff', 'commander': '#f78166',
    'p0': '#58a6ff', 'p1': '#f78166', 'idle': '#21262d', 'setup': '#f85149',
    'grid': '#30363d', 'revenue': '#3fb950', 'cost': '#f85149', 'profit': '#d2a8ff'
}

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(C['card'])
    ax.set_title(title, color=C['text'], fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=C['text'], fontsize=10)
    ax.set_ylabel(ylabel, color=C['text'], fontsize=10)
    ax.tick_params(colors=C['text'], labelsize=9)
    for s in ax.spines.values(): s.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.4, linestyle='--')

def run_episode(env: FabEnv, model=None, deterministic=True) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Any]]:
    """Runs an episode and returns the final metrics, event log dataframe, and machine logs."""
    obs, info = env.reset()
    done = False
    
    # Needs to log events
    env.line.logging_enabled = True
    
    # We will manually collect actions taken
    action_log = []
    
    while not done:
        if model is not None:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        else:
            # Let the env use the heuristic commander
            action = env.line.commander.decide_actions(env.line._state_dict(env.line._t))
            a_idx = []
            for m in range(4):
                a_str = action.get(f'machine{m}', 'None')
                if a_str == 0: a_str = 'Prod0'
                elif a_str == 1: a_str = 'Prod1'
                elif a_str is None or str(a_str).lower() == 'none': a_str = 'None'
                a_idx.append(ACTIONS.index(a_str))
            action = np.array(a_idx)

        # Store intended action
        step_entry = {'t': env.line._t}
        for m in range(4):
            step_entry[f'm{m}_action'] = action[m]
        action_log.append(step_entry)

        # Step
        if model is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            # Manually step heuristic
            r = env.line.run_step(current_time=env.line._t, actions_override=None)
            done = env.line._t >= env.line.H
            
    c = env.line.costs
    throughput_p0 = sum(1 for p, _ in env.line.demand_met_log if p == 0)
    throughput_p1 = sum(1 for p, _ in env.line.demand_met_log if p == 1)

    metrics = {
        "profit": env.line.profit_total(),
        "revenue": c["revenue"],
        "production": c["production"],
        "setup": c["setup"],
        "inventory": c["inventory"],
        "backorder": c["backorder"],
        "throughput_p0": throughput_p0,
        "throughput_p1": throughput_p1,
        "throughput_total": throughput_p0 + throughput_p1,
    }
    
    event_log_df = pd.DataFrame(env.line.event_log)
    action_df = pd.DataFrame(action_log)
    
    # Merge if possible
    if not event_log_df.empty:
        event_log_df = pd.merge(event_log_df, action_df, left_on='Time', right_on='t', how='left')
        
    machine_logs = {
        'm0': env.line.machine0.event_log.copy(),
        'm1': env.line.machine1.event_log.copy(),
        'm2': env.line.machine2.event_log.copy(),
        'm3': env.line.machine3.event_log.copy(),
        'demand_met': env.line.demand_met_log.copy()
    }
    
    return metrics, event_log_df, machine_logs


def plot_cost_breakdown(metrics_agent, metrics_cmd, path):
    fig = plt.figure(figsize=(10, 6), facecolor=C['bg'])
    ax = fig.add_subplot(111)
    style_ax(ax, 'Economic Breakdown: Agent vs Commander', 'Category', 'Value ($)')
    
    categories = ['Revenue', 'Production', 'Setup', 'Inventory', 'Backorder', 'Profit']
    ag_vals = [
        metrics_agent['revenue'], -metrics_agent['production'], -metrics_agent['setup'],
        -metrics_agent['inventory'], -metrics_agent['backorder'], metrics_agent['profit']
    ]
    cmd_vals = [
        metrics_cmd['revenue'], -metrics_cmd['production'], -metrics_cmd['setup'],
        -metrics_cmd['inventory'], -metrics_cmd['backorder'], metrics_cmd['profit']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, ag_vals, width, label='RL Agent Policy', color=C['agent'])
    ax.bar(x + width/2, cmd_vals, width, label='Baseline Heuristic', color=C['commander'])
    
    ax.axhline(0, color=C['grid'], linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'])
    
    plt.tight_layout()
    fig.savefig(path, dpi=150, facecolor=C['bg'], bbox_inches='tight')
    plt.close()


def plot_action_distributions(df_agent, df_cmd, path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), facecolor=C['bg'])
    fig.suptitle('Action Distribution per Machine: Agent vs Commander', color=C['text'], fontsize=14, fontweight='bold')
    
    actions_labels = ['Idle', 'Prod0', 'Prod1']
    colors_ag = [C['idle'], C['p0'], C['p1']]
    colors_cmd = [C['grid'], '#4182cc', '#b85e4a'] # Muted versions
    
    for i in range(4):
        ax = axs[i//2, i%2]
        style_ax(ax, f'Machine {i}', '', 'Percentage (%)')
        
        counts_ag = df_agent[f'm{i}_action'].value_counts(normalize=True).reindex([0,1,2], fill_value=0) * 100
        counts_cmd = df_cmd[f'm{i}_action'].value_counts(normalize=True).reindex([0,1,2], fill_value=0) * 100
        
        x = np.arange(3)
        width = 0.35
        
        ax.bar(x - width/2, counts_ag, width, label='RL Agent Policy', color=colors_ag, edgecolor=C['agent'], linewidth=1)
        ax.bar(x + width/2, counts_cmd, width, label='Baseline Heuristic', color=colors_cmd, edgecolor=C['commander'], hatch='//', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(actions_labels)
        ax.set_ylim(0, 100)
        
        if i == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=C['agent'], label='RL Agent Policy'),
                Patch(facecolor=C['commander'], hatch='//', label='Baseline Heuristic')
            ]
            ax.legend(handles=legend_elements, facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'])
            
    plt.tight_layout()
    fig.savefig(path, dpi=150, facecolor=C['bg'])
    plt.close()


def plot_queue_dynamics(df_agent, df_cmd, path):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), facecolor=C['bg'])
    fig.suptitle('Queue Dynamics: Agent (Solid) vs Commander (Dashed)', color=C['text'], fontsize=14, fontweight='bold')
    
    queues = [
        ('Queue 0', 'Q0 (Raw Arrivals -> M0)'),
        ('Queue 1', 'Q1 (M0 -> M1)'),
        ('Queue 2', 'Q2 (M1 -> M2)'),
        ('Queue 3', 'Q3 (M2 -> M3)'),
        ('Queue Fin', 'Finished Inventory (M3 -> Shipping)'),
        ('Demand Queue', 'Backorders (Unmet Customer Demand)')
    ]
    
    for i, (q_col, title) in enumerate(queues):
        ax = axs[i//2, i%2]
        style_ax(ax, title, 'Simulated Tick (0-2688) / 4 Weeks', 'Units')
        
        for p in [0, 1]:
            col = f"{q_col} Product {p}"
            color = C['p0'] if p == 0 else C['p1']
            
            if col in df_agent.columns:
                ax.plot(df_agent['Time'], df_agent[col], color=color, label=f'Agent P{p}', linewidth=2)
            if col in df_cmd.columns:
                ax.plot(df_cmd['Time'], df_cmd[col], color=color, linestyle='--', label=f'Heuristic P{p}', alpha=0.6)
                
        if i == 0:
            ax.legend(facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'])
            
    plt.tight_layout()
    fig.savefig(path, dpi=150, facecolor=C['bg'])
    plt.close()


def plot_demand_fulfillment(df_agent, df_cmd, common_cfg, path):
    fig = plt.figure(figsize=(10, 6), facecolor=C['bg'])
    ax = fig.add_subplot(111)
    style_ax(ax, 'Cumulative Demand Fulfillment vs Arrivals (M1 vs M3 focus)', 'Day (0-28)', 'Cumulative Units')
    
    H = common_cfg['time_horizon']
    ticks_to_days = np.arange(H) / 96.0
    
    cum_arr_0 = np.cumsum(common_cfg['arrivals_schedule'][0])
    cum_arr_1 = np.cumsum(common_cfg['arrivals_schedule'][1])
    cum_dem_0 = np.cumsum(common_cfg['demand_schedule'][0])
    cum_dem_1 = np.cumsum(common_cfg['demand_schedule'][1])
    
    agent_shipped_p0 = df_agent['Demand Met Product 0'].cumsum()
    cmd_shipped_p0 = df_cmd['Demand Met Product 0'].cumsum()
    
    ax.plot(ticks_to_days, cum_arr_0 + cum_arr_1, color=C['revenue'], linestyle=':', label='Raw Material Inv (Total)')
    ax.step(ticks_to_days, cum_dem_0 + cum_dem_1, where='post', color='white', linewidth=2, label='Total Demand (P0+P1)')
    ax.plot(df_agent['Time'] / 96.0, agent_shipped_p0 + df_agent['Demand Met Product 1'].cumsum(), color=C['agent'], linewidth=2, label='Agent Shipped (Total)')
    ax.plot(df_cmd['Time'] / 96.0, cmd_shipped_p0 + df_cmd['Demand Met Product 1'].cumsum(), color=C['commander'], linestyle='--', linewidth=2, label='Heuristic Shipped (Total)')
    
    ax.legend(facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'])
    plt.tight_layout()
    fig.savefig(path, dpi=150, facecolor=C['bg'], bbox_inches='tight')
    plt.close()


def plot_gantt_chart(logs_ag, logs_cmd, H, path, xlim=None):
    zoom_suffix = " (Zoomed)" if xlim else " (Full Horizon)"
    fig, axs = plt.subplots(2, 1, figsize=(16 if xlim else 24, 8), facecolor=C['bg'])
    fig.suptitle(f'Machine Scheduling Gantt Chart{zoom_suffix}', color=C['text'], fontsize=16, fontweight='bold')
    
    machine_names = ['M3 (Single)', 'M2 (Batch)', 'M1 (Single)', 'M0 (Batch)'] # Bottom to top
    
    def plot_policy(ax, logs, title):
        style_ax(ax, title, 'Tick' if not xlim else 'Tick (Zoomed)', 'Machine')
        ax.set_yticks([10, 20, 30, 40])
        ax.set_yticklabels(machine_names)
        if xlim:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(0, H)
        
        y_pos = {'m0': 38, 'm1': 28, 'm2': 18, 'm3': 8}
        
        for m_key, log in logs.items():
            if m_key == 'demand_met':
                continue
            y = y_pos[m_key]
            
            bars_p0 = []
            bars_p1 = []
            bars_setup = []
            
            for item in log:
                prod = item[0]
                if len(item) == 4 and m_key in ['m1', 'm3']:
                    dec_t, st_t, fin_t = item[1], item[2], item[3]
                    if st_t > dec_t:
                        bars_setup.append((dec_t, st_t - dec_t))
                    bars_to_add = bars_p0 if prod == 0 else bars_p1
                    bars_to_add.append((st_t, fin_t - st_t))
                else:
                    st_t, fin_t = item[1], item[2]
                    bars_to_add = bars_p0 if prod == 0 else bars_p1
                    bars_to_add.append((st_t, fin_t - st_t))
                    
            if bars_setup:
                ax.broken_barh(bars_setup, (y, 4), facecolors=C['setup'])
            if bars_p0:
                ax.broken_barh(bars_p0, (y, 4), facecolors=C['p0'])
            if bars_p1:
                ax.broken_barh(bars_p1, (y, 4), facecolors=C['p1'])
                
        # Shipments as scatter
        ships = logs['demand_met']
        xs_0 = [t for p, t in ships if p == 0]
        xs_1 = [t for p, t in ships if p == 1]
        
        if xs_0:
            ax.scatter(xs_0, np.ones(len(xs_0))*5, color=C['p0'], marker='^', alpha=0.5, s=20)
        if xs_1:
            ax.scatter(xs_1, np.ones(len(xs_1))*5, color=C['p1'], marker='^', alpha=0.5, s=20)

    plot_policy(axs[0], logs_ag, 'RL Agent Policy')
    plot_policy(axs[1], logs_cmd, 'Baseline Heuristic (Commander)')
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=C['p0'], label='Processing P0'),
        Patch(facecolor=C['p1'], label='Processing P1'),
        Patch(facecolor=C['setup'], label='Setup / Changeover'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C['p0'], markersize=8, label='Shipped P0/P1', linestyle='None')
    ]
    fig.legend(handles=legend_elements, loc='upper right', facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'], bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=300 if not xlim else 150, facecolor=C['bg'], bbox_inches='tight')
    plt.close()


def generate_html_report(metrics_ag, metrics_cmd, model_src, out_dir):
    
    def diff_str(val_ag, val_cmd, is_cost=False):
        diff = val_ag - val_cmd
        if diff == 0:
            return "0.00"
        
        # for profit and revenue, pos is good. For costs, neg is good
        bad_class = "negative" if not is_cost else "positive"
        good_class = "positive" if not is_cost else "negative"
        
        if diff > 0:
            return f'<span class="{good_class}">+${diff:,.2f}</span>' if not is_cost and 'throughput' not in str(val_ag) else (f'<span class="{good_class}">+{diff:,.0f}</span>' if 'throughput' in str(val_ag) else f'<span class="{bad_class}">+${diff:,.2f}</span>')
        else:
            return f'<span class="{bad_class}">-${-diff:,.2f}</span>' if not is_cost and 'throughput' not in str(val_ag) else (f'<span class="{bad_class}">-{abs(diff):,.0f}</span>' if 'throughput' in str(val_ag) else f'<span class="{good_class}">-${-diff:,.2f}</span>')

    profit_diff = metrics_ag['profit'] - metrics_cmd['profit']
    
    if profit_diff > 0:
        performance_verdict = f"<p class='positive'><b>The RL Agent is mathematically outperforming the baseline heuristic by ${profit_diff:,.2f}.</b> It has successfully discovered a more optimal policy.</p>"
    else:
        performance_verdict = f"<p class='negative'><b>The RL Agent is currently underperforming the baseline heuristic by ${abs(profit_diff):,.2f}.</b> It is likely still exploring its state space or hasn't fully converged on the optimal batching strategy yet.</p>"

    # Build metrics rows
    rows_data = [
        ("Total Net Profit", True, f"${metrics_ag['profit']:,.2f}", f"${metrics_cmd['profit']:,.2f}", diff_str(metrics_ag['profit'], metrics_cmd['profit']), True),
        ("Revenue", False, f"${metrics_ag['revenue']:,.2f}", f"${metrics_cmd['revenue']:,.2f}", diff_str(metrics_ag['revenue'], metrics_cmd['revenue']), False),
        ("Total Throughput (P0+P1)", False, f"{metrics_ag['throughput_total']:.0f} units", f"{metrics_cmd['throughput_total']:.0f} units", f"{diff_str(metrics_ag['throughput_total'], metrics_cmd['throughput_total'])} units", False),
        ("Production Cost", False, f"${metrics_ag['production']:,.2f}", f"${metrics_cmd['production']:,.2f}", diff_str(metrics_cmd['production'], metrics_ag['production'], is_cost=True), False),
        ("Setup/Changeover Cost", True, f"${metrics_ag['setup']:,.2f}", f"${metrics_cmd['setup']:,.2f}", diff_str(metrics_cmd['setup'], metrics_ag['setup'], is_cost=True), False),
        ("Inventory (WIP) Cost", False, f"${metrics_ag['inventory']:,.2f}", f"${metrics_cmd['inventory']:,.2f}", diff_str(metrics_cmd['inventory'], metrics_ag['inventory'], is_cost=True), False),
        ("Backorder Penalties", False, f"${metrics_ag['backorder']:,.2f}", f"${metrics_cmd['backorder']:,.2f}", diff_str(metrics_cmd['backorder'], metrics_ag['backorder'], is_cost=True), False),
    ]
    
    metrics_rows_html = ""
    for label, bold, ag_val, cmd_val, diff_val, is_large in rows_data:
        name_cell = f"<b>{label}</b>" if bold else label
        ag_cls = ' class="metric-large"' if is_large else ''
        cmd_cls = ' class="metric-large"' if is_large else ''
        metrics_rows_html += f"<tr><td>{name_cell}</td><td{ag_cls}>{ag_val}</td><td{cmd_cls}>{cmd_val}</td><td>{diff_val}</td></tr>\n"

    # Load template and substitute
    template_path = os.path.join(os.path.dirname(__file__), 'report_template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    html = html.replace('{{MODEL_SRC}}', model_src)
    html = html.replace('{{PERFORMANCE_VERDICT}}', performance_verdict)
    html = html.replace('{{METRICS_ROWS}}', metrics_rows_html)


    path = os.path.join(out_dir, "Final_Presentation_Report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"Generated comprehensive report at: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent and Generate HTML presentation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific model checkpoint to load (path to .zip)")
    args = parser.parse_args()

    model_src = "None"
    
    if args.checkpoint:
        model_path = args.checkpoint
        model_src = os.path.basename(model_path)
    else:
        # Auto-find latest checkpoint or final model
        final_model = os.path.join(os.path.dirname(__file__), '..', 'output_data', 'models', 'ppo_fab_policy.zip')
        
        if os.path.exists(final_model):
            model_path = final_model
            model_src = "final model"
        else:
            search_path = os.path.join(os.path.dirname(__file__), '..', 'output_data', 'models', 'ppo_fab_checkpoint_*.zip')
            checkpoints = glob.glob(search_path)
            if checkpoints:
                # Sort numerically
                checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[-2]))
                model_path = checkpoints[-1]
                model_src = os.path.basename(model_path)
            else:
                # Fallback to older test model
                model_path = os.path.join(os.path.dirname(__file__), '..', 'ppo_fab_policy.zip')
                model_src = "old fallback model"

    print(f"Loading Agent: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return
        
    model = MaskablePPO.load(model_path)
    common_cfg = make_common_config()
    
    print("Evaluating Agent...")
    env_ag = FabEnv(common_cfg=common_cfg, normalize_obs=True, invalid_action_penalty=0.0)
    metrics_ag, df_ag, m_logs_ag = run_episode(env_ag, model=model)
    
    print("Evaluating Commander...")
    # Commander doesn't need normalized obs
    env_cmd = FabEnv(common_cfg=common_cfg, normalize_obs=False, invalid_action_penalty=0.0)
    metrics_cmd, df_cmd, m_logs_cmd = run_episode(env_cmd, model=None)
    
    print("\nGenerating visual reports...")
    plot_cost_breakdown(metrics_ag, metrics_cmd, os.path.join(OUT_DIR, "1_cost_breakdown.png"))
    plot_action_distributions(df_ag, df_cmd, os.path.join(OUT_DIR, "2_action_distributions.png"))
    plot_queue_dynamics(df_ag, df_cmd, os.path.join(OUT_DIR, "3_queue_dynamics.png"))
    
    if 'Demand Met Product 0' in df_ag.columns:
        plot_demand_fulfillment(df_ag, df_cmd, common_cfg, os.path.join(OUT_DIR, "4_demand_fulfillment.png"))
        
    plot_gantt_chart(m_logs_ag, m_logs_cmd, common_cfg["time_horizon"], os.path.join(OUT_DIR, "5_gantt_full.png"), xlim=None)
    plot_gantt_chart(m_logs_ag, m_logs_cmd, common_cfg["time_horizon"], os.path.join(OUT_DIR, "6_gantt_zoomed.png"), xlim=(672, 1344))
        
    generate_html_report(metrics_ag, metrics_cmd, model_src, OUT_DIR)
        
    print(f"Done! Open {os.path.join(OUT_DIR, 'Final_Presentation_Report.html')} in a browser.")

if __name__ == "__main__":
    main()
