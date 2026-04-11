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
from simplefab_1p import make_common_config
from simplefab_1p.env import FabEnv
from simplefab_1p.eval import ACTIONS

# Ensure output directory exists
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output_data_1p', 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# Colors styling
C = {
    'bg': '#0d1117', 'card': '#161b22', 'text': '#c9d1d9',
    'agent': '#58a6ff', 'commander': '#f78166',
    'prod': '#58a6ff', 'idle': '#21262d',
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
    
    env.line.logging_enabled = True
    action_log = []
    
    while not done:
        if model is not None:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        else:
            # Let the env use the heuristic commander
            from simplefab_1p.sim import commander_decide
            actions_dict = commander_decide(
                env.line.machine0, env.line.machine1,
                env.line.machine2, env.line.machine3,
                env.line.queues,
            )
            action = np.array([int(actions_dict[f"machine{m}"]) for m in range(4)])

        step_entry = {'t': env.line._t}
        for m in range(4):
            step_entry[f'm{m}_action'] = action[m]
        action_log.append(step_entry)

        if model is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            r = env.line.run_step(current_time=env.line._t, actions_override=None)
            done = env.line._t >= env.line.H
            
    c = env.line.costs
    throughput = len(env.line.demand_met_log)

    metrics = {
        "profit": env.line.profit_total(),
        "revenue": c["revenue"],
        "production": c["production"],
        "inventory": c["inventory"],
        "backorder": c["backorder"],
        "throughput": throughput,
    }
    
    event_log_df = pd.DataFrame(env.line.event_log)
    action_df = pd.DataFrame(action_log)
    
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
    
    categories = ['Revenue', 'Production', 'Inventory', 'Backorder', 'Profit']
    ag_vals = [
        metrics_agent['revenue'], -metrics_agent['production'],
        -metrics_agent['inventory'], -metrics_agent['backorder'], metrics_agent['profit']
    ]
    cmd_vals = [
        metrics_cmd['revenue'], -metrics_cmd['production'],
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
    
    actions_labels = ['Idle', 'Run']
    colors_ag = [C['idle'], C['prod']]
    colors_cmd = [C['grid'], '#4182cc']
    
    for i in range(4):
        ax = axs[i//2, i%2]
        style_ax(ax, f'Machine {i}', '', 'Percentage (%)')
        
        counts_ag = df_agent[f'm{i}_action'].value_counts(normalize=True).reindex([0,1], fill_value=0) * 100
        counts_cmd = df_cmd[f'm{i}_action'].value_counts(normalize=True).reindex([0,1], fill_value=0) * 100
        
        x = np.arange(2)
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
        
        color = C['prod']
        
        if q_col in df_agent.columns:
            ax.plot(df_agent['Time'], df_agent[q_col], color=color, label='Agent', linewidth=2)
        if q_col in df_cmd.columns:
            ax.plot(df_cmd['Time'], df_cmd[q_col], color=color, linestyle='--', label='Heuristic', alpha=0.6)
                
        if i == 0:
            ax.legend(facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'])
            
    plt.tight_layout()
    fig.savefig(path, dpi=150, facecolor=C['bg'])
    plt.close()


def plot_gantt_chart(logs_ag, logs_cmd, H, path, xlim=None):
    zoom_suffix = " (Zoomed)" if xlim else " (Full Horizon)"
    fig, axs = plt.subplots(2, 1, figsize=(16 if xlim else 24, 8), facecolor=C['bg'])
    fig.suptitle(f'Machine Scheduling Gantt Chart{zoom_suffix}', color=C['text'], fontsize=16, fontweight='bold')
    
    machine_names = ['M3 (Single)', 'M2 (Batch)', 'M1 (Single)', 'M0 (Batch)']
    
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
            
            bars = []
            for item in log:
                st_t, fin_t = item[0], item[1]
                bars.append((st_t, fin_t - st_t))
                    
            if bars:
                ax.broken_barh(bars, (y, 4), facecolors=C['prod'])
                
        # Shipments as scatter
        ships = logs['demand_met']
        if ships:
            ax.scatter(ships, np.ones(len(ships))*5, color=C['prod'], marker='^', alpha=0.5, s=20)

    plot_policy(axs[0], logs_ag, 'RL Agent Policy')
    plot_policy(axs[1], logs_cmd, 'Baseline Heuristic (Commander)')
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=C['prod'], label='Processing'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C['prod'], markersize=8, label='Shipped', linestyle='None')
    ]
    fig.legend(handles=legend_elements, loc='upper right', facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'], bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=300 if not xlim else 150, facecolor=C['bg'], bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate 1-Product Agent and Generate Reports")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific model checkpoint to load (path to .zip)")
    args = parser.parse_args()

    model_src = "None"
    
    if args.checkpoint:
        model_path = args.checkpoint
        model_src = os.path.basename(model_path)
    else:
        final_model = os.path.join(os.path.dirname(__file__), '..', 'output_data_1p', 'models', 'ppo_fab_1p_policy.zip')
        
        if os.path.exists(final_model):
            model_path = final_model
            model_src = "final model (1p)"
        else:
            search_path = os.path.join(os.path.dirname(__file__), '..', 'output_data_1p', 'models', 'ppo_fab_1p_checkpoint_*.zip')
            checkpoints = glob.glob(search_path)
            if checkpoints:
                checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[-2]))
                model_path = checkpoints[-1]
                model_src = os.path.basename(model_path)
            else:
                print("Error: No trained model found. Run train_ppo.py first.")
                return

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
    env_cmd = FabEnv(common_cfg=common_cfg, normalize_obs=False, invalid_action_penalty=0.0)
    metrics_cmd, df_cmd, m_logs_cmd = run_episode(env_cmd, model=None)
    
    print("\nGenerating visual reports...")
    plot_cost_breakdown(metrics_ag, metrics_cmd, os.path.join(OUT_DIR, "1_cost_breakdown.png"))
    plot_action_distributions(df_ag, df_cmd, os.path.join(OUT_DIR, "2_action_distributions.png"))
    plot_queue_dynamics(df_ag, df_cmd, os.path.join(OUT_DIR, "3_queue_dynamics.png"))
        
    plot_gantt_chart(m_logs_ag, m_logs_cmd, common_cfg["time_horizon"], os.path.join(OUT_DIR, "4_gantt_full.png"), xlim=None)
    plot_gantt_chart(m_logs_ag, m_logs_cmd, common_cfg["time_horizon"], os.path.join(OUT_DIR, "5_gantt_zoomed.png"), xlim=(672, 1344))
        
    print(f"\n=== Results ===")
    print(f"  Agent   | Profit: ${metrics_ag['profit']:,.2f} | Throughput: {metrics_ag['throughput']}")
    print(f"  Command | Profit: ${metrics_cmd['profit']:,.2f} | Throughput: {metrics_cmd['throughput']}")
    print(f"\nSaved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
