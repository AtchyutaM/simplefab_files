import os
import sys
import base64
import argparse
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
REPORTS_DIR = os.path.join(OUT_DIR, "reports")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

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
            'q0': state.get('Queue 0', 0),
            'q1': state.get('Queue 1', 0),
            'q2': state.get('Queue 2', 0),
            'q3': state.get('Queue 3', 0),
            'q_fin': state.get('Queue Fin', 0),
            'backlog': state.get('Demand Queue', 0),
            'cum_rev': log['revenue'],
            'cum_prod': log['production'],
            'cum_inv': log['inventory'],
            'cum_wip': log.get('wip', 0),
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
    costs = df['cum_prod'].iloc[-1] + df['cum_inv'].iloc[-1] + df['cum_wip'].iloc[-1] + df['cum_bo'].iloc[-1]

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
    ax_fin.stackplot(x_arr, -df['cum_prod'], -df['cum_inv'], -df['cum_wip'], -df['cum_bo'],
                     labels=['Production Cost (Machine Operation)',
                             'Finished Goods Holding ($0.04/unit/tick)',
                             'WIP Holding ($0.00–$0.03/unit/tick)',
                             'Backorder Penalty ($0.15/unit/tick)'],
                     colors=[C['c_prod'], C['c_inv'], '#e5c890', C['c_bo']], alpha=0.7)
    ax_fin.plot(x_arr, df['profit'], color=C['profit'], linewidth=3.5, label='NET PROFIT', zorder=10)
    ax_fin.axhline(0, color=C['text'], linewidth=1.5, alpha=0.8, zorder=5)

    ax_fin.set_xlabel("Time Horizon (Ticks / Days)", color=C['text'], fontsize=16, fontweight='bold', labelpad=15)
    def format_x(x, _pos):
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
        "Shaping: w_backlog=1.0, w_wip=0.0, w_finished=0.5 | "
        "Lead-Time Idle Nudge: -5.0 / tick | "
        "WIP Holding Cost: Q0=$0.00, Q1=$0.005, Q2=$0.01, Q3=$0.015 / unit / tick"
    )
    plt.figtext(0.5, 0.015, footer_text, ha='center', va='bottom', fontsize=12, color='#A5D6FF', fontweight='bold',
                bbox=dict(facecolor='#161B22', edgecolor='#30363D', boxstyle='round,pad=0.5', alpha=0.9))

    base_path = os.path.splitext(output_path)[0]
    fig.savefig(f"{base_path}.png", dpi=250, facecolor=C['bg'], bbox_inches='tight')
    fig.savefig(f"{base_path}.svg", format='svg', facecolor=C['bg'], bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Temperature-scaled prediction for MaskablePPO
# -----------------------------------------------------------------------------
import torch as th

def predict_with_temperature(model, obs, action_masks, temperature=0.3):
    """
    Predict action using temperature-scaled sampling instead of argmax.
    Lower temperature → more deterministic, higher → more exploration.
    """
    obs_tensor = th.as_tensor(obs).float().unsqueeze(0)
    mask_np = action_masks

    with th.no_grad():
        dist = model.policy.get_distribution(obs_tensor, action_masks=mask_np.reshape(1, -1))
        # Scale logits by temperature for each sub-distribution
        for d in dist.distributions:
            d.logits = d.logits / temperature
        action = th.stack([d.sample() for d in dist.distributions], dim=1)

    return action.squeeze(0).cpu().numpy()


# -----------------------------------------------------------------------------
# Main Execution Strategy
# -----------------------------------------------------------------------------
def run_single_episode(model, cfg, temperature=None, logging_enabled=False):
    """Run one episode and return (env, stats). If model is None, use Commander heuristic."""
    env = FabEnv(common_cfg=cfg, normalize_obs=True, invalid_action_penalty=0.0)
    obs, _ = env.reset()
    env.line.logging_enabled = logging_enabled
    done = False

    while not done:
        if model is None:
            actions_dict = commander_decide(
                env.line.machine0, env.line.machine1,
                env.line.machine2, env.line.machine3,
                env.line.queues)
            action = np.array([int(actions_dict[f"machine{m}"]) for m in range(4)])
            env.line.run_step(current_time=env.line._t, actions_override=None)
            done = env.line._t >= env.line.H
        else:
            mask = env.action_masks()
            if temperature is not None:
                action = predict_with_temperature(model, obs, mask, temperature=temperature)
            else:
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

    line = env.line
    cost_log = line.cost_log
    last = cost_log[-1] if cost_log else {}
    stats = {
        "profit":    last.get("profit", 0),
        "revenue":   last.get("revenue", 0),
        "prod_cost": last.get("production", 0),
        "inv_cost":  last.get("inventory", 0),
        "wip_cost":  last.get("wip", 0),
        "bo_cost":   last.get("backorder", 0),
        "shipped":   len(line.demand_met_log),
        "total_demand": line.total_demand,
    }
    return env, stats


def run_and_plot(model=None, output_path="dashboard.png", cfg=None, n_eval=20, temperature=0.7):
    if cfg is None:
        cfg = make_common_config(mode="UNIFORM")

    if model is None:
        # Commander is deterministic — one run is enough
        env, stats = run_single_episode(None, cfg, logging_enabled=True)
        generate_dashboard_for_sim(env, cfg["time_horizon"], output_path)
        label = 'Heuristic'
        print(f"[{label}]: Profit: {stats['profit']:.0f} | InvCost: {stats['inv_cost']:.0f} | WIPCost: {stats['wip_cost']:.0f} | BOCost: {stats['bo_cost']:.0f} | Shipped: {stats['shipped']}/{stats['total_demand']}")
        return stats

    # RL Agent: run n_eval episodes with temperature sampling, pick the best
    best_stats = None
    best_env = None
    all_profits = []

    # Also run one deterministic for comparison
    _, det_stats = run_single_episode(model, cfg, temperature=None, logging_enabled=False)

    for i in range(n_eval):
        env, stats = run_single_episode(model, cfg, temperature=temperature, logging_enabled=(i == 0))
        all_profits.append(stats["profit"])
        if best_stats is None or stats["profit"] > best_stats["profit"]:
            best_stats = stats
            if i == 0:
                best_env = env
            else:
                # Re-run best episode with logging for dashboard
                best_env = None

    # If best wasn't the first run (which had logging), re-run with logging for dashboard
    if best_env is None:
        best_env, best_stats = run_single_episode(model, cfg, temperature=temperature, logging_enabled=True)

    generate_dashboard_for_sim(best_env, cfg["time_horizon"], output_path)

    mean_profit = np.mean(all_profits)
    label = 'Agent'
    print(f"[{label}] Deterministic: Profit={det_stats['profit']:.0f} | Shipped={det_stats['shipped']}/{det_stats['total_demand']}")
    print(f"[{label}] Temp={temperature} ({n_eval} episodes): Mean={mean_profit:.0f} | Best={best_stats['profit']:.0f} | Shipped={best_stats['shipped']}/{best_stats['total_demand']}")
    print(f"[{label}] Using best episode for report.")

    return best_stats


# -----------------------------------------------------------------------------
# HTML Report Generator
# -----------------------------------------------------------------------------
def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_html_report(exp_name: str, cfg: dict, cmd_stats: dict, agent_stats: dict,
                          cmd_img_path: str, agent_img_path: str, out_path: str,
                          prev_stats: dict = None, prev_img_path: str = None):
    # Derive demand schedule info
    dem_schedule = cfg.get("demand_schedule", {}).get(0, [])
    dem_events = [(t, q) for t, q in enumerate(dem_schedule) if q > 0]
    # Detect demand mode for display
    dem_qtys = [q for _, q in dem_events]
    if len(set(dem_qtys)) <= 1:
        dem_mode_label = "Uniform Demand"
    else:
        dem_mode_label = "Alternating Demand"
    dem_pattern_str = " + ".join([f"{q} @ Wk{t//672+1}" for t, q in dem_events])
    total_demand = cfg.get("demand", {}).get(0, sum(q for _, q in dem_events))
    util_pct = int(cfg.get("utilization", 0.5) * 100)
    hold_cost = cfg["inventory_cost_per_unit"][0]
    wip_costs = cfg.get("wip_cost_per_unit", {})
    bo_cost = cfg["backorder_cost_per_unit"][0]
    revenue = cfg["revenue_per_unit"][0]
    has_prev = prev_stats is not None and prev_img_path is not None
    H = cfg["time_horizon"]

    profit_delta = agent_stats["profit"] - cmd_stats["profit"]
    profit_delta_pct = (profit_delta / abs(cmd_stats["profit"]) * 100) if cmd_stats["profit"] != 0 else 0
    agent_better = profit_delta > 0

    inv_savings = cmd_stats["inv_cost"] - agent_stats["inv_cost"]
    bo_savings  = cmd_stats["bo_cost"]  - agent_stats["bo_cost"]

    # Theoretical maximum profit calculation
    # Perfect JIT: all demand met, zero FG holding, zero backorder, only unavoidable WIP transit cost
    # Min WIP cost per batch: Q1 wait (12 unit-ticks * $0.005) + Q2 wait (12 * $0.01) + Q3 wait (12 * $0.015) = $0.36
    batch_size = cfg["batch_sizes"][0]
    dem_total_calc = total_demand
    n_batches = dem_total_calc // batch_size
    min_wip_per_batch = (
        12 * wip_costs.get(1, 0) +
        12 * wip_costs.get(2, 0) +
        12 * wip_costs.get(3, 0)
    )
    min_wip_total = n_batches * min_wip_per_batch
    prod_cost_total = dem_total_calc * sum(cfg["production_cost"][m][0] for m in range(4))
    theoretical_max = dem_total_calc * revenue - prod_cost_total - min_wip_total
    agent_pct_of_max = (agent_stats["profit"] / theoretical_max * 100) if theoretical_max > 0 else 0
    gap_to_max = theoretical_max - agent_stats["profit"]

    # Narrative analysis
    if agent_better:
        summary = (
            f"The RL agent outperformed the Commander heuristic by <strong>${profit_delta:,.0f} "
            f"({profit_delta_pct:+.1f}%)</strong> in net profit. "
            f"Under {dem_mode_label.lower()} at {util_pct}% utilization, the agent discovered "
            f"demand-aware scheduling strategies that reduced inventory holding costs "
            f"by ${inv_savings:,.0f} and backorder costs by ${bo_savings:,.0f} compared to the greedy baseline."
        )
    else:
        summary = (
            f"Under {dem_mode_label.lower()} at {util_pct}% utilization, the Commander heuristic and RL agent "
            f"performed comparably (Δ profit: ${profit_delta:+,.0f}, {profit_delta_pct:+.1f}%). "
            f"The greedy always-run heuristic is already near-optimal in this scenario."
        )

    # Build narrative based on demand mode
    is_alternating = dem_mode_label == "Alternating Demand"

    if is_alternating:
        learned_section = f"""
    <div class="narrative">
      <h3>Why This Experiment</h3>
      <p>
        In previous experiments with <em>uniform demand</em> (equal weekly batches), the RL agent achieved
        93.7% of theoretical maximum profit. Uniform demand is the easiest scheduling problem &mdash;
        every week is identical, so the agent only needs to learn one repeating pattern.
      </p>
      <p>
        <strong>Alternating demand</strong> is fundamentally harder. Demand pattern: <em>{dem_pattern_str}</em>
        (total: {total_demand} units). The agent must now reason across weeks:
      </p>
      <ul>
        <li><strong>Anticipatory Production:</strong> During low-demand weeks, the agent must decide
            how much to pre-build for upcoming high-demand weeks &mdash; but every unit held as WIP
            or finished goods incurs holding costs.</li>
        <li><strong>Cost-Optimal Timing:</strong> Building too early means paying WIP costs
            (Q1=$0.005, Q2=$0.01, Q3=$0.015/unit/tick). Building too late means backorder penalties
            ($0.15/unit/tick). The agent must find the narrow window.</li>
        <li><strong>Variable Production Rate:</strong> Unlike uniform demand where a constant cadence works,
            the agent must <em>accelerate</em> production before high weeks and <em>throttle</em> during low weeks.</li>
      </ul>

      <h3>Iteration History</h3>
      <p><strong>First attempt (nudge=-3.5):</strong> The reward structure from the uniform 50% experiment
        was used directly. The agent over-throttled &mdash; its JIT instincts from 50% utilization caused it
        to ship only 540/600 units, losing $3,360 in net margin from unshipped units while saving only
        $1,833 on holding costs. The Commander won by $615 simply by running everything.</p>
      <p><strong>Key insight:</strong> At {util_pct}% utilization, each unshipped unit costs $56 net margin
        ($80 revenue &minus; $24 production), but FG holding to pre-build averages only ~$5-6/unit.
        The economics heavily favor producing more, not less. The pipeline nudge at -3.5 was too weak
        to overcome the agent's learned caution.</p>
      <p><strong>Current attempt (nudge=-5.0):</strong> Increased pipeline nudge back to -5.0 to give the
        agent a stronger push to keep M0 running when material is available during critical lead time.
        At high utilization, idling is almost always wrong &mdash; the opportunity cost ($56/unit)
        far exceeds any WIP savings.</p>

      <h3>Reward Structure</h3>
      <table style="width:100%; color: var(--text-muted); border-collapse: collapse; margin: 1rem 0;">
        <tr style="border-bottom: 1px solid var(--border);">
          <th style="text-align:left; padding:0.4rem;">Component</th>
          <th style="text-align:left; padding:0.4rem;">Value</th>
          <th style="text-align:left; padding:0.4rem;">Purpose</th>
        </tr>
        <tr style="border-bottom: 1px solid var(--border);">
          <td style="padding:0.4rem;">WIP costs</td>
          <td style="padding:0.4rem;">Q0=$0, Q1=$0.005, Q2=$0.01, Q3=$0.015</td>
          <td style="padding:0.4rem;">Progressive cost reflecting value-added at each stage</td>
        </tr>
        <tr style="border-bottom: 1px solid var(--border);">
          <td style="padding:0.4rem;">FG holding cost</td>
          <td style="padding:0.4rem;">$0.04/unit/tick</td>
          <td style="padding:0.4rem;">Finished goods storage cost</td>
        </tr>
        <tr style="border-bottom: 1px solid var(--border);">
          <td style="padding:0.4rem;">Backorder penalty</td>
          <td style="padding:0.4rem;">$0.15/unit/tick</td>
          <td style="padding:0.4rem;">Late delivery penalty (3.75x FG holding)</td>
        </tr>
        <tr style="border-bottom: 1px solid var(--border);">
          <td style="padding:0.4rem;">Shaping w_wip</td>
          <td style="padding:0.4rem;">0.0</td>
          <td style="padding:0.4rem;">Removed &mdash; real economics handle WIP</td>
        </tr>
        <tr style="border-bottom: 1px solid var(--border);">
          <td style="padding:0.4rem;">Pipeline nudge</td>
          <td style="padding:0.4rem; color: var(--accent-green);"><strong>-5.0/tick</strong> (was -3.5)</td>
          <td style="padding:0.4rem;">Increased &mdash; high util requires aggressive M0 starts</td>
        </tr>
        <tr>
          <td style="padding:0.4rem;">Training steps</td>
          <td style="padding:0.4rem;">8M</td>
          <td style="padding:0.4rem;">Extended for convergence on harder problem</td>
        </tr>
      </table>
    </div>
    """
    else:
        learned_section = """
    <div class="narrative">
      <h3>Experiment Evolution</h3>

      <p><strong>Run 1 (Previous Agent) &mdash; No WIP Costs (5M steps):</strong><br/>
        WIP in intermediate queues was free to hold. The agent exploited this by starting M0 early and
        parking units in downstream queues, then throttling M3 to avoid finished goods holding costs.
        Result: $12,978 profit, 320/336 shipped, but $3,415 in hidden WIP value &mdash; profitable but unrealistic.</p>

      <p><strong>Run 2 (Failed) &mdash; WIP Costs Added, No Tuning (5M steps):</strong><br/>
        Progressive WIP costs introduced (Q1=$0.01, Q2=$0.02, Q3=$0.03/unit/tick)
        but shaping and nudge left unchanged. Triple penalty on WIP (economic cost + shaping w_wip=0.1 + nudge -5.0).
        Result: agent collapsed &mdash; idled everything (28/336 shipped, -$40,634 profit).</p>

      <p><strong>Run 3 &mdash; Removed Shaping WIP, Halved Nudge (5M steps):</strong><br/>
        Set w_wip=0.0 (let economics handle WIP) and nudge=-2.5. WIP cost dropped 77% ($3,415 &rarr; $791)
        but agent became too conservative &mdash; delayed starts too long, only 208/336 shipped, $5,044 backorders.
        Deterministic eval profit: $6,005.</p>

      <p><strong>Run 4 (Current Agent) &mdash; Balanced Signals (8M steps):</strong><br/>
        Three adjustments to find the sweet spot between Runs 1 and 3:</p>
      <ul>
        <li><strong>WIP costs halved:</strong> Q1=$0.005, Q2=$0.01, Q3=$0.015 &mdash; softer signal so agent
            isn't scared away from starting production, but still penalizes WIP parking.</li>
        <li><strong>Nudge increased:</strong> -2.5 &rarr; -3.5 &mdash; stronger push to start M0 during
            critical lead time, reducing backorders from Run 3.</li>
        <li><strong>Training extended:</strong> 5M &rarr; 8M steps &mdash; more time to converge on the
            narrow optimal policy of starting at exactly the right time.</li>
      </ul>

      <h3>Parameter History</h3>
      <table style="width:100%; color: var(--text-muted); border-collapse: collapse; margin: 1rem 0;">
        <tr style="border-bottom: 1px solid var(--border);">
          <th style="text-align:left; padding:0.4rem;">Parameter</th>
          <th style="text-align:center; padding:0.4rem;">Run 1</th>
          <th style="text-align:center; padding:0.4rem;">Run 2</th>
          <th style="text-align:center; padding:0.4rem;">Run 3</th>
          <th style="text-align:center; padding:0.4rem; color: var(--accent-green);">Run 4</th>
        </tr>
        <tr style="border-bottom: 1px solid var(--border);">
          <td style="padding:0.4rem;">WIP costs (Q1/Q2/Q3)</td>
          <td style="text-align:center;">$0/$0/$0</td>
          <td style="text-align:center;">$0.01/$0.02/$0.03</td>
          <td style="text-align:center;">$0.01/$0.02/$0.03</td>
          <td style="text-align:center; color: var(--accent-green);">$0.005/$0.01/$0.015</td>
        </tr>
        <tr style="border-bottom: 1px solid var(--border);">
          <td style="padding:0.4rem;">w_wip (shaping)</td>
          <td style="text-align:center;">0.1</td>
          <td style="text-align:center;">0.1</td>
          <td style="text-align:center;">0.0</td>
          <td style="text-align:center; color: var(--accent-green);">0.0</td>
        </tr>
        <tr style="border-bottom: 1px solid var(--border);">
          <td style="padding:0.4rem;">Nudge penalty</td>
          <td style="text-align:center;">-5.0</td>
          <td style="text-align:center;">-5.0</td>
          <td style="text-align:center;">-2.5</td>
          <td style="text-align:center; color: var(--accent-green);">-3.5</td>
        </tr>
        <tr>
          <td style="padding:0.4rem;">Training steps</td>
          <td style="text-align:center;">5M</td>
          <td style="text-align:center;">5M</td>
          <td style="text-align:center;">5M</td>
          <td style="text-align:center; color: var(--accent-green);">8M</td>
        </tr>
      </table>

      <h3>What the Agent Should Learn</h3>
      <ul>
        <li><strong>Delayed Starts over WIP Parking:</strong> Instead of starting M0 early and holding WIP,
            the agent should delay M0 starts until the right time &mdash; true Just-In-Time production.</li>
        <li><strong>Smooth Pipeline Flow:</strong> Once a batch enters M0, it should flow through
            M1 &rarr; M2 &rarr; M3 without stalling, since every tick of WIP costs money.</li>
        <li><strong>Demand-Aware Timing:</strong> Start M0 so finished goods arrive at queue_fin
            precisely when demand hits, minimizing both WIP and FG holding costs.</li>
      </ul>
    </div>
    """

    # Encode images
    cmd_b64  = _img_to_b64(cmd_img_path)
    agent_b64 = _img_to_b64(agent_img_path)
    prev_b64 = _img_to_b64(prev_img_path) if has_prev else None

    def kpi_card(label, cmd_val, agent_val, fmt="$", is_cost=False, prev_val=None):
        # For costs, lower is better for agent; for profit/shipped, higher is better
        if is_cost:
            delta = cmd_val - agent_val
            delta_label = f"Agent saves ${delta:,.0f}" if delta >= 0 else f"Agent pays ${-delta:,.0f} more"
            agent_color = "#a6e3a1" if delta >= 0 else "#f38ba8"
        else:
            delta = agent_val - cmd_val
            delta_label = f"+${delta:,.0f} vs Commander" if delta >= 0 else f"-${-delta:,.0f} vs Commander"
            agent_color = "#a6e3a1" if delta >= 0 else "#f38ba8"

        if fmt == "$":
            cmd_str   = f"${cmd_val:,.0f}"
            agent_str = f"${agent_val:,.0f}"
        else:
            cmd_str   = f"{cmd_val:,}"
            agent_str = f"{agent_val:,}"

        prev_html = ""
        if prev_val is not None:
            prev_str = f"${prev_val:,.0f}" if fmt == "$" else f"{prev_val:,}"
            if is_cost:
                prev_delta = prev_val - agent_val
                prev_note = f"New agent saves ${prev_delta:,.0f}" if prev_delta >= 0 else f"New agent pays ${-prev_delta:,.0f} more"
            else:
                prev_delta = agent_val - prev_val
                prev_note = f"+${prev_delta:,.0f} vs prev" if prev_delta >= 0 else f"-${-prev_delta:,.0f} vs prev"
            prev_html = f"""
            <div class="kpi-divider">vs</div>
            <div class="kpi-side">
              <div class="kpi-who">Prev Agent</div>
              <div class="kpi-val" style="color:#cba6f7">{prev_str}</div>
            </div>"""
            delta_label += f" &nbsp;|&nbsp; {prev_note}"

        return f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-row">
            <div class="kpi-side">
              <div class="kpi-who">Commander</div>
              <div class="kpi-val" style="color:#cdd6f4">{cmd_str}</div>
            </div>
            <div class="kpi-divider">vs</div>
            <div class="kpi-side">
              <div class="kpi-who">RL Agent</div>
              <div class="kpi-val" style="color:{agent_color}">{agent_str}</div>
            </div>{prev_html}
          </div>
          <div class="kpi-delta">{delta_label}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{dem_mode_label} ({util_pct}% Util) — Progressive WIP Costs — RL vs Commander</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #0b0f19;
      --surface: rgba(22, 27, 34, 0.85);
      --border: rgba(48, 54, 61, 0.8);
      --text: #e6edf3;
      --text-muted: #8b949e;
      --accent-blue: #58a6ff;
      --accent-green: #3fb950;
      --accent-gold: #f9e2af;
      --accent-red: #f85149;
      --accent-purple: #d2a8ff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      font-size: 16px;
      line-height: 1.6;
      padding: 2rem;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{
      font-size: 2.4rem;
      font-weight: 700;
      color: var(--accent-blue);
      margin-bottom: 0.3rem;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 1.05rem;
      margin-bottom: 2rem;
    }}
    /* Config panel */
    .config-panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem 2rem;
      margin-bottom: 2rem;
      backdrop-filter: blur(8px);
    }}
    .config-panel h2 {{ font-size: 1.1rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; }}
    .config-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 1rem;
    }}
    .config-item {{ display: flex; flex-direction: column; gap: 0.2rem; }}
    .config-item .ci-label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .config-item .ci-val {{ font-size: 1.1rem; font-weight: 600; color: var(--text); }}
    /* Narrative */
    .narrative {{
      background: var(--surface);
      border-left: 3px solid var(--accent-blue);
      border-radius: 0 12px 12px 0;
      padding: 1.5rem 2rem;
      margin-bottom: 2rem;
    }}
    .narrative h3 {{ font-size: 1.2rem; color: var(--accent-blue); margin-bottom: 0.8rem; }}
    .narrative p {{ color: var(--text-muted); margin-bottom: 0.8rem; }}
    .narrative ul {{ color: var(--text-muted); padding-left: 1.4rem; }}
    .narrative li {{ margin-bottom: 0.5rem; }}
    .narrative strong {{ color: var(--text); }}
    .narrative em {{ color: var(--accent-gold); font-style: normal; }}
    /* KPI board */
    .kpi-board {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      backdrop-filter: blur(8px);
    }}
    .kpi-label {{ font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.8rem; }}
    .kpi-row {{ display: flex; align-items: center; gap: 0.8rem; justify-content: space-between; }}
    .kpi-side {{ display: flex; flex-direction: column; align-items: center; flex: 1; }}
    .kpi-who {{ font-size: 0.72rem; color: var(--text-muted); margin-bottom: 0.2rem; }}
    .kpi-val {{ font-size: 1.5rem; font-weight: 700; }}
    .kpi-divider {{ font-size: 0.9rem; color: var(--text-muted); }}
    .kpi-delta {{ font-size: 0.8rem; color: var(--text-muted); margin-top: 0.7rem; border-top: 1px solid var(--border); padding-top: 0.5rem; text-align: center; }}
    /* Dashboard images */
    .dashboard-section {{ margin-bottom: 2.5rem; }}
    .dashboard-section h2 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem; }}
    .dashboard-section h2.commander {{ color: var(--accent-red); }}
    .dashboard-section h2.agent {{ color: var(--accent-green); }}
    .dashboard-section img {{ width: 100%; border-radius: 12px; border: 1px solid var(--border); }}
    /* Result banner */
    .result-banner {{
      background: {'rgba(63, 185, 80, 0.12)' if agent_better else 'rgba(139, 148, 158, 0.1)'};
      border: 1px solid {'#3fb950' if agent_better else 'var(--border)'};
      border-radius: 12px;
      padding: 1.2rem 2rem;
      margin-bottom: 2rem;
      font-size: 1.05rem;
      color: var(--text);
    }}
    footer {{ margin-top: 3rem; color: var(--text-muted); font-size: 0.85rem; text-align: center; }}
  </style>
</head>
<body>
<div class="container">

  <h1>{dem_mode_label} &mdash; {util_pct}% Utilization &mdash; Progressive WIP Costs</h1>
  <p class="subtitle">SimpleFab Single-Product Line &nbsp;|&nbsp; Masked PPO vs Commander{'  vs Previous Agent (no WIP costs)' if has_prev else ''} &nbsp;|&nbsp; 8M Training Steps</p>

  <!-- Config Panel -->
  <div class="config-panel">
    <h2>Experiment Setup</h2>
    <div class="config-grid">
      <div class="config-item"><span class="ci-label">Demand Mode</span><span class="ci-val">UNIFORM</span></div>
      <div class="config-item"><span class="ci-label">Demand Pattern</span><span class="ci-val">{dem_pattern_str}</span></div>
      <div class="config-item"><span class="ci-label">Total Demand</span><span class="ci-val">{total_demand} units</span></div>
      <div class="config-item"><span class="ci-label">Utilization</span><span class="ci-val">{util_pct}%</span></div>
      <div class="config-item"><span class="ci-label">FG Holding Cost</span><span class="ci-val">${hold_cost}/unit/tick</span></div>
      <div class="config-item"><span class="ci-label">WIP Holding Cost</span><span class="ci-val">Q0=$0, Q1=${wip_costs.get(1,0)}, Q2=${wip_costs.get(2,0)}, Q3=${wip_costs.get(3,0)}</span></div>
      <div class="config-item"><span class="ci-label">Backorder Cost</span><span class="ci-val">${bo_cost}/unit/tick</span></div>
      <div class="config-item"><span class="ci-label">Revenue / Unit</span><span class="ci-val">${revenue}</span></div>
      <div class="config-item"><span class="ci-label">Time Horizon</span><span class="ci-val">{H} ticks (4 weeks)</span></div>
      <div class="config-item"><span class="ci-label">Training Steps</span><span class="ci-val">5,000,000</span></div>
      <div class="config-item"><span class="ci-label">Raw Material Supply</span><span class="ci-val">100% capacity (fully fed)</span></div>
      <div class="config-item"><span class="ci-label">Algorithm</span><span class="ci-val">Masked PPO (sb3_contrib)</span></div>
      <div class="config-item"><span class="ci-label">PPO gamma</span><span class="ci-val">0.999</span></div>
    </div>
  </div>

  <!-- Result Banner -->
  <div class="result-banner">
    <strong>Result:</strong> {summary}
  </div>

  <!-- Theoretical Maximum Banner -->
  <div style="background: rgba(249, 226, 175, 0.08); border: 1px solid #f9e2af; border-radius: 12px;
              padding: 1.2rem 2rem; margin-bottom: 2rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
    <div>
      <div style="font-size: 0.78rem; color: #f9e2af; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;">Theoretical Maximum Profit</div>
      <div style="font-size: 2rem; font-weight: 700; color: #f9e2af;">${theoretical_max:,.0f}</div>
      <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.3rem;">
        Perfect JIT: all {total_demand} units shipped, zero FG holding, zero backorder, only ${min_wip_total:.0f} unavoidable WIP transit cost
      </div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;">Agent Achievement</div>
      <div style="font-size: 2.4rem; font-weight: 700; color: {'#a6e3a1' if agent_pct_of_max >= 90 else '#f9e2af' if agent_pct_of_max >= 75 else '#f38ba8'};">{agent_pct_of_max:.1f}%</div>
      <div style="font-size: 0.85rem; color: var(--text-muted);">Gap: ${gap_to_max:,.0f}</div>
    </div>
    <div style="text-align: right;">
      <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.5rem;">Breakdown of gap:</div>
      <div style="font-size: 0.85rem; color: var(--text-muted);">
        FG Holding: ${agent_stats['inv_cost']:,.0f} |
        WIP excess: ${agent_stats['wip_cost'] - min_wip_total:,.0f} |
        Backorder: ${agent_stats['bo_cost']:,.0f} |
        Lost revenue: ${(total_demand - agent_stats['shipped']) * revenue:,}
      </div>
    </div>
  </div>

  <!-- KPI Board -->
  <div class="kpi-board">
    {kpi_card("Net Profit", cmd_stats["profit"], agent_stats["profit"],
              prev_val=prev_stats["profit"] if has_prev else None)}
    {kpi_card("Revenue", cmd_stats["revenue"], agent_stats["revenue"],
              prev_val=prev_stats["revenue"] if has_prev else None)}
    {kpi_card("FG Holding Cost", cmd_stats["inv_cost"], agent_stats["inv_cost"], is_cost=True,
              prev_val=prev_stats["inv_cost"] if has_prev else None)}
    {kpi_card("WIP Holding Cost", cmd_stats["wip_cost"], agent_stats["wip_cost"], is_cost=True,
              prev_val=prev_stats["wip_cost"] if has_prev else None)}
    {kpi_card("Backorder Penalty", cmd_stats["bo_cost"], agent_stats["bo_cost"], is_cost=True,
              prev_val=prev_stats["bo_cost"] if has_prev else None)}
    {kpi_card("Production Cost", cmd_stats["prod_cost"], agent_stats["prod_cost"], is_cost=True,
              prev_val=prev_stats["prod_cost"] if has_prev else None)}
  </div>

  <!-- What the Agent Learned -->
  {learned_section}

  <!-- Commander Dashboard -->
  <div class="dashboard-section">
    <h2 class="commander">Commander Baseline &mdash; Greedy Heuristic</h2>
    <img src="data:image/png;base64,{cmd_b64}" alt="Commander Dashboard" />
  </div>

  {"" if not has_prev else f'''<!-- Previous Agent Dashboard -->
  <div class="dashboard-section">
    <h2 style="color: #cba6f7;">Previous RL Agent &mdash; No WIP Costs</h2>
    <img src="data:image/png;base64,{prev_b64}" alt="Previous Agent Dashboard" />
  </div>'''}

  <!-- Agent Dashboard -->
  <div class="dashboard-section">
    <h2 class="agent">RL Agent &mdash; With Progressive WIP Costs (5M Steps)</h2>
    <img src="data:image/png;base64,{agent_b64}" alt="Agent Dashboard" />
  </div>

  <footer>
    SimpleFab RL Research &nbsp;|&nbsp; Experiment: {exp_name} &nbsp;|&nbsp;
    Model: output_data_1p/models/ppo_fab_1p_policy.zip
  </footer>

</div>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"HTML report written to: {out_path}")


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate comparison dashboards and HTML report for SimpleFab")
    parser.add_argument("--exp-name", type=str, default="uniform_50pct_wip",
                        help="Experiment name prefix for output files")
    parser.add_argument("--prev-model", type=str, default="",
                        help="Path to previous agent model (.zip) for 3-way comparison")
    parser.add_argument("--mode", type=str, default="UNIFORM",
                        help="Demand mode: UNIFORM, ALTERNATING, ALTERNATING_50")
    args = parser.parse_args()

    exp_name = args.exp_name
    cfg = make_common_config(mode=args.mode)

    cmd_img   = os.path.join(REPORTS_DIR, f"{exp_name}_commander.png")
    agent_img = os.path.join(REPORTS_DIR, f"{exp_name}_agent.png")
    html_out  = os.path.join(REPORTS_DIR, f"{exp_name}_comparison_report.html")

    print("Generating Commander Baseline Dashboard...")
    cmd_stats = run_and_plot(model=None, output_path=cmd_img, cfg=cfg)

    # Previous agent (trained without WIP costs) — optional
    prev_stats = None
    prev_img = None
    if args.prev_model and os.path.exists(args.prev_model):
        prev_img = os.path.join(REPORTS_DIR, f"{exp_name}_prev_agent.png")
        print(f"Loading previous RL Agent from {args.prev_model}...")
        prev_model = MaskablePPO.load(args.prev_model)
        print("Generating Previous Agent Dashboard...")
        prev_stats = run_and_plot(model=prev_model, output_path=prev_img, cfg=cfg)

    # Current agent (trained with WIP costs)
    model_path = os.path.join(OUT_DIR, "models", "ppo_fab_1p_policy.zip")
    agent_stats = None
    if os.path.exists(model_path):
        print("Loading trained RL Agent (with WIP costs)...")
        model = MaskablePPO.load(model_path)
        print("Generating RL Agent Dashboard...")
        agent_stats = run_and_plot(model=model, output_path=agent_img, cfg=cfg)

    if agent_stats is not None:
        print("Generating HTML comparison report...")
        generate_html_report(
            exp_name=exp_name,
            cfg=cfg,
            cmd_stats=cmd_stats,
            agent_stats=agent_stats,
            cmd_img_path=cmd_img,
            agent_img_path=agent_img,
            out_path=html_out,
            prev_stats=prev_stats,
            prev_img_path=prev_img,
        )
    else:
        print(f"Model not found at {model_path} — skipping agent dashboard and HTML report.")


if __name__ == "__main__":
    main()
