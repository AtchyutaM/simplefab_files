"""
Chart: Weekly arrivals (100% capacity) & demand (92% capacity) with 1-day delay.
H=2688 (4 weeks), initial finished inventory = 8 per product.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Constants ────────────────────────────────────────────
TICK = 0.25          # hours per tick
TICKS_PER_DAY  = 96
TICKS_PER_WEEK = 672
H = 2688             # 4 weeks
ALPHA = 0.5
BATCH = 4
DEMAND_DELAY = TICKS_PER_DAY  # 1-day delay

# ── Arrivals at 100% capacity ────────────────────────────
util_arr = 1.0
denom = 4.0 * ALPHA + 5.0 * (1.0 - ALPHA)  # 4.5
T_arr = util_arr * H / denom                # 597.3
ARR_P0 = (int(ALPHA * T_arr) // BATCH) * BATCH       # 296
ARR_P1 = (int((1.0 - ALPHA) * T_arr) // BATCH) * BATCH  # 296

# ── Demand at 92% capacity ───────────────────────────────
util_dem = 0.92
T_dem = util_dem * H / denom                # 549.5
DEM_P0 = (int(ALPHA * T_dem) // BATCH) * BATCH       # 272
DEM_P1 = (int((1.0 - ALPHA) * T_dem) // BATCH) * BATCH  # 272

N_WEEKS = H // TICKS_PER_WEEK  # 4
INIT_FIN = 8  # per product

# Per-week quantities
def split_weekly(total, n_weeks):
    base = total // n_weeks
    rem = total % n_weeks
    return [base + (1 if w < rem else 0) for w in range(n_weeks)]

arr_weekly_p0 = split_weekly(ARR_P0, N_WEEKS)
arr_weekly_p1 = split_weekly(ARR_P1, N_WEEKS)
dem_weekly_p0 = split_weekly(DEM_P0, N_WEEKS)
dem_weekly_p1 = split_weekly(DEM_P1, N_WEEKS)

# Bottleneck math (demand level)
dem_p0_ticks = DEM_P0 * (16 / BATCH)  # 4 ticks/unit
dem_p1_ticks = DEM_P1 * (20 / BATCH)  # 5 ticks/unit
dem_total_needed = dem_p0_ticks + dem_p1_ticks
dem_slack = H - dem_total_needed

# Bottleneck math (arrival level)
arr_p0_ticks = ARR_P0 * (16 / BATCH)
arr_p1_ticks = ARR_P1 * (20 / BATCH)
arr_total_needed = arr_p0_ticks + arr_p1_ticks

# Excess material
excess_p0 = ARR_P0 - DEM_P0
excess_p1 = ARR_P1 - DEM_P1

# ── Print summary ────────────────────────────────────────
print("=" * 70)
print("WEEKLY ARRIVALS (100%) & DEMAND (92%) — 4-WEEK HORIZON")
print("=" * 70)
print(f"Time horizon: {H} ticks = 28 days = 4 weeks")
print(f"Initial finished inventory: {INIT_FIN} P0 + {INIT_FIN} P1")
print()
print(f"{'':15s} {'P0':>8s} {'P1':>8s} {'Total':>8s}")
print(f"{'Arrivals/week':15s} {arr_weekly_p0[0]:8d} {arr_weekly_p1[0]:8d} {arr_weekly_p0[0]+arr_weekly_p1[0]:8d}")
print(f"{'Demand/week':15s} {dem_weekly_p0[0]:8d} {dem_weekly_p1[0]:8d} {dem_weekly_p0[0]+dem_weekly_p1[0]:8d}")
print(f"{'Arrivals total':15s} {ARR_P0:8d} {ARR_P1:8d} {ARR_P0+ARR_P1:8d}")
print(f"{'Demand total':15s} {DEM_P0:8d} {DEM_P1:8d} {DEM_P0+DEM_P1:8d}")
print(f"{'Excess material':15s} {excess_p0:8d} {excess_p1:8d} {excess_p0+excess_p1:8d}")
print()
print(f"Bottleneck (demand): {int(dem_total_needed)}/{H} ticks = {dem_total_needed/H*100:.1f}%")
print(f"Slack (demand):      {int(dem_slack)} ticks ({dem_slack*TICK:.1f} hrs = {dem_slack/TICKS_PER_DAY:.1f} days)")
print(f"Max changeovers:     {int(dem_slack)} (across M1+M3)")
print()
print(f"Bottleneck (if processing ALL arrivals): {int(arr_total_needed)}/{H} ticks = {arr_total_needed/H*100:.1f}%")
print("=" * 70)

# ── Build schedules ──────────────────────────────────────
arrivals_p0 = np.zeros(H); arrivals_p1 = np.zeros(H)
demand_p0 = np.zeros(H); demand_p1 = np.zeros(H)

for w in range(N_WEEKS):
    arr_t = w * TICKS_PER_WEEK
    dem_t = w * TICKS_PER_WEEK + DEMAND_DELAY
    arrivals_p0[arr_t] = arr_weekly_p0[w]
    arrivals_p1[arr_t] = arr_weekly_p1[w]
    if dem_t < H:
        demand_p0[dem_t] = dem_weekly_p0[w]
        demand_p1[dem_t] = dem_weekly_p1[w]

cum_arr_p0 = np.cumsum(arrivals_p0)
cum_dem_p0 = np.cumsum(demand_p0)
cum_arr_p1 = np.cumsum(arrivals_p1)
cum_dem_p1 = np.cumsum(demand_p1)

# Capacity lines
cap_p0_mixed = np.arange(H) * (BATCH / 16.0) * 0.5   # 0.125/tick
cap_p1_mixed = np.arange(H) * (BATCH / 20.0) * 0.5   # 0.10/tick

ticks_to_days = np.arange(H) / TICKS_PER_DAY
MIN_LEAD_P0 = 42
MIN_LEAD_P1 = 50

# ── COLORS ───────────────────────────────────────────────
C = {
    'bg': '#0d1117', 'card': '#161b22', 'text': '#c9d1d9',
    'p0': '#58a6ff', 'p1': '#f78166',
    'arr': '#3fb950', 'dem': '#d2a8ff',
    'cap': '#8b949e', 'grid': '#21262d', 'accent': '#f0883e',
    'warn': '#f85149',
}

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(C['card'])
    ax.set_title(title, color=C['text'], fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=C['text'], fontsize=10)
    ax.set_ylabel(ylabel, color=C['text'], fontsize=10)
    ax.tick_params(colors=C['text'], labelsize=9)
    for s in ax.spines.values(): s.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.4, linestyle='--')

# ── FIGURE ───────────────────────────────────────────────
fig = plt.figure(figsize=(22, 20), facecolor=C['bg'])
gs = GridSpec(4, 2, figure=fig, hspace=0.32, wspace=0.22,
              left=0.05, right=0.97, top=0.94, bottom=0.04,
              height_ratios=[1, 1.2, 1, 0.8])

# ── Panel 1: Weekly arrivals & demand side by side ───────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Raw Material Arrivals (Weekly, 100% capacity)', '', 'Units')
weeks = np.arange(N_WEEKS)
bw = 0.35
b1 = ax1.bar(weeks - bw/2, arr_weekly_p0, bw, color=C['p0'], alpha=0.85, label=f'P0 ({arr_weekly_p0[0]}/wk)')
b2 = ax1.bar(weeks + bw/2, arr_weekly_p1, bw, color=C['p1'], alpha=0.85, label=f'P1 ({arr_weekly_p1[0]}/wk)')
for w in range(N_WEEKS):
    ax1.text(w - bw/2, arr_weekly_p0[w] + 1, str(arr_weekly_p0[w]), ha='center', color=C['p0'], fontsize=10, fontweight='bold')
    ax1.text(w + bw/2, arr_weekly_p1[w] + 1, str(arr_weekly_p1[w]), ha='center', color=C['p1'], fontsize=10, fontweight='bold')
ax1.set_xticks(weeks)
ax1.set_xticklabels([f'Week {w+1}' for w in range(N_WEEKS)])
ax1.legend(facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'Customer Demand (Weekly, 92% capacity, +1 day delay)', '', 'Units')
b3 = ax2.bar(weeks - bw/2, dem_weekly_p0, bw, color=C['p0'], alpha=0.85, label=f'P0 ({dem_weekly_p0[0]}/wk)')
b4 = ax2.bar(weeks + bw/2, dem_weekly_p1, bw, color=C['p1'], alpha=0.85, label=f'P1 ({dem_weekly_p1[0]}/wk)')
for w in range(N_WEEKS):
    ax2.text(w - bw/2, dem_weekly_p0[w] + 1, str(dem_weekly_p0[w]), ha='center', color=C['p0'], fontsize=10, fontweight='bold')
    ax2.text(w + bw/2, dem_weekly_p1[w] + 1, str(dem_weekly_p1[w]), ha='center', color=C['p1'], fontsize=10, fontweight='bold')
ax2.set_xticks(weeks)
ax2.set_xticklabels([f'Week {w+1}\n(+1 day)' for w in range(N_WEEKS)])
ax2.legend(facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=9)

# ── Panel 2: Cumulative arrivals vs demand vs capacity ───
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, 'P0: Cumulative Arrivals vs Demand vs Capacity', 'Day', 'Cumulative Units')
ax3.step(ticks_to_days, cum_arr_p0, where='post', color=C['arr'], linewidth=2.5, label=f'Arrivals ({ARR_P0})')
ax3.step(ticks_to_days, cum_dem_p0, where='post', color=C['dem'], linewidth=2.5, label=f'Demand ({DEM_P0})')
ax3.plot(ticks_to_days, cap_p0_mixed, color=C['cap'], linewidth=1.5, linestyle='--', label='Bottleneck cap (50/50 mix)')
ax3.fill_between(ticks_to_days, cum_arr_p0, cum_dem_p0,
                 where=cum_arr_p0 >= cum_dem_p0, alpha=0.12, color=C['arr'], step='post')
# Mark the gap = excess material
ax3.annotate(f'Excess: {excess_p0} P0\n(arrivals > demand)',
             xy=(25, (ARR_P0+DEM_P0)/2), fontsize=9, color=C['arr'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor=C['card'], edgecolor=C['arr']))
ax3.axvline(x=MIN_LEAD_P0/TICKS_PER_DAY, color=C['accent'], linestyle='-', alpha=0.6, linewidth=1)
ax3.annotate(f'Min lead\n{MIN_LEAD_P0}t ({MIN_LEAD_P0*TICK:.1f}h)',
             xy=(MIN_LEAD_P0/TICKS_PER_DAY, 5), fontsize=8, color=C['accent'],
             bbox=dict(boxstyle='round,pad=0.2', facecolor=C['card'], edgecolor=C['accent'], alpha=0.8))
for w in range(1, N_WEEKS):
    ax3.axvline(x=w*7, color=C['accent'], linestyle=':', alpha=0.3)
ax3.set_xlim(0, 28)
ax3.legend(facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=9, loc='upper left')

ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, 'P1: Cumulative Arrivals vs Demand vs Capacity', 'Day', 'Cumulative Units')
ax4.step(ticks_to_days, cum_arr_p1, where='post', color=C['arr'], linewidth=2.5, label=f'Arrivals ({ARR_P1})')
ax4.step(ticks_to_days, cum_dem_p1, where='post', color=C['dem'], linewidth=2.5, label=f'Demand ({DEM_P1})')
ax4.plot(ticks_to_days, cap_p1_mixed, color=C['cap'], linewidth=1.5, linestyle='--', label='Bottleneck cap (50/50 mix)')
ax4.fill_between(ticks_to_days, cum_arr_p1, cum_dem_p1,
                 where=cum_arr_p1 >= cum_dem_p1, alpha=0.12, color=C['arr'], step='post')
ax4.annotate(f'Excess: {excess_p1} P1\n(arrivals > demand)',
             xy=(25, (ARR_P1+DEM_P1)/2), fontsize=9, color=C['arr'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor=C['card'], edgecolor=C['arr']))
ax4.axvline(x=MIN_LEAD_P1/TICKS_PER_DAY, color=C['accent'], linestyle='-', alpha=0.6, linewidth=1)
ax4.annotate(f'Min lead\n{MIN_LEAD_P1}t ({MIN_LEAD_P1*TICK:.1f}h)',
             xy=(MIN_LEAD_P1/TICKS_PER_DAY, 5), fontsize=8, color=C['accent'],
             bbox=dict(boxstyle='round,pad=0.2', facecolor=C['card'], edgecolor=C['accent'], alpha=0.8))
for w in range(1, N_WEEKS):
    ax4.axvline(x=w*7, color=C['accent'], linestyle=':', alpha=0.3)
ax4.set_xlim(0, 28)
ax4.legend(facecolor=C['card'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=9, loc='upper left')

# ── Panel 3: Production pipeline ─────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, 'Production Pipeline', '', '')
ax5.set_xlim(0, 10); ax5.set_ylim(0, 6)
ax5.set_xticks([]); ax5.set_yticks([])

machines = [
    ('M0\n(Batch 4)', 'P0: 16t (4h)\nP1: 20t (5h)', 1.0),
    ('M1\n(Single)', 'P0: 2t (0.5h)\nP1: 2t (0.5h)\nSetup: 1t', 3.5),
    ('M2\n(Batch 4)', 'P0: 16t (4h)\nP1: 20t (5h)', 6.0),
    ('M3\n(Single)', 'P0: 2t (0.5h)\nP1: 2t (0.5h)\nSetup: 1t', 8.5),
]
for name, times, x in machines:
    is_batch = 'Batch' in name
    rect = mpatches.FancyBboxPatch((x-0.6, 1.2), 1.2, 3.3,
                                    boxstyle="round,pad=0.1",
                                    facecolor=C['p0'] if is_batch else C['p1'],
                                    alpha=0.25, edgecolor=C['text'], linewidth=1.5)
    ax5.add_patch(rect)
    ax5.text(x, 3.6, name, ha='center', va='center', fontsize=10, fontweight='bold', color=C['text'])
    ax5.text(x, 2.0, times, ha='center', va='center', fontsize=7.5, color=C['text'], style='italic')

for i in range(len(machines)-1):
    x1 = machines[i][2] + 0.7; x2 = machines[i+1][2] - 0.7
    ax5.annotate('', xy=(x2, 2.9), xytext=(x1, 2.9),
                 arrowprops=dict(arrowstyle='->', color=C['text'], lw=1.5))
    ax5.text((x1+x2)/2, 3.2, f'Q{i+1}', ha='center', fontsize=8, color=C['cap'])

ax5.text(0.15, 2.9, 'Raw\nMat', ha='center', fontsize=9, color=C['arr'], fontweight='bold')
ax5.text(9.85, 2.9, 'Ship/\nInv', ha='center', fontsize=9, color=C['dem'], fontweight='bold')
ax5.text(5.0, 0.4, f'Min Lead: P0={MIN_LEAD_P0}t ({MIN_LEAD_P0*TICK:.1f}h) | P1={MIN_LEAD_P1}t ({MIN_LEAD_P1*TICK:.1f}h)',
         ha='center', fontsize=10, color=C['accent'], fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=C['card'], edgecolor=C['accent']))

# ── Panel 4: Key numbers & impact analysis ───────────────
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, 'Impact Analysis', '', '')
ax6.set_xlim(0, 10); ax6.set_ylim(0, 6)
ax6.set_xticks([]); ax6.set_yticks([])

lines = [
    (f'Demand util: {dem_total_needed/H*100:.1f}% | Slack: {int(dem_slack)} ticks ({dem_slack*TICK:.0f}h, {dem_slack/TICKS_PER_DAY:.1f} days)', C['accent'], 5.2, 'bold'),
    (f'Excess raw material: {excess_p0} P0 + {excess_p1} P1 = {excess_p0+excess_p1} units', C['arr'], 4.4, 'normal'),
    (f'  -> Sits in WIP queues at NO cost (only finished inv costs $)', C['text'], 3.8, 'normal'),
    (f'  -> Agent must decide: process excess or leave it idle', C['text'], 3.2, 'normal'),
    (f'Initial finished inv: {INIT_FIN} P0 + {INIT_FIN} P1 (cold-start buffer)', C['dem'], 2.4, 'normal'),
    (f'  -> Covers first ~1 day before demand delay kicks in', C['text'], 1.8, 'normal'),
    (f'Max changeovers (M1+M3): ~{int(dem_slack)} before backorders', C['warn'], 1.0, 'bold'),
]
for text, color, y, weight in lines:
    ax6.text(0.3, y, text, ha='left', va='center', fontsize=9.5, color=color,
             fontweight=weight, family='monospace')

# ── Panel 5: Summary bar ────────────────────────────────
ax7 = fig.add_subplot(gs[3, :])
style_ax(ax7, '', '', '')
ax7.set_xlim(0, 10); ax7.set_ylim(0, 2)
ax7.set_xticks([]); ax7.set_yticks([])

summary = (
    f"H={H}t (4wk)  |  Arrivals: {ARR_P0}+{ARR_P1}={ARR_P0+ARR_P1} (100%)  |  "
    f"Demand: {DEM_P0}+{DEM_P1}={DEM_P0+DEM_P1} (92%)  |  "
    f"Per week: {arr_weekly_p0[0]}+{arr_weekly_p1[0]} arr, {dem_weekly_p0[0]}+{dem_weekly_p1[0]} dem  |  "
    f"Init inv: {INIT_FIN}+{INIT_FIN}"
)
ax7.text(5.0, 1.0, summary, ha='center', va='center', fontsize=12,
         color=C['accent'], fontweight='bold', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=C['card'], edgecolor=C['accent'], linewidth=2))

fig.suptitle('SimpleFab: Arrivals (100%) & Demand (92%) — Weekly, 4-Week Horizon',
             fontsize=18, fontweight='bold', color=C['text'], y=0.97)

out_path = os.path.join(os.path.dirname(__file__), '..', 'output_data', 'weekly_92pct_overview.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150, facecolor=C['bg'])
plt.close()
print(f"\nSaved chart to: {os.path.abspath(out_path)}")
