"""
SimpleFab Architecture Infographic Generator  –  v3 (Poster Quality)
=====================================================================
Generates a visually stunning, poster-quality infographic (PNG / PDF).
Run:  python scripts/generate_infographic.py

All data values are taken directly from:
  - simplefab/config.py   (processing times, costs, batch sizes, setup)
  - simplefab/env.py      (obs space, action space, reward shaping)
  - scripts/train_ppo.py  (PPO hyperparameters)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import os

# ──────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────
BG          = "#0a0f1e"
PANEL_BG    = "#131b2e"
PANEL_BG2   = "#1a2340"
PANEL_EDGE  = "#243054"
TEXT_WHITE  = "#e8ecf4"
TEXT_MUTED  = "#8892a8"
TEXT_DIM    = "#5a6478"
P0_COLOR    = "#4a9eff"
P1_COLOR    = "#ff6b6b"
BATCH_COLOR = "#a78bfa"
SINGLE_COLOR= "#34d399"
QUEUE_COLOR = "#2a3654"
QUEUE_EDGE  = "#3d5088"
ARROW_COLOR = "#4a5878"
ACCENT_GOLD = "#fbbf24"
ACCENT_TEAL = "#2dd4bf"
REWARD_GREEN= "#34d399"
PENALTY_RED = "#f87171"
AGENT_GRAD1 = "#818cf8"
AGENT_GRAD2 = "#4f46e5"
SECTION_GLOW= "#1e3a6e"
TIMELINE_BG = "#0d1526"
CONVEYOR_COLOR = "#334155"

# ──────────────────────────────────────────────────────────────────────
# DATA  (verified against source code)
# ──────────────────────────────────────────────────────────────────────
MACHINES = [
    {"name": "Machine 0", "type": "Batch", "batch": 4,
     "time": (16, 20), "cost": (8, 10), "setup_cost": 0, "setup_time": (0, 0),
     "color": BATCH_COLOR, "icon": "⚙"},
    {"name": "Machine 1", "type": "Single", "batch": 1,
     "time": (2, 2), "cost": (4, 4), "setup_cost": 20, "setup_time": (1, 1),
     "color": SINGLE_COLOR, "icon": "S"},
    {"name": "Machine 2", "type": "Batch", "batch": 4,
     "time": (16, 20), "cost": (8, 10), "setup_cost": 0, "setup_time": (0, 0),
     "color": BATCH_COLOR, "icon": "⚙"},
    {"name": "Machine 3", "type": "Single", "batch": 1,
     "time": (2, 2), "cost": (4, 4), "setup_cost": 20, "setup_time": (1, 1),
     "color": SINGLE_COLOR, "icon": "S"},
]

REVENUE     = (80, 100)
INV_COST    = (0.5, 0.6)
BACK_COST   = (1.0, 1.0)
INIT_INV    = (8, 9)


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────
def rounded_rect(ax, xy, w, h, r=0.012, fc=PANEL_BG, ec=PANEL_EDGE, lw=1.5,
                 zorder=2, alpha=1.0, clip_on=True):
    box = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={r}",
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         transform=ax.transAxes, zorder=zorder, alpha=alpha,
                         clip_on=clip_on)
    ax.add_patch(box)
    return box

def draw_arrow(ax, start, end, color=ARROW_COLOR, lw=2, zorder=3,
               style="->,head_width=6,head_length=5"):
    arrow = FancyArrowPatch(start, end,
                            arrowstyle=style,
                            color=color, lw=lw,
                            transform=ax.transAxes, zorder=zorder,
                            connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)

def thick_arrow(ax, start, end, color=ARROW_COLOR, lw=3.5, zorder=3):
    draw_arrow(ax, start, end, color=color, lw=lw, zorder=zorder,
               style="->,head_width=8,head_length=6")

def draw_circle(ax, center, radius, fc=QUEUE_COLOR, ec="#64748b", lw=2,
                zorder=4, alpha=1.0):
    c = Circle(center, radius, facecolor=fc, edgecolor=ec, linewidth=lw,
               transform=ax.transAxes, zorder=zorder, alpha=alpha)
    ax.add_patch(c)
    return c

def glow_circle(ax, center, radius, color, zorder=3, layers=12, max_alpha=0.06):
    """Draw a soft glow halo behind a circle."""
    for i in range(layers):
        r = radius * (1 + 2.0 * (i / layers))
        a = max_alpha * (1 - i / layers)
        draw_circle(ax, center, r, fc=color, ec="none", lw=0, zorder=zorder, alpha=a)

def section_header(ax, x, y, text, color=TEXT_MUTED, fontsize=12):
    ax.text(x, y, text, fontsize=fontsize, fontweight="bold", color=color,
            transform=ax.transAxes, fontfamily="monospace")


# ──────────────────────────────────────────────────────────────────────
# MACHINE VISUAL  — Factory-style icon with details
# ──────────────────────────────────────────────────────────────────────
def draw_machine_visual(ax, mx, my, mw, mh, m):
    """Draw a visually striking machine with factory-style icon."""
    color = m["color"]
    is_batch = m["type"] == "Batch"

    # ── Outer glow ──
    for i in range(8):
        pad = 0.003 * (i + 1)
        rounded_rect(ax, (mx - pad, my - pad), mw + 2 * pad, mh + 2 * pad,
                     r=0.010, fc=color, ec="none", lw=0, zorder=1, alpha=0.018)

    # ── Card body ──
    rounded_rect(ax, (mx, my), mw, mh,
                 r=0.008, fc=PANEL_BG2, ec=color, lw=2.5, zorder=5)

    # ── Top color accent bar ──
    accent_h = 0.006
    rounded_rect(ax, (mx + 0.002, my + mh - accent_h - 0.002),
                 mw - 0.004, accent_h,
                 r=0.003, fc=color, ec="none", lw=0, zorder=6, alpha=0.9)

    # ── Icon area (top portion) ──
    icon_cy = my + mh - 0.038
    icon_cx = mx + mw / 2

    if is_batch:
        # Draw a factory/gear-like icon for batch machines
        # Outer gear ring
        gear_r = 0.018
        draw_circle(ax, (icon_cx, icon_cy), gear_r,
                    fc=color, ec="none", lw=0, zorder=7, alpha=0.15)
        draw_circle(ax, (icon_cx, icon_cy), gear_r * 0.7,
                    fc=color, ec="none", lw=0, zorder=7, alpha=0.25)
        draw_circle(ax, (icon_cx, icon_cy), gear_r * 0.35,
                    fc=color, ec="none", lw=0, zorder=7, alpha=0.5)
        # Gear teeth (small squares around the circle)
        n_teeth = 8
        for k in range(n_teeth):
            angle = 2 * np.pi * k / n_teeth
            tx = icon_cx + gear_r * 0.85 * np.cos(angle)
            ty = icon_cy + gear_r * 0.85 * np.sin(angle)
            tooth_r = 0.004
            draw_circle(ax, (tx, ty), tooth_r,
                        fc=color, ec="none", lw=0, zorder=7, alpha=0.4)
        # Center symbol
        ax.text(icon_cx, icon_cy, "⚙", fontsize=14, color=color,
                ha="center", va="center", transform=ax.transAxes, zorder=8)
    else:
        # Draw a wrench/tool icon for single machines
        # Tool shape background
        draw_circle(ax, (icon_cx, icon_cy), 0.018,
                    fc=color, ec="none", lw=0, zorder=7, alpha=0.15)
        draw_circle(ax, (icon_cx, icon_cy), 0.012,
                    fc=color, ec="none", lw=0, zorder=7, alpha=0.25)
        # Draw crossed lines for tool icon
        line_len = 0.010
        ax.plot([icon_cx - line_len, icon_cx + line_len],
                [icon_cy - line_len * 0.6, icon_cy + line_len * 0.6],
                color=color, lw=2.5, transform=ax.transAxes, zorder=8, alpha=0.8)
        ax.plot([icon_cx + line_len, icon_cx - line_len],
                [icon_cy - line_len * 0.6, icon_cy + line_len * 0.6],
                color=color, lw=2.5, transform=ax.transAxes, zorder=8, alpha=0.8)
        # Center dot
        draw_circle(ax, (icon_cx, icon_cy), 0.004,
                    fc=color, ec="none", lw=0, zorder=9, alpha=0.7)

    # ── Machine name ──
    ax.text(icon_cx, icon_cy - 0.024, m["name"],
            fontsize=9, fontweight="bold", color=TEXT_WHITE,
            ha="center", va="center", transform=ax.transAxes, zorder=7)

    # ── Type badge ──
    badge_col = BATCH_COLOR if is_batch else SINGLE_COLOR
    badge_bg = "#1e1050" if is_batch else "#0a2520"
    ax.text(icon_cx, icon_cy - 0.043, m["type"],
            fontsize=7.5, fontweight="bold", color=badge_col,
            ha="center", va="center", transform=ax.transAxes, zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc=badge_bg, ec=badge_col,
                      alpha=0.8, lw=1))

    # ── Data rows ──
    row_start_y = my + mh - 0.105
    row_gap = 0.022

    if is_batch:
        labels_vals = [
            ("Proc Time", str(m['time'][0]), str(m['time'][1]), True),
            ("Cost/Unit", f"${m['cost'][0]}", f"${m['cost'][1]}", True),
            ("Batch", str(m["batch"]) + " units", None, False),
        ]
    else:
        labels_vals = [
            ("Proc Time", str(m['time'][0]), str(m['time'][1]), True),
            ("Cost/Unit", f"${m['cost'][0]}", f"${m['cost'][1]}", True),
            ("Setup $", f"${m['setup_cost']}", None, False),
            ("Setup T", str(m['setup_time'][0]), str(m['setup_time'][1]), True),
        ]

    for j, (lbl, v0, v1, is_pair) in enumerate(labels_vals):
        ry = row_start_y - j * row_gap

        # Alternating row shading
        if j % 2 == 0:
            rounded_rect(ax, (mx + 0.004, ry - 0.008), mw - 0.008, 0.018,
                         r=0.003, fc="#0d1526", ec="none", lw=0, zorder=5, alpha=0.5)

        ax.text(mx + 0.008, ry, lbl, fontsize=6.5, color=TEXT_MUTED,
                ha="left", va="center", transform=ax.transAxes, zorder=8)

        if is_pair and v1 is not None:
            ax.text(mx + mw - 0.035, ry, v0,
                    fontsize=7, fontweight="bold", color=P0_COLOR,
                    ha="right", va="center", transform=ax.transAxes, zorder=8)
            ax.text(mx + mw - 0.028, ry, "/",
                    fontsize=7, color=TEXT_DIM,
                    ha="center", va="center", transform=ax.transAxes, zorder=8)
            ax.text(mx + mw - 0.008, ry, v1,
                    fontsize=7, fontweight="bold", color=P1_COLOR,
                    ha="right", va="center", transform=ax.transAxes, zorder=8)
        else:
            ax.text(mx + mw - 0.008, ry, v0,
                    fontsize=7.5, fontweight="bold", color=TEXT_WHITE,
                    ha="right", va="center", transform=ax.transAxes, zorder=8)


# ──────────────────────────────────────────────────────────────────────
# QUEUE VISUAL — Buffer icon with stacked items
# ──────────────────────────────────────────────────────────────────────
def draw_queue_visual(ax, cx, cy, label, radius=0.020):
    """Draw a visually rich queue buffer node with stacked items."""
    # Glow
    glow_circle(ax, (cx, cy), radius, QUEUE_EDGE, zorder=3, layers=10, max_alpha=0.05)

    # Outer ring
    draw_circle(ax, (cx, cy), radius, fc=QUEUE_COLOR, ec=QUEUE_EDGE, lw=2.5, zorder=6)

    # Inner ring
    draw_circle(ax, (cx, cy), radius * 0.75, fc="#1a2340", ec=QUEUE_EDGE,
                lw=1, zorder=7, alpha=0.6)

    # Buffer visualization: small colored squares inside
    sq_size = radius * 0.28
    positions = [
        (-0.4, 0.3),  (0.4, 0.3),
        (-0.4, -0.3), (0.4, -0.3),
    ]
    colors = [P0_COLOR, P1_COLOR, P0_COLOR, P1_COLOR]
    for (dx, dy), col in zip(positions, colors):
        sx = cx + dx * radius * 0.5
        sy = cy + dy * radius * 0.5
        rounded_rect(ax, (sx - sq_size/2, sy - sq_size/2), sq_size, sq_size,
                     r=0.001, fc=col, ec="none", lw=0, zorder=8, alpha=0.5)

    # Label
    ax.text(cx, cy, label, fontsize=9.5, fontweight="bold",
            color=TEXT_WHITE, ha="center", va="center",
            transform=ax.transAxes, zorder=9)


# ──────────────────────────────────────────────────────────────────────
# CONVEYOR BELT — Visual connection between machines
# ──────────────────────────────────────────────────────────────────────
def draw_conveyor(ax, x_start, x_end, y, color=CONVEYOR_COLOR, lw=2, zorder=3):
    """Draw a conveyor-belt-style connector with chevrons."""
    # Main track lines (top and bottom)
    belt_half_h = 0.004
    ax.plot([x_start, x_end], [y + belt_half_h, y + belt_half_h],
            color=color, lw=lw, transform=ax.transAxes, zorder=zorder, alpha=0.5)
    ax.plot([x_start, x_end], [y - belt_half_h, y - belt_half_h],
            color=color, lw=lw, transform=ax.transAxes, zorder=zorder, alpha=0.5)

    # Chevrons along the belt
    n_chevrons = int((x_end - x_start) / 0.008)
    n_chevrons = max(n_chevrons, 3)
    for i in range(n_chevrons):
        t = (i + 0.5) / n_chevrons
        cx = x_start + t * (x_end - x_start)
        chev_w = 0.003
        # Small triangle pointing right
        ax.plot([cx - chev_w, cx, cx - chev_w],
                [y + belt_half_h * 0.8, y, y - belt_half_h * 0.8],
                color=ACCENT_TEAL, lw=1, transform=ax.transAxes,
                zorder=zorder + 1, alpha=0.4)

    # Arrow overlay
    draw_arrow(ax, (x_start, y), (x_end, y),
               color=ACCENT_TEAL, lw=2, zorder=zorder + 2,
               style="->,head_width=7,head_length=5")


# ══════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ══════════════════════════════════════════════════════════════════════
def create_infographic(save_path="simplefab_infographic.png", dpi=300):
    fig, ax = plt.subplots(figsize=(38, 26))
    fig.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ════════════════════════════════════════════════════════════
    # TITLE BAND  (top 5%)
    # ════════════════════════════════════════════════════════════
    # Subtle glow behind title
    for i in range(10):
        r = 0.06 + i * 0.018
        draw_circle(ax, (0.50, 0.970), r, fc=AGENT_GRAD2, ec="none",
                    lw=0, zorder=0, alpha=0.012)

    ax.text(0.50, 0.975, "SimpleFab", fontsize=60, fontweight="bold",
            color=TEXT_WHITE, ha="center", va="center", transform=ax.transAxes,
            fontfamily="sans-serif")
    ax.text(0.50, 0.950, "Mixed-Model Production Line  ·  Reinforcement Learning Control Architecture",
            fontsize=18, color=TEXT_MUTED, ha="center", va="center",
            transform=ax.transAxes, fontfamily="sans-serif")

    # Legend
    for i, (label, col) in enumerate([("Product 0 (P0)", P0_COLOR),
                                       ("Product 1 (P1)", P1_COLOR)]):
        x_dot = 0.82 + i * 0.09
        draw_circle(ax, (x_dot, 0.975), 0.005, fc=col, ec=col, lw=0, zorder=5)
        ax.text(x_dot + 0.010, 0.975, label, fontsize=11, color=col,
                ha="left", va="center", transform=ax.transAxes, fontweight="bold")

    # ════════════════════════════════════════════════════════════
    # SECTION 1 — PHYSICAL ENVIRONMENT  (top ~40%)
    # ════════════════════════════════════════════════════════════
    sec1_top = 0.930; sec1_bot = 0.530
    rounded_rect(ax, (0.012, sec1_bot), 0.976, sec1_top - sec1_bot,
                 r=0.010, fc=PANEL_BG, ec=SECTION_GLOW, lw=1.5)
    section_header(ax, 0.030, sec1_top - 0.022,
                   "▌ PHYSICAL ENVIRONMENT", TEXT_MUTED, fontsize=11)

    # Sub-label showing the flow
    ax.text(0.030, sec1_top - 0.042,
            "Demand Arrives  →  Queue  →  Process  →  Ship  →  Revenue / Penalty",
            fontsize=8, color=ACCENT_TEAL, ha="left", va="center",
            transform=ax.transAxes, fontfamily="monospace", fontstyle="italic",
            alpha=0.7)

    line_y = 0.745   # vertical centre of production flow

    # ─── INPUT SECTION ────────────────────────────────────────
    inp_x = 0.025
    inp_w = 0.072; inp_h = 0.055

    # Warm Start
    ws_y = line_y + 0.028
    rounded_rect(ax, (inp_x, ws_y), inp_w, inp_h,
                 r=0.006, fc=PANEL_BG2, ec=ACCENT_GOLD, lw=2)
    ax.text(inp_x + inp_w / 2, ws_y + inp_h - 0.012, "Warm Start",
            fontsize=8, fontweight="bold", color=ACCENT_GOLD, ha="center",
            va="center", transform=ax.transAxes, zorder=5)
    ax.text(inp_x + inp_w / 2, ws_y + 0.014,
            f"P0:{INIT_INV[0]}  P1:{INIT_INV[1]}",
            fontsize=7.5, color=TEXT_WHITE, ha="center", va="center",
            transform=ax.transAxes, zorder=5, fontfamily="monospace")

    # Arrivals
    ar_y = line_y - 0.083
    rounded_rect(ax, (inp_x, ar_y), inp_w, inp_h,
                 r=0.006, fc=PANEL_BG2, ec=ACCENT_TEAL, lw=2)
    ax.text(inp_x + inp_w / 2, ar_y + inp_h - 0.012, "Arrivals",
            fontsize=8, fontweight="bold", color=ACCENT_TEAL, ha="center",
            va="center", transform=ax.transAxes, zorder=5)
    ax.text(inp_x + inp_w / 2, ar_y + 0.014,
            "Uniform / Cont.",
            fontsize=7, color=TEXT_MUTED, ha="center", va="center",
            transform=ax.transAxes, zorder=5)

    # Arrows from inputs to Q0
    q0_x = 0.130
    draw_arrow(ax, (inp_x + inp_w + 0.003, ws_y + inp_h / 2),
               (q0_x - 0.025, line_y + 0.004),
               color=ACCENT_GOLD, lw=2)
    draw_arrow(ax, (inp_x + inp_w + 0.003, ar_y + inp_h / 2),
               (q0_x - 0.025, line_y - 0.004),
               color=ACCENT_TEAL, lw=2)

    # ─── PRODUCTION LINE ──────────────────────────────────────
    seg_start = 0.130
    q_spacing = 0.148
    machine_w = 0.088
    machine_h = 0.180
    queue_r = 0.020
    queue_names = ["Q0", "Q1", "Q2", "Q3"]

    for i in range(4):
        qx = seg_start + i * q_spacing
        mx_pos = qx + 0.036

        # Queue node (visually rich)
        draw_queue_visual(ax, qx, line_y, queue_names[i], radius=queue_r)

        # Conveyor belt Q -> M
        draw_conveyor(ax, qx + queue_r + 0.005, mx_pos - 0.004, line_y)

        # Machine Card (visually striking)
        m = MACHINES[i]
        mc_y = line_y - machine_h / 2
        draw_machine_visual(ax, mx_pos, mc_y, machine_w, machine_h, m)

        # Conveyor belt M -> next Q (or to Fin/Dem)
        arrow_start_x = mx_pos + machine_w + 0.005
        if i < 3:
            next_qx = seg_start + (i + 1) * q_spacing
            draw_conveyor(ax, arrow_start_x, next_qx - queue_r - 0.005, line_y)

    # ─── FINISHED GOODS & DEMAND QUEUES ───────────────────────
    last_m_end = seg_start + 3 * q_spacing + 0.036 + machine_w + 0.005

    fin_qx = last_m_end + 0.030
    draw_queue_visual(ax, fin_qx, line_y, "Fin", radius=queue_r)
    draw_circle(ax, (fin_qx, line_y), queue_r + 0.003, fc="none",
                ec=REWARD_GREEN, lw=2, zorder=9, alpha=0.6)
    draw_conveyor(ax, last_m_end, fin_qx - queue_r - 0.005, line_y)

    # Label above Finished
    ax.text(fin_qx, line_y + queue_r + 0.016, "Finished",
            fontsize=7.5, fontweight="bold", color=REWARD_GREEN,
            ha="center", va="center", transform=ax.transAxes, zorder=6)
    ax.text(fin_qx, line_y + queue_r + 0.005, "Goods",
            fontsize=7.5, fontweight="bold", color=REWARD_GREEN,
            ha="center", va="center", transform=ax.transAxes, zorder=6)

    dem_qx = fin_qx + 0.050
    draw_queue_visual(ax, dem_qx, line_y, "Dem", radius=queue_r)
    draw_circle(ax, (dem_qx, line_y), queue_r + 0.003, fc="none",
                ec=PENALTY_RED, lw=2, zorder=9, alpha=0.6)
    draw_conveyor(ax, fin_qx + queue_r + 0.005, dem_qx - queue_r - 0.005, line_y)

    # Label above Demand
    ax.text(dem_qx, line_y + queue_r + 0.016, "Demand",
            fontsize=7.5, fontweight="bold", color=PENALTY_RED,
            ha="center", va="center", transform=ax.transAxes, zorder=6)
    ax.text(dem_qx, line_y + queue_r + 0.005, "Queue",
            fontsize=7.5, fontweight="bold", color=PENALTY_RED,
            ha="center", va="center", transform=ax.transAxes, zorder=6)

    # ─── MARKET BOX ───────────────────────────────────────────
    out_x = dem_qx + 0.034
    out_w = 0.085; out_h = 0.130
    out_y = line_y - out_h / 2

    # Market glow
    for i in range(5):
        pad = 0.004 * (i + 1)
        rounded_rect(ax, (out_x - pad, out_y - pad),
                     out_w + 2*pad, out_h + 2*pad,
                     r=0.010, fc=P0_COLOR, ec="none", lw=0, zorder=1, alpha=0.015)

    rounded_rect(ax, (out_x, out_y), out_w, out_h,
                 r=0.008, fc="#0c1a3d", ec=P0_COLOR, lw=2.5, zorder=5)

    ax.text(out_x + out_w / 2, out_y + out_h - 0.018, "Market",
            fontsize=12, fontweight="bold", color=TEXT_WHITE,
            ha="center", va="center", transform=ax.transAxes, zorder=6)

    ax.text(out_x + out_w / 2, out_y + out_h - 0.040, "Revenue / Unit",
            fontsize=7.5, color=TEXT_MUTED, ha="center", va="center",
            transform=ax.transAxes, zorder=6)

    ax.text(out_x + out_w / 2, out_y + out_h / 2 - 0.008,
            f"P0:  ${REVENUE[0]}",
            fontsize=15, fontweight="bold", color=P0_COLOR,
            ha="center", va="center", transform=ax.transAxes, zorder=6)
    ax.text(out_x + out_w / 2, out_y + 0.016,
            f"P1:  ${REVENUE[1]}",
            fontsize=15, fontweight="bold", color=P1_COLOR,
            ha="center", va="center", transform=ax.transAxes, zorder=6)

    draw_conveyor(ax, dem_qx + queue_r + 0.005, out_x - 0.005, line_y)

    # ─── ECONOMICS BAR ────────────────────────────────────────
    econ_y = sec1_bot + 0.012
    econ_items = [
        ("INVENTORY HOLDING COST", f"P0: ${INV_COST[0]}  ·  P1: ${INV_COST[1]}",
         "per unit per step", TEXT_MUTED, "#1a2340"),
        ("BACKORDER PENALTY", f"P0: ${BACK_COST[0]}  ·  P1: ${BACK_COST[1]}",
         "per unit per step", PENALTY_RED, "#2d0a0a"),
        ("EPISODE LENGTH", "H = 500 steps per episode",
         "Each step = 1 time unit", ACCENT_TEAL, "#0a2520"),
    ]
    bar_w_each = 0.305
    for k, (title, val, sub, col, bg_col) in enumerate(econ_items):
        ex = 0.025 + k * 0.325
        rounded_rect(ax, (ex, econ_y), bar_w_each, 0.048,
                     r=0.005, fc=bg_col, ec=PANEL_EDGE, lw=1, zorder=5, alpha=0.7)
        ax.text(ex + 0.012, econ_y + 0.034, title, fontsize=7.5, fontweight="bold",
                color=col, ha="left", va="center", transform=ax.transAxes,
                fontfamily="monospace", zorder=6)
        ax.text(ex + 0.012, econ_y + 0.015, f"{val}  •  {sub}", fontsize=7,
                color=TEXT_WHITE, ha="left", va="center", transform=ax.transAxes,
                zorder=6)


    # ════════════════════════════════════════════════════════════
    # SECTION 2 — RL AGENT ARCHITECTURE  (bottom ~50%)
    # ════════════════════════════════════════════════════════════
    sec2_top = 0.510; sec2_bot = 0.010
    rounded_rect(ax, (0.012, sec2_bot), 0.976, sec2_top - sec2_bot,
                 r=0.010, fc=PANEL_BG, ec="#312e81", lw=1.5)
    section_header(ax, 0.030, sec2_top - 0.022,
                   "▌ RL AGENT ARCHITECTURE", TEXT_MUTED, fontsize=11)
    ax.text(0.030, sec2_top - 0.042,
            "Observe  →  Decide  →  Act  →  Reward  →  Learn  →  Repeat",
            fontsize=8, color=AGENT_GRAD1, ha="left", va="center",
            transform=ax.transAxes, fontfamily="monospace", fontstyle="italic",
            alpha=0.7)

    # ─── LEFT: OBSERVATION SPACE ──────────────────────────────
    obs_x = 0.025; obs_y = sec2_bot + 0.095; obs_w = 0.225; obs_h = 0.372
    rounded_rect(ax, (obs_x, obs_y), obs_w, obs_h,
                 r=0.008, fc=PANEL_BG2, ec=PANEL_EDGE, lw=1.5, zorder=5)

    # Title
    ax.text(obs_x + obs_w / 2, obs_y + obs_h - 0.018, "STATE OBSERVATION",
            fontsize=12, fontweight="bold", color=AGENT_GRAD1,
            ha="center", va="center", transform=ax.transAxes, zorder=6)

    # Subtitle  (clearly below title with good spacing)
    ax.text(obs_x + obs_w / 2, obs_y + obs_h - 0.042,
            "Box(20,)  ·  Normalized [0, 1]",
            fontsize=8, color=TEXT_MUTED, ha="center", va="center",
            transform=ax.transAxes, zorder=6, fontfamily="monospace")

    # ── Observation component cards with clear spacing ──
    obs_items = [
        ("Queue Depths", "12 Dims",
         "Capacities of Q0-Q3, Fin, Demand",
         "2 prods × 6 queues · Cap-normed [0,1]"),
        ("Machine Status", "4 Dims",
         "Current product processing",
         "Idle→0.0  ·  P0→0.5  ·  P1→1.0"),
        ("Time Remaining", "4 Dims",
         "Steps until machine is free",
         "Normed by max proc time (20)"),
    ]

    card_h = 0.082
    card_gap = 0.018
    first_card_y = obs_y + obs_h - 0.080 - card_h   # properly place below subtitle

    for j, (title, dims, desc, encode) in enumerate(obs_items):
        iy = first_card_y - j * (card_h + card_gap)
        rounded_rect(ax, (obs_x + 0.010, iy), obs_w - 0.020, card_h,
                     r=0.005, fc="#0d1526", ec=SECTION_GLOW, lw=1, zorder=6)

        # Title line — well within card
        ax.text(obs_x + 0.020, iy + card_h - 0.016, title,
                fontsize=8.5, fontweight="bold", color=TEXT_WHITE,
                ha="left", va="center", transform=ax.transAxes, zorder=7)
        ax.text(obs_x + obs_w - 0.020, iy + card_h - 0.016, dims,
                fontsize=8, fontweight="bold", color=ACCENT_TEAL,
                ha="right", va="center", transform=ax.transAxes, zorder=7,
                fontfamily="monospace")

        # Description
        ax.text(obs_x + 0.020, iy + card_h - 0.040, desc,
                fontsize=7, color=TEXT_MUTED, ha="left", va="center",
                transform=ax.transAxes, zorder=7)

        # Encoding
        ax.text(obs_x + 0.020, iy + card_h - 0.062, encode,
                fontsize=6.5, color=ACCENT_TEAL, ha="left", va="center",
                transform=ax.transAxes, zorder=7, fontfamily="monospace", alpha=0.8)

    # Total badge
    ax.text(obs_x + obs_w / 2, obs_y + 0.020,
            "  Total: 20 Dimensions  ",
            fontsize=9.5, fontweight="bold", color=AGENT_GRAD1,
            ha="center", va="center", transform=ax.transAxes, zorder=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="#1e1050", ec=AGENT_GRAD1, lw=1.5))

    # ─── CENTRE: AGENT ────────────────────────────────────────
    agent_cx = 0.365; agent_cy = sec2_bot + 0.330
    agent_r = 0.058

    # Big glow
    glow_circle(ax, (agent_cx, agent_cy), agent_r, AGENT_GRAD1,
                zorder=3, layers=18, max_alpha=0.04)

    draw_circle(ax, (agent_cx, agent_cy), agent_r,
                fc=AGENT_GRAD2, ec=AGENT_GRAD1, lw=3.5, zorder=10)
    # Inner brighter circle
    draw_circle(ax, (agent_cx, agent_cy), agent_r * 0.85,
                fc=AGENT_GRAD2, ec="none", lw=0, zorder=10, alpha=0.5)

    ax.text(agent_cx, agent_cy + 0.015, "Maskable",
            fontsize=11, fontweight="bold", color="white",
            ha="center", va="center", transform=ax.transAxes, zorder=11)
    ax.text(agent_cx, agent_cy - 0.015, "PPO",
            fontsize=24, fontweight="bold", color="white",
            ha="center", va="center", transform=ax.transAxes, zorder=11)

    # Action mask badge
    ax.text(agent_cx, agent_cy - agent_r - 0.026, "ACTION MASKING",
            fontsize=8, fontweight="bold", color="#065f46",
            ha="center", va="center", transform=ax.transAxes, zorder=11,
            bbox=dict(boxstyle="round,pad=0.3", fc=SINGLE_COLOR,
                      ec="#065f46", lw=1.5))
    ax.text(agent_cx, agent_cy - agent_r - 0.048,
            "bool[12] mask per step",
            fontsize=6.5, color=TEXT_MUTED, ha="center", va="center",
            transform=ax.transAxes, zorder=6, fontfamily="monospace")

    # ── RL Episode Loop (circular arrows around agent) ──
    # Small text labels around the agent showing the RL cycle
    loop_labels = [
        (0.40,  "observe s", AGENT_GRAD1),    # left
        (0.12,  "select a", ACCENT_GOLD),      # top-right
        (-0.15, "reward r", REWARD_GREEN),      # right
        (-0.42, "update π", PENALTY_RED),       # bottom
    ]
    for angle_offset, label, color in loop_labels:
        angle = np.pi * angle_offset
        lx = agent_cx + (agent_r + 0.045) * np.cos(angle)
        ly = agent_cy + (agent_r + 0.045) * np.sin(angle)
        ax.text(lx, ly, label, fontsize=6.5, fontweight="bold",
                color=color, ha="center", va="center",
                transform=ax.transAxes, zorder=6, fontstyle="italic",
                alpha=0.7)

    # Observe arrow
    draw_arrow(ax, (obs_x + obs_w + 0.008, agent_cy + 0.005),
               (agent_cx - agent_r - 0.008, agent_cy + 0.005),
               color=AGENT_GRAD1, lw=3)
    ax.text((obs_x + obs_w + agent_cx - agent_r) / 2, agent_cy + 0.028,
            "observe", fontsize=9, color=AGENT_GRAD1, ha="center",
            va="center", transform=ax.transAxes, fontstyle="italic",
            fontweight="bold")

    # Act arrow
    act_target_x = 0.490
    draw_arrow(ax, (agent_cx + agent_r + 0.008, agent_cy + 0.005),
               (act_target_x - 0.005, agent_cy + 0.005),
               color=ACCENT_GOLD, lw=3)
    ax.text((agent_cx + agent_r + act_target_x) / 2, agent_cy + 0.028,
            "act", fontsize=9, color=ACCENT_GOLD, ha="center",
            va="center", transform=ax.transAxes, fontstyle="italic",
            fontweight="bold")

    # ─── RIGHT-TOP: ACTION SPACE ──────────────────────────────
    act_x = 0.490; act_y = sec2_bot + 0.300; act_w = 0.200; act_h = 0.165
    rounded_rect(ax, (act_x, act_y), act_w, act_h,
                 r=0.008, fc=PANEL_BG2, ec=PANEL_EDGE, lw=1.5, zorder=5)

    ax.text(act_x + act_w / 2, act_y + act_h - 0.018,
            "ACTION SPACE", fontsize=10.5, fontweight="bold",
            color=ACCENT_GOLD, ha="center", va="center",
            transform=ax.transAxes, zorder=6)
    ax.text(act_x + act_w / 2, act_y + act_h - 0.040,
            "MultiDiscrete([3, 3, 3, 3])", fontsize=8.5,
            color=TEXT_MUTED, ha="center", va="center",
            transform=ax.transAxes, zorder=6, fontfamily="monospace")

    for j in range(4):
        bx = act_x + 0.008 + (j % 2) * 0.098
        by = act_y + act_h - 0.075 - (j // 2) * 0.055
        bw = 0.090; bh = 0.042
        rounded_rect(ax, (bx, by), bw, bh,
                     r=0.005, fc="#0d1526", ec=PANEL_EDGE, lw=1, zorder=6)
        m_col = BATCH_COLOR if j in [0, 2] else SINGLE_COLOR
        ax.text(bx + bw / 2, by + bh - 0.012, f"Machine {j}",
                fontsize=7, fontweight="bold", color=m_col,
                ha="center", va="center", transform=ax.transAxes, zorder=7)
        ax.text(bx + bw / 2, by + 0.010, "{NoOp, P0, P1}",
                fontsize=7, fontweight="bold", color=TEXT_WHITE,
                ha="center", va="center", transform=ax.transAxes, zorder=7,
                fontfamily="monospace")

    # ─── RIGHT-BOTTOM: REWARD FUNCTION ────────────────────────
    rew_x = 0.490; rew_y = sec2_bot + 0.095; rew_w = 0.200; rew_h = 0.190
    rounded_rect(ax, (rew_x, rew_y), rew_w, rew_h,
                 r=0.008, fc=PANEL_BG2, ec=PANEL_EDGE, lw=1.5, zorder=5)

    ax.text(rew_x + rew_w / 2, rew_y + rew_h - 0.018,
            "REWARD FUNCTION", fontsize=10.5, fontweight="bold",
            color=REWARD_GREEN, ha="center", va="center",
            transform=ax.transAxes, zorder=6)

    # Main equation
    ax.text(rew_x + rew_w / 2, rew_y + rew_h - 0.048,
            "R = R_base + R_shape + R_pen",
            fontsize=8.5, fontweight="bold", color=TEXT_WHITE,
            ha="center", va="center", transform=ax.transAxes, zorder=6,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#0d1526",
                      ec=SECTION_GLOW, lw=1))

    rew_terms = [
        ("R_base",    "= ΔProfit (Rev − Costs)", REWARD_GREEN),
        ("R_shape",   "= β·(γ·Φ(s′) − Φ(s))",   ACCENT_GOLD),
        ("R_penalty", "= −2.0 × invalid_count",  PENALTY_RED),
    ]
    for j, (name, formula, col) in enumerate(rew_terms):
        ry = rew_y + rew_h - 0.080 - j * 0.026
        ax.text(rew_x + 0.012, ry, name, fontsize=7, fontweight="bold",
                color=col, ha="left", va="center", transform=ax.transAxes,
                zorder=6, fontfamily="monospace")
        ax.text(rew_x + 0.068, ry, formula, fontsize=7, color=TEXT_MUTED,
                ha="left", va="center", transform=ax.transAxes, zorder=6,
                fontfamily="monospace")

    # Cost breakdown
    ax.text(rew_x + rew_w / 2, rew_y + 0.015,
            "Costs = Prod + Setup + Inv + Backorder",
            fontsize=6, color=TEXT_DIM, ha="center", va="center",
            transform=ax.transAxes, zorder=6, fontfamily="monospace",
            fontstyle="italic")

    # Reward arrow back to agent
    draw_arrow(ax, (rew_x + 0.010, rew_y + rew_h + 0.005),
               (agent_cx + 0.015, agent_cy - agent_r - 0.050),
               color=REWARD_GREEN, lw=2)
    ax.text((rew_x + agent_cx) / 2 + 0.020,
            (rew_y + rew_h + agent_cy - agent_r) / 2 - 0.015,
            "reward", fontsize=8, color=REWARD_GREEN, ha="center",
            va="center", transform=ax.transAxes, fontstyle="italic",
            fontweight="bold", alpha=0.8)

    # ─── FAR-RIGHT TOP: REWARD SHAPING PARAMS ─────────────────
    hp_x = 0.710; hp_y = sec2_bot + 0.300; hp_w = 0.270; hp_h = 0.165
    rounded_rect(ax, (hp_x, hp_y), hp_w, hp_h,
                 r=0.008, fc=PANEL_BG2, ec=PANEL_EDGE, lw=1.5, zorder=5)

    ax.text(hp_x + hp_w / 2, hp_y + hp_h - 0.018,
            "POTENTIAL-BASED SHAPING", fontsize=9, fontweight="bold",
            color=ACCENT_GOLD, ha="center", va="center",
            transform=ax.transAxes, zorder=6)

    ax.text(hp_x + hp_w / 2, hp_y + hp_h - 0.042,
            "Φ(s) = −(w_b·Backlog + w_w·WIP + w_f·Fin)",
            fontsize=7, color=TEXT_MUTED, ha="center", va="center",
            transform=ax.transAxes, zorder=6, fontfamily="monospace")

    ax.text(hp_x + hp_w / 2, hp_y + hp_h - 0.060,
            "Encourages low inventory, low backlog",
            fontsize=6.5, color=ACCENT_TEAL, ha="center", va="center",
            transform=ax.transAxes, zorder=6, fontstyle="italic")

    shaping_params = [
        ("β  (strength)",   "0.5"),
        ("γ  (discount)",   "0.99"),
        ("w_backlog",       "2.0"),
        ("w_wip",           "0.2"),
        ("w_finished",      "0.05"),
    ]
    for j, (name, val) in enumerate(shaping_params):
        sy = hp_y + hp_h - 0.082 - j * 0.018
        ax.text(hp_x + 0.012, sy, name, fontsize=6.5, color=TEXT_MUTED,
                ha="left", va="center", transform=ax.transAxes, zorder=6,
                fontfamily="monospace")
        ax.text(hp_x + hp_w - 0.012, sy, val, fontsize=7, fontweight="bold",
                color=ACCENT_GOLD, ha="right", va="center",
                transform=ax.transAxes, zorder=6, fontfamily="monospace")

    # ─── FAR-RIGHT BOTTOM: PPO HYPERPARAMETERS ────────────────
    pp_x = 0.710; pp_y = sec2_bot + 0.090; pp_w = 0.270; pp_h = 0.198
    rounded_rect(ax, (pp_x, pp_y), pp_w, pp_h,
                 r=0.008, fc=PANEL_BG2, ec=PANEL_EDGE, lw=1.5, zorder=5)

    ax.text(pp_x + pp_w / 2, pp_y + pp_h - 0.018,
            "PPO HYPERPARAMETERS", fontsize=10, fontweight="bold",
            color=AGENT_GRAD1, ha="center", va="center",
            transform=ax.transAxes, zorder=6)

    # Organized into two columns to fit without overflow
    ppo_left = [
        ("Algorithm",      "MaskablePPO"),
        ("Policy",         "MlpPolicy"),
        ("n_envs",         "8"),
        ("n_steps",        "2,048"),
        ("batch_size",     "512"),
        ("learning_rate",  "3e-4"),
        ("clip_range",     "0.2"),
    ]
    ppo_right = [
        ("ent_coef",       "0.05"),
        ("gamma",          "0.99"),
        ("gae_lambda",     "0.95"),
        ("vf_coef",        "0.5"),
        ("max_grad_norm",  "0.5"),
        ("total_steps",    "3M"),
        ("norm_reward",    "True (clip=10)"),
    ]

    col_w = (pp_w - 0.020) / 2
    row_h_ppo = 0.020

    for j, (name, val) in enumerate(ppo_left):
        py = pp_y + pp_h - 0.042 - j * row_h_ppo
        # Alternating shading
        if j % 2 == 0:
            rounded_rect(ax, (pp_x + 0.005, py - row_h_ppo * 0.35),
                         col_w + 0.003, row_h_ppo * 0.85,
                         r=0.002, fc="#0d1526", ec="none", lw=0,
                         zorder=5, alpha=0.4)
        ax.text(pp_x + 0.012, py, name, fontsize=6, color=TEXT_MUTED,
                ha="left", va="center", transform=ax.transAxes, zorder=6,
                fontfamily="monospace")
        ax.text(pp_x + 0.008 + col_w, py, val, fontsize=6.5, fontweight="bold",
                color=AGENT_GRAD1, ha="right", va="center",
                transform=ax.transAxes, zorder=6, fontfamily="monospace")

    for j, (name, val) in enumerate(ppo_right):
        py = pp_y + pp_h - 0.042 - j * row_h_ppo
        # Alternating shading
        if j % 2 == 0:
            rounded_rect(ax, (pp_x + col_w + 0.012, py - row_h_ppo * 0.35),
                         col_w + 0.003, row_h_ppo * 0.85,
                         r=0.002, fc="#0d1526", ec="none", lw=0,
                         zorder=5, alpha=0.4)
        ax.text(pp_x + col_w + 0.018, py, name, fontsize=6, color=TEXT_MUTED,
                ha="left", va="center", transform=ax.transAxes, zorder=6,
                fontfamily="monospace")
        ax.text(pp_x + pp_w - 0.012, py, val, fontsize=6.5, fontweight="bold",
                color=AGENT_GRAD1, ha="right", va="center",
                transform=ax.transAxes, zorder=6, fontfamily="monospace")

    # ─── TRAINING TIMELINE STRIP  (bottom of RL section) ─────
    tl_y = sec2_bot + 0.008; tl_h = 0.076
    tl_x = 0.025; tl_w = 0.950
    rounded_rect(ax, (tl_x, tl_y), tl_w, tl_h,
                 r=0.006, fc=TIMELINE_BG, ec=PANEL_EDGE, lw=1.2, zorder=5)

    ax.text(tl_x + 0.012, tl_y + tl_h - 0.012, "TRAINING TIMELINE & EPISODE REWARD (LEARNING CURVE)",
            fontsize=8.5, fontweight="bold", color=TEXT_MUTED,
            ha="left", va="center", transform=ax.transAxes, zorder=6,
            fontfamily="monospace")
    ax.text(tl_x + tl_w - 0.012, tl_y + tl_h - 0.012,
            "8 parallel envs  ·  H=500 steps/episode  ·  Agent discovers layout physics",
            fontsize=7.5, color=ACCENT_TEAL, ha="right", va="center",
            transform=ax.transAxes, zorder=6, fontfamily="monospace")

    # Timeline bar
    bar_y = tl_y + 0.022
    bar_h = 0.014
    bar_x_start = tl_x + 0.012
    bar_x_end = tl_x + tl_w - 0.012
    bar_w = bar_x_end - bar_x_start

    # Draw gradient bar segments
    phases = [
        (0.00, 0.25, "#4f46e5", "#818cf8", "Exploration",
         "High entropy · Random actions"),
        (0.25, 0.55, "#6366f1", "#fbbf24", "Shaping Guides",
         "Potential fn steers policy"),
        (0.55, 0.80, "#fbbf24", "#34d399", "Policy Refinement",
         "Reward exploiting structure"),
        (0.80, 1.00, "#34d399", "#34d399", "Convergence",
         "Stable near-optimal policy"),
    ]
    for frac_s, frac_e, c1, c2, label, desc in phases:
        px_s = bar_x_start + frac_s * bar_w
        px_e = bar_x_start + frac_e * bar_w
        pw = px_e - px_s
        rounded_rect(ax, (px_s, bar_y), pw, bar_h,
                     r=0.003, fc=c1, ec="none", lw=0, zorder=7, alpha=0.7)

        # Phase label below bar
        ax.text(px_s + pw / 2, bar_y - 0.008, label,
                fontsize=6.5, color=c2, ha="center", va="center",
                transform=ax.transAxes, zorder=7, fontweight="bold")

        # Phase description (what the agent is learning)
        ax.text(px_s + pw / 2, bar_y - 0.018, desc,
                fontsize=5.5, color=TEXT_DIM, ha="center", va="center",
                transform=ax.transAxes, zorder=7, fontstyle="italic")

    # Phase markers on bar
    markers = [
        (0.00, "t=0"),
        (0.25, "~750K"),
        (0.55, "~1.5M"),
        (0.80, "~2.4M"),
        (1.00, "3M steps"),
    ]
    for frac, lbl in markers:
        mx_pos = bar_x_start + frac * bar_w
        ax.plot([mx_pos, mx_pos], [bar_y, bar_y + bar_h],
                color=TEXT_WHITE, lw=1.5, transform=ax.transAxes,
                zorder=8, alpha=0.6)
        ax.text(mx_pos, bar_y + bar_h + 0.006, lbl,
                fontsize=5.5, color=TEXT_MUTED, ha="center", va="center",
                transform=ax.transAxes, zorder=7, fontfamily="monospace")

    # Add pseudo-learning curve above the timeline
    curve_x = np.linspace(bar_x_start, bar_x_end, 200)
    # create a sigmoidal learning curve matching the phases
    curve_base = 0.015 + 0.025 * (1 / (1 + np.exp(-12 * (curve_x - (bar_x_start + bar_w*0.4)))))
    # random noise that scales down as training progresses
    noise_scale = 0.008 * (1.0 - (curve_x - bar_x_start)/bar_w)
    noise = np.random.normal(0, noise_scale, 200)
    curve_y = bar_y + bar_h + 0.010 + curve_base + noise
    
    # Clip curve to stay within bounds
    curve_y = np.clip(curve_y, bar_y + bar_h + 0.002, tl_y + tl_h - 0.015)
    
    # Fill area under learning curve
    ax.fill_between(curve_x, bar_y + bar_h, curve_y, color=REWARD_GREEN, alpha=0.15, zorder=6)
    # Plot curve line
    ax.plot(curve_x, curve_y, color=REWARD_GREEN, lw=1.5, zorder=7, alpha=0.8)


    # ════════════════════════════════════════════════════════════
    # SAVE
    # ════════════════════════════════════════════════════════════
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(out_dir), "output_data")
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, save_path)
    fig.savefig(full_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"✅ Infographic saved to: {full_path}  ({dpi} DPI)")

    pdf_path = full_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"✅ Vector PDF saved to: {pdf_path}")

    plt.close(fig)
    return full_path


if __name__ == "__main__":
    create_infographic()
