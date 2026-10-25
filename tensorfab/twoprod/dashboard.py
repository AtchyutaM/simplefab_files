"""
Full-detail schedule dashboard for the two-product line.

Follows the layout used for the single-product PPO runs — KPI strip, machine
Gantt with item tracing, per-queue work-in-process, finished goods against
backlog, and a cumulative financial breakdown — extended for two products.

The one real design problem is the Gantt. With one product the original used a
distinct colour per batch so individual jobs could be traced. With two products
that alone turns into a blob, because nothing separates the recipes. So colour
carries two things at once:

    HUE FAMILY  ->  which product   (P0 cool / cyan-blue, P1 warm / amber-red)
    SHADE       ->  which batch     (cycled within the family)

Product identity therefore reads at a glance from across the room, while
individual jobs stay traceable up close. Rendered wide and at high DPI so a
2-tick litho job is still several pixels across.

    python -m tensorfab.twoprod.dashboard --alpha 0.5 --method cpsat
    python -m tensorfab.twoprod.dashboard --all
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

from simplefab.config import make_common_config
from tensorfab.twoprod.naming import dash_name, parse, run_label, sched_name
from tensorfab.twoprod.spec import spec2p_from_common_config

HERE = pathlib.Path(__file__).parent
SCH = HERE / "schedules"

C = {"bg": "#0b0f19", "card": "#131a28", "text": "#e2e8f0", "dim": "#94a3b8",
     "grid": "#1e293b", "arrival": "#06b6d4", "demand": "#ef4444",
     "ship": "#10b981", "profit": "#fbbf24", "wip": "#38bdf8",
     "fin": "#10b981", "back": "#ef4444", "prod": "#8b5cf6",
     "hold": "#f59e0b", "setup": "#ec4899"}

# product colour families: P0 cool, P1 warm — shade cycles within each
P0_CMAP = LinearSegmentedColormap.from_list("p0", ["#0ea5e9", "#22d3ee", "#2563eb",
                                                   "#38bdf8", "#0891b2", "#60a5fa"])
P1_CMAP = LinearSegmentedColormap.from_list("p1", ["#f59e0b", "#ef4444", "#fb923c",
                                                   "#f43f5e", "#fbbf24", "#e11d48"])
NSHADE = 14
SHADES = {1: [P0_CMAP(i / (NSHADE - 1)) for i in range(NSHADE)],
          2: [P1_CMAP(i / (NSHADE - 1)) for i in range(NSHADE)]}
MACH = ["M0 DIFF (batch)", "M1 LITHO (single)", "M2 DIFF (batch)", "M3 LITHO (single)"]


def sty(ax, title=None, ylabel=None):
    ax.set_facecolor(C["card"])
    for s in ax.spines.values():
        s.set_color(C["grid"]); s.set_linewidth(1)
    ax.tick_params(colors=C["dim"], labelsize=11)
    ax.grid(True, color=C["grid"], alpha=0.75, linewidth=1)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=C["text"], fontsize=17, fontweight="bold",
                     loc="left", pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["dim"], fontsize=12, fontweight="bold")


def week_marks(ax, H, y=None, lab=True):
    for w in range(0, H + 1, 672):
        ax.axvline(w, color=C["grid"], lw=2, zorder=0)
        if lab and w < H:
            ax.text(w + 18, ax.get_ylim()[1] * 0.94 if y is None else y,
                    f"Wk {w // 672 + 1}", color=C["dim"], alpha=0.75,
                    fontsize=11, fontweight="bold")


def jobs_of(d, m, spec):
    """Reconstruct every job as (machine, product, start, duration)."""
    sj = d[m + "__started_jobs"]                                   # [H, M, P] cumulative
    per = np.diff(np.concatenate([np.zeros((1,) + sj.shape[1:]), sj]), axis=0)
    lastp = d[m + "__lastp"]
    out = []
    for t in range(per.shape[0]):
        for mm in range(spec.n_machines):
            for p in range(spec.n_products):
                if per[t, mm, p] > 0:
                    dur = int(spec.setup_time[mm, int(lastp[t, mm]), p] + spec.pt[mm, p])
                    out.append((mm, p, t, dur))
    return out


def draw(d, m, spec, title, out_path):
    H = int(d["H"])
    arrivals, demand = d["arrivals"], d["demand"]
    rev, pc = spec.revenue, spec.prod_cost * spec.take[:, None]

    sj = d[m + "__started_jobs"]; q = d[m + "__q"]
    qfin = d[m + "__qfin"]; back = d[m + "__backlog"]
    cum_ship = d[m + "__cum_ship"]; setups = d[m + "__setups"]

    # cumulative money
    c_rev = (cum_ship * rev).sum(-1)
    c_prod = (sj * pc[None]).sum((1, 2))
    c_set = (setups * spec.setup_cost[None]).sum(-1)
    c_fg = (np.cumsum(qfin, 0) * spec.fg_holding).sum(-1)
    c_wip = (np.cumsum(q, 0) * spec.wip_holding[None]).sum((1, 2))
    lag = spec.demand_lag_cum
    late = np.maximum(lag - cum_ship, 0)
    c_bo = (np.cumsum(late, 0) * spec.backorder).sum(-1)
    net = c_rev - (c_prod + c_set + c_fg + c_wip + c_bo)

    fig = plt.figure(figsize=(30, 27), facecolor=C["bg"])
    # The KPI strip is drawn in figure coordinates above the grid, so the grid
    # itself holds only the five plotting rows -- no empty spacer row, which is
    # what previously opened a dead band between the numbers and the Gantt.
    gs = GridSpec(4, 1, height_ratios=[2.5, 2.2, 1.25, 1.5],
                  hspace=0.30, left=0.052, right=0.986, top=0.903, bottom=0.058)

    # ---- KPI strip -------------------------------------------------------
    tot_d = int(demand.sum())
    kpis = [("TOTAL DEMAND", f"{tot_d}", C["text"]),
            ("UNITS SHIPPED", f"{int(cum_ship[-1].sum())}", C["ship"]),
            ("REVENUE", f"${c_rev[-1]:,.0f}", C["ship"]),
            ("TOTAL COSTS", f"${c_rev[-1]-net[-1]:,.0f}", C["back"]),
            ("CHANGEOVERS", f"{int(setups[-1].sum())}", C["setup"]),
            ("NET PROFIT", f"${net[-1]:,.0f}", C["profit"])]
    for i, (k, v, col) in enumerate(kpis):
        x = 0.055 + i * 0.158
        fig.text(x, 0.963, k, color=C["dim"], fontsize=13, fontweight="bold",
                 ha="left", family="monospace")
        fig.text(x, 0.940, v, color=col, fontsize=30, fontweight="bold", ha="left")

    # ---- Gantt -----------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    # y grows upward, so map each tool to a row with M3 at the top
    ypos = {3: 3, 0: 2, 1: 1, 2: 0}
    seen = {mm: 0 for mm in range(spec.n_machines)}
    for (mm, p, t, dur) in jobs_of(d, m, spec):
        y = ypos[mm]
        col = SHADES[p + 1][seen[mm] % NSHADE]; seen[mm] += 1
        ax.broken_barh([(t, dur)], (y + 0.12, 0.76), facecolors=col,
                       edgecolors=C["bg"], linewidth=0.35, zorder=3)
    ax.set_yticks([i + 0.5 for i in range(4)])
    ax.set_yticklabels([MACH[2], MACH[1], MACH[0], MACH[3]], color=C["text"],
                       fontsize=13, fontweight="bold")
    ax.set_ylim(0, 4.6); ax.set_xlim(0, H)
    sty(ax, "Machine scheduling and item tracing   —   hue = product, shade = individual batch",
        "Tool")
    for w in range(0, H, 672):
        a = int(arrivals[w].sum())
        if a:
            ax.annotate(f"+{a} raw", (w, -0.40), color=C["arrival"], fontsize=12,
                        fontweight="bold", ha="left", annotation_clip=False)
        dd = int(demand[w].sum())
        if dd:
            ax.annotate(f"\u2212{dd} demand", (w, 4.42), color=C["demand"], fontsize=12,
                        fontweight="bold", ha="left", annotation_clip=False)
    week_marks(ax, H, y=4.20)
    ax.legend(handles=[Patch(facecolor=SHADES[1][2], label="Product P0  (16t batch, \\$80/unit)"),
                       Patch(facecolor=SHADES[2][2], label="Product P1  (20t batch, \\$100/unit)")],
              loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=12,
              facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"],
              framealpha=0.96)

    # ---- work in process, one row per queue ------------------------------
    inner = gs[1].subgridspec(4, 1, hspace=0.12)
    for i in range(4):
        axq = fig.add_subplot(inner[i])
        t = np.arange(H)
        axq.fill_between(t, 0, q[:, i, 0], color=SHADES[1][2], alpha=0.75, lw=0,
                         label="P0" if i == 0 else None)
        axq.fill_between(t, q[:, i, 0], q[:, i].sum(-1), color=SHADES[2][2],
                         alpha=0.75, lw=0, label="P1" if i == 0 else None)
        axq.plot(t, q[:, i].sum(-1), color=C["text"], lw=1.0, alpha=0.6)
        sty(axq, "Work-in-process waiting at each tool" if i == 0 else None,
            f"queue {i}")
        # headroom above the peak so the annotation never sits on the trace --
        # a saturating queue would otherwise run straight through the label
        qmax = max(4, q[:, i].sum(-1).max() * 1.32)
        axq.set_xlim(0, H); axq.set_ylim(0, qmax)
        # queue 0 carries the P0/P1 legend in the same corner, so drop the peak
        # label clear of it rather than letting the two overlap
        axq.text(0.995, 0.52 if i == 0 else 0.80,
                 f"peak {int(q[:, i].sum(-1).max())}", transform=axq.transAxes,
                 ha="right", va="top", color=C["dim"], fontsize=10.5, family="monospace")
        for w in range(0, H + 1, 672):
            axq.axvline(w, color=C["grid"], lw=2, zorder=0)
        if i == 0:
            axq.legend(loc="upper right", fontsize=11, facecolor=C["card"],
                       edgecolor=C["grid"], labelcolor=C["text"], ncol=2)
        if i < 3:
            axq.set_xticklabels([])

    # ---- finished goods vs backlog ---------------------------------------
    axf = fig.add_subplot(gs[2])
    t = np.arange(H)
    axf.fill_between(t, 0, qfin.sum(-1), color=C["fin"], alpha=0.30, lw=0)
    axf.plot(t, qfin.sum(-1), color=C["fin"], lw=2.4, label="Finished goods on shelf")
    axf.fill_between(t, 0, -back.sum(-1), color=C["back"], alpha=0.28, lw=0)
    axf.plot(t, -back.sum(-1), color=C["back"], lw=2.4, label="Backordered demand")
    axf.axhline(0, color=C["dim"], lw=1.2)
    sty(axf, "Finished goods and demand backlog", "Units")
    axf.set_xlim(0, H)
    for w in range(0, H + 1, 672):
        axf.axvline(w, color=C["grid"], lw=2, zorder=0)
    axf.legend(loc="upper left", fontsize=12, facecolor=C["card"],
               edgecolor=C["grid"], labelcolor=C["text"])

    # ---- money -----------------------------------------------------------
    axm = fig.add_subplot(gs[3])
    axm.fill_between(t, 0, c_rev, color=C["ship"], alpha=0.22, lw=0)
    axm.plot(t, c_rev, color=C["ship"], lw=2.6, label="Cumulative revenue")
    stack, cols, labs = [c_prod, c_set, c_fg, c_wip, c_bo], \
        [C["prod"], C["setup"], C["hold"], C["wip"], C["back"]], \
        ["Production", "Changeover", "Finished-goods holding", "WIP holding", "Backorder"]
    base = np.zeros(H)
    for arr, col, lab in zip(stack, cols, labs):
        axm.fill_between(t, -base, -(base + arr), color=col, alpha=0.80, lw=0, label=lab)
        base = base + arr
    axm.plot(t, net, color=C["profit"], lw=3.0, label="NET PROFIT")
    axm.axhline(0, color=C["dim"], lw=1.2)
    sty(axm, "Cumulative revenue against costs", "Cumulative $")
    axm.set_xlim(0, H)
    for w in range(0, H + 1, 672):
        axm.axvline(w, color=C["grid"], lw=2, zorder=0)
    axm.set_xlabel("Time horizon (ticks;  672 ticks = one week)", color=C["text"],
                   fontsize=14, fontweight="bold")
    axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=11.5, ncol=7,
               facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"],
               framealpha=0.95)

    fig.suptitle(title, color=C["text"], fontsize=26, fontweight="bold",
                 x=0.052, ha="left", y=0.992)
    fig.text(0.052, 0.004,
             "Two products on four tools   |   P0: 16t batch, \\$80/unit, \\$24 cost      "
             "P1: 20t batch, \\$100/unit, \\$28 cost   |   changeover \\$20 + 1 tick on the "
             "litho tools   |   FG holding \\$0.04\u20130.05,  WIP \\$0\u20130.0094,  "
             "backorder \\$0.15\u20130.225 per unit-tick after a one-week grace period",
             color=C["dim"], fontsize=11.5, family="monospace")
    fig.savefig(out_path, dpi=170, facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--util", type=float, default=0.80)
    ap.add_argument("--method", default="cpsat", choices=["cpsat", "gradst", "commander"])
    ap.add_argument("--all", action="store_true",
                    help="every schedule file present, whatever mix or utilisation")
    args = ap.parse_args()

    names = {"cpsat": "CP-SAT", "gradst": "Grad ST", "commander": "Commander"}
    if args.all:
        runs = sorted(parse(f.stem.split("schedules_")[1])
                      for f in SCH.glob("schedules_a*_u*.npz"))
    else:
        runs = [(args.alpha, args.util)]
    meths = ["cpsat", "gradst", "commander"] if args.all else [args.method]

    for a, u in runs:
        f = SCH / sched_name(a, u)
        if not f.exists():
            print(f"skip {run_label(a, u)}: {f.name} not generated yet"); continue
        d = np.load(f)
        common = make_common_config(mode="UNIFORM", alpha=a, utilization=u)
        spec = spec2p_from_common_config(common)
        print(f"{run_label(a, u)}  (demand {spec.demand.sum(0).tolist()})")
        for meth in meths:
            if meth + "__profit" not in d.files:
                continue
            wall = f"{float(d[meth+'__wall']):.0f} s" if meth + "__wall" in d.files else "\u2014"
            title = (f"{names[meth]}   |   {run_label(a, u)}   |   "
                     f"${float(d[meth+'__profit']):,.0f}   |   {wall}")
            draw(d, meth, spec, title, SCH / dash_name(a, u, meth))


if __name__ == "__main__":
    main()
