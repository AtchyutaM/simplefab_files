"""
Schedule anatomy: what each method actually does, not just what it scores.

The centrepiece is a CUMULATIVE FLOW DIAGRAM. Each curve counts units that have
passed one milestone — arrived, started at a tool, finished at a tool, shipped —
so the geometry carries the information directly:

    vertical gap between two curves  = units waiting at that stage right now
    horizontal gap between two curves = how long a unit sat there

One chart therefore answers all of: when material arrives, when it starts
processing, how deep the queue in front of each tool is, when processing ends,
and when finished goods leave the line.

Below it, tool activity as rolling utilisation rather than raw bars — at 2,688
ticks a 2-tick litho job is 0.07% of the width, so individual bars merge into a
meaningless block. A zoomed week at the bottom restores the real bars, where
changeovers are visible as gaps.

    python -m tensorfab.twoprod.plot_schedules
"""
from __future__ import annotations
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).parent
NPZ = HERE / "schedules" / "schedules.npz"
OUT = HERE / "schedules"

# deck palette
C = dict(bg="#FAF7F0", panel="#FFFFFF", ink="#1A1712", ink2="#4A4238", ink3="#7D7264",
         rule="#E2D9C9", diff="#A87400", litho="#2C63C4", fg="#0A7D5C",
         hot="#B92A45", alt="#7A3FB8", accent="#A87400")
QUEUE_SHADES = ["#E8DFCB", "#D8CCB0", "#C7B894", "#B5A279"]


def style(ax, title=None, ylabel=None, xlab=False):
    ax.set_facecolor(C["panel"])
    for s in ax.spines.values():
        s.set_color(C["rule"]); s.set_linewidth(0.8)
    ax.tick_params(colors=C["ink3"], labelsize=8.5, length=3)
    ax.grid(True, color=C["rule"], alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=C["ink"], fontsize=11, fontweight="600",
                     loc="left", pad=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["ink2"], fontsize=9)
    if xlab:
        ax.set_xlabel("tick  (2,688 = four weeks)", color=C["ink2"], fontsize=9)


def cfd(ax, d, m, H, init_fin, show_legend=False):
    """Cumulative flow: one nested band per stage."""
    t = np.arange(H)
    arrived = np.cumsum(d["arrivals"].sum(-1))[:H]
    su, du = d[m + "__started_units"], d[m + "__done_units"]
    shipped = d[m + "__cum_ship"].sum(-1)
    avail = du[:, 3] + init_fin

    curves = [arrived, su[:, 0], du[:, 0], su[:, 1], du[:, 1],
              su[:, 2], du[:, 2], su[:, 3], avail, shipped]
    labels = ["q0 waiting", "in DIFF-1", "q1 waiting", "in LITHO-1", "q2 waiting",
              "in DIFF-2", "q3 waiting", "in LITHO-2", "finished goods"]
    cols = [QUEUE_SHADES[0], C["diff"], QUEUE_SHADES[1], C["litho"],
            QUEUE_SHADES[2], C["diff"], QUEUE_SHADES[3], C["litho"], C["fg"]]
    alphas = [1, .55, 1, .55, 1, .40, 1, .40, .45]

    for i in range(len(curves) - 1):
        ax.fill_between(t, curves[i + 1], curves[i], color=cols[i],
                        alpha=alphas[i], linewidth=0, zorder=2)
    ax.plot(t, arrived, color=C["ink"], lw=1.5, zorder=4, label="raw material in")
    ax.plot(t, shipped, color=C["fg"], lw=1.8, zorder=4, label="shipped")
    for w in range(672, H, 672):
        ax.axvline(w, color=C["ink3"], lw=0.8, ls=(0, (4, 3)), alpha=0.5, zorder=1)
    style(ax, ylabel="cumulative units")
    ax.set_xlim(0, H); ax.set_ylim(0, arrived.max() * 1.03)
    if show_legend:
        handles = [Patch(facecolor=c, alpha=a, label=l)
                   for c, a, l in zip(cols, alphas, labels)]
        handles += [Line2D([], [], color=C["ink"], lw=1.5, label="raw material in"),
                    Line2D([], [], color=C["fg"], lw=1.8, label="shipped")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.08),
                  fontsize=7.2, ncol=5,
                  frameon=True, facecolor=C["panel"], edgecolor=C["rule"],
                  labelcolor=C["ink2"], framealpha=0.94, borderpad=0.6)
        ax.set_zorder(3)


def queues(ax, d, m, H, ymax):
    """Queue depths stacked. In the flow diagram these are thin slivers against
    a 600-unit axis; plotted directly, the difference between the methods in how
    much material they leave standing becomes obvious."""
    t = np.arange(H)
    q = d[m + "__q"].sum(-1)                                  # [H, M] over products
    labels = ["q0 (raw)", "q1", "q2", "q3"]
    ax.stackplot(t, [q[:, i] for i in range(4)], colors=QUEUE_SHADES,
                 labels=labels, edgecolor="none")
    ax.plot(t, q.sum(-1), color=C["ink2"], lw=1.1)
    for w in range(672, H, 672):
        ax.axvline(w, color=C["ink3"], lw=0.8, ls=(0, (4, 3)), alpha=0.5)
    style(ax, ylabel="units queued")
    ax.set_xlim(0, H); ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=7.2, ncol=4, frameon=True,
              facecolor=C["panel"], edgecolor=C["rule"], labelcolor=C["ink2"],
              framealpha=0.94)


def activity(ax, d, m, H, win=96):
    """Rolling fraction of time each tool is busy, plus changeover marks."""
    busy = d[m + "__busy"]                                   # [H, M]
    curp = d[m + "__curp"]                                   # [H, M]
    names = ["DIFF-1", "LITHO-1", "DIFF-2", "LITHO-2"]
    cols = [C["diff"], C["litho"], C["diff"], C["litho"]]
    k = np.ones(win) / win
    for i in range(4):
        u = np.convolve(busy[:, i].astype(float), k, mode="same")
        ax.fill_between(np.arange(H), 3 - i, 3 - i + u * 0.86,
                        color=cols[i], alpha=0.30 if i % 2 else 0.42, linewidth=0)
        ax.plot(np.arange(H), 3 - i + u * 0.86, color=cols[i], lw=1.0)
        # a changeover = the product in the tool differs from the previous job
        run = curp[:, i]
        chg = np.where((run[1:] != run[:-1]) & (run[1:] > 0))[0] + 1
        ax.scatter(chg, np.full(chg.shape, 3 - i + 0.93), s=2.2,
                   color=C["hot"], alpha=0.75, marker="|", linewidths=0.7)
        ax.text(-70, 3 - i + 0.38, names[i], ha="right", va="center",
                fontsize=8, color=C["ink2"], family="monospace")
    for w in range(672, H, 672):
        ax.axvline(w, color=C["ink3"], lw=0.8, ls=(0, (4, 3)), alpha=0.5)
    style(ax, ylabel="tool busy (rolling)")
    ax.set_xlim(0, H); ax.set_ylim(-0.1, 4.05); ax.set_yticks([])


def stocks(ax, d, m, H, ylim=None):
    fin = d[m + "__qfin"].sum(-1); back = d[m + "__backlog"].sum(-1)
    t = np.arange(H)
    ax.fill_between(t, 0, fin, color=C["fg"], alpha=0.30, linewidth=0)
    ax.plot(t, fin, color=C["fg"], lw=1.4, label="finished goods")
    ax.fill_between(t, 0, -back, color=C["hot"], alpha=0.25, linewidth=0)
    ax.plot(t, -back, color=C["hot"], lw=1.4, label="unmet demand")
    ax.axhline(0, color=C["ink3"], lw=0.8)
    for w in range(672, H, 672):
        ax.axvline(w, color=C["ink3"], lw=0.8, ls=(0, (4, 3)), alpha=0.5)
    style(ax, ylabel="units")
    ax.set_xlim(0, H)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor=C["panel"],
              edgecolor=C["rule"], labelcolor=C["ink2"])


def zoom(ax, d, m, t0, t1):
    """Real bars for a short window — where changeovers become visible."""
    curp = d[m + "__curp"]
    names = ["DIFF-1", "LITHO-1", "DIFF-2", "LITHO-2"]
    pcol = {1: C["litho"], 2: C["alt"]}
    for i in range(4):
        run = curp[t0:t1, i]
        s = 0
        while s < len(run):
            if run[s] == 0:
                s += 1; continue
            e = s
            while e + 1 < len(run) and run[e + 1] == run[s]:
                e += 1
            ax.broken_barh([(t0 + s, e - s + 1)], (3 - i + 0.12, 0.72),
                           facecolors=pcol[int(run[s])], edgecolor="none", alpha=0.9)
            s = e + 1
        ax.text(t0 - (t1 - t0) * 0.035, 3 - i + 0.48, names[i], ha="right",
                va="center", fontsize=8, color=C["ink2"], family="monospace")
    style(ax, ylabel=None)
    ax.set_xlim(t0, t1); ax.set_ylim(-0.05, 4.05); ax.set_yticks([])
    ax.set_xlabel(f"tick  (zoom: {t0}\u2013{t1})", color=C["ink2"], fontsize=9)
    ax.legend(handles=[Patch(facecolor=pcol[1], label="running P0"),
                       Patch(facecolor=pcol[2], label="running P1")],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=7.6, ncol=2,
              frameon=False, labelcolor=C["ink2"])


def main():
    d = np.load(NPZ)
    H, init_fin = int(d["H"]), 16
    methods = [("cpsat", "CP-SAT  —  600 s"), ("gradst", "Grad ST  —  36 s")]

    qmax = max(d[m + "__q"].sum(-1).sum(-1).max() for m, _ in methods) * 1.08
    lo = min(-d[m + "__backlog"].sum(-1).max() for m, _ in methods) * 1.1
    hi = max(d[m + "__qfin"].sum(-1).max() for m, _ in methods) * 1.25
    fig = plt.figure(figsize=(17.5, 16.4), facecolor=C["bg"])
    gs = fig.add_gridspec(5, 2, height_ratios=[2.9, 1.25, 1.45, 1.0, 1.0],
                          hspace=0.42, wspace=0.13,
                          left=0.065, right=0.985, top=0.935, bottom=0.045)
    for j, (m, title) in enumerate(methods):
        profit = float(d[m + "__profit"])
        chg = int(d[m + "__setups"][-1].sum())
        ax = fig.add_subplot(gs[0, j])
        cfd(ax, d, m, H, init_fin, show_legend=(j == 0))
        ax.set_title(f"{title}      ${profit:,.0f}      {chg} changeovers",
                     color=C["ink"], fontsize=13, fontweight="700", loc="left", pad=10)
        queues(fig.add_subplot(gs[1, j]), d, m, H, qmax)
        activity(fig.add_subplot(gs[2, j]), d, m, H)
        stocks(fig.add_subplot(gs[3, j]), d, m, H, ylim=(lo, hi))
        zoom(fig.add_subplot(gs[4, j]), d, m, 1344, 1344 + 340)

    fig.suptitle("What the two schedules actually do differently",
                 color=C["ink"], fontsize=17, fontweight="700", x=0.065, ha="left", y=0.978)
    fig.text(0.065, 0.956,
             "Two products, four tools, 80% utilisation.  In the flow diagram the vertical gap "
             "between two curves is the queue at that stage; the horizontal gap is how long a unit "
             "waited there.  Red ticks mark changeovers.",
             color=C["ink3"], fontsize=9.8, ha="left")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"schedule_anatomy.{ext}", dpi=150,
                    facecolor=C["bg"], bbox_inches="tight")
    print(f"wrote {OUT/'schedule_anatomy.png'} and .svg")


if __name__ == "__main__":
    main()
