"""
How each method feeds the line, across utilisations.

The dashboards show what happens inside the line. This asks the question one
step earlier: at what rate does each method push raw material in, and how does
that change when the line goes from slack to nearly saturated?

Release is measured at M0 -- the gating diffusion tool -- in units, so a job
counts for the four wafers it actually consumes. Two views per utilisation:

  left    cumulative units released, against the cumulative demand staircase.
          Slope is release rate; vertical distance from the demand line is how
          far ahead of (or behind) the order book a method is running.

  right   release rate in a rolling window, which is where pacing shows up:
          a flat line is a metered feed, spikes are burst-and-wait.

Writes release_patterns.png plus the per-run numbers the deck quotes.

    python -m tensorfab.twoprod.release_patterns
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfab.twoprod.naming import parse, sched_name  # noqa: E402

HERE = pathlib.Path(__file__).parent
SCH = HERE / "schedules"
DATA = HERE.parent / "presentation_data.json"

C = {"bg": "#0b0f19", "card": "#131a28", "text": "#e2e8f0", "dim": "#94a3b8",
     "grid": "#1e293b"}
METHOD = {"cpsat": ("CP-SAT", "#22d3ee"),
          "gradst": ("Grad ST", "#f59e0b"),
          "commander": ("Commander", "#a78bfa")}
WEEK = 672
WIN = 336          # half a week; smooths batch granularity without hiding the
                   # weekly rhythm. Shorter windows turn the rate panels into a
                   # comb of single-job spikes that hides the trend.


def released_units(z, method: str) -> np.ndarray:
    """Per-tick units pushed into M0 by `method`."""
    cum = z[f"{method}__started_jobs"][:, 0, :]        # [H, P] cumulative jobs
    per_tick = np.diff(cum, axis=0, prepend=0).sum(-1)  # jobs started each tick
    return per_tick * int(z["take"][0])                 # units per job at M0


def rolling(x: np.ndarray, win: int) -> np.ndarray:
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same") * win        # units per window


def sty(ax, title=None, ylab=None):
    ax.set_facecolor(C["card"])
    for sp in ax.spines.values():
        sp.set_color(C["grid"])
    ax.tick_params(colors=C["dim"], labelsize=10)
    ax.grid(True, color=C["grid"], lw=1, alpha=0.75)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=C["text"], fontsize=13, fontweight="bold",
                     loc="left", pad=9)
    if ylab:
        ax.set_ylabel(ylab, color=C["dim"], fontsize=11)


def build(runs: list[tuple[float, float]]) -> dict:
    n = len(runs)
    fig, axes = plt.subplots(n, 2, figsize=(21, 4.4 * n), facecolor=C["bg"],
                             squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.965, top=0.90, bottom=0.075,
                        hspace=0.36, wspace=0.14)
    stats = {}

    for r, (alpha, util) in enumerate(runs):
        z = np.load(SCH / sched_name(alpha, util))
        H = int(z["H"])
        t = np.arange(H)
        demand_cum = z["demand"].sum(-1).cumsum()
        arrive_cum = z["arrivals"].sum(-1).cumsum()
        axL, axR = axes[r][0], axes[r][1]

        axL.plot(t, arrive_cum, color=C["dim"], lw=1.4, ls=":",
                 label="raw material available")
        axL.plot(t, demand_cum, color=C["text"], lw=1.6, ls="--",
                 label="cumulative demand")

        row = {}
        for meth, (lab, col) in METHOD.items():
            if f"{meth}__started_jobs" not in z.files:
                continue
            rel = released_units(z, meth)
            cum = rel.cumsum()
            # widths descend so a curve underneath still shows where they coincide
            lw = {"cpsat": 3.0, "gradst": 2.2, "commander": 1.6}[meth]
            axL.plot(t, cum, color=col, lw=lw, label=lab, solid_capstyle="round")
            axR.plot(t, rolling(rel, WIN), color=col, lw=lw, alpha=0.9)

            # how far ahead of the order book it runs, on average, in units
            lead = float(np.mean(cum - demand_cum))
            first_week = int(cum[WEEK - 1])
            # burstiness: spread of the smoothed rate relative to its own mean
            rr = rolling(rel, WIN)
            active = rr[rr > 0]
            cv = float(active.std() / active.mean()) if active.size else 0.0
            row[meth] = {"released": int(cum[-1]), "lead_units": round(lead, 1),
                         "first_week": first_week, "rate_cv": round(cv, 3),
                         # how much material it puts in relative to what was
                         # ordered -- above 1.0 is building for nobody
                         "vs_demand": round(float(cum[-1] / demand_cum[-1]), 3)}

        # end-of-horizon ratio, nudged apart where methods finish close together
        # (at high load they nearly coincide and the labels would overprint)
        labels = sorted(((row[m]["released"], row[m]["vs_demand"], METHOD[m][1])
                         for m in row if m in METHOD), key=lambda x: x[0])
        gap = 0.045 * max(arrive_cum[-1], 1)
        placed = []
        for y, ratio, col in labels:
            if placed and y - placed[-1][0] < gap:
                y = placed[-1][0] + gap
            placed.append((y, ratio, col))
        for y, ratio, col in placed:
            axL.annotate(f"{ratio:.2f}x", xy=(H, y), xytext=(6, 0),
                         textcoords="offset points", color=col, fontsize=11,
                         fontweight="bold", va="center", ha="left",
                         annotation_clip=False)

        for ax in (axL, axR):
            for w in range(WEEK, H, WEEK):
                ax.axvline(w, color=C["grid"], lw=2, zorder=0)
            ax.set_xlim(0, H)
        sty(axL, f"{util:.0%} utilisation  —  cumulative units released",
            "units")
        sty(axR, f"{util:.0%} utilisation  —  release rate "
                 f"(units per {WIN}-tick window)", "units / window")
        if r == 0:
            axL.legend(loc="upper left", fontsize=10.5, facecolor=C["card"],
                       edgecolor=C["grid"], labelcolor=C["text"], framealpha=0.95,
                       ncol=2)
        if r == n - 1:
            for ax in (axL, axR):
                ax.set_xlabel("Time horizon (ticks;  672 ticks = one week)",
                              color=C["dim"], fontsize=11)
        stats[f"{util:.2f}"] = {"alpha": alpha, "util": util,
                                "demand": int(demand_cum[-1]), **row}

    fig.suptitle("How each method feeds the line  —  50/50 product mix, "
                 "three utilisations", color=C["text"], fontsize=19,
                 fontweight="bold", x=0.055, ha="left", y=0.972)
    fig.text(0.055, 0.938,
             "Release measured at M0, the gating diffusion tool, in units. "
             "Left: cumulative, against the demand staircase, labelled with total "
             "released as a multiple of demand. Right: the same releases as a rolling rate.",
             color=C["dim"], fontsize=12, ha="left")
    out = SCH / "release_patterns.png"
    fig.savefig(out, dpi=165, facecolor=C["bg"])
    plt.close(fig)
    print(f"  wrote {out.name}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--utils", type=float, nargs="+", default=[0.60, 0.80, 0.92])
    args = ap.parse_args()

    runs = [(args.alpha, u) for u in sorted(args.utils)
            if (SCH / sched_name(args.alpha, u)).exists()]
    missing = [u for u in args.utils if (args.alpha, u) not in runs]
    for u in missing:
        print(f"  skip {u:.0%}: {sched_name(args.alpha, u)} not generated yet")
    if not runs:
        print("nothing to plot")
        return 1

    stats = build(runs)
    d = json.loads(DATA.read_text())
    d.setdefault("twoprod", {})["release_patterns"] = stats
    DATA.write_text(json.dumps(d, indent=2))

    for u, row in stats.items():
        print(f"\nutil {float(u):.0%}   demand {row['demand']} units")
        for meth, (lab, _) in METHOD.items():
            if meth not in row:
                continue
            m = row[meth]
            print(f"   {lab:10s} released {m['released']:>4d} units "
                  f"= {m['vs_demand']:.2f}x demand   "
                  f"mean lead {m['lead_units']:>7.1f} units   "
                  f"rate CV {m['rate_cv']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
