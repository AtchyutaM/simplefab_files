"""
Pull the comparative numbers off the saved schedules and into the deck data.

Everything the dashboard slides assert is measured here from the replayed
schedules rather than typed in by hand, so the claims on the slides cannot drift
away from the runs that produced them. Re-running this after regenerating any
schedule updates the deck.

    python -m tensorfab.twoprod.dash_stats
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

from tensorfab.twoprod.naming import parse, run_label, sched_name, tag

HERE = pathlib.Path(__file__).parent
SCH = HERE / "schedules"
DATA = HERE.parent / "presentation_data.json"
METHODS = (("cpsat", "CP-SAT"), ("gradst", "Grad-ST"), ("commander", "Commander"))


def stats_for(path: pathlib.Path) -> dict:
    z = np.load(path)
    bound = float(z["cpsat_bound"])
    rows = []
    for key, label in METHODS:
        if f"{key}__profit" not in z.files:
            continue
        q = z[f"{key}__q"]                      # [H, M, P] units waiting
        busy = z[f"{key}__busy"]                # [H, M]
        profit = float(z[f"{key}__profit"])
        mid = q[:, 1:3].sum(-1)                 # the two interior queues
        rows.append({
            "method": label,
            "profit": profit,
            "pct_bound": profit / bound,
            "changeovers": int(z[f"{key}__setups"][-1].sum()),
            # mean units waiting at the two interior queues -- the clearest
            # single number for whether the line is flowing or congesting
            "midline_wip": round(float(mid.mean()), 1),
            "midline_peak": int(mid.sum(-1).max()),
            # units still stuck mid-line when the horizon ends: bought,
            # processed, never sold
            "stranded": int(q[-1, 1:3].sum()),
            # how hard the gating tool is pushed
            "m0_duty": round(float(busy[:, 0].mean()), 3),
            "shipped": int(z[f"{key}__cum_ship"][-1].sum()),
            # finished goods still on the shelf when the horizon closes: built,
            # held, and with the order book already satisfied, never sold
            "fg_left": int(z[f"{key}__qfin"][-1].sum()),
            "overbuilt": int(z[f"{key}__qfin"][-1].sum() + z[f"{key}__q"][-1, 1:3].sum()),
        })
    return {"bound": bound, "wall": float(z["cpsat__wall"]),
            "demand": [int(v) for v in z["demand"].sum(axis=0)], "rows": rows}


def util_study(out: dict, alpha: float = 0.5) -> dict:
    """The utilisation sweep at one mix, assembled from the runs already stat-ed.

    Bottleneck ticks are what M0 must be occupied for to convert the whole order
    book -- demand / batch size x processing time, summed over products -- which
    is the quantity the utilisation label actually refers to.
    """
    import numpy as np
    rows = []
    for t, rec in sorted(out.items(), key=lambda kv: kv[1]["util"]):
        if abs(rec["alpha"] - alpha) > 1e-9:
            continue
        z = np.load(SCH / sched_name(rec["alpha"], rec["util"]))
        take0, pt0 = int(z["take"][0]), z["pt"][0]
        dem = z["demand"].sum(axis=0)
        ticks = float(sum(dem[j] / take0 * float(pt0[j]) for j in range(len(dem))))
        by = {r["method"]: r for r in rec["rows"]}
        gs = by.get("Grad-ST", {}).get("profit")
        cm = by.get("Commander", {}).get("profit")
        cp = by.get("CP-SAT", {}).get("profit")
        margin = (gs - cm) if (gs is not None and cm is not None) else None
        rows.append({
            "tag": t, "util_f": rec["util"], "util": f"{rec['util']:.0%}",
            "demand": int(dem.sum()), "ticks": int(round(ticks)),
            "pct_ceiling": f"{ticks / int(z['H']):.1%}",
            "commander": f"{cm:,.0f}" if cm is not None else "&mdash;",
            "gradst": f"{gs:,.0f}" if gs is not None else "&mdash;",
            "cpsat": f"{cp:,.0f}" if cp is not None else "&mdash;",
            "bound": f"{rec['bound']:,.0f}",
            "margin": (f"{margin:+,.0f}" if margin is not None else "&mdash;"),
            "st_wins": bool(margin is not None and margin > 0),
            "pct_bound": (f"{gs / rec['bound']:.1%}" if gs is not None else "&mdash;"),
            "m0_duty": {m: by[m]["m0_duty"] for m in by},
        })
    return {"alpha": alpha, "rows": rows}


def main() -> int:
    d = json.loads(DATA.read_text())
    out = {}
    for path in sorted(SCH.glob("schedules_a*_u*.npz")):
        t = path.stem.split("schedules_")[1]
        alpha, util = parse(t)
        rec = stats_for(path)
        rec["alpha"], rec["util"], rec["label"] = alpha, util, run_label(alpha, util)
        out[t] = rec
        print(f"{t}: {run_label(alpha, util)}  bound ${rec['bound']:,.0f}")
        for r in rec["rows"]:
            print(f"   {r['method']:10s} ${r['profit']:>9,.0f}  {r['pct_bound']:6.1%}  "
                  f"chg {r['changeovers']:>3d}  midWIP {r['midline_wip']:>5.1f}  "
                  f"stranded {r['stranded']:>3d}  FG left {r['fg_left']:>4d}  "
                  f"M0 duty {r['m0_duty']:.0%}")
    d.setdefault("twoprod", {})["dashboards"] = out
    us = util_study(out, alpha=0.5)
    d["twoprod"]["util_study"] = {**d["twoprod"].get("util_study", {}), **us}
    print("\nutilisation study (50/50 mix):")
    for r in us["rows"]:
        print(f"   {r['util']:>4s}  demand {r['demand']:>4d}  {r['pct_ceiling']:>6s} of horizon"
              f"   commander {r['commander']:>7s}   grad-ST {r['gradst']:>7s}"
              f"   CP-SAT {r['cpsat']:>7s}   ST-CMD {r['margin']:>8s}")
    DATA.write_text(json.dumps(d, indent=2))
    print(f"\nwrote {len(out)} run(s) into {DATA.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
