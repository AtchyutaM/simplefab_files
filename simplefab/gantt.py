from __future__ import annotations

from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


def extract_gantt_data(production_line) -> pd.DataFrame:
    """
    Build a tidy DataFrame with columns:
      ['Machine', 'Start', 'Finish', 'Product', 'Duration']

    Works with:
      - Batch machines (machine0, machine2) events: (product, start, finish, batch_size)
      - Single machines (machine1, machine3) events: (product, decision_time, start, finish)
    """
    rows = []
    machine_order = ["machine0", "machine1", "machine2", "machine3"]

    for machine_name in machine_order:
        m = getattr(production_line, machine_name, None)
        if m is None or not hasattr(m, "event_log"):
            continue

        for e in getattr(m, "event_log", []):
            if not isinstance(e, (tuple, list)):
                continue

            if machine_name in ("machine0", "machine2"):
                if len(e) < 3:
                    continue
                product, start, finish = e[0], e[1], e[2]
            else:
                if len(e) < 4:
                    continue
                product, start, finish = e[0], e[2], e[3]

            if start is None or finish is None:
                continue
            duration = max(0, finish - start)

            rows.append({
                "Machine": machine_name,
                "Start": float(start),
                "Finish": float(finish),
                "Product": int(product) if product is not None else None,
                "Duration": float(duration),
            })

    df = pd.DataFrame(rows, columns=["Machine", "Start", "Finish", "Product", "Duration"])
    if not df.empty:
        df = df.sort_values(by=["Machine", "Start"], kind="mergesort").reset_index(drop=True)
    return df


def plot_gantt_chart(gantt_df: pd.DataFrame) -> None:
    """
    Plot a horizontal Gantt using matplotlib.
    One lane per machine in order: M0, M1, M2, M3.
    Color encodes Product (0/1). Unknown -> gray.
    """
    if gantt_df is None or gantt_df.empty:
        print("Gantt: no events to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    name_map = {
        "machine0": "Machine 0 (Batch)",
        "machine1": "Machine 1 (Single)",
        "machine2": "Machine 2 (Batch)",
        "machine3": "Machine 3 (Single)",
    }
    lane_order = ["machine0", "machine1", "machine2", "machine3"]
    lanes = {m: i for i, m in enumerate(lane_order)}

    color_map = {0: "tab:blue", 1: "tab:orange", None: "gray"}

    for _, row in gantt_df.iterrows():
        m_key = row["Machine"]
        y = lanes.get(m_key, 0)
        start = float(row["Start"])
        finish = float(row["Finish"])
        width = max(0.0, finish - start)
        prod = row["Product"] if pd.notna(row["Product"]) else None
        color = color_map.get(prod, "gray")

        ax.barh(y=y, width=width, left=start, height=0.6, color=color, edgecolor="black", linewidth=0.8)

        if width >= 1.0:
            ax.text(start + width / 2, y, f"P{int(prod)}" if prod is not None else "P?",
                    va="center", ha="center", fontsize=9, color="white")

    y_ticks = [lanes[m] for m in lane_order if m in set(gantt_df["Machine"])]
    y_labels = [name_map[m] for m in lane_order if m in set(gantt_df["Machine"])]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Time")
    ax.set_title("Gantt: Products Being Manufactured at Each Machine")
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    plt.show()
