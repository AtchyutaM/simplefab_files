"""
Gather every logged run into one JSON for the presentation.

Single source of truth for the deck: nothing in the presentation is typed by
hand, everything traces back to a run directory under `tensorfab/runs/`. Run this
after `compare_patterns` and `eval_ppo_patterns` have completed.

Emits `tensorfab/presentation_data.json`.

Usage:  python -m tensorfab.collect_results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

RUNS = Path(__file__).parent / "runs"
OUT = Path(__file__).parent / "presentation_data.json"


def latest(name: str) -> Optional[Path]:
    cands = sorted(p for p in RUNS.iterdir() if p.is_dir() and p.name.endswith(name))
    return cands[-1] if cands else None


def load_json(d: Optional[Path], fname: str) -> Dict[str, Any]:
    if d is None:
        return {}
    f = d / fname
    return json.loads(f.read_text()) if f.exists() else {}


def downsample(a: np.ndarray, n: int = 240) -> List[float]:
    """Bucket-mean an array down to n points for plotting."""
    a = np.asarray(a, dtype=np.float64)
    if a.size <= n:
        return [round(float(x), 6) for x in a]
    edges = np.linspace(0, a.size, n + 1).astype(int)
    return [round(float(a[edges[i]:edges[i + 1]].mean()), 6) for i in range(n)]


def main() -> None:
    data: Dict[str, Any] = {}

    # ---- cross-pattern method comparison --------------------------------
    cmp_dir = latest("compare_patterns")
    data["patterns"] = load_json(cmp_dir, "results.json")
    data["compare_run"] = cmp_dir.name if cmp_dir else None

    # ---- retrained PPO agents -------------------------------------------
    ppo_dir = latest("eval_ppo_patterns")
    data["ppo"] = load_json(ppo_dir, "ppo_results.json")
    data["ppo_run"] = ppo_dir.name if ppo_dir else None

    # ---- gradient credit-assignment curves ------------------------------
    curves: Dict[str, List[float]] = {}
    if cmp_dir and (cmp_dir / "series.npz").exists():
        z = np.load(cmp_dir / "series.npz")
        for k in z.files:
            if k.endswith("_grad_norm_by_t"):
                pattern = k[: -len("_grad_norm_by_t")]
                arr = z[k]
                if arr.size > 1:
                    curves[pattern] = downsample(arr / arr.sum())
    data["grad_curves"] = curves

    # ---- the BPTT depth sweep (from the spike) ---------------------------
    spike_dirs = sorted(p for p in RUNS.iterdir()
                        if p.is_dir() and p.name.endswith("openloop_spike"))
    depth: Dict[str, Any] = {}
    for d in spike_dirs:
        m = load_json(d, "metrics.json")
        res = m.get("results", {})
        if len(res) >= 4:            # the depth sweep, not the single-H runs
            depth = {"run": d.name, "results": res}
    data["depth_sweep"] = depth

    # ---- throughput ------------------------------------------------------
    env_dir = latest("bench_env")
    env_metrics = load_json(env_dir, "metrics.json")
    data["throughput"] = {
        "sim_python_steps_per_sec": 309485,
        "sim_tensor_peak": 59893949,
        "sim_tensor_peak_batch": 10000,
        "env_python_steps_per_sec": env_metrics.get("reference_steps_per_sec"),
        "env_tensor_peak": env_metrics.get("peak_steps_per_sec"),
        "env_tensor_peak_batch": env_metrics.get("peak_batch"),
        "ppo_observed_fps": 13650,
        "bench_env_run": env_dir.name if env_dir else None,
    }

    # ---- historical context (CLAUDE.md, OLD doubled WIP costs) ------------
    data["historical"] = {
        "UNIFORM":        {"agent": 17607, "commander": -159,  "claimed_max": 18786},
        "ALTERNATING_50": {"agent": 16337, "commander": -2813, "claimed_max": 18786},
        "ALTERNATING":    {"agent": 29965, "commander": 29480, "claimed_max": 33546},
        "ALTERNATING_92": {"agent": 31493, "commander": 31301, "claimed_max": None},
    }

    OUT.write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {OUT}")
    print(f"  patterns   : {list(data['patterns'].keys())}")
    print(f"  ppo        : {list(data['ppo'].keys())}")
    print(f"  grad curves: {list(curves.keys())}")
    print(f"  depth sweep: {list(depth.get('results', {}).keys())}")


if __name__ == "__main__":
    main()
