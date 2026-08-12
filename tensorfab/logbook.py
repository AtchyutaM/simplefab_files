"""
Run logging for tensorfab experiments.

Every experiment writes a self-contained, timestamped directory under
`tensorfab/runs/` and appends one row to `tensorfab/EXPERIMENTS.md`. The point is
a defensible audit trail: months later, or in a write-up, you can point at a run
and say exactly what code, hardware, and parameters produced a number.

Each run directory contains:
    meta.json     environment capture — git commit + dirty flag, platform, JAX
                  version and backend, python version, timestamp, wall time
    params.json   every experiment parameter, verbatim
    log.txt       the full console transcript
    metrics.json  final scalar results
    series.npz    per-iteration arrays (loss curves, gradient norms, ...)

Usage:
    with Run("openloop_spike", params={"H": 672, "mode": "st"}) as run:
        run.log("starting")
        run.record(step=0, profit=123.4)
        run.set_metrics(final_profit=456.7)
        run.save_series(grad_norm_by_t=arr)
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

RUNS_DIR = Path(__file__).parent / "runs"
JOURNAL = Path(__file__).parent / "EXPERIMENTS.md"

JOURNAL_HEADER = """# tensorfab experiment log

Auto-appended by `tensorfab/logbook.py`. One row per run; full detail in
`tensorfab/runs/<run-id>/`. Newest rows at the bottom.

Narrative write-ups of what these runs *mean* live in `tensorfab/README.md`.

| run id | experiment | headline | wall s | backend | commit |
|---|---|---|---|---|---|
"""


def _git_info() -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                args, cwd=Path(__file__).parent, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    commit = run("git", "rev-parse", "--short", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def _env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import jax

        info["jax"] = jax.__version__
        info["backend"] = jax.default_backend()
        info["devices"] = [str(d) for d in jax.devices()]
        info["x64"] = bool(jax.config.jax_enable_x64)
    except Exception as exc:  # pragma: no cover
        info["jax_error"] = str(exc)
    return info


class Run:
    """Context manager for one logged experiment."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None,
                 note: str = ""):
        self.name = name
        self.params = dict(params or {})
        self.note = note
        self.run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
        self.dir = RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self._lines: List[str] = []
        self._series: Dict[str, List[Any]] = {}
        self._metrics: Dict[str, Any] = {}
        self._headline = ""
        self._t0 = time.perf_counter()

    # -- lifecycle -----------------------------------------------------
    def __enter__(self) -> "Run":
        self.log(f"=== run {self.run_id} ===")
        env = _env_info()
        self.log(f"backend={env.get('backend')} devices={env.get('devices')} "
                 f"jax={env.get('jax')}")
        if self.params:
            self.log("params: " + json.dumps(self.params, default=str))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        wall = time.perf_counter() - self._t0
        if exc is not None:
            self.log(f"!! FAILED: {exc_type.__name__}: {exc}")
            self._headline = self._headline or f"FAILED: {exc_type.__name__}"
        self.log(f"=== done in {wall:.1f}s ===")

        meta = {
            "run_id": self.run_id,
            "name": self.name,
            "note": self.note,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": round(wall, 3),
            "failed": exc is not None,
            "git": _git_info(),
            "env": _env_info(),
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        (self.dir / "params.json").write_text(json.dumps(self.params, indent=2, default=str))
        (self.dir / "metrics.json").write_text(json.dumps(self._metrics, indent=2, default=str))
        (self.dir / "log.txt").write_text("\n".join(self._lines) + "\n")
        if self._series:
            np.savez_compressed(
                self.dir / "series.npz",
                **{k: np.asarray(v) for k, v in self._series.items()},
            )

        self._append_journal(meta, wall)
        return False  # never swallow exceptions

    # -- recording -----------------------------------------------------
    def log(self, msg: str = "") -> None:
        print(msg)
        sys.stdout.flush()
        self._lines.append(str(msg))

    def record(self, **kwargs: Any) -> None:
        """Append one point to each named per-iteration series."""
        for k, v in kwargs.items():
            self._series.setdefault(k, []).append(v)

    def save_series(self, **kwargs: Any) -> None:
        """Store a whole array at once (overwrites any same-named series)."""
        for k, v in kwargs.items():
            self._series[k] = np.asarray(v)

    def set_metrics(self, **kwargs: Any) -> None:
        self._metrics.update(kwargs)

    def headline(self, text: str) -> None:
        """The one-line result that lands in EXPERIMENTS.md."""
        self._headline = text
        self.log(f">>> {text}")

    # -- journal -------------------------------------------------------
    def _append_journal(self, meta: Dict[str, Any], wall: float) -> None:
        if not JOURNAL.exists():
            JOURNAL.write_text(JOURNAL_HEADER)
        git = meta["git"]
        commit = git.get("commit") or "?"
        if git.get("dirty"):
            commit += "+dirty"
        backend = meta["env"].get("backend", "?")
        headline = (self._headline or "(no headline)").replace("|", "\\|")
        row = (f"| `{self.run_id}` | {self.name} | {headline} | "
               f"{wall:.0f} | {backend} | {commit} |\n")
        with JOURNAL.open("a") as fh:
            fh.write(row)


def latest_run(name_filter: str = "") -> Optional[Path]:
    """Most recent run directory, optionally filtered by name substring."""
    if not RUNS_DIR.exists():
        return None
    cands = sorted(p for p in RUNS_DIR.iterdir()
                   if p.is_dir() and name_filter in p.name)
    return cands[-1] if cands else None
