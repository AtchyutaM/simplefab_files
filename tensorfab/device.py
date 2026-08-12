"""
Device / platform configuration for tensorfab.

This module is the single place that knows about hardware differences between
the macOS development machine and the Windows + NVIDIA box. Nothing else in
`tensorfab` imports platform-specific anything, so porting is: install the CUDA
wheel, run `python -m tensorfab.device` to confirm, and everything else works
unchanged.

See tensorfab/README.md § "Running on the Windows/NVIDIA machine".
"""

from __future__ import annotations

import os
import platform
from typing import Optional


def configure(x64: bool = False, platform_name: Optional[str] = None) -> None:
    """
    Configure JAX. MUST be called before any other JAX import/use takes effect
    for `x64`, which is why it lives here and is called at the top of the
    entrypoint scripts rather than inside library code.

    x64:
        Enable float64/int64. The simulator does not need it — all state is
        integer-valued and profit is reconciled in numpy float64 on the host —
        but it is useful when debugging gradient work in Phase 2b, where
        float32 accumulation over 2688 timesteps can drift.

    platform_name:
        Force a backend ("cpu", "gpu", "metal"). Leave as None to let JAX pick
        the best available. Forcing "cpu" on the NVIDIA box is a useful A/B
        control when benchmarking.
    """
    from jax import config as jax_config

    if x64:
        jax_config.update("jax_enable_x64", True)
    if platform_name is not None:
        jax_config.update("jax_platform_name", platform_name)


def describe() -> str:
    """Human-readable summary of what JAX is actually going to run on."""
    import jax

    devs = jax.devices()
    lines = [
        f"host       : {platform.system()} {platform.machine()} (python {platform.python_version()})",
        f"jax        : {jax.__version__}",
        f"backend    : {jax.default_backend()}",
        f"devices    : {devs}",
        f"x64 enabled: {jax.config.jax_enable_x64}",
    ]

    if platform.system() == "Darwin":
        lines.append("")
        lines.append(
            "NOTE (macOS): the CUDA backend does not exist on this platform. JAX runs on\n"
            "      the XLA *CPU* backend here, which still compiles the tick loop into a\n"
            "      single fused program (the main structural win) but cannot deliver the\n"
            "      10^6-10^8 steps/sec figures that need a real GPU. `jax-metal` exists\n"
            "      but is experimental and routinely hits unimplemented ops on scan/scatter\n"
            "      patterns like this one — do not rely on it. Treat macOS numbers as the\n"
            "      CPU reference point and re-run the benchmark on the NVIDIA box for the\n"
            "      GPU claim."
        )

    if jax.default_backend() == "cpu" and platform.system() != "Darwin":
        lines.append("")
        lines.append(
            "WARNING: running on CPU despite not being on macOS. If this machine has an\n"
            "      NVIDIA GPU, the CUDA wheel is probably not installed. See\n"
            "      tensorfab/README.md § 'Running on the Windows/NVIDIA machine'."
        )

    return "\n".join(lines)


def thread_note() -> str:
    """
    XLA's CPU backend parallelises across cores, but the reference Python sim is
    single-threaded. Report the setting so benchmark comparisons are honest.
    """
    n = os.cpu_count()
    flags = os.environ.get("XLA_FLAGS", "(unset)")
    return f"cpu cores: {n}   XLA_FLAGS: {flags}"


if __name__ == "__main__":
    print(describe())
    print()
    print(thread_note())
