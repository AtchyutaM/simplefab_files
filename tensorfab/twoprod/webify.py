"""
Downscale the dashboard PNGs for embedding in the deck.

The dashboards are rendered at 5112 px wide so the Gantt stays legible when
someone zooms into a 2-tick litho job. That resolution is right for the PNG on
disk and wrong for an HTML deck that inlines nine of them as data URIs, so this
writes a second, narrower copy alongside each one. The full-resolution file
stays the artifact of record.

    python -m tensorfab.twoprod.webify
"""
from __future__ import annotations
import pathlib, sys
from PIL import Image

SCH = pathlib.Path(__file__).parent / "schedules"
WIDTH = 2000
COLOURS = 256


def main() -> int:
    srcs = sorted(p for p in list(SCH.glob("dash_*.png"))
                  + list(SCH.glob("release_patterns.png"))
                  if not p.stem.endswith("_web"))
    if not srcs:
        print("no figures found -- run dashboard.py first")
        return 1
    total = 0
    for src in srcs:
        im = Image.open(src).convert("RGB")
        h = round(im.height * WIDTH / im.width)
        out = src.with_name(src.stem + "_web.png")
        # These are flat-colour plots on a dark ground, so a 256-entry palette is
        # visually lossless here and cuts the file by about two thirds -- which
        # matters when sixteen of them are inlined as base64 in one document.
        (im.resize((WIDTH, h), Image.LANCZOS)
           .quantize(colors=256, method=Image.MEDIANCUT,
                     dither=Image.FLOYDSTEINBERG)
           .save(out, optimize=True))
        kb = out.stat().st_size / 1024
        total += kb
        print(f"  {out.name:34s} {im.width}x{im.height} -> {WIDTH}x{h}  {kb:6.0f} KB")
    print(f"\n{len(srcs)} images, {total/1024:.1f} MB total "
          f"(~{total*1.34/1024:.1f} MB once base64-encoded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
