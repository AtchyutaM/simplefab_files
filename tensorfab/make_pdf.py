"""
Render a deck HTML file to PDF.

The decks are self-contained HTML whose charts are drawn by JavaScript at load
time, so a static converter (weasyprint, wkhtmltopdf) would produce blank
figures. This drives headless Chromium via Playwright, which executes the page
exactly as a browser would.

Two details that matter:

* The deck is theme-aware. Chromium is forced to the light colour scheme and the
  root is stamped with data-theme="light" before printing, so the PDF never
  comes out with a dark background.
* Charts redraw on theme change via a MutationObserver, so we stamp the theme
  first and then wait for the redraw to settle before printing.

Usage:
    python -m tensorfab.make_pdf
    python -m tensorfab.make_pdf --html tensorfab/plan_deck.html --out plan.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent


def render(html_path: Path, out_path: Path, landscape: bool = True,
           scale: float = 1.0) -> None:
    from playwright.sync_api import sync_playwright

    url = html_path.resolve().as_uri()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1400, "height": 900},
            color_scheme="light",              # deck styles both themes; pick light for paper
            device_scale_factor=2,
        )
        page.goto(url, wait_until="networkidle")
        # Pin the theme explicitly — the deck's toggle stamps data-theme on <html>,
        # and its MutationObserver redraws every chart when that changes.
        page.evaluate("document.documentElement.setAttribute('data-theme','light')")
        page.wait_for_timeout(1200)            # let the chart redraw settle
        # Sanity check: charts are injected into these hosts; empty means JS failed.
        drawn = page.evaluate("document.querySelectorAll('svg').length")
        page.pdf(
            path=str(out_path),
            format="A4",
            landscape=landscape,
            print_background=True,
            scale=scale,
            margin={"top": "13mm", "bottom": "13mm", "left": "14mm", "right": "14mm"},
        )
        browser.close()
    print(f"  charts rendered: {drawn}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=str(HERE / "deck.html"))
    ap.add_argument("--out", default=str(HERE / "deck.pdf"))
    ap.add_argument("--portrait", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    html = Path(args.html)
    if not html.exists():
        print(f"missing {html}", file=sys.stderr)
        return 1
    out = Path(args.out)
    render(html, out, landscape=not args.portrait, scale=args.scale)
    size = out.stat().st_size
    print(f"wrote {out}  ({size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
