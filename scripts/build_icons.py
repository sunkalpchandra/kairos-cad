#!/usr/bin/env python3
"""Vendor the icon sprite from the Lucide set.

    python3 scripts/build_icons.py --source /path/to/lucide-static/icons

Writes `kairos/dashboard/assets/icons.html`, a sprite of `<symbol>` elements.

Vendored rather than linked: the page has to hold with no network, both for a
file:// open and under an artifact host's CSP. Only the icons actually used are
included, so the sprite stays a few KB rather than the full 2,000.

Lucide is ISC licensed; the licence travels with the sprite in a comment, and
the full text is in `licenses/lucide.txt`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: symbol id -> Lucide icon name. The left column is the vocabulary the UI
#: speaks (CAD operations, viewport commands); the right is what Lucide calls
#: the nearest glyph.
ICONS: dict[str, str] = {
    # Feature operations, matching the executor's action names.
    "i-sketch": "pen-tool",
    "i-line": "spline",
    "i-circle": "circle",
    "i-rect": "square",
    "i-pad": "box",
    "i-pocket": "layers",
    "i-revolve": "rotate-3d",
    "i-fillet": "spline",
    "i-chamfer": "triangle",
    "i-pattern": "grid-2x2",
    "i-finish": "check",
    # Viewport
    "i-iso": "box",
    "i-front": "square",
    "i-top": "scan-line",
    "i-right": "rectangle-vertical",
    "i-fit": "maximize",
    "i-grid": "grid-3x3",
    "i-shaded": "contrast",
    "i-measure": "ruler",
    "i-orbit": "orbit",
    "i-pan": "move",
    "i-zoom": "zoom-in",
    # Browser
    "i-eye": "eye",
    "i-folder": "folder",
    "i-body": "boxes",
    "i-prev": "chevron-left",
    "i-next": "chevron-right",
    "i-down": "chevron-down",
    # Data workspaces
    "i-leaderboard": "chart-line",
    "i-curve": "chart-spline",
    "i-intervals": "git-compare",
    "i-loss": "trending-down",
    "i-ppo": "cpu",
    "i-ablate": "flask-conical",
}

#: The brand mark. Drawn here rather than borrowed: an extruded L-bracket in
#: isometric, the part family this project is built around, with three tones
#: for the three visible faces the way the viewport shades a solid.
MARK = """  <symbol id="mark" viewBox="0 0 32 32">
    <g stroke="none">
      <path d="M4 10 L16 3 L28 10 L16 17 Z" fill="currentColor" opacity="0.95"/>
      <path d="M4 10 L16 17 L16 29 L4 22 Z" fill="currentColor" opacity="0.55"/>
      <path d="M28 10 L16 17 L16 29 L28 22 Z" fill="currentColor" opacity="0.28"/>
    </g>
  </symbol>"""

_BODY = re.compile(r"<svg[^>]*>(.*)</svg>", re.DOTALL)


def extract(path: Path) -> str:
    """The drawing commands inside a Lucide svg, without its wrapper."""
    text = path.read_text()
    match = _BODY.search(text)
    if not match:
        raise ValueError(f"{path.name} is not a single <svg> element")
    body = match.group(1).strip()
    # Lucide sets stroke attributes on the <svg>; the sprite applies them via
    # CSS on `use` instead, so a symbol inherits the surrounding colour.
    body = re.sub(r'\s(stroke|fill|stroke-width|stroke-linecap|stroke-linejoin)="[^"]*"',
                  "", body)
    return "\n".join("    " + line.strip() for line in body.splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="lucide-static/icons")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "kairos/dashboard/assets/icons.html")
    parser.add_argument("--version", default="1.30.0")
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"error: no such directory {args.source}", file=sys.stderr)
        return 1

    symbols = [MARK]
    missing = []
    for symbol_id, icon in sorted(ICONS.items()):
        path = args.source / f"{icon}.svg"
        if not path.exists():
            missing.append(icon)
            continue
        symbols.append(
            f'  <symbol id="{symbol_id}" viewBox="0 0 24 24">\n'
            f"{extract(path)}\n  </symbol>"
        )

    if missing:
        print(f"error: not in the icon set: {sorted(set(missing))}", file=sys.stderr)
        return 1

    sprite = (
        "<!-- Icon sprite.\n"
        f"     Glyphs from Lucide v{args.version}, ISC licensed, vendored so the page\n"
        "     holds with no network. Full licence in licenses/lucide.txt.\n"
        "     Regenerate with scripts/build_icons.py; do not edit by hand. -->\n"
        '<svg width="0" height="0" style="position:absolute" aria-hidden="true">\n'
        "  <defs>\n" + "\n".join(symbols) + "\n  </defs>\n</svg>\n"
    )
    args.out.write_text(sprite)
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.1f} KB, {len(ICONS)} icons + mark)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
