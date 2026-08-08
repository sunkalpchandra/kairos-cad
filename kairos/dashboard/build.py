"""Inline the assets and the data bundle into one standalone HTML file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ASSETS = Path(__file__).parent / "assets"

#: Placeholder -> asset filename. Each placeholder sits inside a comment in the
#: template so `index.html` stays valid and openable on its own during editing.
_SLOTS = {
    "/*__STYLE__*/": "style.css",
    "/*__VIEWER__*/": "viewer.js",
    "/*__CHARTS__*/": "charts.js",
    "/*__APP__*/": "app.js",
}


def _guard(payload: str) -> str:
    """Neutralize sequences that would end the enclosing <script> early.

    A design's requirement text is arbitrary and lands inside a `<script>` block
    verbatim. HTML tokenizes `</script>` inside script content regardless of
    JSON quoting, so an unescaped one truncates the page and the dashboard
    renders blank. Escaping the slash keeps the JSON byte-identical after parse.
    """
    return (
        payload.replace("</", "<\\/")
        .replace(" ", "\\u2028")  # JS line terminators, invalid in string literals
        .replace(" ", "\\u2029")
    )


def render(data: dict[str, Any], template: Path | None = None) -> str:
    """Render the dashboard HTML with `data` inlined."""
    html = (template or ASSETS / "index.html").read_text()
    for slot, filename in _SLOTS.items():
        if slot not in html:
            raise ValueError(f"template is missing the {slot} slot")
        html = html.replace(slot, (ASSETS / filename).read_text())
    if "/*__DATA__*/" not in html:
        raise ValueError("template is missing the /*__DATA__*/ slot")
    payload = _guard(json.dumps(data, separators=(",", ":"), allow_nan=False))
    return html.replace("/*__DATA__*/null", payload)


def write_dashboard(data: dict[str, Any], path: str | Path) -> Path:
    """Render and write the dashboard; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(data))
    return path
