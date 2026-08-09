"""The CI gates, tested for the thing that matters: that they FAIL.

A gate with no test is the most dangerous untested code in a repository,
because a broken one reports success. Every defect this project has shipped had
that shape, so these tests drive each gate with input it must reject rather
than only checking it runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def run(script: str, *args: str, cwd: Path | None = None):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True, text=True, cwd=cwd or ROOT,
    )


# ------------------------------------------------------------------ check_text


def test_check_text_passes_on_the_repository():
    assert run("check_text.py").returncode == 0


@pytest.mark.parametrize(
    "char, name",
    [("—", "em dash"), ("’", "curly quote"),
     ("​", "zero-width space"), ("×", "multiplication sign")],
)
def test_check_text_rejects_disallowed_characters(tmp_path, char, name):
    """The sweep that collapsed [x*] to [xx] is what this gate exists for."""
    from importlib import util

    spec = util.spec_from_file_location("check_text", SCRIPTS / "check_text.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tmp_path / "sample.py"
    target.write_text(f"# a comment with a {char} in it\n")
    found = module.offenders(target)
    assert found, f"{name} was not flagged"
    assert found[0][2] == char


def test_check_text_allows_technical_symbols(tmp_path):
    """Degree, plus-minus and box-drawing carry meaning and must not trip it."""
    from importlib import util

    spec = util.spec_from_file_location("check_text", SCRIPTS / "check_text.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tmp_path / "sample.py"
    target.write_text("# tolerance ±0.1 mm at 90°\n# ┌─┐\n")
    assert module.offenders(target) == []


# --------------------------------------------------------------- check_station


def test_check_station_passes_on_the_repository():
    assert run("check_station.py").returncode == 0


def test_check_station_fails_when_a_page_lags_its_assets(tmp_path):
    """Editing an asset without rebuilding ships a page that renders fine and
    is not the thing in the repository."""
    from importlib import util

    spec = util.spec_from_file_location("check_station", SCRIPTS / "check_station.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    page = tmp_path / "built.html"
    page.write_text("<style>.a { color: red; } /* built from an older asset */</style>")

    # Point the checker at an asset the page does not contain.
    asset = module.ASSETS / "studio.css"
    assert asset.exists(), "studio.css moved; this test needs a real asset"
    problems = module.stale(page, ("studio.css",))
    assert problems, "a page missing its asset's content was reported as current"
    assert "studio.css" in problems[0]


def test_check_station_reports_a_page_that_was_never_built(tmp_path):
    from importlib import util

    spec = util.spec_from_file_location("check_station", SCRIPTS / "check_station.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    problems = module.stale(tmp_path / "absent.html", ("studio.css",))
    assert problems and "has not been built" in problems[0]


# ------------------------------------------------------------------ sync_docs


TRACES = list((ROOT / "runs/benchmark_core").glob("*_traces.jsonl"))


@pytest.mark.skipif(
    not TRACES,
    reason="benchmark traces are not committed, so the generators cannot run",
)
def test_sync_docs_reports_the_repository_as_current():
    """--check exits 0 only when every generated block matches its generator.

    Skipped where the traces are absent, which includes CI: they are run
    artifacts, not source. The marker test below still runs everywhere and is
    the part that catches a gate wired to nothing.
    """
    result = run("sync_docs.py", "--check")
    # Exit 2 means a block marker is missing, which is a wiring bug, not drift.
    assert result.returncode != 2, result.stderr
    assert result.returncode == 0, f"docs are stale:\n{result.stdout}{result.stderr}"


def test_sync_docs_requires_a_mode():
    """Neither --check nor --write must be an error, not a silent no-op."""
    assert run("sync_docs.py").returncode != 0


def test_sync_docs_blocks_point_at_real_docs():
    from importlib import util

    spec = util.spec_from_file_location("sync_docs", SCRIPTS / "sync_docs.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.BLOCKS, "no generated blocks are registered"
    for block_id, spec_ in module.BLOCKS.items():
        doc = ROOT / spec_["doc"]
        assert doc.exists(), f"{block_id} targets a missing doc"
        text = doc.read_text()
        assert module._MARKER.format(block_id) in text, (
            f"{spec_['doc']} has no {block_id} marker, so --check guards nothing"
        )


# ----------------------------------------------------------------- build_icons


def test_every_icon_the_markup_uses_is_in_the_sprite():
    """A missing symbol renders as an empty box, not an error."""
    import re

    assets = ROOT / "kairos/dashboard/assets"
    sprite = (assets / "icons.html").read_text()
    available = set(re.findall(r'<symbol id="([^"]+)"', sprite))

    used: set[str] = set()
    for name in ("studio.html", "studio.js"):
        used |= set(re.findall(r'href="#([a-z0-9-]+)"', (assets / name).read_text()))

    missing = sorted(used - available)
    assert not missing, f"markup references symbols not in the sprite: {missing}"


def test_icon_map_and_sprite_agree():
    """build_icons.py is the sprite's only author; a hand edit would drift."""
    import re
    from importlib import util

    spec = util.spec_from_file_location("build_icons", SCRIPTS / "build_icons.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sprite = (ROOT / "kairos/dashboard/assets/icons.html").read_text()
    available = set(re.findall(r'<symbol id="([^"]+)"', sprite))
    expected = set(module.ICONS) | {"mark"}
    assert expected == available, (
        f"sprite and ICONS disagree; regenerate with build_icons.py. "
        f"only in map: {sorted(expected - available)}, "
        f"only in sprite: {sorted(available - expected)}"
    )


def test_vendored_icons_carry_their_licence():
    """Lucide is ISC; the licence has to travel with the glyphs."""
    sprite = (ROOT / "kairos/dashboard/assets/icons.html").read_text()
    assert "ISC" in sprite and "Lucide" in sprite
    assert (ROOT / "licenses/lucide.txt").exists()
