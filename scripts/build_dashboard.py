#!/usr/bin/env python3
"""Build the standalone KAIROS dashboard from artifacts on disk.

    python3 scripts/build_dashboard.py --out docs/dashboard.html

Reads only committed artifacts, designs, benchmark traces, training reports, so the page
can never show a number that is not also on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kairos.dashboard.build import write_dashboard  # noqa: E402
from kairos.dashboard.bundle import MAX_DESIGNS, build_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset", help="dataset root")
    parser.add_argument("--benchmark", default="runs/benchmark_core", help="benchmark run dir")
    parser.add_argument("--ablation", default="runs/ablation", help="ablation run dir")
    parser.add_argument("--runs", default="runs", help="training runs root")
    parser.add_argument("--out", default="docs/dashboard.html", help="output HTML path")
    parser.add_argument("--limit", type=int, default=MAX_DESIGNS, help="designs to embed")
    parser.add_argument("--no-meshes", action="store_true", help="skip 3D geometry")
    parser.add_argument("--stamp", default="", help="generation stamp shown in the header")
    parser.add_argument(
        "--layout", default="dashboard", choices=("dashboard", "studio"),
        help="studio is the review station; dashboard is the plain report",
    )
    parser.add_argument("--bundle-out", default="", help="also write the raw JSON bundle here")
    args = parser.parse_args()

    bundle = build_bundle(
        dataset=args.dataset,
        benchmark_runs=args.benchmark,
        ablation_runs=args.ablation,
        runs_root=args.runs,
        limit=args.limit,
        meshes=not args.no_meshes,
    )
    bundle["generated_at"] = args.stamp

    if args.bundle_out:
        Path(args.bundle_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.bundle_out).write_text(json.dumps(bundle, indent=2))

    path = write_dashboard(bundle, args.out, layout=args.layout)
    size_kb = path.stat().st_size / 1024
    counts = bundle["counts"]
    policies = len(bundle["benchmark"]["leaderboard"]["policies"])

    print(f"wrote {path} ({size_kb:.0f} KB)")
    print(f"  designs embedded : {counts['designs_embedded']}")
    print(f"  meshes attached  : {counts['meshes_attached']}")
    scrubbable = counts.get("designs_with_step_meshes", 0)
    print(f"  scrubbable       : {scrubbable}"
          + ("" if scrubbable else "  (run scripts/build_steps.py under FreeCAD)"))
    print(f"  policies scored  : {policies}")
    print(f"  comparisons      : {len(bundle['comparisons']['rows'])}")
    print(f"  ablation rows    : {len(bundle['ablations']['rows'])}")

    missing = [d["design_id"] for d in bundle["designs"] if d.get("mesh_error")]
    if missing:
        print(f"  WARNING: {len(missing)} designs have no mesh: {', '.join(missing[:5])}")
    if not policies:
        print("  WARNING: no benchmark leaderboard found; the benchmark tab will be empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
