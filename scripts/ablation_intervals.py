#!/usr/bin/env python3
"""Print the ablation intervals as a markdown table, for docs to embed.

    python3 scripts/ablation_intervals.py --runs runs/ablation

Each condition is paired against the intact run it perturbs, over the same
tasks, and the interval is oriented ablated minus intact so a negative number
always means the ablation cost the policy something.

This exists so the table in `docs/phase7.md` is generated rather than typed.
The number it reports is exactly the one this project has retracted twice, and
a typed copy of it goes stale silently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--baseline", default="bc")
    args = parser.parse_args()

    from kairos.dashboard.bundle import collect_ablation_intervals

    result = collect_ablation_intervals(args.runs, baseline=args.baseline)
    rows = result.get("rows") or []
    if not rows:
        print(f"No ablation traces under `{args.runs}`.")
        return 0

    print(f"Paired bootstrap on the per-task difference against `{args.baseline}`, "
          "oriented ablated minus intact.\n")
    print("| condition | difference | 95% interval | tasks | separates |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        sign = "+" if row["difference"] > 0 else ""
        low = f"{row['low']:+.3f}"
        high = f"{row['high']:+.3f}"
        print(f"| `{row['condition']}` | {sign}{row['difference']:.3f} "
              f"| [{low}, {high}] | {row['n_pairs']} "
              f"| {'yes' if row['separates'] else 'no'} |")

    separating = [r for r in rows if r["separates"]]
    print()
    if separating:
        print("Separating from zero: "
              + ", ".join(f"`{r['condition']}`" for r in separating) + ".")
    else:
        print("No condition separates from zero on this suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
