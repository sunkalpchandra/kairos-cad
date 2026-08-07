#!/usr/bin/env python
"""Compare a trained policy's choices against a recorded expert trajectory.

    python3 scripts/replay_policy.py --design dataset/designs/design_070003
    python3 scripts/replay_policy.py --requirement "Design a plate 60 x 40 x 5 mm"

Replay is **teacher forced**: every step is scored from the expert's recorded
state, not from what the policy's own previous action would have produced, so
it measures per-step agreement rather than closed-loop success.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/bc/checkpoint.pt"))
    parser.add_argument("--design", type=Path, default=None, help="a design_* directory")
    parser.add_argument("--requirement", default=None, help="predict just the first action")
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--sample", type=int, default=0, help="replay N random designs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        from kairos.training.bc_train import load_checkpoint
        from kairos.training.rollout import format_replay, predict_action, replay_trajectory
    except ImportError as err:  # pragma: no cover - depends on the environment
        print(f"error: the learning stack needs torch ({err}).", file=sys.stderr)
        return 2

    if not args.checkpoint.exists():
        print(f"error: no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1
    model = load_checkpoint(args.checkpoint, device="cpu").eval()

    if args.requirement:
        action, probabilities = predict_action(model, args.requirement)
        ranked = sorted(enumerate(probabilities), key=lambda kv: -kv[1])[:3]
        from kairos.rl.action_space import OPERATIONS

        print(f"requirement: {args.requirement}")
        print(f"first action: {action.operation.value} {action.parameters}")
        print(
            "top-3: "
            + ", ".join(f"{OPERATIONS[i].value} {p:.3f}" for i, p in ranked)
        )
        return 0

    designs: list[Path] = []
    if args.design:
        designs = [args.design / "trajectory.json"]
    else:
        every = sorted(args.root.glob("designs/design_*/trajectory.json"))
        if not every:
            print(f"error: no trajectories under {args.root}", file=sys.stderr)
            return 1
        count = args.sample or 1
        designs = random.Random(args.seed).sample(every, min(count, len(every)))

    reports = []
    for path in designs:
        if not path.exists():
            print(f"error: no trajectory at {path}", file=sys.stderr)
            return 1
        report = replay_trajectory(model, path)
        reports.append(report)
        if len(designs) == 1:
            print(format_replay(report))
        else:
            print(
                f"{str(report['design_id']):>8} [{report['family']:>16}] "
                f"agreement {report['agreement']:.3f} "
                f"({len(report['steps'])} steps)"
            )

    if len(reports) > 1:
        total = sum(s["agrees"] for r in reports for s in r["steps"])
        steps = sum(len(r["steps"]) for r in reports)
        print(f"\n{len(reports)} designs: {total}/{steps} steps agree ({total / steps:.3f})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
