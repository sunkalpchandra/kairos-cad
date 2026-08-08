#!/usr/bin/env python
"""Compare BC, PPO, and a legal-random baseline in the live CAD environment.

    python3 scripts/evaluate_ppo.py --episodes 12

Every policy faces the same held-out requirements in the same order with the
same step budget. Reports closed-loop success rate, a much harder number than
the teacher-forced next-action accuracy Phase 4 reports, because here per-step
errors compound.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc", type=Path, default=Path("runs/bc/checkpoint.pt"))
    parser.add_argument("--ppo", type=Path, default=Path("runs/ppo/best.pt"))
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=Path("runs/ppo/comparison.json"))
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-episode-steps", type=int, default=30)
    parser.add_argument("--requirements", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-random", action="store_true")
    args = parser.parse_args()

    try:
        from kairos.models.actor_critic import ActorCritic, load_actor_critic
        from kairos.rl.env_client import RemoteCADEnv
        from kairos.rl.evaluate import RandomPolicy, compare_policies, format_comparison
        from kairos.rl.requirements import requirement_pools
    except ImportError as err:  # pragma: no cover - depends on the environment
        print(f"error: the learning stack needs torch ({err}).", file=sys.stderr)
        return 2

    policies: dict = {}
    if args.bc.exists():
        policies["bc"] = ActorCritic.from_bc_checkpoint(args.bc, device=args.device).eval()
        print(f"bc:  {args.bc}")
    else:
        print(f"note: no BC checkpoint at {args.bc}, skipping", file=sys.stderr)
    if args.ppo.exists():
        policies["ppo"] = load_actor_critic(args.ppo, device=args.device).eval()
        print(f"ppo: {args.ppo}")
    else:
        print(f"note: no PPO checkpoint at {args.ppo}, skipping", file=sys.stderr)
    if not args.skip_random:
        max_text = 64
        if policies:
            max_text = next(iter(policies.values())).vla.config.max_text_length
        policies["random"] = RandomPolicy(seed=args.seed, max_text_length=max_text)

    if not policies:
        print("error: nothing to evaluate", file=sys.stderr)
        return 1

    # Re-deriving the split here is how the published comparison ended up
    # measuring PPO on requirements it had trained on: train_ppo defaulted
    # to a 64-requirement pool and this script to 40, and requirement_pools
    # permutes over the pool, so the boundary moved. Use the run's own
    # recorded pool whenever it exists.
    report_path = args.ppo.parent / "report.json"
    held_out: list[str] = []
    trained_on: set[str] = set()
    if report_path.exists():
        payload = json.loads(report_path.read_text())
        held_out = list(payload.get("held_out_requirements") or [])
        trained_on = set(payload.get("train_requirements") or [])
    if held_out:
        print(f"held-out requirements: {len(held_out)} (from {report_path})")
    else:
        _, held_out = requirement_pools(args.root, limit=args.requirements, seed=args.seed)
        print(
            f"held-out requirements: {len(held_out)} (re-derived; "
            f"{report_path} has no recorded pool, so overlap is unverified)"
        )
    leaked = [r for r in held_out if r in trained_on]
    if leaked:
        print(
            f"error: {len(leaked)} of {len(held_out)} evaluation requirements were "
            "trained on; refusing to report a contaminated comparison.",
            file=sys.stderr,
        )
        return 1

    try:
        env = RemoteCADEnv(max_steps=args.max_episode_steps)
    except Exception as err:
        print(f"error: could not start the CAD environment: {err}", file=sys.stderr)
        return 1

    try:
        results = compare_policies(
            env, policies, held_out,
            episodes=args.episodes,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            device=args.device,
        )
    finally:
        env.close()

    print()
    print(format_comparison(results))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "episodes": args.episodes,
                "max_episode_steps": args.max_episode_steps,
                "held_out_requirements": len(held_out),
                "checkpoints": {"bc": str(args.bc), "ppo": str(args.ppo)},
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
