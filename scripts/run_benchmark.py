#!/usr/bin/env python
"""Run the KAIROS-CAD benchmark and write a leaderboard.

    python3 scripts/run_benchmark.py --preset smoke
    python3 scripts/run_benchmark.py --preset core --policies oracle-replay,scripted-spec

Tasks come from the frozen test split, so every policy faces the same work in
the same order. Two baselines audit the harness: `oracle-replay` must score
1.000 and `immediate-finish` must score bottom; the run reports whether those
invariants held rather than assuming they did.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRESETS = {"smoke": 1, "core": 4, "full": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--suite", type=Path, default=Path("benchmark/kairos-cad-v1"))
    parser.add_argument("--out", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--policies", default="", help="comma-separated; default all baselines")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--bc", type=Path, default=Path("runs/bc/checkpoint.pt"))
    parser.add_argument("--ppo", type=Path, default=Path("runs/ppo/best.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ablate", default="",
        help="comma-separated policies to also run ablated (shuffled/blank "
             "requirement, no action mask)",
    )
    args = parser.parse_args()

    from kairos.benchmark import SUITE_VERSION, SplitSet, format_scores, score_policy
    from kairos.benchmark.baselines import registry
    from kairos.benchmark.runner import run_suite, write_traces
    from kairos.benchmark.tasks import build_tasks, select

    splits_path = args.suite / "splits.json"
    if not splits_path.exists():
        print(f"error: no frozen split at {splits_path}; run `make benchmark-suite`",
              file=sys.stderr)
        return 1
    splits = SplitSet.load(splits_path)

    tasks = build_tasks(args.root, splits["test"].design_ids)
    tasks = select(tasks, limit_per_group=PRESETS[args.preset])
    if not tasks:
        print(f"error: no tasks built from {args.root}", file=sys.stderr)
        return 1
    print(f"{SUITE_VERSION}: {len(tasks)} tasks ({args.preset}) from the held-out test split")

    policies = registry(seed=args.seed)
    _add_learned_policies(policies, args)
    # Ablations are wrappers, so the perturbed and unperturbed runs share
    # every other condition and the difference is the ablation alone.
    for name in [a.strip() for a in args.ablate.split(",") if a.strip()]:
        if name not in policies:
            print(f"error: cannot ablate unknown policy {name!r}", file=sys.stderr)
            return 1
        from kairos.benchmark.ablations import build_ablations

        policies.update(
            build_ablations(policies[name], [t.requirement for t in tasks], args.seed)
        )

    wanted = [p.strip() for p in args.policies.split(",") if p.strip()] or list(policies)
    missing = [w for w in wanted if w not in policies]
    if missing:
        print(f"error: unknown policies {missing}; have {sorted(policies)}", file=sys.stderr)
        return 1

    try:
        from kairos.rl.env_client import RemoteCADEnv

        env = RemoteCADEnv(max_steps=40)
    except Exception as err:
        print(f"error: could not start the CAD environment: {err}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    scores = []
    by_type: dict = {}
    try:
        for name in wanted:
            started = time.perf_counter()
            done = {"n": 0}
            total_runs = len(tasks) * args.repeats

            def progress(result, total=total_runs, state=done) -> None:
                state["n"] += 1
                if state["n"] % 10 == 0 or state["n"] == total:
                    print(f"    {state['n']}/{total}", flush=True)

            print(f"\n{name}:")
            results = run_suite(
                env, policies[name], tasks, SUITE_VERSION, args.repeats, on_result=progress
            )
            write_traces(results, args.out / f"{name}_traces.jsonl")

            scored = [r.outcome for r in results if not r.aborted]
            aborted = sum(r.aborted for r in results)
            score = score_policy(name, scored)
            scores.append(score)
            # Segmented too: BUILD is where quitting cannot be correct, so it
            # is the only place the immediate-finish invariant is meaningful.
            build_ids = {t.task_id for t in tasks if t.task_type.value == "build"}
            by_type[name] = {
                "build": score_policy(
                    name, [r.outcome for r in results
                           if not r.aborted and r.task_id in build_ids]
                ),
                "complete": score_policy(
                    name, [r.outcome for r in results
                           if not r.aborted and r.task_id not in build_ids]
                ),
            }
            print(
                f"  progress {score.progress_score:.3f} | success {score.success_rate:.3f} "
                f"| {aborted} not attemptable | {time.perf_counter() - started:.0f}s"
            )
    finally:
        env.close()

    print("\n" + format_scores(scores))
    invariants = _check_invariants(scores, by_type)
    print("\nharness invariants:")
    for line in invariants["messages"]:
        print(f"  {line}")

    (args.out / "leaderboard.json").write_text(
        json.dumps(
            {
                "suite_version": SUITE_VERSION,
                "preset": args.preset,
                "tasks": len(tasks),
                "repeats": args.repeats,
                "scores": [s.to_dict() for s in scores],
                "by_task_type": {
                    name: {k: v.to_dict() for k, v in seg.items()}
                    for name, seg in by_type.items()
                },
                "invariants": invariants,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}/leaderboard.json")
    return 0 if invariants["ok"] else 2


def _add_learned_policies(policies: dict, args) -> None:
    """Register BC/PPO if their checkpoints exist and torch is installed."""
    try:
        from kairos.benchmark.policies import load_bc_policy, load_ppo_policy
    except ImportError:
        return
    if args.bc.exists():
        try:
            policies["bc"] = load_bc_policy(args.bc)
        except Exception as err:  # pragma: no cover - depends on artifacts
            print(f"note: could not load BC policy ({err})", file=sys.stderr)
    if args.ppo.exists():
        try:
            policies["ppo"] = load_ppo_policy(args.ppo)
        except Exception as err:  # pragma: no cover
            print(f"note: could not load PPO policy ({err})", file=sys.stderr)


def _check_invariants(scores, by_type: dict) -> dict:
    """The baselines that audit the benchmark rather than a policy.

    Both invariants are checked on BUILD tasks only. On COMPLETE(k=1) the
    expert's own last action is FINISH_DESIGN, so finishing immediately is the
    *correct* answer there — scoring well is not evidence of a broken metric,
    and the oracle's codec losses are likewise a BUILD-only phenomenon.
    """
    messages, ok = [], True
    builds = {name: s["build"] for name, s in by_type.items() if s["build"].episodes}

    oracle = builds.get("oracle-replay")
    if oracle is not None:
        # Below 1.000 means the codec cannot reproduce the expert build: six of
        # eight families draw irregular polygons the action space cannot express.
        passed = oracle.progress_score > 0.99
        ok = ok and passed
        messages.append(
            f"{'PASS' if passed else 'CEILING'}  oracle-replay scores "
            f"{oracle.progress_score:.3f} on BUILD. Below 1.000 this is the "
            "codec ceiling, not a policy result: no policy can beat it."
        )
    quitter = builds.get("immediate-finish")
    if quitter is not None and len(builds) > 1:
        worst = min(s.progress_score for s in builds.values())
        passed = quitter.progress_score <= worst + 1e-9
        ok = ok and passed
        messages.append(
            f"{'PASS' if passed else 'FAIL'}  immediate-finish scores "
            f"{quitter.progress_score:.3f} on BUILD, the lowest of {len(builds)} "
            "policies (any metric it wins there is a broken metric)"
        )
    if not messages:
        messages.append("no auditing baseline was run; invariants unchecked")
    return {"ok": ok, "messages": messages}


if __name__ == "__main__":
    raise SystemExit(main())
