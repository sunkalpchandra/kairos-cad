"""Bundle run artifacts into one self-contained payload for the dashboard.

The dashboard is a single HTML file with the data inlined, not a server reading
the repo, so it opens from disk with nothing running. Inlining also forces the
bundle to come from artifacts on disk rather than from a live process, the same
rule benchmark_report.py follows.

Pure python, so it builds under either interpreter.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

#: Cap on designs embedded in one bundle. Each carries a tessellated mesh, and
#: a browser opening a 200 MB HTML file is not a dashboard.
MAX_DESIGNS = 24


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


#: Lines skipped by the last _read_jsonl calls, surfaced in the bundle counts.
_SKIPPED: dict[str, int] = {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a trace file line by line, skipping only the lines that fail.

    Wrapping the whole comprehension in one try meant a single malformed line
    discarded every trace for that policy and the build still exited 0, so the
    dashboard showed a policy that had simply not been run.
    """
    try:
        text = path.read_text()
    except OSError:
        _SKIPPED[path.name] = -1  # unreadable is different from empty
        return []
    rows, skipped = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    if skipped:
        _SKIPPED[path.name] = skipped
    return rows


def _measured(results: list[dict[str, Any]], kind: str) -> float | None:
    """The measured value of a named constraint, if it was measured at all."""
    for result in results:
        if (result.get("constraint") or {}).get("kind") == kind:
            return result.get("measured")
    return None


def _stratify(designs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Take `limit` designs spread evenly across families, not the first N.

    Design ids cycle through families, so slicing the first N *looks* balanced
    and is not: at limit=12 it silently dropped `reinforced_plate` entirely, and
    a viewer that never shows a family reads as a family that does not exist.
    Round-robin guarantees every family appears before any family repeats.
    """
    by_family: dict[str, list[dict[str, Any]]] = {}
    for design in designs:
        by_family.setdefault(design["family"], []).append(design)

    picked: list[dict[str, Any]] = []
    families = sorted(by_family)
    for rank in range(max(len(v) for v in by_family.values()) if by_family else 0):
        for family in families:
            if rank < len(by_family[family]) and len(picked) < limit:
                picked.append(by_family[family][rank])
        if len(picked) >= limit:
            break
    return sorted(picked, key=lambda d: d["design_id"])


def collect_designs(root: str | Path, limit: int = MAX_DESIGNS) -> list[dict[str, Any]]:
    """Design summaries with their requirement and constraint outcome."""
    designs: list[dict[str, Any]] = []
    for path in sorted(Path(root).glob("designs/design_*/state.json")):
        state = _read_json(path)
        requirement = _read_json(path.parent / "requirements.json") or {}
        trajectory = _read_json(path.parent / "trajectory.json") or {}
        if not state:
            continue
        metrics = trajectory.get("final_metrics", {})
        constraints = metrics.get("constraints", {})
        results = constraints.get("results", [])
        box = state.get("bounding_box") or {}
        designs.append({
            "design_id": path.parent.name,
            "family": (requirement.get("spec") or {}).get("kind", "unknown"),
            "requirement": requirement.get("text", ""),
            # What the parser pulled out of that sentence. Only `kind` was ever
            # read from it, so the whole language-grounding step -- the first
            # phase of this project -- had no representation on the page at
            # all. A number the parser missed shows here as a field that is
            # absent, which is the only way to see it short of reading code.
            "spec": requirement.get("spec") or {},
            "mass_g": state.get("mass_g"),
            "volume_mm3": state.get("volume_mm3"),
            "surface_area_mm2": state.get("surface_area_mm2"),
            "material": state.get("material"),
            # The state stores span per axis as *_len; there is no
            # length/width/height, and reading those names yields silent nulls.
            "extent_mm": [box.get("x_len"), box.get("y_len"), box.get("z_len")],
            "faces": (state.get("topology") or {}).get("faces"),
            "hole_count": state.get("hole_count"),
            # Wall thickness is measured by the constraint checker, not stored
            # on the state, so it has to be lifted back out of the result.
            "min_wall_thickness_mm": _measured(results, "min_wall_thickness"),
            "steps": metrics.get("steps"),
            "total_reward": metrics.get("total_reward"),
            "satisfaction_rate": constraints.get("satisfaction_rate"),
            "all_satisfied": constraints.get("all_measured_satisfied"),
            "constraints": [
                {
                    "kind": r["constraint"]["kind"],
                    "status": r["status"],
                    "detail": r.get("detail", "")[:120],
                }
                for r in constraints.get("results", [])
            ],
            "operations": [a.get("operation") for a in trajectory.get("actions", [])],
        })
    return _stratify(designs, limit)


def _traces_by_policy(runs: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        path.name.replace("_traces.jsonl", ""): _read_jsonl(path)
        for path in sorted(runs.glob("*_traces.jsonl"))
    }


def _normalize_scores(leaderboard: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten `leaderboard.json` scores into the fields the viewer renders.

    The renaming happens here rather than in JavaScript so the field mapping is
    covered by python tests; a silent rename on the JS side would show blank
    columns rather than fail.
    """
    return [
        {
            "policy": score.get("policy"),
            "progress_mean": score.get("progress_score"),
            "success_rate": score.get("success_rate"),
            "validity": score.get("validity_rate"),
            "satisfaction": score.get("satisfaction_rate"),
            "efficiency": score.get("efficiency"),
            "crash_rate": score.get("crash_rate"),
            "tasks": score.get("episodes"),
            "milestones": score.get("milestone_rates", {}),
        }
        for score in leaderboard.get("scores", [])
    ]


def _bucket(task_id: str) -> str:
    """Bucket a task id: `build-*` -> "build", `complete-k<N>-*` -> "<N>".

    The `k` prefix is the whole reason this needs a function. Splitting on "-"
    and testing `isdigit()` on the middle field looks right and quietly files
    every COMPLETE task under "build", which flattens the success(k) curve into
    a single point without any error.
    """
    parts = task_id.split("-")
    if len(parts) > 1 and parts[0] == "complete":
        stage = parts[1]
        if stage.startswith("k") and stage[1:].isdigit():
            return stage[1:]
    return "build"


def collect_benchmark(runs: str | Path) -> dict[str, Any]:
    """Leaderboard, the success(k) curve, and paired comparisons."""
    runs = Path(runs)
    leaderboard = _read_json(runs / "leaderboard.json") or {}
    traces = _traces_by_policy(runs)

    curves: dict[str, dict[str, list[bool]]] = {}
    for policy, rows in traces.items():
        for row in rows:
            if row.get("aborted"):
                continue
            curves.setdefault(policy, {}).setdefault(_bucket(row["task_id"]), []).append(
                bool(row.get("finished_successfully"))
            )

    return {
        "suite_version": leaderboard.get("suite_version"),
        "preset": leaderboard.get("preset"),
        "tasks": leaderboard.get("tasks"),
        "leaderboard": {"policies": _normalize_scores(leaderboard)},
        "success_curve": {
            policy: {k: sum(v) / len(v) for k, v in sorted(buckets.items())}
            for policy, buckets in curves.items()
        },
    }


#: Milestones in the order a build reaches them, matching the scorer.
MILESTONES = (
    "opened_a_sketch", "drew_geometry", "made_a_solid", "solid_is_valid",
    "has_any_hole", "all_constraints_met", "finished_successfully",
)


def collect_rollouts(runs: str | Path, per_family: int = 1) -> dict[str, Any]:
    """What each policy actually did, step by step, on a handful of tasks.

    The leaderboard says bc scores 0.458. It does not say why, and the answer
    is not subtle once you can see it: on some tasks the policy emits one
    operation for the rest of the episode and the environment rejects every
    one. That is visible in the traces and in nothing else the page shows.

    One BUILD task per family: BUILD starts from nothing, so the whole episode
    is the policy's, where a COMPLETE task is mostly replayed expert prefix.
    """
    traces = _traces_by_policy(Path(runs))
    if not traces:
        return {"tasks": []}

    # Pick the tasks first, from whichever policy ran, so every policy is
    # reported on the same set rather than on whatever each one happened to do.
    chosen: dict[str, str] = {}
    for rows in traces.values():
        for row in rows:
            task = row.get("task_id", "")
            if not task.startswith("build-"):
                continue
            family = row.get("family") or "unknown"
            if len([t for t in chosen.values() if t == family]) < per_family:
                chosen.setdefault(task, family)
        break

    # Solids rebuilt from the traces by scripts/build_rollout_meshes.py. Absent
    # until that has run, which the page reports rather than papering over.
    from kairos.dashboard.mesh import mesh_from_stl

    meshes_root = Path(runs) / "rollout_meshes"

    def _mesh(task_id: str, policy: str):
        path = meshes_root / task_id / f"{policy}.stl"
        if not path.is_file():
            return None
        try:
            return mesh_from_stl(path)
        except (OSError, ValueError):
            return None

    out = []
    for task_id, family in sorted(chosen.items()):
        episodes = []
        for policy in sorted(traces):
            row = next((r for r in traces[policy] if r.get("task_id") == task_id), None)
            if row is None:
                continue
            operations = row.get("operations") or []
            accepted = row.get("accepted") or []
            episodes.append({
                "policy": policy,
                "operations": operations,
                # Older traces have no per-step record; an empty list reads as
                # "not recorded" in the page rather than as "all accepted".
                "accepted": [bool(a) for a in accepted][: len(operations)],
                "rejections": (row.get("rejections") or [])[: len(operations)],
                "steps": row.get("steps"),
                "expert_steps": row.get("expert_steps"),
                "invalid_actions": row.get("invalid_actions"),
                "progress_score": row.get("progress_score"),
                "milestones": [m for m in MILESTONES if row.get(m)],
                "aborted": bool(row.get("aborted")),
                "abort_reason": row.get("abort_reason") or "",
                "mesh": _mesh(task_id, policy),
            })
        if episodes:
            out.append({
                "task_id": task_id,
                "family": family,
                "requirement": next(
                    (r.get("requirement") for rows in traces.values()
                     for r in rows if r.get("task_id") == task_id and r.get("requirement")),
                    "",
                ),
                "episodes": episodes,
            })
    return {"tasks": out, "milestones": list(MILESTONES)}


#: A rejection message carries the FreeCAD object that failed, e.g. "Pad002".
#: Two episodes that failed the same way in different places would otherwise
#: count as two different failures and neither would look common.
_INSTANCE = re.compile(r"\b([A-Za-z]+?)\d+\b")

#: Trailing index tuples, e.g. "Perpendicular constraint (8, 8)". They name
#: which geometry the constraint touched, not what went wrong, and leaving them
#: in split one failure kind across 150 rows of count 1.
_INDICES = re.compile(r"\s*\(\s*\d+(?:\s*,\s*\d+)*\s*\)")


def _failure_kind(message: str) -> str:
    """A rejection message reduced to the failure it describes."""
    kind = _INDICES.sub("", message.strip())
    kind = _INSTANCE.sub(r"\1", kind)
    # Everything after the colon is FreeCAD's internal state dump; the part
    # before it is the sentence a reader needs.
    head = kind.split(":", 1)[0].strip()
    return head or kind[:80]


def collect_dataset(root: str | Path, buckets: int = 12) -> dict[str, Any]:
    """Shape of the whole dataset, not the 24 designs the page can carry.

    The browser shows a capped sample and says 24. Nothing said what it is a
    sample *of*, so a reader had no way to know whether the families are
    balanced or whether the sample is representative of anything.

    Read from the trajectory files rather than the embedded bundle, so the
    numbers describe the dataset and not the cap.
    """
    root = Path(root)
    families: dict[str, int] = {}
    masses: list[float] = []
    steps: list[int] = []
    for path in sorted(root.glob("designs/design_*/state.json")):
        state = _read_json(path) or {}
        requirement = _read_json(path.parent / "requirements.json") or {}
        family = (requirement.get("spec") or {}).get("kind", "unknown")
        families[family] = families.get(family, 0) + 1
        mass = state.get("mass_g")
        if isinstance(mass, (int, float)) and mass > 0:
            masses.append(float(mass))
        trajectory = _read_json(path.parent / "trajectory.json") or {}
        count = (trajectory.get("final_metrics") or {}).get("steps")
        if isinstance(count, int):
            steps.append(count)

    if not families:
        return {}

    histogram: list[dict[str, Any]] = []
    if masses:
        low, high = min(masses), max(masses)
        width = (high - low) / buckets or 1.0
        counts = [0] * buckets
        for mass in masses:
            index = min(buckets - 1, int((mass - low) / width))
            counts[index] += 1
        histogram = [
            {"from": low + i * width, "to": low + (i + 1) * width, "count": n}
            for i, n in enumerate(counts)
        ]

    ordered = sorted(families.items())
    return {
        "designs": sum(families.values()),
        "families": [{"name": name, "count": n} for name, n in ordered],
        "mass_histogram": histogram,
        "mass_min": min(masses) if masses else None,
        "mass_max": max(masses) if masses else None,
        "steps_mean": sum(steps) / len(steps) if steps else None,
        "steps_min": min(steps) if steps else None,
        "steps_max": max(steps) if steps else None,
    }


def collect_codec(runs_root: str | Path) -> dict[str, Any]:
    """The action codec's audit: what it cannot express, and how far it drifts.

    Every learned policy speaks through this codec, so anything it cannot
    represent is a ceiling no policy can pass. It has been the cause rather
    than the symptom twice -- the oracle sat at 0.431 and then 0.858 because
    of it, not because of any policy -- and it lived in a script's exit code.
    """
    audit = _read_json(Path(runs_root) / "codec_audit.json") or {}
    if not audit:
        return {}
    return {
        "steps": audit.get("steps"),
        "unrepresentable": audit.get("unrepresentable"),
        "unrepresentable_rate": audit.get("unrepresentable_rate"),
        "drifted": audit.get("drifted"),
        "worst_round_trip_mm": audit.get("worst_round_trip_mm"),
        "operations_used": audit.get("operations_used"),
        "affected_designs": audit.get("affected_designs"),
    }


def collect_families(runs: str | Path) -> dict[str, Any]:
    """Progress per part family, per policy.

    The suite mixes eight families of very different difficulty. A mean over
    all of them says a policy scores 0.458 without saying that it is competent
    on plates and helpless on flanges, which is the difference between a
    policy that has learned something and one that has learned one thing.
    """
    traces = _traces_by_policy(Path(runs))
    if not traces:
        return {"families": [], "policies": [], "cells": {}}

    families: dict[str, int] = {}
    cells: dict[str, dict[str, list[float]]] = {}
    for policy, rows in traces.items():
        for row in rows:
            if row.get("aborted"):
                continue
            family = row.get("family") or "unknown"
            families[family] = families.get(family, 0) + 1
            cells.setdefault(policy, {}).setdefault(family, []).append(
                float(row.get("progress_score") or 0.0))

    order = sorted(families)
    return {
        "families": [{"name": name, "episodes": families[name] // max(1, len(traces))}
                     for name in order],
        "policies": sorted(cells),
        "cells": {
            policy: [
                (sum(scores[family]) / len(scores[family])) if family in scores else None
                for family in order
            ]
            for policy, scores in cells.items()
        },
    }


def collect_jam(runs: str | Path) -> dict[str, Any]:
    """When a policy's actions start being refused, and whether it recovers.

    A refusal rate says how much of an episode was wasted. It does not say
    whether the waste is scattered or is one unbroken run at the end, and those
    are different failures: scattered refusals are a policy making mistakes,
    an unbroken tail is a policy that has stopped.
    """
    traces = _traces_by_policy(Path(runs))
    rows = []
    for policy, episodes in sorted(traces.items()):
        firsts, tails, recovered, counted = [], [], 0, 0
        for episode in episodes:
            accepted = episode.get("accepted") or []
            if not accepted:
                continue
            counted += 1
            refused = [i for i, ok in enumerate(accepted) if not ok]
            if not refused:
                continue
            firsts.append(refused[0] / len(accepted))
            # The unbroken run of refusals at the end of the episode.
            tail = 0
            for ok in reversed(accepted):
                if ok:
                    break
                tail += 1
            tails.append(tail / len(accepted))
            if any(accepted[refused[0]:]):
                recovered += 1
        if not firsts:
            rows.append({"policy": policy, "episodes": counted, "jammed": 0})
            continue
        rows.append({
            "policy": policy,
            "episodes": counted,
            "jammed": len(firsts),
            "first_refusal": sum(firsts) / len(firsts),
            "tail_share": sum(tails) / len(tails),
            "recovered": recovered,
        })
    return {"rows": rows}


def collect_funnel(runs: str | Path) -> dict[str, Any]:
    """Milestone reach rates as a funnel, with the drop at each rung.

    Progress is prefix-scored, so the rung a policy stops at is the whole
    story of its score. The rates are in the leaderboard already; the drops
    are not, and the drop is what names the wall.
    """
    leaderboard = _read_json(Path(runs) / "leaderboard.json") or {}
    rows = []
    for score in leaderboard.get("scores") or []:
        rates = score.get("milestone_rates") or {}
        if not rates:
            continue
        steps = []
        previous = 1.0
        for name in MILESTONES:
            rate = rates.get(name)
            if rate is None:
                continue
            steps.append({"milestone": name, "rate": rate, "drop": previous - rate})
            previous = rate
        worst = max(steps, key=lambda s: s["drop"], default=None)
        rows.append({
            "policy": score.get("policy"),
            "steps": steps,
            # The rung where the most episodes were lost. For a policy that
            # never finishes, this is the sentence the funnel is drawing.
            "wall": worst["milestone"] if worst and worst["drop"] > 0 else None,
            "wall_drop": worst["drop"] if worst and worst["drop"] > 0 else 0.0,
        })
    return {"milestones": list(MILESTONES), "rows": rows}


def collect_task_types(runs: str | Path) -> dict[str, Any]:
    """The leaderboard split into BUILD and COMPLETE.

    This is the number the headline hides. bc scores 0.458 over the suite and
    0.069 on BUILD alone, because a COMPLETE task hands it an expert prefix and
    asks for the last k actions. Both are honest; only one of them is a policy
    building a part from a sentence.
    """
    leaderboard = _read_json(Path(runs) / "leaderboard.json") or {}
    split = leaderboard.get("by_task_type") or {}
    if not split:
        return {"kinds": [], "rows": []}

    kinds = sorted({kind for policy in split.values() for kind in policy})
    rows = []
    for policy, kinds_for_policy in split.items():
        entry: dict[str, Any] = {"policy": policy}
        for kind in kinds:
            scores = kinds_for_policy.get(kind) or {}
            entry[kind] = {
                "progress": scores.get("progress_score"),
                "success": scores.get("success_rate"),
                "episodes": scores.get("episodes"),
            }
        rows.append(entry)
    rows.sort(key=lambda r: -( (r.get(kinds[0]) or {}).get("progress") or 0))
    return {"kinds": kinds, "rows": rows}


def collect_failures(runs: str | Path, top: int = 8) -> dict[str, Any]:
    """How each policy's actions were refused, across the whole suite.

    The rollout strips show seven tasks in detail. This is the same data over
    all of them, which is the only way to see whether a failure is this task's
    or the policy's.
    """
    traces = _traces_by_policy(Path(runs))
    out: dict[str, Any] = {"policies": {}, "top": top}
    for policy, rows in sorted(traces.items()):
        counts: dict[str, int] = {}
        steps = rejected = 0
        for row in rows:
            operations = row.get("operations") or []
            steps += len(operations)
            for message in (row.get("rejections") or []):
                if not message:
                    continue
                rejected += 1
                kind = _failure_kind(str(message))
                counts[kind] = counts.get(kind, 0) + 1
        ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        out["policies"][policy] = {
            "steps": steps,
            "rejected": rejected,
            "kinds": [{"kind": k, "count": n} for k, n in ranked[:top]],
            # Say what the top-N left out rather than letting the listed counts
            # look like the whole of it.
            "other": sum(n for _, n in ranked[top:]),
            "distinct": len(ranked),
        }
    return out


def collect_matrix(runs: str | Path) -> dict[str, Any]:
    """Every task against every policy, as milestones reached out of seven.

    The leaderboard is a mean over 76 tasks. Two policies with the same mean
    can be solving disjoint halves of the suite, and nothing on the page could
    tell them apart. This is the un-averaged version of the same number.

    Milestones rather than success: success is 0 for three of six policies, so
    a success matrix is mostly one colour and says nothing about where the
    difference is.
    """
    traces = _traces_by_policy(Path(runs))
    if not traces:
        return {"tasks": [], "policies": [], "cells": {}}

    tasks: dict[str, str] = {}
    for rows in traces.values():
        for row in rows:
            tasks.setdefault(row["task_id"], row.get("family") or "unknown")

    order = sorted(tasks)
    cells: dict[str, list[int | None]] = {}
    for policy, rows in traces.items():
        by_task = {row["task_id"]: row for row in rows}
        column = []
        for task_id in order:
            row = by_task.get(task_id)
            if row is None or row.get("aborted"):
                # A task a policy never attempted is not a task it failed.
                column.append(None)
                continue
            column.append(sum(1 for m in MILESTONES if row.get(m)))
        cells[policy] = column

    return {
        "tasks": [{"id": task_id, "family": tasks[task_id], "kind": _bucket(task_id)}
                  for task_id in order],
        "policies": sorted(cells),
        "cells": cells,
        "milestones": len(MILESTONES),
    }


def collect_comparisons(runs: str | Path) -> dict[str, Any]:
    """Recompute paired bootstrap intervals from the traces on disk.

    Recomputed rather than read from a saved table on purpose: the intervals in
    docs/phase7.md were produced by a one-off call and live only in prose, so a
    stored copy could silently disagree with the traces beside it. numpy is a
    hard dependency of the statistics module, so this degrades to an empty list
    rather than failing the build under FreeCAD's interpreter.
    """
    try:
        from kairos.benchmark.statistics import compare_all
    except ImportError:
        return {"rows": [], "unavailable": "numpy is not importable here"}

    traces = _traces_by_policy(Path(runs))
    if len(traces) < 2:
        return {"rows": []}
    return {
        "rows": [
            {
                "a": c.policy_a,
                "b": c.policy_b,
                "difference": c.mean_difference,
                "low": c.ci_low,
                "high": c.ci_high,
                "wins": c.wins,
                "losses": c.losses,
                "ties": c.ties,
                "separates": c.separates,
                "pairs": c.n_pairs,
            }
            for c in compare_all(traces)
            if c.n_pairs
        ]
    }


def collect_ablations(runs: str | Path, baseline: str = "bc") -> dict[str, Any]:
    """Ablation leaderboard, expressed as deltas against the intact policy.

    The absolute score of `bc+shuffled-req` means little on its own; what the
    ablation answers is how far the policy falls when the requirement is
    corrupted, so the delta is the number worth showing.
    """
    leaderboard = _read_json(Path(runs) / "leaderboard.json") or {}
    scores = _normalize_scores(leaderboard)
    reference = next((s for s in scores if s["policy"] == baseline), None)
    base = (reference or {}).get("progress_mean")

    rows = []
    for score in scores:
        delta = None
        if base not in (None, 0) and score["progress_mean"] is not None:
            delta = (score["progress_mean"] - base) / base
        rows.append({
            "name": score["policy"],
            "progress_mean": score["progress_mean"],
            "success_rate": score["success_rate"],
            "validity": score["validity"],
            "delta": 0.0 if score["policy"] == baseline else delta,
            "baseline": score["policy"] == baseline,
        })
    return {"rows": rows, "baseline": baseline}


def collect_training(runs_root: str | Path) -> dict[str, Any]:
    """Per-epoch BC history and per-iteration PPO history."""
    runs_root = Path(runs_root)
    bc = _read_json(runs_root / "bc" / "report.json") or {}
    ppo = _read_json(runs_root / "ppo" / "report.json") or {}
    return {
        "bc": {
            # operation_accuracy is the held-out next-action accuracy;
            # train_accuracy is the training one. Nothing is named dev_/val_
            # accuracy, so guessing those names silently drops the held-out
            # series, which is the one the plot exists to show.
            "history": [
                {
                    "epoch": r.get("epoch"),
                    "train_accuracy": r.get("train_accuracy"),
                    "held_out_accuracy": r.get("operation_accuracy"),
                    "held_out_top3": r.get("operation_top3"),
                    "train_loss": r.get("train_loss"),
                    "val_loss": r.get("val_loss"),
                    "parameter_mae": r.get("parameter_mae"),
                }
                for r in bc.get("history", [])
            ],
            "dataset": bc.get("dataset", {}),
            "parameters": (bc.get("model") or {}).get("parameters"),
            "best_held_out_accuracy": max(
                (r.get("operation_accuracy") or 0.0 for r in bc.get("history", [])),
                default=None,
            ),
        },
        "ppo": {
            "history": [
                {
                    "iteration": r.get("iteration"),
                    "reward_mean": (r.get("rollout") or {}).get("reward_mean"),
                    "success_rate": (r.get("rollout") or {}).get("success_rate"),
                    "invalid_action_rate": (r.get("rollout") or {}).get("invalid_action_rate"),
                    "approx_kl": (r.get("update") or {}).get("approx_kl"),
                }
                for r in ppo.get("history", [])
            ],
            "best_success_rate": ppo.get("best_success_rate"),
        },
    }


def attach_step_meshes(designs: list[dict[str, Any]], root: str | Path) -> int:
    """Attach per-action meshes where scripts/build_steps.py produced them.

    A parametric timeline shows the part as it was at a feature. Without these
    the station's timeline can only highlight a node. Present for a subset of
    designs on purpose: every step mesh is bytes in a page that has to stay
    openable, so `build_steps.py` covers one design per family.
    """
    from kairos.dashboard.mesh import mesh_from_stl

    attached = 0
    for design in designs:
        steps_dir = Path(root) / "designs" / design["design_id"] / "steps"
        if not steps_dir.is_dir():
            continue
        steps: dict[str, Any] = {}
        for path in sorted(steps_dir.glob("*.stl")):
            try:
                steps[str(int(path.stem))] = mesh_from_stl(path)
            except (OSError, ValueError):
                continue
        if steps:
            design["step_meshes"] = steps
            attached += 1
    return attached


def attach_meshes(designs: list[dict[str, Any]], root: str | Path) -> int:
    """Attach a viewer mesh to each design; returns how many succeeded.

    A design whose STL is missing or unreadable keeps every one of its metrics
    and simply renders without geometry, a bundle is still useful without a
    picture, and failing the whole build over one bad mesh is not.
    """
    from kairos.dashboard.mesh import mesh_from_stl

    attached = 0
    for design in designs:
        stl = Path(root) / "designs" / design["design_id"] / "model.stl"
        try:
            design["mesh"] = mesh_from_stl(stl)
            attached += 1
        except (OSError, ValueError) as err:
            design["mesh"] = None
            design["mesh_error"] = str(err)
    return attached


def build_bundle(
    dataset: str | Path = "dataset",
    benchmark_runs: str | Path = "runs/benchmark_core",
    ablation_runs: str | Path = "runs/ablation",
    runs_root: str | Path = "runs",
    limit: int = MAX_DESIGNS,
    meshes: bool = True,
) -> dict[str, Any]:
    """Everything the dashboard renders, in one JSON-serializable dict."""
    designs = collect_designs(dataset, limit=limit)
    attached = attach_meshes(designs, dataset) if meshes else 0
    scrubbable = attach_step_meshes(designs, dataset) if meshes else 0
    return {
        "designs": designs,
        "benchmark": collect_benchmark(benchmark_runs),
        "comparisons": collect_comparisons(benchmark_runs),
        "rollouts": collect_rollouts(benchmark_runs),
        "matrix": collect_matrix(benchmark_runs),
        "failures": collect_failures(benchmark_runs),
        "task_types": collect_task_types(benchmark_runs),
        "funnel": collect_funnel(benchmark_runs),
        "jam": collect_jam(benchmark_runs),
        "families_scored": collect_families(benchmark_runs),
        "codec": collect_codec(runs_root),
        "dataset": collect_dataset(dataset),
        "ablations": collect_ablations(ablation_runs),
        "training": collect_training(runs_root),
        "families": sorted({d["family"] for d in designs}),
        "counts": {
            "designs_embedded": len(designs),
            "meshes_attached": attached,
            "designs_with_step_meshes": scrubbable,
            # Non-empty means a trace file had lines that would not parse. The
            # affected policy is still charted, from fewer episodes than it ran.
            "unparsable_trace_lines": dict(_SKIPPED),
        },
    }
