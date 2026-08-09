"""Bundle run artifacts into one self-contained payload for the dashboard.

The dashboard is a single HTML file with the data inlined, not a server reading
the repo, so it opens from disk with nothing running. Inlining also forces the
bundle to come from artifacts on disk rather than from a live process, the same
rule benchmark_report.py follows.

Pure python, so it builds under either interpreter.
"""

from __future__ import annotations

import json
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
