"""Bundle run artifacts into one self-contained payload for the dashboard.

The dashboard is a single HTML file with the data inlined, not a server reading
the repo. Two reasons: a static file can be opened from disk, mailed, or
attached to a paper without anything running, and inlining forces the bundle to
be **derived from the artifacts on disk** rather than from whatever a live
process happens to hold in memory — the same discipline that keeps
`benchmark_report.py` reading only traces.

Everything here is pure python, so the bundle can be built under either
interpreter.
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _measured(results: list[dict[str, Any]], kind: str) -> float | None:
    """The measured value of a named constraint, if it was measured at all."""
    for result in results:
        if (result.get("constraint") or {}).get("kind") == kind:
            return result.get("measured")
    return None


def collect_designs(root: str | Path, limit: int = MAX_DESIGNS) -> list[dict[str, Any]]:
    """Design summaries with their requirement and constraint outcome."""
    designs: list[dict[str, Any]] = []
    for path in sorted(Path(root).glob("designs/design_*/state.json"))[:limit]:
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
    return designs


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
            "history": bc.get("history", []),
            "dataset": bc.get("dataset", {}),
            "parameters": (bc.get("model") or {}).get("parameters"),
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


def attach_meshes(designs: list[dict[str, Any]], root: str | Path) -> int:
    """Attach a viewer mesh to each design; returns how many succeeded.

    A design whose STL is missing or unreadable keeps every one of its metrics
    and simply renders without geometry — a bundle is still useful without a
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
    return {
        "designs": designs,
        "benchmark": collect_benchmark(benchmark_runs),
        "comparisons": collect_comparisons(benchmark_runs),
        "ablations": collect_ablations(ablation_runs),
        "training": collect_training(runs_root),
        "families": sorted({d["family"] for d in designs}),
        "counts": {"designs_embedded": len(designs), "meshes_attached": attached},
    }
