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
        designs.append({
            "design_id": path.parent.name,
            "family": (requirement.get("spec") or {}).get("kind", "unknown"),
            "requirement": requirement.get("text", ""),
            "mass_g": state.get("mass_g"),
            "volume_mm3": state.get("volume_mm3"),
            "bounding_box": state.get("bounding_box"),
            "hole_count": state.get("hole_count"),
            "min_wall_thickness_mm": state.get("min_wall_thickness_mm"),
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


def collect_benchmark(runs: str | Path) -> dict[str, Any]:
    """Leaderboard, per-policy traces and the success(k) curve."""
    runs = Path(runs)
    leaderboard = _read_json(runs / "leaderboard.json") or {}

    curves: dict[str, dict[str, list[bool]]] = {}
    for path in sorted(runs.glob("*_traces.jsonl")):
        policy = path.name.replace("_traces.jsonl", "")
        for row in _read_jsonl(path):
            if row.get("aborted"):
                continue
            task = row["task_id"]
            key = "build" if task.startswith("build-") else task.split("-")[1]
            curves.setdefault(policy, {}).setdefault(key, []).append(
                bool(row.get("finished_successfully"))
            )

    return {
        "leaderboard": leaderboard,
        "success_curve": {
            policy: {k: sum(v) / len(v) for k, v in sorted(buckets.items())}
            for policy, buckets in curves.items()
        },
    }


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


def build_bundle(
    dataset: str | Path = "dataset",
    benchmark_runs: str | Path = "runs/benchmark_core",
    runs_root: str | Path = "runs",
    limit: int = MAX_DESIGNS,
) -> dict[str, Any]:
    """Everything the dashboard renders, in one JSON-serializable dict."""
    designs = collect_designs(dataset, limit=limit)
    return {
        "designs": designs,
        "benchmark": collect_benchmark(benchmark_runs),
        "training": collect_training(runs_root),
        "families": sorted({d["family"] for d in designs}),
        "counts": {"designs_embedded": len(designs)},
    }
