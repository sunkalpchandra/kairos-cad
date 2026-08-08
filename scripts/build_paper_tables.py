#!/usr/bin/env python3
"""Generate the paper's tables and figures from artifacts on disk.

    python3 scripts/build_paper_tables.py --out paper/generated

Every number in the paper comes from here. Hand-typing results into LaTeX is
how a paper ends up claiming a figure the repo no longer produces — this repo
has already corrected a published "100% of designs satisfy their constraints"
that covered 266 designs where nothing was measured, and a "0.286 closed-loop
success" measured on a contaminated split. A generated table cannot drift from
its artifacts without the build failing.

Outputs one `.tex` fragment per table plus a `facts.tex` of macros, so prose can
say \\OracleCeiling instead of a literal that quietly goes stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kairos.dashboard.bundle import (  # noqa: E402
    collect_ablations,
    collect_benchmark,
    collect_comparisons,
    collect_training,
)

#: Human-readable names; the raw keys are terse for the CLI, not for a reader.
POLICY_LABELS = {
    "oracle-replay": r"Oracle replay\,$^\dagger$",
    "bc": "Behavioural cloning",
    "ppo": "PPO (BC-anchored)",
    "scripted-spec": "Scripted from spec",
    "immediate-finish": "Immediate finish",
    "legal-random": "Legal random",
}

ABLATION_LABELS = {
    "bc": "Behavioural cloning (intact)",
    "bc+shuffled-req": "\\quad requirement shuffled",
    "bc+blank-req": "\\quad requirement blanked",
    "bc+no-mask": "\\quad action mask removed",
}


def _fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def leaderboard_table(benchmark: dict) -> str:
    """Main results table, ordered by progress score."""
    rows = sorted(
        benchmark["leaderboard"]["policies"],
        key=lambda r: -(r["progress_mean"] or 0.0),
    )
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Policy & Progress & Success & Validity & Satisfaction & Efficiency \\",
        r"\midrule",
    ]
    for row in rows:
        label = POLICY_LABELS.get(row["policy"], row["policy"].replace("_", r"\_"))
        cells = " & ".join([
            label,
            _fmt(row["progress_mean"]),
            _fmt(row["success_rate"]),
            _fmt(row["validity"]),
            _fmt(row["satisfaction"]),
            _fmt(row["efficiency"]),
        ])
        lines.append(cells + r" \\")
        if row["policy"] == "oracle-replay":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def comparison_table(comparisons: dict, limit: int = 8) -> str:
    """Paired bootstrap intervals, strongest separation first.

    Set in \\small: policy names are long and five columns of them overflow the
    5.5in NeurIPS column, which LaTeX reports only as an overfull hbox warning
    while happily printing text into the margin.
    """
    lines = [
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Comparison & $\Delta$ progress & 95\% CI & W/L/T & Separates \\",
        r"\midrule",
    ]
    for row in comparisons["rows"][:limit]:
        name = f"{row['a']} vs.\\ {row['b']}".replace("_", r"\_")
        # Bold the negatives: "cannot separate" is the finding worth noticing,
        # and it is the one a skimming reader drops.
        verdict = "yes" if row["separates"] else r"\textbf{no}"
        lines.append(
            f"\\texttt{{{name}}} & {row['difference']:+.3f} & "
            f"$[{row['low']:+.3f}, {row['high']:+.3f}]$ & "
            f"{row['wins']}/{row['losses']}/{row['ties']} & "
            f"{verdict} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ablation_table(ablations: dict) -> str:
    """Corrupted-input conditions as deltas against the intact policy."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Condition & Progress & $\Delta$ & Validity \\",
        r"\midrule",
    ]
    for row in ablations["rows"]:
        label = ABLATION_LABELS.get(row["name"], row["name"].replace("_", r"\_"))
        delta = "--" if row["delta"] is None else (
            "---" if row["baseline"] else f"{row['delta'] * 100:+.1f}\\%"
        )
        lines.append(
            f"{label} & {_fmt(row['progress_mean'])} & {delta} & "
            f"{_fmt(row['validity'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def success_curve_table(benchmark: dict) -> str:
    """Success against the number of trailing actions the policy must supply."""
    curves = benchmark["success_curve"]
    ks = sorted({int(k) for c in curves.values() for k in c if k != "build"})
    order = [p for p in ("oracle-replay", "bc", "ppo", "immediate-finish") if p in curves]

    header = " & ".join([f"$k{{=}}{k}$" for k in ks])
    lines = [
        r"\begin{tabular}{l" + "c" * (len(ks) + 1) + "}",
        r"\toprule",
        f"Policy & {header} & full build \\\\",
        r"\midrule",
    ]
    for policy in order:
        curve = curves[policy]
        cells = " & ".join(_fmt(curve.get(str(k)), 3) for k in ks)
        lines.append(
            f"{POLICY_LABELS.get(policy, policy)} & {cells} & "
            f"{_fmt(curve.get('build'), 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def facts(benchmark: dict, training: dict, comparisons: dict, codec: dict | None) -> str:
    """LaTeX macros so prose never hard-codes a number."""
    scores = {r["policy"]: r for r in benchmark["leaderboard"]["policies"]}

    def score(policy: str, field: str = "progress_mean") -> str:
        return _fmt((scores.get(policy) or {}).get(field))

    bc_ppo = next(
        (r for r in comparisons["rows"]
         if {r["a"], r["b"]} == {"bc", "ppo"}), None
    )
    macros = {
        "OracleCeiling": score("oracle-replay"),
        "OracleSuccess": score("oracle-replay", "success_rate"),
        "BCProgress": score("bc"),
        "BCSuccess": score("bc", "success_rate"),
        "PPOProgress": score("ppo"),
        "PPOSuccess": score("ppo", "success_rate"),
        "RandomProgress": score("legal-random"),
        "BenchmarkTasks": str(benchmark.get("tasks") or "--"),
        "BCHeldOutAccuracy": _fmt(training["bc"].get("best_held_out_accuracy")),
        "BCEpochs": str(len(training["bc"].get("history", []))),
        "PPOIterations": str(len(training["ppo"].get("history", []))),
    }
    if bc_ppo:
        macros["BCvsPPODelta"] = f"{bc_ppo['difference']:+.3f}"
        macros["BCvsPPOCI"] = f"[{bc_ppo['low']:+.3f}, {bc_ppo['high']:+.3f}]"
        macros["BCvsPPOWLT"] = f"{bc_ppo['wins']}/{bc_ppo['losses']}/{bc_ppo['ties']}"
    if codec:
        macros["CodecSteps"] = f"{codec['steps']:,}"
        macros["CodecUnusable"] = str(codec["unusable"])
        macros["CodecWorstDrift"] = f"{codec['worst_round_trip_mm']:.4f}"

    return "\n".join(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="runs/benchmark_core")
    parser.add_argument("--ablation", default="runs/ablation")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--codec", default="runs/codec_audit.json")
    parser.add_argument("--out", default="paper/generated")
    args = parser.parse_args()

    benchmark = collect_benchmark(args.benchmark)
    comparisons = collect_comparisons(args.benchmark)
    ablations = collect_ablations(args.ablation)
    training = collect_training(args.runs)
    codec = None
    if Path(args.codec).exists():
        codec = json.loads(Path(args.codec).read_text())

    if not benchmark["leaderboard"]["policies"]:
        print(f"error: no leaderboard under {args.benchmark}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {
        "leaderboard.tex": leaderboard_table(benchmark),
        "comparisons.tex": comparison_table(comparisons),
        "ablations.tex": ablation_table(ablations),
        "success_curve.tex": success_curve_table(benchmark),
        "facts.tex": facts(benchmark, training, comparisons, codec),
    }
    for name, body in written.items():
        (out / name).write_text(body + "\n")
        print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
