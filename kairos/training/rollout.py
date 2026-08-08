"""Teacher-forced replay of a recorded trajectory against a trained policy.

Closed-loop rollout needs torch and FreeCAD in one interpreter, which this
project does not have (the CAD stack runs under FreeCAD's bundled python, which
has no torch). Replay is the honest stand-in: walk a recorded design's states
and ask the policy what it would do at each one, then line its answers up
against what the expert actually did.

It is *teacher forced*, every step is scored from the expert's state, not from
whatever the policy's own previous action would have produced, so it measures
per-step agreement, not the compounding error a real rollout would show. That
distinction matters when reading the numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kairos.actions.masking import operation_mask
from kairos.language import parse_requirement
from kairos.language import tokenizer as tk
from kairos.models.policy import MASK_FILL
from kairos.representation.feature_encoder import encode_history
from kairos.representation.numerical_encoder import encode_numeric
from kairos.rl.action_space import OPERATIONS, decode
from kairos.training.bc_dataset import _flags_from_numeric


@dataclass
class ReplayStep:
    """One step of a replay: what the expert did, what the policy would do."""

    index: int
    expert: str
    predicted: str
    confidence: float
    agrees: bool
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "expert": self.expert,
            "predicted": self.predicted,
            "confidence": round(self.confidence, 4),
            "agrees": self.agrees,
            "parameters": self.parameters,
        }


@torch.no_grad()
def predict_action(
    model,
    requirement: str,
    numeric: np.ndarray | None = None,
    history: list[str] | None = None,
    max_text_length: int = 64,
    history_length: int = 16,
    max_steps: int = 40,
    targets: dict[str, list[str]] | None = None,
    apply_mask: bool = True,
):
    """Predict the next action for a requirement and geometry state.

    ``numeric=None`` means an empty document, the state before any action.
    Returns ``(Action, probabilities)``.
    """
    spec = parse_requirement(requirement)
    if numeric is None:
        numeric = encode_numeric({}, spec, step=0, max_steps=max_steps)
    numeric = np.asarray(numeric, dtype=np.float32)

    ids, values, mask = tk.encode(requirement, max_length=max_text_length)
    history_ids, _ = encode_history(list(history or []), max_length=history_length)

    device = next(model.parameters()).device
    batch = {
        "token_ids": torch.tensor(np.asarray([ids]), dtype=torch.long, device=device),
        "token_values": torch.tensor(np.asarray([values]), dtype=torch.float32, device=device),
        "token_mask": torch.tensor(np.asarray([mask]), dtype=torch.long, device=device),
        "numeric": torch.tensor(numeric[None], dtype=torch.float32, device=device),
        "history": torch.tensor(history_ids[None], dtype=torch.long, device=device),
    }
    legal_mask: torch.Tensor | None = None
    if apply_mask:
        legal = operation_mask(_flags_from_numeric(numeric), list(OPERATIONS))
        legal_mask = torch.tensor(
            np.asarray([legal], dtype=np.int64), dtype=torch.long, device=device
        )
        batch["operation_mask"] = legal_mask

    outputs = model(**batch)
    logits = outputs["operation_logits"]
    if legal_mask is not None:
        # Re-applied here rather than trusted to the model: the argmax below is
        # this function's own decision, so it enforces its own legality. For a
        # real policy the logits are already masked and this changes nothing.
        logits = logits.masked_fill(legal_mask == 0, MASK_FILL)

    probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    operation_index = int(probabilities.argmax())
    action = decode(
        operation_index,
        outputs["parameters"][0].cpu().numpy(),
        0,
        targets or {"edges": ["Edge1"], "faces": ["Face1"], "features": ["Pad"]},
    )
    return action, probabilities


def replay_trajectory(
    model,
    trajectory_path: str | Path,
    max_steps: int = 40,
    **kwargs,
) -> dict[str, Any]:
    """Score a policy against one recorded design, step by step."""
    trajectory = json.loads(Path(trajectory_path).read_text())
    requirement = trajectory["requirement"]
    spec = parse_requirement(requirement)
    states = trajectory["states"]
    summaries = trajectory.get("step_summaries", [])
    initial = encode_numeric({}, spec, step=0, max_steps=max_steps)

    steps: list[ReplayStep] = []
    for i, expert_action in enumerate(trajectory["actions"]):
        numeric = np.asarray(states[i - 1], dtype=np.float32) if i > 0 else initial
        history = summaries[i - 1].get("feature_history", []) if i > 0 else []
        action, probabilities = predict_action(
            model, requirement, numeric=numeric, history=history, max_steps=max_steps, **kwargs
        )
        expert_name = expert_action["operation"]
        steps.append(
            ReplayStep(
                index=i,
                expert=expert_name,
                predicted=action.operation.value,
                confidence=float(probabilities.max()),
                agrees=action.operation.value == expert_name,
                parameters=action.parameters,
            )
        )

    agreement = sum(s.agrees for s in steps) / len(steps) if steps else 0.0
    return {
        "design_id": trajectory.get("design_id"),
        "family": trajectory.get("family"),
        "requirement": requirement,
        "steps": [s.to_dict() for s in steps],
        "agreement": round(agreement, 4),
        "teacher_forced": True,
    }


def format_replay(report: dict[str, Any]) -> str:
    """Render a replay as an expert-vs-policy table."""
    lines = [
        f"design {report.get('design_id')} [{report.get('family')}]",
        f"  {report['requirement'][:96]}",
        "",
        f"{'step':>4}  {'expert':>18}  {'policy':>18}  {'conf':>6}  ok",
    ]
    for step in report["steps"]:
        lines.append(
            f"{step['index']:>4}  {step['expert']:>18}  {step['predicted']:>18}  "
            f"{step['confidence']:>6.3f}  {'y' if step['agrees'] else 'N'}"
        )
    lines.append("")
    lines.append(
        f"agreement: {report['agreement']:.3f} "
        f"({sum(s['agrees'] for s in report['steps'])}/{len(report['steps'])} steps, "
        "teacher forced)"
    )
    return "\n".join(lines)
