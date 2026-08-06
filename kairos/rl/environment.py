"""KairosCADEnv: Gymnasium-compatible CAD design environment.

Episode: an engineering requirement (natural language → EngineeringSpec) plus
a fresh (or preloaded) document. Each step the agent emits
``{"operation": Discrete, "params": Box[0,1]^6, "target": Discrete}``; the
codec decodes it into a structured Action, the executor runs it against
FreeCAD, and the shaped reward tracker scores the transition.

Observations are ``{"numeric": Box, "action_mask": MultiBinary}``; the mask
comes from ``kairos.actions.masking`` and collapses the operation space to
what is legal in the current CAD state.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from kairos.actions.executor import ActionExecutor
from kairos.actions.masking import flags_from_engine, operation_mask
from kairos.cad.engine import CADEngine
from kairos.evaluation.constraints import check_constraints
from kairos.language import parse_requirement
from kairos.language.spec import EngineeringSpec
from kairos.representation.numerical_encoder import ENCODING_DIM, encode_numeric
from kairos.representation.observation import observe
from kairos.rl.action_space import (
    MAX_TARGETS,
    NUM_OPERATIONS,
    OPERATIONS,
    PARAM_SLOTS,
    TARGET_KIND,
    decode,
)
from kairos.rl.rewards import RewardTracker, RewardWeights

#: Default benchmark requirement (Task A – mounting bracket).
DEFAULT_REQUIREMENT = (
    "Create an L-bracket with 4 M5 mounting holes, 3 mm minimum wall "
    "thickness, 90 degree angle, and minimum possible mass."
)


class KairosCADEnv(gym.Env):
    """Sequential CAD design as a POMDP over structured actions."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        requirement: str = DEFAULT_REQUIREMENT,
        max_steps: int = 40,
        reward_weights: RewardWeights | None = None,
        material: str = "aluminum",
    ) -> None:
        super().__init__()
        self.requirement = requirement
        self.max_steps = int(max_steps)
        self.reward_weights = reward_weights
        self.material = material

        self.observation_space = spaces.Dict(
            {
                "numeric": spaces.Box(-np.inf, np.inf, shape=(ENCODING_DIM,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(NUM_OPERATIONS),
            }
        )
        self.action_space = spaces.Dict(
            {
                "operation": spaces.Discrete(NUM_OPERATIONS),
                "params": spaces.Box(0.0, 1.0, shape=(PARAM_SLOTS,), dtype=np.float32),
                "target": spaces.Discrete(MAX_TARGETS),
            }
        )

        self._engine: CADEngine | None = None
        self._executor: ActionExecutor | None = None
        self._tracker: RewardTracker | None = None
        self._spec: EngineeringSpec | None = None
        self._step_count = 0

    # ------------------------------------------------------------------ api

    @property
    def spec(self) -> EngineeringSpec:
        if self._spec is None:
            self._spec = parse_requirement(self.requirement)
        return self._spec

    @property
    def engine(self) -> CADEngine:
        if self._engine is None:
            raise RuntimeError("environment not reset")
        return self._engine

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if self._engine is not None:
            self._engine.close()
        requirement = options.get("requirement", self.requirement)
        self.requirement = requirement
        self._spec = parse_requirement(requirement)
        self._engine = CADEngine("episode", material=self.material)
        self._executor = ActionExecutor(self._engine)
        self._tracker = RewardTracker(self._spec, self.reward_weights)
        self._step_count = 0
        observation = observe(self._engine)
        return self._encode_obs(observation), {"requirement": requirement}

    def step(self, action: dict[str, Any]):
        if self._executor is None or self._tracker is None:
            raise RuntimeError("environment not reset")
        self._step_count += 1

        structured = decode(
            int(action["operation"]),
            np.asarray(action["params"], dtype=np.float64),
            int(action.get("target", 0)),
            targets=self._current_targets(OPERATIONS[int(action["operation"]) % NUM_OPERATIONS]),
        )
        result = self._executor.execute(structured)
        observation = observe(self.engine)
        breakdown = self._tracker.step(result, observation)
        report = self._tracker.last_report or check_constraints(observation, self.spec)

        terminated = bool(result.done)
        truncated = self._step_count >= self.max_steps and not terminated
        info = {
            "action": structured.to_dict(),
            "result": result.to_dict(),
            "reward_breakdown": breakdown.to_dict(),
            "constraints": report.to_dict(),
            "observation": observation,
        }
        return (
            self._encode_obs(observation, report),
            float(breakdown.total),
            terminated,
            truncated,
            info,
        )

    def render(self):  # pragma: no cover - thin convenience
        """Return the iso view as an RGB array (rgb_array mode)."""
        from kairos.cad.rendering import rasterize, tessellate

        shape = self.engine.document.tip_shape()
        if shape is None:
            return np.full((256, 256, 3), 245, dtype=np.uint8)
        vertices, triangles = tessellate(shape)
        return rasterize(vertices, triangles, view="iso", size=256)

    def close(self):
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    # -------------------------------------------------------------- helpers

    def _current_targets(self, op) -> dict[str, list[str]]:
        kind = TARGET_KIND.get(op)
        if kind is None:
            return {}
        try:
            if kind == "edges" and self.engine.has_solid():
                return {"edges": [e["name"] for e in self.engine.list_edges()]}
            if kind == "faces" and self.engine.has_solid():
                return {"faces": [f["name"] for f in self.engine.list_faces()]}
            if kind == "features":
                return {
                    "features": [
                        f["name"]
                        for f in self.engine.feature_history()
                        if not f["type"].startswith("Sketcher")
                    ]
                }
        except Exception:
            pass
        return {}

    def _encode_obs(self, observation: dict, report=None) -> dict[str, np.ndarray]:
        if report is None:
            report = check_constraints(observation, self.spec)
        numeric = encode_numeric(
            observation,
            self.spec,
            step=self._step_count,
            max_steps=self.max_steps,
            satisfaction_rate=report.satisfaction_rate,
            all_satisfied=report.all_measured_satisfied,
        )
        flags = flags_from_engine(self.engine)
        mask = np.asarray(operation_mask(flags, list(OPERATIONS)), dtype=np.int8)
        return {"numeric": numeric, "action_mask": mask}
