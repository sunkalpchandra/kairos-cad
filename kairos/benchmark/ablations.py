"""Ablations: perturb one input and measure whether the policy notices.

Ablations test whether a requirement-conditioned policy actually reads the
requirement. With eight families and near-fixed build recipes, a policy can
score well by learning what CAD builds look like; aggregate scores do not
separate that from requirement following.

:class:`ShuffledRequirement` hands the policy another task's requirement while
scoring against the real one. A score that does not move means the requirement
is unused, which is what the current run reports.

Ablations are wrappers rather than separate policies, so tasks, ordering and
seeds are identical between the perturbed and unperturbed runs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from kairos.benchmark.baselines import BenchmarkPolicy


class AblationWrapper(BenchmarkPolicy):
    """Base wrapper: delegates everything, perturbs one thing."""

    def __init__(self, inner: BenchmarkPolicy, suffix: str) -> None:
        self.inner = inner
        self.name = f"{inner.name}+{suffix}"

    def begin_episode(self, task, seed: int = 0) -> None:
        self.inner.begin_episode(self._perturb_task(task), seed=seed)

    def act(self, observation: dict, task, step: int) -> tuple[int, np.ndarray, int]:
        return self.inner.act(
            self._perturb_observation(observation), self._perturb_task(task), step
        )

    def _perturb_task(self, task):
        return task

    def _perturb_observation(self, observation: dict) -> dict:
        return observation


class ShuffledRequirement(AblationWrapper):
    """Hands the policy a different task's requirement.

    Scoring still uses the real task, so a policy that reads the text should
    collapse. One that has memorized a build prior will not move, and that
    result would mean the language encoder is decorative, which no aggregate
    score can reveal.
    """

    def __init__(self, inner: BenchmarkPolicy, requirements: list[str], seed: int = 0) -> None:
        super().__init__(inner, "shuffled-req")
        self._pool = list(requirements)
        self._rng = np.random.default_rng(seed)
        self._swap: str | None = None

    def begin_episode(self, task, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        # Draw a requirement that is genuinely different from this task's.
        others = [r for r in self._pool if r != task.requirement] or self._pool
        self._swap = str(others[int(self._rng.integers(0, len(others)))])
        self.inner.begin_episode(self._perturb_task(task), seed=seed)

    def _perturb_task(self, task):
        if self._swap is None:
            return task
        return replace(task, requirement=self._swap)


class BlankRequirement(AblationWrapper):
    """Replaces the requirement with a contentless one.

    Distinguishes "reads the requirement" from "reacts to any text at all":
    shuffling swaps meaning, blanking removes it.
    """

    BLANK = "Design a part."

    def __init__(self, inner: BenchmarkPolicy) -> None:
        super().__init__(inner, "blank-req")

    def _perturb_task(self, task):
        return replace(task, requirement=self.BLANK)


class NoActionMask(AblationWrapper):
    """Strips the legality mask from the observation.

    PPO's headline Phase 5 gain was a zero invalid-action rate. That is only a
    policy result if the policy is what produced it, with the mask removed,
    whatever remains is the policy's own doing.
    """

    def __init__(self, inner: BenchmarkPolicy) -> None:
        super().__init__(inner, "no-mask")

    def _perturb_observation(self, observation: dict) -> dict:
        if "action_mask" not in observation:
            return observation
        stripped = dict(observation)
        stripped["action_mask"] = np.ones_like(
            np.asarray(observation["action_mask"], dtype=np.int64)
        )
        return stripped


def build_ablations(
    policy: BenchmarkPolicy, requirements: list[str], seed: int = 0
) -> dict[str, BenchmarkPolicy]:
    """Every ablation of one policy, by wrapped name."""
    wrapped = [
        ShuffledRequirement(policy, requirements, seed=seed),
        BlankRequirement(policy),
        NoActionMask(policy),
    ]
    return {w.name: w for w in wrapped}
