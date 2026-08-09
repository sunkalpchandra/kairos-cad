"""CAD integration tests: trajectory recording during generation."""

import json
import random

import pytest

from kairos.actions.executor import ActionExecutor
from kairos.data.families import get_family
from kairos.data.generator import GenerationStats, generate_design
from kairos.data.trajectories import TrajectoryRecorder
from kairos.representation.numerical_encoder import ENCODING_DIM

pytestmark = pytest.mark.cad


def test_recorder_captures_states_rewards_and_actions(engine):
    family = get_family("plate")
    params = family.params_cls()
    requirement = family.requirements(params)["text"]
    executor = ActionExecutor(engine)
    recorder = TrajectoryRecorder(executor, requirement)
    family.build(executor, params)

    data = recorder.to_dict()
    n = len(data["actions"])
    assert n == len(data["states"]) == len(data["rewards"]) > 4
    assert all(len(s) == ENCODING_DIM for s in data["states"])
    # The expert recipe satisfies its own requirement: final constraints ok.
    final = data["final_metrics"]
    assert final["invalid_actions"] == 0
    assert final["constraints"]["counts"]["violated"] == 0
    assert final["summary"]["valid"] is True
    # Expert trajectories should end with strongly positive finish reward.
    assert data["rewards"][-1] >= 4.0
    assert final["total_reward"] > 3.0


def test_recorder_scores_expert_shaping_events(engine):
    family = get_family("l_bracket")
    params = family.params_cls(fillet_radius=2.0)
    executor = ActionExecutor(engine)
    recorder = TrajectoryRecorder(executor, family.requirements(params)["text"])
    family.build(executor, params)
    components = set()
    for breakdown in recorder.to_dict()["reward_breakdowns"]:
        components |= set(breakdown["components"])
    assert "valid_sketch" in components
    assert "first_solid" in components
    assert "all_constraints" in components
    assert "finish" in components


def test_generator_writes_trajectory_files(tmp_path):
    rng = random.Random(5)
    stats = GenerationStats()
    designs_dir = tmp_path / "designs"
    written = False
    design_id = 0
    while not written and design_id < 20:
        written = generate_design("plate", rng, designs_dir, design_id, stats)
        design_id += 1
    assert written

    # The trajectory lives beside its design. A separate trajectories/ tree
    # used to hold a byte-identical copy of each one; the generator no longer
    # writes it, and this test used to pass that directory into what is now the
    # `render` parameter, so it silently asked for PNGs and then looked for
    # json files that were never going to be there.
    trajectory_files = sorted(designs_dir.glob("design_*/trajectory.json"))
    assert len(trajectory_files) == 1
    data = json.loads(trajectory_files[0].read_text())
    assert data["family"] == "plate"
    assert data["requirement"].startswith("Design a rectangular")
    assert len(data["actions"]) == len(data["rewards"])
    assert data["design_id"].startswith("design_")
