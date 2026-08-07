# KAIROS — build and experiment workflows.
#
# CAD-dependent targets run under FreeCAD's bundled Python interpreter,
# resolved automatically below (override with `make FREECAD_PY=/path/to/python`).

PYTHON ?= python3
FREECAD_APP ?= /Applications/FreeCAD.app
FREECAD_PY ?= $(shell ls $(FREECAD_APP)/Contents/Resources/bin/python* 2>/dev/null | head -1)

.PHONY: setup setup-learn test test-cad test-all lint generate-data dataset-report train-bc eval-bc train-ppo eval-ppo optimize benchmark-suite benchmark demo clean

setup:
	$(PYTHON) -m pip install -e ".[dev]"

## Phase 4 learning stack (torch). Not installable under FreeCAD's python.
setup-learn:
	$(PYTHON) -m pip install -e ".[dev,learn]"

## Pure-python tests: action schema, parameter validation, masking, planning.
test:
	$(PYTHON) -m pytest -m "not cad" -q

## CAD integration tests: require FreeCAD, run under its bundled interpreter.
test-cad:
	@test -n "$(FREECAD_PY)" || (echo "FreeCAD python not found at $(FREECAD_APP); set FREECAD_PY=" && exit 1)
	PYTHONPATH=$(CURDIR) $(FREECAD_PY) -m pytest -m cad -q

test-all: test test-cad

lint:
	$(PYTHON) -m ruff check kairos tests scripts

## Procedurally generate a validated design dataset across all families.
generate-data:
	@test -n "$(FREECAD_PY)" || (echo "FreeCAD python not found at $(FREECAD_APP); set FREECAD_PY=" && exit 1)
	PYTHONPATH=$(CURDIR) $(FREECAD_PY) scripts/generate_brackets.py --count 135 --out dataset/designs

## Audit a generated dataset and regenerate docs/dataset.md.
dataset-report:
	$(PYTHON) scripts/audit_dataset.py --root dataset
	$(PYTHON) scripts/dataset_stats.py --root dataset --out docs/dataset.md

## Behavioral cloning: fit the VLA policy to the expert trajectories.
train-bc:
	$(PYTHON) scripts/train_bc.py --root dataset --out runs/bc

## Held-out evaluation and expert-vs-policy replay for a trained checkpoint.
eval-bc:
	$(PYTHON) scripts/evaluate_bc.py --checkpoint runs/bc/checkpoint.pt \
		--out runs/bc/evaluation.json
	$(PYTHON) scripts/replay_policy.py --sample 12 --out runs/bc/replay.json

## PPO fine-tuning of the BC policy against the live CAD environment.
## Runs under the torch interpreter; the environment is served out of FreeCAD's.
train-ppo:
	$(PYTHON) scripts/train_ppo.py --bc runs/bc/checkpoint.pt --out runs/ppo

## Closed-loop comparison: BC vs PPO vs a legal-random baseline.
eval-ppo:
	$(PYTHON) scripts/evaluate_ppo.py --episodes 14

## Phase 6: fit a surrogate, search for the lightest manufacturable design,
## then build the winner and report the VERIFIED numbers.
optimize:
	@test -n "$(FREECAD_PY)" || (echo "FreeCAD python not found" && exit 1)
	PYTHONPATH=$(CURDIR) $(FREECAD_PY) scripts/optimize_design.py \
		--family plate --min-thickness 5.0 --out runs/optimize

## Phase 7: freeze the benchmark split. Commit the result; it must not move.
benchmark-suite:
	$(PYTHON) scripts/benchmark_build.py --root dataset --out benchmark/kairos-cad-v1

## Run the benchmark. PRESET=smoke|core|full.
benchmark:
	$(PYTHON) scripts/run_benchmark.py --preset $(or $(PRESET),smoke) \
		--suite benchmark/kairos-cad-v1 --out runs/benchmark

## End-to-end demo: requirement → spec → build → rewards → exports (spec §50).
demo:
	@test -n "$(FREECAD_PY)" || (echo "FreeCAD python not found at $(FREECAD_APP); set FREECAD_PY=" && exit 1)
	PYTHONPATH=$(CURDIR) $(FREECAD_PY) scripts/demo.py --out outputs/demo

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
