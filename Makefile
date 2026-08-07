# KAIROS — build and experiment workflows.
#
# CAD-dependent targets run under FreeCAD's bundled Python interpreter,
# resolved automatically below (override with `make FREECAD_PY=/path/to/python`).

PYTHON ?= python3
FREECAD_APP ?= /Applications/FreeCAD.app
FREECAD_PY ?= $(shell ls $(FREECAD_APP)/Contents/Resources/bin/python* 2>/dev/null | head -1)

.PHONY: setup setup-learn test test-cad test-all lint generate-data dataset-report train-bc demo clean

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

## End-to-end demo: requirement → spec → build → rewards → exports (spec §50).
demo:
	@test -n "$(FREECAD_PY)" || (echo "FreeCAD python not found at $(FREECAD_APP); set FREECAD_PY=" && exit 1)
	PYTHONPATH=$(CURDIR) $(FREECAD_PY) scripts/demo.py --out outputs/demo

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
