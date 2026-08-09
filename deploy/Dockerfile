# KAIROS live sandbox.
#
# Two interpreters in one image, which is the whole difficulty:
#
#   * FreeCAD ships its own Python and cannot import torch.
#   * The learning stack needs torch and cannot import FreeCAD.
#
# They talk over the newline-delimited JSON bridge in kairos/rl/. The CAD side
# runs as env_server under FreeCAD's interpreter; the trainer runs under the
# system interpreter and spawns it.
#
# Build:  docker build -f deploy/Dockerfile -t kairos:latest .
# Shell:  docker run --rm -it kairos:latest bash
# Bench:  docker run --rm -v "$PWD/runs:/app/runs" kairos:latest \
#             python3 scripts/run_benchmark.py --preset smoke
#
# The image carries no dataset. Generating 1,080 designs takes ~9 minutes on
# 8 shards and writes ~138 MB, so it is a volume mount, not a layer.

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KAIROS_PYTHON=/usr/bin/freecadcmd

# freecad-python3 provides FreeCADCmd, the headless console binary. No X server
# and no GUI libraries: nothing in the agent path opens a window, and the
# renders are opt-in and produced by the repo's own software rasterizer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        freecad-python3 \
        python3 \
        python3-pip \
        python3-venv \
        libgl1 \
        libglu1-mesa \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so a source edit does not re-resolve torch.
COPY pyproject.toml requirements.txt ./
RUN python3 -m pip install --break-system-packages -r requirements.txt \
    && python3 -m pip install --break-system-packages \
        "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

COPY kairos ./kairos
COPY scripts ./scripts
COPY configs ./configs
COPY benchmark ./benchmark
COPY Makefile ./

ENV PYTHONPATH=/app

# Fail the build if the two interpreters are not both wired up. A container
# that starts and then cannot import FreeCAD is the failure this catches.
RUN python3 -c "import kairos.rl.action_space, kairos.benchmark; print('torch side ok')" \
    && freecadcmd -c "import sys; sys.path.insert(0, '/app'); \
import kairos.cad.engine; print('freecad side ok')"

CMD ["python3", "-c", "print('KAIROS image. See deploy/README.md for entry points.')"]
