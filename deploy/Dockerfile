# KAIROS live sandbox.
#
# The two-interpreter split that shapes this project is a macOS artefact, not a
# universal one. There, FreeCAD.app bundles its own Python which cannot import
# torch, so the CAD half runs as env_server behind the JSON bridge in
# kairos/rl/ and the trainer spawns it.
#
# Debian's freecad-python3 installs into the SYSTEM python instead, so both
# stacks can live in one interpreter here. The bridge still works unchanged and
# is what the benchmark and PPO use; it simply is not forced by packaging.
# Anything importing FreeCAD still needs the module directory on PYTHONPATH,
# which the Debian package does not add (see below).
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

# FreeCAD's modules are not on the default sys.path under Debian's package.
# The real directory is whatever holds FreeCAD.so; CI discovers it with
# `dpkg -L freecad-python3`, and the import check below fails the build if this
# value is wrong rather than producing an image that dies at runtime.
ENV PYTHONPATH=/app:/usr/lib/freecad-python3/lib

# Fail the build if the two interpreters are not both wired up. A container
# that starts and then cannot import FreeCAD is the failure this catches.
RUN python3 -c "import kairos.rl.action_space, kairos.benchmark; print('torch side ok')" \
    && python3 -c "import FreeCAD; print('freecad', '.'.join(FreeCAD.Version()[:2]))" \
    && python3 -c "import kairos.cad.engine; print('cad side ok')"

CMD ["python3", "-c", "print('KAIROS image. See deploy/README.md for entry points.')"]
