"""Wire protocol between the CAD environment and a torch-side trainer.

Phase 5 needs one process holding both FreeCAD and torch, and no such process
exists: the CAD stack runs under FreeCAD's bundled interpreter, which has no
torch and cannot practically get one. So the environment is served *out* of
that interpreter over newline-delimited JSON on stdin/stdout, and the trainer
drives it from the interpreter that does have torch.

JSON rather than pickle deliberately — the two ends run different Python
versions (3.11 vs 3.12), and pickle across versions is a portability trap. The
payloads are small (a 24-float state, two boolean masks), so encoding cost is
irrelevant next to a FreeCAD recompute.

This module is imported by **both** sides, so it must stay dependency-free:
no torch, no FreeCAD, no numpy.
"""

from __future__ import annotations

import json
from typing import Any

#: Bumped whenever a message's shape changes. The client refuses to talk to a
#: server that does not match, because a silent field mismatch would surface as
#: a training bug days later rather than a startup error now.
PROTOCOL_VERSION = 1

RESET = "reset"
STEP = "step"
CLOSE = "close"
HANDSHAKE = "handshake"


class ProtocolError(RuntimeError):
    """The peer sent something unparseable, or refused a request."""


def encode_message(payload: dict[str, Any]) -> str:
    """Serialize one message to a single line (no embedded newlines)."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def decode_message(line: str) -> dict[str, Any]:
    """Parse one line into a message, raising :class:`ProtocolError`."""
    line = line.strip()
    if not line:
        raise ProtocolError("empty message")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as err:
        raise ProtocolError(f"malformed message: {err}") from err
    if not isinstance(payload, dict):
        raise ProtocolError(f"expected an object, got {type(payload).__name__}")
    return payload


# ------------------------------------------------------------------ requests


def handshake_request() -> dict[str, Any]:
    return {"cmd": HANDSHAKE, "version": PROTOCOL_VERSION}


def reset_request(requirement: str | None = None, seed: int | None = None) -> dict[str, Any]:
    return {"cmd": RESET, "requirement": requirement, "seed": seed}


def step_request(operation: int, params: list[float], target: int = 0) -> dict[str, Any]:
    return {
        "cmd": STEP,
        "operation": int(operation),
        "params": [float(v) for v in params],
        "target": int(target),
    }


def close_request() -> dict[str, Any]:
    return {"cmd": CLOSE}


# ----------------------------------------------------------------- responses


def ok_response(**fields: Any) -> dict[str, Any]:
    return {"ok": True, **fields}


def error_response(message: str, kind: str = "error") -> dict[str, Any]:
    return {"ok": False, "error": str(message), "kind": kind}


def raise_for_error(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload, or raise if the peer reported a failure."""
    if not payload.get("ok", False):
        raise ProtocolError(payload.get("error", "peer reported an unspecified failure"))
    return payload


def check_version(payload: dict[str, Any]) -> None:
    """Verify a handshake reply came from a matching protocol version."""
    raise_for_error(payload)
    version = payload.get("version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol mismatch: client speaks v{PROTOCOL_VERSION}, server speaks v{version}"
        )
