"""Drive a FreeCAD-hosted :class:`KairosCADEnv` from the torch interpreter.

Spawns ``kairos.rl.env_server`` under FreeCAD's python and speaks the JSON
protocol to it, presenting an ordinary ``reset``/``step``/``close`` surface so
PPO never learns that an interpreter boundary exists.

Two failure modes get explicit handling because both are routine here rather
than exceptional:

- **The server dies mid-rollout.** FreeCAD can segfault on a pathological
  recompute. The client detects the closed pipe, restarts the subprocess, and
  reports the episode as truncated instead of losing the run.
- **A step hangs.** Some recomputes do not return. Each request carries a
  timeout; exceeding it restarts the server rather than blocking training
  forever.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from kairos.rl.protocol import (
    ProtocolError,
    check_version,
    close_request,
    decode_message,
    encode_message,
    handshake_request,
    raise_for_error,
    replay_request,
    reset_request,
    step_request,
)

#: Where FreeCAD's interpreter usually lives on macOS; override with
#: KAIROS_FREECAD_PYTHON or the constructor argument.
DEFAULT_FREECAD_PYTHON = "/Applications/FreeCAD.app/Contents/Resources/bin/python"


def resolve_freecad_python(explicit: str | None = None) -> str:
    """Find the interpreter that can import FreeCAD.

    An explicitly named interpreter that does not exist is an error, not a
    reason to fall back: silently training against a different interpreter than
    the one asked for is worse than refusing to start.
    """
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"requested interpreter does not exist: {explicit}")
        return explicit
    for candidate in (os.environ.get("KAIROS_FREECAD_PYTHON"), DEFAULT_FREECAD_PYTHON):
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "no FreeCAD python found; set KAIROS_FREECAD_PYTHON to its interpreter"
    )


class RemoteEnvError(RuntimeError):
    """The environment subprocess could not be started or spoke nonsense."""


class RemoteCADEnv:
    """Client-side handle on an out-of-process CAD environment."""

    def __init__(
        self,
        requirement: str | None = None,
        max_steps: int = 40,
        python: str | None = None,
        repo_root: str | Path | None = None,
        timeout: float = 120.0,
        auto_restart: bool = True,
        material: str = "aluminum",
        config: str | Path | None = None,
    ) -> None:
        self.requirement = requirement
        self.max_steps = int(max_steps)
        self.python = resolve_freecad_python(python)
        self.repo_root = str(repo_root or Path(__file__).resolve().parent.parent.parent)
        self.timeout = float(timeout)
        self.auto_restart = auto_restart
        self.material = material
        # Default to the project config so the `reward:` and `environment:`
        # sections actually reach the server. Without this a reward-weight
        # ablation is a silent no-op.
        if config is None:
            default = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
            config = default if default.exists() else None
        self.config = str(config) if config else None

        self.restarts = 0
        self.timeouts = 0
        self._process: subprocess.Popen | None = None
        self._start()

    # ------------------------------------------------------------- lifecycle

    def _start(self) -> None:
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            f"{self.repo_root}{os.pathsep}{existing}" if existing else self.repo_root
        )
        try:
            self._process = subprocess.Popen(
                [
                    self.python, "-u", "-m", "kairos.rl.env_server",
                    # Without these the server silently builds a default
                    # environment and the client's max_steps is a no-op.
                    "--max-steps", str(self.max_steps),
                    "--material", self.material,
                    *(["--config", self.config] if self.config else []),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=self.repo_root,
                env=environment,
            )
        except OSError as err:
            raise RemoteEnvError(f"could not start {self.python}: {err}") from err

        try:
            check_version(self._request(handshake_request(), allow_restart=False))
        except (ProtocolError, RemoteEnvError, TimeoutError, OSError) as err:
            self.close()
            raise RemoteEnvError(f"handshake failed: {err}") from err

    def _restart(self) -> None:
        self._terminate()
        self.restarts += 1
        self._start()

    def _terminate(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.stdin.close()
        except Exception:
            pass
        try:
            process.wait(timeout=5)
        except Exception:
            process.kill()

    def close(self) -> None:
        """Ask the server to shut down, then make sure it actually did."""
        if self._process is not None and self._process.poll() is None:
            try:
                self._send(close_request())
                self._read_line(timeout=5.0)
            except Exception:
                pass
        self._terminate()

    def __enter__(self) -> RemoteCADEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------- messaging

    def _send(self, payload: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RemoteEnvError("environment process is not running")
        self._process.stdin.write(encode_message(payload) + "\n")
        self._process.stdin.flush()

    def _read_line(self, timeout: float | None = None) -> str:
        """Read one response line, raising on timeout or a closed pipe."""
        if self._process is None or self._process.stdout is None:
            raise RemoteEnvError("environment process is not running")

        result: list[str] = []
        error: list[BaseException] = []

        def reader() -> None:
            try:
                result.append(self._process.stdout.readline())
            except BaseException as err:  # pragma: no cover - pipe teardown
                error.append(err)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout if timeout is not None else self.timeout)
        if thread.is_alive():
            self.timeouts += 1
            raise TimeoutError(f"environment did not answer within {self.timeout}s")
        if error:
            raise RemoteEnvError(f"read failed: {error[0]}")
        line = result[0] if result else ""
        if not line:
            raise RemoteEnvError("environment process closed its output")
        return line

    def _request(self, payload: dict[str, Any], allow_restart: bool = True) -> dict[str, Any]:
        try:
            self._send(payload)
            return decode_message(self._read_line())
        except (RemoteEnvError, ProtocolError, TimeoutError, OSError, ValueError) as err:
            # ProtocolError included deliberately: a malformed line or a
            # server-side error_response leaves the stream desynced, which
            # is exactly the case a restart exists for.
            if not (allow_restart and self.auto_restart):
                raise
            try:
                self._restart()
            except Exception as restart_err:
                # Even a failed restart must reach the caller as one error type,
                # so a rollout loop has a single thing to catch.
                raise RemoteEnvError(
                    f"environment died ({err}) and could not restart: {restart_err}"
                ) from err
            raise RemoteEnvError(f"environment restarted after: {err}") from err

    # ------------------------------------------------------------------ api

    def reset(self, requirement: str | None = None, seed: int | None = None) -> dict[str, Any]:
        """Start a new episode; returns the decoded observation."""
        payload = raise_for_error(
            self._request(reset_request(requirement or self.requirement, seed))
        )
        self.requirement = payload.get("requirement", self.requirement)
        return self._decode_observation(payload["observation"])

    def replay(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Execute raw expert actions; returns the observation, or None on failure."""
        if not actions:
            return None
        try:
            payload = raise_for_error(self._request(replay_request(actions)))
        except (RemoteEnvError, ProtocolError):
            return None
        observation = payload.get("observation")
        return self._decode_observation(observation) if observation else None

    def step(self, operation: int, params, target: int = 0):
        """Take one action; returns ``(obs, reward, terminated, truncated, info)``."""
        try:
            payload = raise_for_error(
                self._request(step_request(operation, list(np.asarray(params).ravel()), target))
            )
        except (RemoteEnvError, ProtocolError) as err:
            # The subprocess died or hung and has been restarted; the episode
            # is over, but training continues.
            return None, 0.0, False, True, {"ok": False, "message": str(err), "crashed": True}

        observation = payload.get("observation")
        return (
            self._decode_observation(observation) if observation else None,
            float(payload.get("reward", 0.0)),
            bool(payload.get("terminated", False)),
            bool(payload.get("truncated", False)),
            payload.get("info", {}),
        )

    @staticmethod
    def _decode_observation(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "numeric": np.asarray(payload["numeric"], dtype=np.float32),
            "action_mask": np.asarray(payload["action_mask"], dtype=np.int64),
            # Left as plain lists of strings: these are names to match against,
            # not values to do arithmetic on.
            "targets": payload.get("targets") or {},
            "feature_history": list(payload.get("feature_history", [])),
            "has_solid": bool(payload.get("has_solid", False)),
            "valid": bool(payload.get("valid", False)),
            "mass_g": float(payload.get("mass_g", 0.0)),
        }


def probe_environment(python: str | None = None) -> tuple[bool, str]:
    """Check whether a bridged environment can start; never raises.

    Used by CLIs and tests to skip cleanly on machines without FreeCAD.
    """
    try:
        env = RemoteCADEnv(python=python, auto_restart=False)
    except Exception as err:
        return False, str(err)
    try:
        env.reset()
        return True, "ok"
    except Exception as err:
        return False, str(err)
    finally:
        env.close()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    ok, message = probe_environment()
    print(f"bridged environment: {'ok' if ok else 'unavailable'} ({message})")
    sys.exit(0 if ok else 1)
