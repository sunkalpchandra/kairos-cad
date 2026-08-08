"""Serve :class:`KairosCADEnv` over stdin/stdout for an out-of-process trainer.

Runs under FreeCAD's bundled interpreter:

    PYTHONPATH=. /Applications/FreeCAD.app/Contents/Resources/bin/python \\
        -m kairos.rl.env_server

Reads one JSON request per line, writes one JSON response per line. Anything
the environment prints (FreeCAD is chatty on import) would corrupt that stream,
so stdout is captured at startup and every stray write is redirected to stderr.

A step that raises is reported as a failed *step*, not a dead server: a broken
recompute is an ordinary event in CAD RL, and the trainer should score it and
move on rather than lose the whole rollout.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

from kairos.representation.observation import observe
from kairos.rl.protocol import (
    CLOSE,
    HANDSHAKE,
    PROTOCOL_VERSION,
    REPLAY,
    RESET,
    STEP,
    ProtocolError,
    decode_message,
    encode_message,
    error_response,
    ok_response,
)


def _observation_payload(env, observation: dict[str, Any], info: dict | None = None) -> dict:
    """Flatten an observation into JSON-safe lists the policy can consume.

    Carries the feature history too: the environment's own observation space
    does not include it, but the VLA's state encoder needs it, and recovering
    it on the trainer side would mean a second source of truth.
    """
    summary = (info or {}).get("observation", {}).get("summary", {})
    if not summary:
        try:
            from kairos.representation.observation import observe

            summary = observe(env.engine).get("summary", {})
        except Exception:
            summary = {}
    return {
        "numeric": [float(v) for v in observation["numeric"]],
        "action_mask": [int(v) for v in observation["action_mask"]],
        # Target names, so a policy emitting a target INDEX can know what the
        # indices refer to. Without these the choice is blind and the oracle
        # cannot reproduce the expert's own edge.
        "targets": {
            kind: list(names)
            for kind, names in (observation.get("targets") or {}).items()
        },
        "feature_history": [str(f) for f in summary.get("feature_history", [])],
        "has_solid": bool(summary.get("has_solid", False)),
        # Counted geometry, not inferred. The benchmark's has_any_hole
        # milestone used to be awarded from satisfaction_rate > 0, so a
        # satisfied mounting_angle (true of any prismatic solid) bought credit
        # for holes that were never drilled.
        #
        # Read from `summary`, which every path here computes. Reading it from
        # info["observation"]["holes"] instead left RESET and REPLAY at zero,
        # and since credit is prefix-scored a zero here also zeroed every later
        # milestone: the oracle finished 94% of tasks and scored 0.111.
        "hole_count": int(summary.get("hole_count", 0) or 0),
        "valid": bool(summary.get("valid", False)),
        "mass_g": float(summary.get("mass_g") or 0.0),
    }


def _step_info(info: dict[str, Any]) -> dict[str, Any]:
    """Trim step info to what training and logging actually read."""
    constraints = info.get("constraints") or {}
    result = info.get("result") or {}
    return {
        "operation": (info.get("action") or {}).get("operation"),
        "ok": bool(result.get("ok", False)),
        "message": str(result.get("message", ""))[:200],
        "reward_components": (info.get("reward_breakdown") or {}).get("components", {}),
        "satisfaction_rate": float(constraints.get("satisfaction_rate", 0.0)),
        "all_satisfied": bool(constraints.get("all_measured_satisfied", False)),
    }


class EnvironmentServer:
    """Request loop over a single :class:`KairosCADEnv`."""

    def __init__(self, stdin=None, stdout=None, env=None, **env_kwargs) -> None:
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self.env_kwargs = env_kwargs
        self._env = env
        self._closed = False

    # ------------------------------------------------------------------ env

    @property
    def env(self):
        if self._env is None:
            from kairos.rl.environment import KairosCADEnv

            self._env = KairosCADEnv(**self.env_kwargs)
        return self._env

    # ------------------------------------------------------------- handlers

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("cmd")
        if command == HANDSHAKE:
            return ok_response(version=PROTOCOL_VERSION)
        if command == RESET:
            return self._handle_reset(request)
        if command == STEP:
            return self._handle_step(request)
        if command == REPLAY:
            return self._handle_replay(request)
        if command == CLOSE:
            self._closed = True
            try:
                if self._env is not None:
                    self._env.close()
            except Exception as err:  # pragma: no cover - teardown best effort
                return error_response(f"close failed: {err}")
            return ok_response(closed=True)
        return error_response(f"unknown command {command!r}", kind="unknown_command")

    def _handle_reset(self, request: dict[str, Any]) -> dict[str, Any]:
        options = {}
        if request.get("requirement"):
            options["requirement"] = request["requirement"]
        try:
            observation, info = self.env.reset(seed=request.get("seed"), options=options)
        except Exception as err:
            return error_response(f"reset failed: {err}", kind="reset_failed")
        return ok_response(
            observation=_observation_payload(self.env, observation),
            requirement=info.get("requirement", ""),
        )

    def _handle_step(self, request: dict[str, Any]) -> dict[str, Any]:
        action = {
            "operation": int(request.get("operation", 0)),
            "params": [float(v) for v in request.get("params", [])],
            "target": int(request.get("target", 0)),
        }
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
        except Exception as err:
            # A failed recompute is a normal event, not a dead server: report it
            # as a step outcome so the trainer can penalize and continue.
            return ok_response(
                observation=None,
                reward=0.0,
                terminated=False,
                truncated=True,
                info={"ok": False, "message": f"step raised: {err}"[:200], "crashed": True},
            )
        return ok_response(
            observation=_observation_payload(self.env, observation, info),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=_step_info(info),
        )

    def _handle_replay(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute raw expert actions, bypassing the policy action space."""
        from kairos.actions.schema import Action, Operation

        executed = 0
        for raw in request.get("actions", []):
            try:
                action = Action(
                    Operation(raw["operation"]),
                    target=raw.get("target"),
                    parameters=raw.get("parameters") or {},
                )
                result = self.env._executor.execute(action)
            except Exception as err:
                return error_response(
                    f"replay failed at action {executed}: {err}", kind="replay_failed"
                )
            if not result.ok:
                return error_response(
                    f"replay rejected at action {executed}: {result.message}"[:200],
                    kind="replay_rejected",
                )
            executed += 1

        observation = self.env._encode_obs(observe(self.env.engine))
        return ok_response(
            observation=_observation_payload(self.env, observation),
            executed=executed,
        )

    # ------------------------------------------------------------- run loop

    def serve(self) -> int:
        """Process requests until stdin closes or a close command arrives."""
        for line in self.stdin:
            if not line.strip():
                continue
            try:
                request = decode_message(line)
            except ProtocolError as err:
                self._write(error_response(str(err), kind="protocol"))
                continue
            try:
                response = self.handle(request)
            except Exception as err:  # pragma: no cover - defensive
                traceback.print_exc(file=sys.stderr)
                response = error_response(f"unhandled: {err}")
            self._write(response)
            if self._closed:
                break
        return 0

    def _write(self, payload: dict[str, Any]) -> None:
        self.stdout.write(encode_message(payload) + "\n")
        self.stdout.flush()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--material", default="aluminum")
    parser.add_argument("--config", default=None, help="YAML supplying reward weights")
    args = parser.parse_args(argv)

    env_kwargs = {"max_steps": args.max_steps, "material": args.material}
    if args.config:
        # The reward weights live on this side of the bridge, so the config
        # has to be read here or the `reward:` section is inert.
        from kairos.config import load_config, reward_weights_from

        env_kwargs["reward_weights"] = reward_weights_from(load_config(args.config))

    # FreeCAD writes banners to stdout on import; anything landing there would
    # be parsed as a response. Hand the real stdout to the protocol only.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    server = EnvironmentServer(stdin=sys.stdin, stdout=real_stdout, **env_kwargs)
    return server.serve()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
