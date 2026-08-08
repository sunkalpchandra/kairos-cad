"""Protocol tests, must pass under both interpreters (no torch, no FreeCAD)."""

import math

import pytest

from kairos.rl import protocol as p


def test_messages_round_trip():
    for message in (
        p.handshake_request(),
        p.reset_request("Design a plate", seed=3),
        p.step_request(4, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], target=2),
        p.close_request(),
        p.ok_response(reward=1.5, terminated=False),
        p.error_response("boom"),
    ):
        assert p.decode_message(p.encode_message(message)) == message


def test_encoded_messages_are_single_lines():
    """Framing is newline-delimited; an embedded newline would desynchronize."""
    line = p.encode_message(p.reset_request("multi\nline\nrequirement"))
    assert "\n" not in line
    assert p.decode_message(line)["requirement"] == "multi\nline\nrequirement"


def test_non_finite_values_are_rejected_at_encode_time():
    """NaN is not valid JSON; catching it here beats a parse error mid-episode."""
    with pytest.raises(ValueError):
        p.encode_message(p.ok_response(reward=math.nan))


def test_malformed_input_raises_protocol_error():
    for bad in ("", "   ", "not json", "[1, 2, 3]", "42"):
        with pytest.raises(p.ProtocolError):
            p.decode_message(bad)


def test_raise_for_error_surfaces_the_peer_message():
    assert p.raise_for_error(p.ok_response(value=1))["value"] == 1
    with pytest.raises(p.ProtocolError, match="engine exploded"):
        p.raise_for_error(p.error_response("engine exploded"))


def test_version_mismatch_is_refused():
    p.check_version(p.ok_response(version=p.PROTOCOL_VERSION))  # must not raise
    with pytest.raises(p.ProtocolError, match="protocol mismatch"):
        p.check_version(p.ok_response(version=p.PROTOCOL_VERSION + 1))


def test_step_request_coerces_numpy_like_values():
    """Callers pass numpy scalars; the wire must carry plain JSON types."""
    request = p.step_request(operation=True, params=(0, 1), target=False)
    assert request["operation"] == 1 and request["target"] == 0
    assert request["params"] == [0.0, 1.0]
    p.encode_message(request)  # must not raise
