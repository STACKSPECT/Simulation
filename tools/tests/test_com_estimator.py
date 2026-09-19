import numpy as np
import pytest

from stable_pallet.com_estimator import (
    NO_PAYLOAD,
    OK,
    PARALLEL_POSES,
    SLIPPED,
    estimate_com,
    hidden_planar_error,
    pose_condition,
)
from stable_pallet.tare import TareCalibration, calibrate_tare
from stable_pallet.wrench import WrenchSample, average_wrench, skew

GRAVITY = 9.81


def _tilted_gravities(count: int) -> list[np.ndarray]:
    down = np.array([0.0, 0.0, -GRAVITY])
    tilts = ((1.0, 0.0), (0.0, 1.0), (0.7, 0.7))
    out = [down]
    for index in range(count - 1):
        axis_x, axis_y = tilts[index % len(tilts)]
        out.append(np.array([axis_x * GRAVITY * 0.7, axis_y * GRAVITY * 0.7, -GRAVITY * 0.7]))
    return out


def _reading(mass: float, com: np.ndarray, gravity: np.ndarray) -> WrenchSample:
    return WrenchSample(force=-mass * gravity, torque=skew(gravity) @ (mass * com), gravity=gravity)


def _null_tare() -> TareCalibration:
    return TareCalibration(1e-12, np.zeros(3), np.zeros(3), np.zeros(3), 1.0, 1.0, 0.0, 0.0)


def test_two_tilted_poses_recover_the_centre_of_mass() -> None:
    com = np.array([0.012, -0.031, 0.163])
    samples = [_reading(8.5, com, gravity) for gravity in _tilted_gravities(2)]
    estimate = estimate_com(samples, _null_tare(), prior=np.zeros(3))
    assert np.allclose(estimate.com, com, atol=1e-9)
    assert estimate.mass == pytest.approx(8.5, abs=1e-9)
    assert estimate.reason == OK
    assert estimate.rank == 3


def test_one_plumb_pose_recovers_the_horizontal_centre_and_keeps_the_prior() -> None:
    true = np.array([0.01, 0.02, 0.16])
    prior = np.array([0.00, 0.00, 0.18])
    samples = [_reading(8.5, true, _tilted_gravities(1)[0])]
    estimate = estimate_com(samples, _null_tare(), prior=prior)
    assert estimate.com[0] == pytest.approx(true[0], abs=1e-9)
    assert estimate.com[1] == pytest.approx(true[1], abs=1e-9)
    assert estimate.com[2] == pytest.approx(prior[2], abs=1e-9)
    assert estimate.mass == pytest.approx(8.5, abs=1e-9)
    assert estimate.reason == OK
    assert estimate.rank == 2


def test_the_prior_does_not_pull_the_observable_plane() -> None:
    """Truncation, not ridge: a wrong prior must leave the horizontal components exact."""
    true = np.array([0.04, -0.03, 0.12])
    prior = np.array([0.20, 0.20, 0.12])
    samples = [_reading(8.5, true, np.array([0.0, 0.0, -GRAVITY]))]
    estimate = estimate_com(samples, _null_tare(), prior=prior)
    assert estimate.com[0] == pytest.approx(true[0], abs=1e-9)
    assert estimate.com[1] == pytest.approx(true[1], abs=1e-9)


def test_a_zero_prior_parks_the_unseen_component_at_the_sensor() -> None:
    true = np.array([0.01, 0.02, 0.16])
    samples = [_reading(8.5, true, np.array([0.0, 0.0, -GRAVITY]))]
    estimate = estimate_com(samples, _null_tare(), prior=np.zeros(3))
    assert estimate.com[2] == pytest.approx(0.0, abs=1e-9)


def test_a_full_rank_solve_ignores_the_prior() -> None:
    com = np.array([0.012, -0.031, 0.163])
    samples = [_reading(8.5, com, gravity) for gravity in _tilted_gravities(3)]
    estimate = estimate_com(samples, _null_tare(), prior=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(estimate.com, com, atol=1e-9)


def test_moving_the_centre_of_mass_along_gravity_changes_no_reading() -> None:
    gravity = np.array([0.0, 0.0, -GRAVITY])
    near = _reading(8.5, np.array([0.01, 0.02, 0.16]), gravity)
    far = _reading(8.5, np.array([0.01, 0.02, 0.44]), gravity)
    assert np.allclose(near.torque, far.torque)


def test_nearly_parallel_poses_are_flagged() -> None:
    down = np.array([0.0, 0.0, -GRAVITY])
    samples = [_reading(8.5, np.array([0.01, 0.02, 0.16]), g) for g in (down, down + np.array([1e-6, 0.0, 0.0]))]
    assert estimate_com(samples, _null_tare()).reason == PARALLEL_POSES


def test_an_empty_tool_is_reported_as_no_payload() -> None:
    samples = [_reading(1e-4, np.zeros(3), gravity) for gravity in _tilted_gravities(3)]
    assert estimate_com(samples, _null_tare()).reason == NO_PAYLOAD


def test_a_package_that_shifts_between_poses_is_flagged() -> None:
    gravities = _tilted_gravities(3)
    samples = [
        _reading(8.5, np.array([0.01, 0.02, 0.16]), gravities[0]),
        _reading(8.5, np.array([0.01, 0.02, 0.16]), gravities[1]),
        _reading(12.0, np.array([0.09, 0.02, 0.16]), gravities[2]),
    ]
    assert estimate_com(samples, _null_tare()).reason == SLIPPED


def test_spread_poses_are_better_conditioned_than_close_ones() -> None:
    down = np.array([0.0, 0.0, -GRAVITY])
    assert pose_condition(_tilted_gravities(3)) < pose_condition([down, down + np.array([0.01, 0.0, 0.0])])


def test_a_plumb_blind_axis_hides_nothing_in_the_plane() -> None:
    lean, hidden = hidden_planar_error(
        np.array([0.0, 0.0, 1.0]),
        np.eye(3),
        np.array([0.2, 0.15, 0.1]),
    )
    assert lean == pytest.approx(0.0, abs=1e-12)
    assert hidden == pytest.approx(0.0, abs=1e-12)


def test_a_tilted_blind_axis_leaks_the_half_height_into_the_plane() -> None:
    angle = np.deg2rad(20.0)
    blind = np.array([np.sin(angle), 0.0, np.cos(angle)])
    half = np.array([0.2, 0.15, 0.1])
    lean, hidden = hidden_planar_error(blind, np.eye(3), half)
    assert lean == pytest.approx(np.sin(angle), abs=1e-9)
    assert hidden == pytest.approx(float(np.abs(blind) @ half) * np.sin(angle), abs=1e-9)


def test_tare_recovers_tool_mass_moment_and_offsets() -> None:
    mass, com = 2.0, np.array([0.001, -0.002, 0.040])
    force_offset, torque_offset = np.array([0.3, -0.1, 0.2]), np.array([0.02, 0.01, -0.03])
    samples = [
        WrenchSample(
            force=_reading(mass, com, gravity).force + force_offset,
            torque=_reading(mass, com, gravity).torque + torque_offset,
            gravity=gravity,
        )
        for gravity in _tilted_gravities(4)
    ]
    tare = calibrate_tare(samples)
    assert tare.mass == pytest.approx(mass, abs=1e-9)
    assert np.allclose(tare.com, com, atol=1e-9)
    assert np.allclose(tare.force_offset, force_offset, atol=1e-9)
    assert np.allclose(tare.torque_offset, torque_offset, atol=1e-9)
    assert tare.well_conditioned


def test_tare_needs_three_poses_for_the_torque_block() -> None:
    samples = [_reading(2.0, np.zeros(3), gravity) for gravity in _tilted_gravities(2)]
    with pytest.raises(ValueError, match="need >= 3 poses"):
        calibrate_tare(samples)


def test_subtracting_the_tare_leaves_only_the_payload() -> None:
    tool_mass, tool_com = 2.0, np.array([0.0, 0.0, 0.04])
    payload_mass, payload_com = 8.5, np.array([0.01, -0.02, 0.16])
    gravity = _tilted_gravities(2)[1]
    combined = WrenchSample(
        force=-(tool_mass + payload_mass) * gravity,
        torque=skew(gravity) @ (tool_mass * tool_com + payload_mass * payload_com),
        gravity=gravity,
    )
    tare = TareCalibration(tool_mass, tool_mass * tool_com, np.zeros(3), np.zeros(3), 1.0, 1.0, 0.0, 0.0)
    payload = tare.subtract(combined)
    assert np.allclose(payload.force, -payload_mass * gravity)
    assert np.allclose(payload.torque, skew(gravity) @ (payload_mass * payload_com))


def test_averaging_refuses_to_mix_two_poses() -> None:
    samples = [_reading(8.5, np.zeros(3), gravity) for gravity in _tilted_gravities(2)]
    with pytest.raises(ValueError, match="same pose"):
        average_wrench(samples)


def test_averaging_accepts_a_few_milliradians_of_creep() -> None:
    down = np.array([0.0, 0.0, -GRAVITY])
    angle = np.deg2rad(0.2)
    crept = np.array([GRAVITY * np.sin(angle), 0.0, -GRAVITY * np.cos(angle)])
    average_wrench([_reading(8.5, np.zeros(3), down), _reading(8.5, np.zeros(3), crept)])
