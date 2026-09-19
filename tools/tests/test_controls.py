from stable_pallet.controls import SPEED_PRESETS, ViewerControls


def test_speed_and_fast_forward_are_independent_settings() -> None:
    """Halving the speed must not turn the trajectories off, and vice versa."""
    controls = ViewerControls(speed=0.5)
    assert controls.paced
    assert not controls.fast_forward

    controls.fast_forward = True
    assert controls.speed == 0.5
    assert controls.paced


def test_asking_for_no_pacing_is_not_the_same_as_fast_forward() -> None:
    controls = ViewerControls(speed=0.0)
    assert not controls.paced
    assert not controls.fast_forward


def test_every_speed_preset_is_usable() -> None:
    values = dict(SPEED_PRESETS)
    assert values["x1"] == 1.0
    assert values["max"] == 0.0
    assert all(value >= 0 for value in values.values())


def test_starting_a_run_clears_what_the_previous_one_left() -> None:
    controls = ViewerControls()
    controls.cancel()
    controls.scrub_to(12)
    controls.play("backward")
    controls.frame_count = 900
    controls.activity = "Planificando…"

    controls.start_run(hold_at_end=True)

    assert not controls.cancelled
    assert not controls.paused
    assert controls.review_index is None
    assert controls.playback == "stopped"
    assert controls.frame_count == 0
    assert controls.activity == ""
    assert controls.hold_at_end


def test_scrubbing_pauses_and_resuming_returns_to_the_live_frame() -> None:
    controls = ViewerControls()
    controls.scrub_to(40)
    assert controls.paused
    assert controls.review_index == 40

    controls.resume()
    assert not controls.paused
    assert controls.review_index is None
    assert controls.playback == "stopped"


def test_cancelling_releases_a_run_parked_for_review() -> None:
    controls = ViewerControls(holding=True, paused=True)
    controls.cancel()
    assert controls.cancelled
    assert not controls.holding
    assert not controls.paused


def test_markers_are_drawn_only_when_one_of_them_is_asked_for() -> None:
    assert not ViewerControls().draws_markers
    assert ViewerControls(show_true_com=True).draws_markers
    assert ViewerControls(show_estimated_com=True).draws_markers
