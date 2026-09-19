"""Switches an operator can flip while a run is already under way.

The cell reads this object on every frame instead of taking the settings as constructor
arguments, so a checkbox can turn a marker on, halve the speed or pause the arm without
restarting the experiment. Everything here is a plain attribute written from the UI
thread and read from the thread that steps MuJoCo; a torn read costs one stale frame,
which is why none of it needs a lock.

Speed and fast-forward are different things and both are useful:

- **Speed** is wall-clock pacing. The cell runs exactly the same trajectories, just
  slower or faster against the clock.
- **Fast-forward** skips the trajectories altogether: the arm jumps from waypoint to
  waypoint instead of driving between them. Physics still runs for the parts that decide
  the outcome -- the release, the settle, the jolts -- so the stack it builds is the
  stack the robot would have built, reached in a fraction of the time.
"""

from __future__ import annotations

from dataclasses import dataclass

# What the speed selector offers. `0.0` means "no pacing at all": step as fast as the
# machine manages, which is still every trajectory, unlike fast-forward.
SPEED_PRESETS: tuple[tuple[str, float], ...] = (
    ("x0.5", 0.5),
    ("x1", 1.0),
    ("x2", 2.0),
    ("x4", 4.0),
    ("max", 0.0),
)


class RunCancelled(RuntimeError):
    """Raised inside the stepping loop when the operator stops the run."""


@dataclass(slots=True)
class ViewerControls:
    """Live state shared between a dashboard and the cell it is driving."""

    speed: float = 1.0
    fast_forward: bool = False
    show_true_com: bool = False
    show_estimated_com: bool = False

    paused: bool = False
    cancelled: bool = False
    hold_at_end: bool = False
    holding: bool = False

    # What the cell is doing when it is doing something other than stepping. The
    # planner takes seconds and nothing moves while it thinks, so without this the
    # window looks hung rather than busy.
    activity: str = ""

    # Frame the operator is reviewing while paused, and how the timeline is playing.
    # `review_index` is written by both sides: the UI to scrub, the cell to report
    # where automatic playback has got to.
    review_index: int | None = None
    playback: str = "stopped"  # stopped | forward | backward
    frame_count: int = 0

    @property
    def draws_markers(self) -> bool:
        return self.show_true_com or self.show_estimated_com

    @property
    def paced(self) -> bool:
        """Whether the viewer waits for the wall clock between steps."""
        return self.speed > 0.0

    def start_run(self, *, hold_at_end: bool) -> None:
        """Reset everything a previous run may have left behind."""
        self.cancelled = False
        self.paused = False
        self.holding = False
        self.activity = ""
        self.hold_at_end = hold_at_end
        self.review_index = None
        self.playback = "stopped"
        self.frame_count = 0

    def cancel(self) -> None:
        """Stop the run, and release it first if it is parked for review."""
        self.cancelled = True
        self.holding = False
        self.paused = False

    def resume(self) -> None:
        self.paused = False
        self.playback = "stopped"
        self.review_index = None

    def scrub_to(self, index: int) -> None:
        self.paused = True
        self.playback = "stopped"
        self.review_index = index

    def play(self, direction: str) -> None:
        if direction not in {"forward", "backward"}:
            raise ValueError(f"Playback direction must be forward or backward, not {direction!r}")
        self.paused = True
        self.playback = direction
