from stable_pallet.models import Package, Pallet, Placement, StackState
from stable_pallet.stability import has_top_down_access, validate_stack


def box(name: str, size=(0.4, 0.4, 0.2), mass=5.0, com=(0.0, 0.0, 0.0)) -> Package:
    return Package(name, size, mass, com)


def test_rejects_floating_package() -> None:
    state = StackState(Pallet(1.2, 1.0), [Placement(box("floating"), 0.2, 0.2, 0.4)])
    report = validate_stack(state)
    assert not report.stable
    assert "no support" in report.reason


def test_accepts_centered_bridge_on_two_supports() -> None:
    supports = [
        Placement(box("left"), 0.1, 0.3, 0.0),
        Placement(box("right"), 0.7, 0.3, 0.0),
    ]
    bridge = Placement(box("bridge", (1.0, 0.4, 0.2), 4.0), 0.1, 0.3, 0.2)
    report = validate_stack(StackState(Pallet(1.2, 1.0), [*supports, bridge]), 0.7, 0.001)
    assert report.stable, report.reason


def test_rejects_offset_com_outside_support_polygon() -> None:
    base = Placement(box("base", (0.3, 0.4, 0.2)), 0.1, 0.3, 0.0)
    top = Placement(box("top", (0.6, 0.4, 0.2), com=(0.20, 0.0, 0.0)), 0.1, 0.3, 0.2)
    report = validate_stack(StackState(Pallet(1.2, 1.0), [base, top]), 0.45, 0.001)
    assert not report.stable
    assert "tipping margin" in report.reason


def test_upper_load_is_propagated_to_lower_interface() -> None:
    base = Placement(box("base", (0.4, 0.4, 0.2), 2.0), 0.3, 0.3, 0.0)
    heavy = Placement(
        box("heavy", (0.35, 0.35, 0.2), 30.0, com=(0.12, 0.0, 0.0)),
        0.3,
        0.325,
        0.2,
    )
    report = validate_stack(StackState(Pallet(1.2, 1.0), [base, heavy]), 0.7, 0.001)
    assert report.stable
    base_interface = next(item for item in report.interfaces if item.package_id == "base")
    assert base_interface.load_mass == 32.0


def test_rejects_top_down_insertion_below_an_overhang() -> None:
    lower = Placement(box("lower"), 0.1, 0.2, 0.0)
    overhang = Placement(box("overhang", (0.6, 0.4, 0.2)), 0.1, 0.2, 0.2)
    state = StackState(Pallet(1.2, 1.0), [lower, overhang])
    candidate = Placement(box("candidate", (0.3, 0.3, 0.2)), 0.55, 0.25, 0.0)
    assert not has_top_down_access(state, candidate)
