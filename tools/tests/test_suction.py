from stable_pallet.models import Package
from stable_pallet.scenario import load_scenario
from stable_pallet.suction import SuctionArray


def test_vgp20_covers_more_cups_on_a_larger_footprint() -> None:
    array = SuctionArray()
    small = array.plan(Package("small", (0.22, 0.20, 0.15), 2.0))
    large = array.plan(Package("large", (0.50, 0.40, 0.20), 10.0))
    assert 0 < len(small.active_cups) < len(large.active_cups)
    assert small.feasible
    assert large.feasible


def test_pickup_uses_infeed_orientation_before_box_rotation() -> None:
    array = SuctionArray()
    package = Package("book", (0.24, 0.18, 0.10), 1.2)
    # 240 x 180 across the array takes a 3x3 block; turned to 180 x 240 the third
    # column no longer fits, so the infeed orientation is the one worth picking in.
    assert len(array.plan(package, yaw=0).active_cups) == 9
    assert len(array.plan(package, yaw=90).active_cups) == 8


def test_offset_block_beats_centring_the_housing() -> None:
    """A carton smaller than the 264 x 184 mm array loses its outer cups when the housing
    is centred on it; sliding the arm onto the largest block that fits wins them back."""
    array = SuctionArray()
    package = Package("book", (0.24, 0.18, 0.10), 1.2)
    width, depth, _ = package.oriented_size(0)
    centred = sum(
        1
        for _column, _row, x, y in array.cup_offsets()
        if abs(x) + array.cup_radius <= width / 2 and abs(y) + array.cup_radius <= depth / 2
    )
    grip = array.plan(package, 0)
    assert len(grip.active_cups) > centred
    assert grip.tool_offset != (0.0, 0.0)


def test_a_carton_wider_than_the_array_needs_no_offset() -> None:
    array = SuctionArray()
    grip = array.plan(Package("large", (0.50, 0.40, 0.20), 10.0), 0)
    assert len(grip.active_cups) == 16
    assert grip.tool_offset == (0.0, 0.0)


def test_active_cups_land_inside_the_carton_once_the_arm_offsets() -> None:
    """The offset is what the arm cancels, so every engaged cup has to be fully on the
    cardboard after it is applied. A cup past the edge would never seal."""
    array = SuctionArray()
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    for package in scenario.packages:
        for yaw in (0, 90):
            grip = array.plan(package, yaw)
            width, depth, _ = package.oriented_size(yaw)
            offset_x, offset_y = grip.tool_offset
            assert grip.active_cups, package.id
            for x, y in grip.active_cups:
                assert abs(x - offset_x) + array.cup_radius <= width / 2 + 1e-9, package.id
                assert abs(y - offset_y) + array.cup_radius <= depth / 2 + 1e-9, package.id


def test_a_carton_narrower_than_one_cup_is_refused() -> None:
    array = SuctionArray()
    grip = array.plan(Package("sliver", (0.03, 0.30, 0.10), 1.0), 0)
    assert grip.active_cups == ()
    assert not grip.feasible


def test_mixed_boxes_fit_the_vgp20() -> None:
    array = SuctionArray()
    scenario = load_scenario("scenarios/mixed_boxes.yaml")
    assert array.name == "OnRobot VGP20"
    assert array.columns * array.rows == 16
    for package in scenario.packages:
        grip = array.plan(package, yaw=0)
        assert grip.feasible, package.id
        assert package.mass + array.mass <= 12.5
