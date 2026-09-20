from stable_pallet.geometry import convex_hull, signed_polygon_margin


def test_convex_hull_and_signed_margin() -> None:
    hull = convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    assert hull == [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert signed_polygon_margin((0.5, 0.5), hull) == 0.5
    assert signed_polygon_margin((1.1, 0.5), hull) < 0
