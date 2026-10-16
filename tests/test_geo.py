import math

from src.geo import drone_path_km, haversine_km, path_length_around_circle, to_xy_km


def test_haversine_one_degree_latitude():
    assert abs(haversine_km(0, 0, 1, 0) - 111.195) < 0.05


def test_haversine_zero_and_symmetry():
    assert haversine_km(28.6, 77.2, 28.6, 77.2) == 0
    a = haversine_km(28.5, 77.1, 28.7, 77.3)
    b = haversine_km(28.7, 77.3, 28.5, 77.1)
    assert abs(a - b) < 1e-9


def test_projection_close_to_haversine_at_city_scale():
    p1 = to_xy_km(28.5672, 77.2100, 28.6, 77.2)
    p2 = to_xy_km(28.6835, 77.3080, 28.6, 77.2)
    planar = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    assert abs(planar - haversine_km(28.5672, 77.2100, 28.6835, 77.3080)) < 0.2


def test_circle_not_blocking_returns_straight():
    assert abs(path_length_around_circle((-10, 5), (10, 5), (0, 0), 2) - 20) < 1e-9


def test_circle_blocking_detour_matches_hand_calc():
    # tangents 2*sqrt(96) = 19.5959 plus arc 2*(pi - 2*acos(0.2)) = 0.8054 -> 20.4013
    got = path_length_around_circle((-10, 0), (10, 0), (0, 0), 2)
    assert abs(got - 20.4013) < 1e-3
    assert got > 20


def test_endpoint_inside_zone_is_inf():
    assert math.isinf(path_length_around_circle((0.5, 0), (10, 0), (0, 0), 2))


def test_drone_path_no_zones_equals_straight():
    d = drone_path_km(28.5672, 77.21, 28.6835, 77.308, [], 28.6, 77.2)
    p1 = to_xy_km(28.5672, 77.21, 28.6, 77.2)
    p2 = to_xy_km(28.6835, 77.308, 28.6, 77.2)
    assert abs(d - math.hypot(p2[0] - p1[0], p2[1] - p1[1])) < 1e-9


def test_drone_path_detours_around_zone():
    zone = [("z", 28.60, 77.20, 3.0)]
    blocked = drone_path_km(28.60, 77.13, 28.60, 77.27, zone, 28.6, 77.2)
    free = drone_path_km(28.60, 77.13, 28.60, 77.27, [], 28.6, 77.2)
    assert blocked > free
