"""Geometry helpers: great-circle distance, local projection, no-fly detours."""
import math

EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def to_xy_km(lat, lon, ref_lat, ref_lon):
    """Equirectangular projection to km. Accurate enough at city scale."""
    x = EARTH_R_KM * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
    y = EARTH_R_KM * math.radians(lat - ref_lat)
    return x, y


def _dist_point_to_segment(c, a, b):
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 == 0:
        return math.hypot(cx - ax, cy - ay)
    t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / seg2))
    return math.hypot(cx - (ax + t * dx), cy - (ay + t * dy))


def path_length_around_circle(p1, p2, center, radius):
    """Shortest path p1 -> p2 (planar, km) that does not enter a circle.

    Returns the straight length if the segment misses the circle, math.inf if
    an endpoint lies inside it, otherwise tangent + arc + tangent length.
    """
    straight = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    d1 = math.hypot(p1[0] - center[0], p1[1] - center[1])
    d2 = math.hypot(p2[0] - center[0], p2[1] - center[1])
    if d1 < radius or d2 < radius:
        return math.inf
    if _dist_point_to_segment(center, p1, p2) >= radius:
        return straight
    v1 = (p1[0] - center[0], p1[1] - center[1])
    v2 = (p2[0] - center[0], p2[1] - center[1])
    cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)
    theta = math.acos(max(-1.0, min(1.0, cos_theta)))
    a1 = math.acos(radius / d1)
    a2 = math.acos(radius / d2)
    arc = max(0.0, theta - a1 - a2)
    return math.sqrt(d1 ** 2 - radius ** 2) + math.sqrt(d2 ** 2 - radius ** 2) + radius * arc


def drone_path_km(lat1, lon1, lat2, lon2, zones, ref_lat, ref_lon):
    """Drone flight distance in km, detouring around circular no-fly zones.

    Approximation: each blocking zone adds its own detour extra to the straight
    line. Fine for non-overlapping zones; documented limitation.
    """
    p1 = to_xy_km(lat1, lon1, ref_lat, ref_lon)
    p2 = to_xy_km(lat2, lon2, ref_lat, ref_lon)
    straight = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    total_extra = 0.0
    for _name, zlat, zlon, zr in zones:
        c = to_xy_km(zlat, zlon, ref_lat, ref_lon)
        length = path_length_around_circle(p1, p2, c, zr)
        if math.isinf(length):
            return math.inf
        total_extra += length - straight
    return straight + total_extra
