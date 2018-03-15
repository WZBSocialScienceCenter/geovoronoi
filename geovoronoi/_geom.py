"""
Geometry helper functions in cartesian 2D space.

"shapely" refers to the [Shapely Python package for computational geometry](http://toblerity.org/shapely/index.html).

Author: Markus Konrad <markus.konrad@wzb.eu>
"""


import numpy as np
from shapely.geometry import Polygon


def angle_between_pts(a, b, ref_vec=(1.0, 0.0)):
    """
    Angle *theta* between two points (numpy arrays) `a` and `b` in relation to a reference vector `ref_vec`.
    By default, `ref_vec` is the x-axis (i.e. unit vector (1, 0)).
    *theta* is in [0, 2Pi] unless `a` and `b` are very close. In this case this function returns NaN.
    """
    ang = inner_angle_between_vecs(a - b, np.array(ref_vec))
    if not np.isnan(ang) and a[1] < b[1]:
        ang = 2 * np.pi - ang

    return ang


def inner_angle_between_vecs(a, b):
    """
    Return the inner angle *theta* between numpy vectors `a` and `b`. *theta* is in [0, Pi] if both `a` and `b` are
    not at the origin (0, 0), otherwise this function returns NaN.
    """
    origin = np.array((0, 0))
    if np.allclose(a, origin) or np.allclose(b, origin):
        return np.nan

    au = a / np.linalg.norm(a)
    bu = b / np.linalg.norm(b)
    ang = np.arccos(np.clip(np.dot(au, bu), -1.0, 1.0))

    return ang


def polygon_around_center(points, center=None, fix_nan_angles=True):
    """
    Order numpy array of coordinates `points` around `center` so that they form a valid polygon. Return that as
    shapely `Polygon` object. If no valid polygon can be formed, return `None`.
    If `center` is None (default), use midpoint of `points` as center.
    """
    if center is None:
        center = points.mean(axis=0)
    else:
        center = np.array(center)

    # angle between each point in `points` and `center`
    angles = np.apply_along_axis(angle_between_pts, 1, points, b=center)

    # sort by angles and generate polygon
    if fix_nan_angles:
        for repl in (0, np.pi):
            tmp_angles = angles.copy()
            tmp_angles[np.isnan(tmp_angles)] = repl
            poly = Polygon(points[np.argsort(tmp_angles)])
            if poly.is_simple and poly.is_valid:
                return poly

        return None
    else:
        poly = Polygon(points[np.argsort(angles)])

        if poly.is_simple and poly.is_valid:
            return poly
        else:
            return None


def calculate_polygon_areas(poly_shapes, m2_to_km2=False):
    """
    Return the area of the respective polygons in `poly_shapes`. Returns a NumPy array of areas in m² (if `m2_to_km2` is
    False) or km² (otherwise).
    """
    areas = np.array([p.area for p in poly_shapes])
    if m2_to_km2:
        return areas / 1000000    # = 1000²
    else:
        return areas
