"""
Geometry helper functions in cartesian 2D space.

"Shapely" refers to the [Shapely Python package for computational geometry](http://toblerity.org/shapely/index.html).

Author: Markus Konrad <markus.konrad@wzb.eu>
"""


import numpy as np


def line_segment_intersection(l_off, l_dir, segm_a, segm_b):
    """
    Check for line - segment intersection between line defined as `l_off + t * l_dir` and line segment between points
    `segm_a` and `segm_b`. Hence the line is conceived as line starting at `l_off` and heading in direction `l_dir`
    towards infinity. The function returns the intersection point as 1D NumPy array `[x, y]` if the given line hits
    the defined segment between points `segm_a` and `segm_b` (endpoints inclusive). If there's no intersection, the
    function returns None.

    All arguments must be 1D NumPy arrays of size 2.

    :param l_off: line offset point
    :param l_dir: line direction vector
    :param segm_a: segment start point
    :param segm_b: segment end point
    :return: intersection point as 1D NumPy array `[x, y]` or None if there is no intersection.
    """

    if not all(isinstance(arg, np.ndarray) and arg.shape == (2,) for arg in (l_off, l_dir, segm_a, segm_b)):
        raise ValueError('all arguments must be 1D NumPy arrays of size 2')

    if np.isclose(np.linalg.norm(l_dir), 0):
        raise ValueError('vector length of `l_dir` must be greater than 0')

    if np.isclose(np.linalg.norm(segm_b - segm_a), 0):
        raise ValueError('vector length between `segm_a` and `segm_b` must be greater than 0')

    segm_dir = segm_b - segm_a
    v = np.array([l_dir, -segm_dir]).T
    p = segm_a - l_off

    if np.isclose(np.linalg.det(v), 0):   # det of direction vector matrix is zero -> parallel direction vectors
        if np.isclose(np.linalg.det(np.array([p, l_dir])), 0):  # det of vector offset matrix is zero
                                                                # -> possible overlap
            # order segment end points either horizontally or vertically
            if segm_a[0] > segm_b[0] or \
                    (np.isclose(segm_a[0], segm_b[0]) and segm_a[1] > segm_b[1]):  # if horizontally aligned,
                                                                                   # order on y-axis
                segm_b, segm_a = segm_a, segm_b

            nonzero_ind = np.nonzero(l_dir)[0]   # norm is > 0 so there must be a nonzero index
            t_a = (segm_a - l_off)[nonzero_ind] / l_dir[nonzero_ind]        # segm_a = l_off + t_a * l_dir
            t_b = (segm_b - l_off)[nonzero_ind] / l_dir[nonzero_ind]        # segm_b = l_off + t_b * l_dir

            t = np.array([t_a, t_b])
            t = t[t >= 0]
            if len(t) > 0:
                return l_off + np.min(t) * l_dir

        return None   # either parallel directions or line doesn't intersect with any segment endpoint
    else:
        t = np.matmul(np.linalg.inv(v), p.T)
        # intersection at l_off + t_0 * l_dir and segm_a + t_1 * segm_dir
        # -> we're only interested if l_dir hits the segment (segm_a,segm_b) when it goes in positive direction,
        #    hence if t_0 is positive and t_1 is in [0, 1]

        if t[0] >= 0 and 0 <= t[1] <= 1:
            return segm_a + t[1] * segm_dir
        else:
            return None


def calculate_polygon_areas(region_polys, m2_to_km2=False):
    """
    Return the area of the respective polygons in `poly_shapes` either in unit square meters (`m2_to_km2` is False) or
    in square kilometers (`m2_to_km2` is True). Does not really calculate the area but uses the `area` property of
    the Shapely polygons.

    Note: It is important to use an *equal area* projection with meters as units before using the areas of the
          Voronoi regions!

    :param region_polys: dict mapping Voronoi region IDs to Shapely Polygon/MultiPolygon objects
    :param m2_to_km2: if True, return results as square kilometers, otherwise square meters
    :return: dict mapping Voronoi region IDs to area in square meters or square kilometers
    """

    if not isinstance(region_polys, dict):
        raise ValueError('`poly_shapes` must be a dict')

    unit_convert = 1000000 if m2_to_km2 else 1
    return {i_poly: p.area / unit_convert for i_poly, p in region_polys.items()}
