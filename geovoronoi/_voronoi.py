"""
Functions to create Voronoi regions from points inside a geographic area.

"shapely" refers to the [Shapely Python package for computational geometry](http://toblerity.org/shapely/index.html).

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import logging

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import LineString, asPoint
from shapely.ops import polygonize

from ._geom import polygon_around_center


logger = logging.getLogger('geovoronoi')
logger.addHandler(logging.NullHandler())


def coords_to_points(coords):
    return list(map(asPoint, coords))


def points_to_coords(pts):
    return np.array([p.coords[0] for p in pts])


def voronoi_regions_from_coords(coords, geo_shape, accept_n_coord_duplicates=0, flatten_assignments=True):
    logger.info('running Voronoi tesselation for %d points' % len(coords))
    vor = Voronoi(coords)
    logger.info('generated %d Voronoi regions' % (len(vor.regions)-1))

    logger.info('generating Voronoi polygon lines')
    poly_lines = polygon_lines_from_voronoi(vor, geo_shape)

    logger.info('generating Voronoi polygon shapes')
    poly_shapes = polygon_shapes_from_voronoi_lines(poly_lines, geo_shape)

    logger.info('assigning %d points to %d Voronoi polygons' % (len(coords), len(poly_shapes)))
    points = coords_to_points(coords)
    poly_to_pt_assignments = assign_points_to_voronoi_polygons(points, poly_shapes,
                                                               accept_n_coord_duplicates=accept_n_coord_duplicates,
                                                               flatten_assignments=flatten_assignments)

    return poly_shapes, points, poly_to_pt_assignments


def polygon_lines_from_voronoi(vor, geo_shape, return_only_poly_lines=True):
    """
    Takes a scipy Voronoi result object `vor` (see [1]) and a shapely Polygon `hull_shape` the represents the geographic
    area in which the Voronoi regions shall be placed. Calculates the following three lists:
    1. Polygon lines of the Voronoi regions. These can be used to generate all Voronoi region polygons via
       `assign_points_to_voronoi_polygons`.
    2. Loose ridges of Voronoi regions.
    3. Far points of loose ridges of Voronoi regions.

    If `return_only_poly_lines` is True, only the first list is returned, otherwise a tuple of all thress lists is
    returned.

    Calculation of Voronoi region polygon lines taken and adopted from [2].

    [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html#scipy.spatial.Voronoi
    [2]: https://github.com/scipy/scipy/blob/v1.0.0/scipy/spatial/_plotutils.py
    """

    xmin, ymin, xmax, ymax = geo_shape.bounds
    xrange = xmax - xmin
    yrange = ymax - ymin
    max_dim_extend = max(xrange, yrange)
    center = np.array(geo_shape.centroid)

    # generate lists of full polygon lines, loose ridges and far points of loose ridges from scipy Voronoi result object
    poly_lines = []
    loose_ridges = []
    far_points = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):       # full finite polygon line
            poly_lines.append(LineString(vor.vertices[simplex]))
        else:   # "loose ridge": contains infinite Voronoi vertex
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            direction = direction / np.linalg.norm(direction)   # to unit vector
            far_point = vor.vertices[i] + direction * max_dim_extend

            loose_ridges.append(LineString(np.vstack((vor.vertices[i], far_point))))
            far_points.append(far_point)

    #
    # confine the infinite Voronoi regions by constructing a "hull" from loose ridge far points around the centroid of
    # the geographic area
    #

    # first append the loose ridges themselves
    for l in loose_ridges:
        poly_lines.append(l)

    # now create the "hull" of far points: `far_points_hull`
    far_points = np.array(far_points)
    far_points_hull = polygon_around_center(far_points, center)

    if far_points_hull is None:
        raise RuntimeError('no polygonal hull of far points could be created')

    # sometimes, this hull does not completely encompass the geographic area `geo_shape`
    if not far_points_hull.contains(geo_shape):   # if that's the case, merge it by taking the union
        far_points_hull = far_points_hull.union(geo_shape)

    # now add the lines that make up `far_points_hull` to the final `poly_lines` list
    far_points_hull_coords = far_points_hull.exterior.coords
    for i, pt in list(enumerate(far_points_hull_coords))[1:]:
        poly_lines.append(LineString((far_points_hull_coords[i-1], pt)))
    poly_lines.append(LineString((far_points_hull_coords[-1], far_points_hull_coords[0])))

    if return_only_poly_lines:
        return poly_lines
    else:
        return poly_lines, loose_ridges, far_points


def polygon_shapes_from_voronoi_lines(poly_lines, geo_shape=None):
    """
    Form shapely Polygons objects from a list of shapely LineString objects in `poly_lines` by using
    [`polygonize`](http://toblerity.org/shapely/manual.html#shapely.ops.polygonize). If `geo_shape` is not None, then
    the intersection between any generated polygon and `geo_shape` is taken in case they overlap (i.e. the Voronoi
    regions at the border are "cut" to the `geo_shape` polygon that represents the geographic area holding the
    Voronoi regions).
    Returns a list of shapely Polygons objects.
    """
    poly_shapes = []
    for p in polygonize(poly_lines):
        if geo_shape is not None and not geo_shape.contains(p):
            p = p.intersection(geo_shape)

        poly_shapes.append(p)

    return poly_shapes


def assign_points_to_voronoi_polygons(points, poly_shapes, accept_n_coord_duplicates=0, flatten_assignments=True):
    """
    Assign a list/array of shapely Point objects `points` to their respective Voronoi polygons passed as list
    `poly_shapes`. Return a list of `assignments` of size `len(poly_shapes)` where ith element in `assignments`
    contains the index of the point in `points` that resides in the ith Voronoi region.
    Normally, 1-to-1 assignments are expected, i.e. for each Voronoi region in `poly_shapes` there's exactly one point
    in `points` belonging to this Voronoi region. However, if there are duplicate coordinates in `points`, then those
    duplicates will be assigned together to their Voronoi region and hence there is a 1-to-n relationship between
    Voronoi regions and points. In this case `accept_n_coord_duplicates` should be set to the number of duplicates in
    `points` and a nested list of assignments will be returned (this is also the case when `flatten_assignments` is
    False).
    """
    n_polys = len(poly_shapes)
    n_geocoords = len(points)
    expected_n_geocoords = n_geocoords - accept_n_coord_duplicates

    if accept_n_coord_duplicates >= 0 and n_polys != expected_n_geocoords:
        raise ValueError('Unexpected number of geo-coordinates: %d (got %d polygons and expected %d geo-coordinates' %
                         (expected_n_geocoords, n_geocoords, expected_n_geocoords))

    unassigned = dict(enumerate(points))
    assignments = []
    for i_poly, vor_poly in enumerate(poly_shapes):
        tmp_assigned = [i_pt for i_pt, pt in unassigned.items() if pt.intersects(vor_poly)]

        if not tmp_assigned:
            raise RuntimeError('Polygon %d does not contain any point' % i_poly)

        if accept_n_coord_duplicates == 0 and len(tmp_assigned) > 1:
            raise RuntimeError('No duplicate points allowed but polygon %d contains several points: %s'
                               % (i_poly, str(tmp_assigned)))

        assignments.append(tmp_assigned)
        for i_pt in tmp_assigned:
            del unassigned[i_pt]

    assert sum(map(len, assignments)) == len(poly_shapes) + accept_n_coord_duplicates
    assert len(unassigned) == 0

    if flatten_assignments and accept_n_coord_duplicates == 0:
        flattened = sum(assignments, [])
        assert len(flattened) == len(assignments)
        return flattened
    else:
        return assignments
