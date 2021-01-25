"""
Functions to create Voronoi regions from points inside a geographic area.

"shapely" refers to the [Shapely Python package for computational geometry](http://toblerity.org/shapely/index.html).

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, asPoint, MultiPoint, Polygon, MultiPolygon
from shapely.errors import TopologicalError
from shapely.ops import polygonize, cascaded_union


logger = logging.getLogger('geovoronoi')
logger.addHandler(logging.NullHandler())


def coords_to_points(coords):
    """Convert a NumPy array of 2D coordinates `coords` to a list of shapely Point objects"""
    return list(map(asPoint, coords))


def points_to_coords(pts):
    """Convert a list of shapely Point objects to a NumPy array of 2D coordinates `coords`"""
    return np.array([p.coords[0] for p in pts])


def voronoi_regions_from_coords(coords, geo_shape, farpoints_max_extend_factor=10):
    """
    Calculate Voronoi regions from NumPy array of 2D coordinates `coord` that lie within a shape `geo_shape`. Setting
    `shapes_from_diff_with_min_area` fixes rare errors where the Voronoi shapes do not fully cover `geo_shape`. Set this
    to a small number that indicates the minimum valid area of "fill up" Voronoi region shapes.
    Set `accept_n_coord_duplicates` to accept exactly this number of points with exactly the same coordinates. Such
    duplicates will belong to the same Voronoi region. If set to `None` then any number of duplicate coordinates is
    accepted. Set `return_unassigned_points` to True to additionally return a list of shapely Point objects that could
    not be assigned to any Voronoi region.

    This function returns the following values in a tuple:

    1. `poly_shapes`: a list of shapely Polygon/MultiPolygon objects that represent the generated Voronoi regions
    2. `points`: the input coordinates converted to a list of shapely Point objects
    3. `poly_to_pt_assignments`: a nested list that for each Voronoi region in `poly_shapes` contains a list of indices
       into `points` (or `coords`) that represent the points that belong to this Voronoi region. Usually, this is only
       a single point. However, in case of duplicate points (e.g. both or more points have exactly the same coordinates)
       then all these duplicate points are listed for the respective Voronoi region.
    4. optional if `return_unassigned_points` is True: a list of points that could not be assigned to any Voronoi region

    When calculating the far points of loose ridges for the Voronoi regions, `farpoints_max_extend_factor` is the
    factor that is multiplied with the maximum extend per dimension. Increase this number in case the hull of far points
    doesn't intersect with `geo_shape`.
    """

    logger.info('running Voronoi tesselation for %d points' % len(coords))

    if not isinstance(coords, np.ndarray):
        coords = points_to_coords(coords)

    vor = Voronoi(coords)
    logger.info('generated %d Voronoi regions' % (len(vor.regions)-1))

    logger.info('generating Voronoi polygon regions')
    region_polys, region_pts = region_polygons_from_voronoi(vor, geo_shape,
                                                            return_point_assignments=True,
                                                            farpoints_max_extend_factor=farpoints_max_extend_factor)

    return region_polys, region_pts


def region_polygons_from_voronoi(vor, geo_shape, return_point_assignments=False, farpoints_max_extend_factor=10):
    if isinstance(geo_shape, Polygon):
        geoms = geo_shape
    elif isinstance(geo_shape, MultiPolygon):
        geoms = geo_shape.geoms
    else:
        raise ValueError('`geo_shape` must be a Polygon or MultiPolygon')

    max_dim_extend = vor.points.ptp(axis=0).max() * farpoints_max_extend_factor
    center = np.array(MultiPoint(vor.points).convex_hull.centroid)
    ridge_vert = np.array(vor.ridge_vertices)

    region_pts = defaultdict(list)
    region_polys = {}
    for i_reg, reg_vert in enumerate(vor.regions):
        pt_indices = np.where(vor.point_region == i_reg)[0]
        if len(pt_indices) == 0:  # skip regions w/o points in them
            continue

        region_pts[i_reg].extend(pt_indices.tolist())

        geom = {g for pt in vor.points[pt_indices] for g in geoms if g.contains(pt)}
        if len(geom) == 0:
            raise RuntimeError('no sub-geometry of `geo_shape` contains a point of %s' % str(vor.points[pt_indices]))
        elif len(geom) > 1:
            raise RuntimeError('more than one sub-geometry of `geo_shape` contains a point of %s'
                               % str(vor.points[pt_indices]))
        else:
            geom = geom.pop()

        if np.all(np.array(reg_vert) >= 0):  # fully finite-bounded region
            p = Polygon(vor.vertices[reg_vert])
        else:
            p_vertices = []
            i_pt = pt_indices[0]  # only consider one point, not a duplicate
            enclosing_ridge_pts_mask = (vor.ridge_points[:, 0] == i_pt) | (vor.ridge_points[:, 1] == i_pt)
            for pointidx, simplex in zip(vor.ridge_points[enclosing_ridge_pts_mask],
                                         ridge_vert[enclosing_ridge_pts_mask]):

                if np.all(simplex >= 0):
                    p_vertices.extend(vor.vertices[simplex])
                else:
                    # "loose ridge": contains infinite Voronoi vertex
                    i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
                    finite_pt = vor.vertices[i]

                    t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[pointidx].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    direction = direction / np.linalg.norm(direction)  # to unit vector

                    far_point = finite_pt + direction * max_dim_extend * farpoints_max_extend_factor

                    p_vertices.extend([finite_pt, far_point])

            p = MultiPoint(p_vertices).convex_hull  # Voronoi regions are convex

        assert p.is_valid and p.is_simple, 'generated polygon is valid and simple'

        if not geo_shape.contains(p):
            p = p.intersection(geo_shape)
            assert p.is_valid and p.is_simple, 'generated polygon is valid and simple after ' \
                                               'intersection with `geo_shape`'

        region_polys[i_reg] = p

    if return_point_assignments:
        return region_polys, region_pts
    else:
        return region_polys


# def polygon_lines_from_voronoi(vor, geo_shape, return_only_poly_lines=True, farpoints_max_extend_factor=10):
#     """
#     Takes a scipy Voronoi result object `vor` (see [1]) and a shapely Polygon `geo_shape` the represents the geographic
#     area in which the Voronoi regions shall be placed. Calculates the following three lists:
#
#     1. Polygon lines of the Voronoi regions. These can be used to generate all Voronoi region polygons via
#        `polygon_shapes_from_voronoi_lines`.
#     2. Loose ridges of Voronoi regions.
#     3. Far points of loose ridges of Voronoi regions.
#
#     If `return_only_poly_lines` is True, only the first list is returned, otherwise a tuple of all three lists is
#     returned.
#
#     When calculating the far points of loose ridges, `farpoints_max_extend_factor` is the factor that is multiplied
#     with the maximum extend per dimension. Increase this number in case the hull of far points doesn't intersect
#     with `geo_shape`.
#
#     Calculation of Voronoi region polygon lines taken and adopted from [2].
#
#     [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html#scipy.spatial.Voronoi
#     [2]: https://github.com/scipy/scipy/blob/v1.0.0/scipy/spatial/_plotutils.py
#     """
#
#     max_dim_extend = vor.points.ptp(axis=0).max() * farpoints_max_extend_factor
#     center = np.array(MultiPoint(vor.points).convex_hull.centroid)
#
#     # generate lists of full polygon lines, loose ridges and far points of loose ridges from scipy Voronoi result object
#     poly_lines = []
#     loose_ridges = []
#     far_points = []
#     for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
#         simplex = np.asarray(simplex)
#         if np.all(simplex >= 0):       # full finite polygon line
#             poly_lines.append(LineString(vor.vertices[simplex]))
#         else:   # "loose ridge": contains infinite Voronoi vertex
#             i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
#
#             t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
#             t /= np.linalg.norm(t)
#             n = np.array([-t[1], t[0]])  # normal
#
#             midpoint = vor.points[pointidx].mean(axis=0)
#             direction = np.sign(np.dot(midpoint - center, n)) * n
#             direction = direction / np.linalg.norm(direction)   # to unit vector
#
#             far_point = vor.vertices[i] + direction * max_dim_extend * farpoints_max_extend_factor
#
#             loose_ridges.append(LineString(np.vstack((vor.vertices[i], far_point))))
#             far_points.append(far_point)
#
#     #
#     # confine the infinite Voronoi regions by constructing a "hull" from loose ridge far points around the centroid of
#     # the geographic area
#     #
#
#     # first append the loose ridges themselves
#     for l in loose_ridges:
#         poly_lines.append(l)
#
#     # now create the "hull" of far points: `far_points_hull`
#     far_points = np.array(far_points)
#     far_points_hull = polygon_around_center(far_points, center)
#
#     if far_points_hull is None:
#         raise RuntimeError('no polygonal hull of far points could be created')
#
#     # sometimes, this hull does not completely encompass the geographic area `geo_shape`
#     if not far_points_hull.contains(geo_shape):   # if that's the case, merge it by taking the union
#         far_points_hull = far_points_hull.union(geo_shape)
#
#     if not isinstance(far_points_hull, Polygon):
#         raise RuntimeError('hull of far points is not Polygon as it should be; try increasing '
#                            '`farpoints_max_extend_factor`')
#
#     # now add the lines that make up `far_points_hull` to the final `poly_lines` list
#     far_points_hull_coords = far_points_hull.exterior.coords
#     for i, pt in list(enumerate(far_points_hull_coords))[1:]:
#         poly_lines.append(LineString((far_points_hull_coords[i-1], pt)))
#     poly_lines.append(LineString((far_points_hull_coords[-1], far_points_hull_coords[0])))
#
#     if return_only_poly_lines:
#         return poly_lines
#     else:
#         return poly_lines, loose_ridges, far_points
#
#
# def polygon_shapes_from_voronoi_lines(poly_lines, geo_shape=None, shapes_from_diff_with_min_area=None):
#     """
#     Form shapely Polygons objects from a list of shapely LineString objects in `poly_lines` by using
#     [`polygonize`](http://toblerity.org/shapely/manual.html#shapely.ops.polygonize). If `geo_shape` is not None, then
#     the intersection between any generated polygon and `geo_shape` is taken in case they overlap (i.e. the Voronoi
#     regions at the border are "cut" to the `geo_shape` polygon that represents the geographic area holding the
#     Voronoi regions). Setting `shapes_from_diff_with_min_area` fixes rare errors where the Voronoi shapes do not fully
#     cover `geo_shape`. Set this to a small number that indicates the minimum valid area of "fill up" Voronoi region
#     shapes.
#     Returns a list of shapely Polygons objects.
#     """
#
#     # generate shapely Polygon objects from the LineStrings of the Voronoi shapes in `poly_lines`
#     poly_shapes = []
#     for p in polygonize(poly_lines):
#         if geo_shape is not None and not geo_shape.contains(p):    # if `geo_shape` contains polygon `p`,
#             p = p.intersection(geo_shape)                          # intersect it with `geo_shape` (i.e. "cut" it)
#
#         if not p.is_empty:
#             poly_shapes.append(p)
#
#     if geo_shape is not None and shapes_from_diff_with_min_area is not None:
#         # fix rare cases where the generated polygons of the Voronoi regions don't fully cover `geo_shape`
#         vor_polys_union = cascaded_union(poly_shapes)   # union of Voronoi regions
#         try:
#             diff = np.array(geo_shape.difference(vor_polys_union), dtype=object)    # "gaps"
#         except TopologicalError:
#             diff = np.array([], dtype=object)
#
#         if diff.shape and len(diff) > 0:
#             diff_areas = np.array([p.area for p in diff])    # areas of "gaps"
#             # use only those "gaps" bigger than `shapes_from_diff_with_min_area` because very tiny areas are generated
#             # at the borders due to floating point errors
#             poly_shapes.extend(diff[diff_areas >= shapes_from_diff_with_min_area])
#
#     return poly_shapes
#
#
# def assign_points_to_voronoi_polygons(points, poly_shapes, vor=None,
#                                       accept_n_coord_duplicates=None,
#                                       return_unassigned_points=False,
#                                       coords=None):
#     """
#     Assign a list/array of shapely Point objects `points` to their respective Voronoi polygons passed as list
#     `poly_shapes`. Use already generated assignments from SciPy Voronoi in `vor` (SciPy Voronoi object).
#
#     Return a list of `assignments` of size `len(poly_shapes)` where ith element in `assignments`
#     contains the index of the point in `points` that resides in the ith Voronoi region.
#
#     Normally, 1-to-1 assignments are expected, i.e. for each Voronoi region in `poly_shapes` there's exactly one point
#     in `points` belonging to this Voronoi region. However, if there are duplicate coordinates in `points`, then those
#     duplicates will be assigned together to their Voronoi region and hence there is a 1-to-n relationship between
#     Voronoi regions and points. If `accept_n_coord_duplicates` is set to None, then an unspecified number of
#     duplicates are allowed. If `accept_n_coord_duplicates` is 0, then no point duplicates are allowed, otherwise
#     up to `accept_n_coord_duplicates` duplicates are allowed.
#
#     Set `return_unassigned_points` to additionally return a list of points that could not be assigned to any Voronoi
#     region. `coords` can be passed in order to avoid another conversion from Point objects to NumPy coordinate array.
#     """
#
#     # do some checks first
#
#     n_polys = len(poly_shapes)
#     n_points = len(points)
#
#     if n_polys > n_points:
#         raise ValueError('The number of voronoi regions must be smaller or equal to the number of points')
#
#     if accept_n_coord_duplicates is None:
#         dupl_restricted = False
#         accept_n_coord_duplicates = n_points - n_polys
#     else:
#         dupl_restricted = True
#
#     expected_n_geocoords = n_points - accept_n_coord_duplicates
#
#     if coords is None:
#         coords = points_to_coords(points)
#     elif len(coords) != n_points:
#         raise ValueError('`coords` and `points` must have the same length')
#
#     if accept_n_coord_duplicates >= 0 and n_polys != expected_n_geocoords:
#         raise ValueError('Unexpected number of geo-coordinates: %d (got %d polygons and expected %d geo-coordinates)' %
#                          (n_points, n_polys, expected_n_geocoords))
#
#     # generate the assignments
#     assignments = [[] for _ in range(n_polys)]    # region polygon -> point indices
#     already_assigned = set()                      # already assigned point indices
#     n_assigned_dupl = 0                           # number of duplicate points assigned to same region
#
#     # handle already assigned points passed from generated Voronoi object
#     if vor is not None:
#         finite_regions = np.where([r and all([i >= 0 for i in r]) for r in vor.regions])[0]   # indices
#         logger.info('using already calculated assignments to %d finite Voronoi regions' % len(finite_regions))
#
#         # find all points assigned to finite regions
#         for i_poly in finite_regions:
#             assigned_pts = np.where(vor.point_region == i_poly)[0].tolist()
#
#             if not assigned_pts:
#                 continue
#
#             if accept_n_coord_duplicates == 0 and len(assigned_pts) > 1:
#                 raise RuntimeError('No duplicate points allowed but polygon %d contains several points: %s'
#                                    % (i_poly, str(assigned_pts)))
#
#             assignments[i_poly].extend(assigned_pts)
#             already_assigned.update(assigned_pts)
#             n_assigned_dupl += len(assigned_pts) - 1
#
#         unassigned_indices = np.array(list(set(range(n_points)) - already_assigned), dtype=np.int)
#         coords = coords[unassigned_indices]
#         points = [p for i, p in enumerate(points) if i in unassigned_indices]
#     else:
#         unassigned_indices = np.arange(n_points)
#
#     assert len(coords) == len(points) == len(unassigned_indices)
#
#     if len(unassigned_indices) > 0:
#         logger.info('assigning remaining %d points via intersection tests' % len(unassigned_indices))
#         # for the remaining points:
#         # get the Voronoi regions' centroids and calculate the distance between all centroid – input coordinate pairs
#         poly_centroids = np.array([p.centroid.coords[0] for p in poly_shapes])
#         poly_pt_dists = cdist(poly_centroids, coords)
#
#         for i_poly, vor_poly in enumerate(poly_shapes):
#             # indices to points sorted by distance to this Voronoi region's centroid
#             closest_pt_indices = np.argsort(poly_pt_dists[i_poly])
#             assigned_pts = []
#             for i_pt in closest_pt_indices:      # check each point with increasing distance
#                 pt = points[i_pt]
#
#                 if pt.intersects(vor_poly):      # if this point is inside this Voronoi region, assign it to the region
#                     i_pt_orig = unassigned_indices[i_pt]
#                     if i_pt_orig in already_assigned:
#                         raise RuntimeError('Point %d cannot be assigned to more than one voronoi region' % i_pt_orig)
#                     assigned_pts.append(i_pt_orig)
#                     already_assigned.add(i_pt_orig)
#                     if len(assigned_pts) >= accept_n_coord_duplicates - n_assigned_dupl:
#                         break
#
#             if assigned_pts:
#                 # add the assignments for this Voronoi region
#                 assignments[i_poly].extend(assigned_pts)
#
#                 if accept_n_coord_duplicates == 0 and len(assignments[i_poly]) > 1:
#                     raise RuntimeError('No duplicate points allowed but polygon %d contains several points: %s'
#                                        % (i_poly, str(assignments[i_poly])))
#
#                 n_assigned_dupl += len(assigned_pts)-1
#
#     # make some final checks
#
#     assert len(assignments) == len(poly_shapes)
#
#     if dupl_restricted:
#         assert n_assigned_dupl == accept_n_coord_duplicates
#         assert sum(map(len, assignments)) == len(poly_shapes) + accept_n_coord_duplicates
#         assert len(already_assigned) == n_points   # make sure all points were assigned
#         unassigned_pt_indices = set()
#     else:
#         unassigned_pt_indices = set(range(n_points)) - already_assigned
#
#     if return_unassigned_points:
#         return assignments, unassigned_pt_indices
#     else:
#         return assignments


def get_points_to_poly_assignments(poly_to_pt_assignments):
    """
    Reverse of poly to points assignments: Returns a list of size N, which is the number of unique points in
    `poly_to_pt_assignments`. Each list element is an index into the list of Voronoi regions.
    """

    pt_to_poly = {}
    for poly, pts in poly_to_pt_assignments.items():
        for pt in pts:
            pt_to_poly[pt] = poly

    return pt_to_poly
