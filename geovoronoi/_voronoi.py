"""
Functions to create Voronoi regions from points inside a geographic area.

"shapely" refers to the [Shapely Python package for computational geometry](http://toblerity.org/shapely/index.html).

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.qhull import QhullError
from shapely.geometry import box, LineString, asPoint, MultiPoint, Polygon, MultiPolygon
from shapely.ops import cascaded_union

from ._geom import line_segment_intersection


logger = logging.getLogger('geovoronoi')
logger.addHandler(logging.NullHandler())


def coords_to_points(coords):
    """Convert a NumPy array of 2D coordinates `coords` to a list of shapely Point objects"""
    return list(map(asPoint, coords))


def points_to_coords(pts):
    """Convert a list of shapely Point objects to a NumPy array of 2D coordinates `coords`"""
    return np.array([p.coords[0] for p in pts])


def voronoi_regions_from_coords(coords, geo_shape, per_geom=True, return_unassigned_points=False):
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

    logger.info('running Voronoi tesselation for %d points / treating geoms separately: %s' % (len(coords), per_geom))

    if isinstance(coords, np.ndarray):
        pts = coords_to_points(coords)
    else:
        pts = coords
        coords = points_to_coords(pts)

    if not isinstance(geo_shape, (Polygon, MultiPolygon)):
        raise ValueError('`geo_shape` must be a Polygon or MultiPolygon')

    if not per_geom or isinstance(geo_shape, Polygon):
        geoms = [geo_shape]
    else:   # Multipolygon
        geoms = geo_shape.geoms

    pts_indices = set(range(len(pts)))
    region_polys = {}
    region_pts = {}
    unassigned_pts = set()
    for i_geom, geom in enumerate(geoms):
        pts_in_geom = [i for i in pts_indices if geom.contains(pts[i])]
        logger.info('%d of %d points in geometry #%d of %d' % (len(pts_in_geom), len(pts), i_geom + 1, len(geoms)))
        if not pts_in_geom:
            continue

        pts_indices.difference_update(pts_in_geom)

        logger.info('generating Voronoi regions')
        try:
            vor = Voronoi(coords[pts_in_geom])
        except QhullError as exc:
            if exc.args and 'QH6214' in exc.args[0]:
                logger.error('not enough input points (%d) for Voronoi generation; original error message: %s' %
                             (len(pts_in_geom), exc.args[0]))
                unassigned_pts.update(pts_in_geom)
                continue
            raise exc

        logger.info('generating Voronoi region polygons')
        geom_polys, geom_pts = region_polygons_from_voronoi(vor, geom, return_point_assignments=True)

        region_ids = range(len(region_polys), len(region_polys) + len(geom_polys))
        region_ids_mapping = dict(zip(region_ids, geom_polys.keys()))
        region_polys.update(dict(zip(region_ids, geom_polys.values())))

        pt_ids_mapping = dict(zip(range(len(pts_in_geom)), pts_in_geom))
        region_pts.update(
            {reg_id: [pt_ids_mapping[pt_id] for pt_id in geom_pts[old_reg_id]]
                     for reg_id, old_reg_id in region_ids_mapping.items()}
        )

    if return_unassigned_points:
        return region_polys, region_pts, unassigned_pts
    else:
        return region_polys, region_pts


def region_polygons_from_voronoi(vor, geom, return_point_assignments=False):
    geom_bb = box(*geom.bounds)
    center = np.array(MultiPoint(vor.points).convex_hull.centroid)
    ridge_vert = np.array(vor.ridge_vertices)

    region_pts = defaultdict(list)
    region_neighbor_pts = defaultdict(set)
    region_polys = {}

    logger.debug('generating polygons')

    covered_area = 0
    for i_reg, reg_vert in enumerate(vor.regions):
        pt_indices, = np.nonzero(vor.point_region == i_reg)
        if len(pt_indices) == 0:  # skip regions w/o points in them
            continue

        region_pts[i_reg].extend(pt_indices.tolist())

        if np.all(np.array(reg_vert) >= 0):  # fully finite-bounded region
            p = Polygon(vor.vertices[reg_vert])
        else:
            p_vert_indices = set()
            p_vert_farpoints = set()
            for i_pt in pt_indices:     # also consider duplicates
                enclosing_ridge_pts_mask = (vor.ridge_points[:, 0] == i_pt) | (vor.ridge_points[:, 1] == i_pt)
                for pointidx, simplex in zip(vor.ridge_points[enclosing_ridge_pts_mask],
                                             ridge_vert[enclosing_ridge_pts_mask]):

                    region_neighbor_pts[i_reg].update([x for x in pointidx if x != i_pt])

                    if np.all(simplex >= 0):   # both vertices of the ridge are finite points
                        p_vert_indices.update(simplex)
                    else:
                        # "loose ridge": contains infinite Voronoi vertex
                        # we calculate the far point, i.e. the point of intersection with a surrounding polygon boundary
                        i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
                        finite_pt = vor.vertices[i]
                        p_vert_indices.add(i)

                        t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                        t /= np.linalg.norm(t)
                        n = np.array([-t[1], t[0]])  # normal

                        midpoint = vor.points[pointidx].mean(axis=0)
                        direction = np.sign(np.dot(midpoint - center, n)) * n

                        isects = []
                        for i_ext_coord in range(len(geom_bb.exterior.coords) - 1):
                            isect = line_segment_intersection(midpoint, direction,
                                                              np.array(geom_bb.exterior.coords[i_ext_coord]),
                                                              np.array(geom_bb.exterior.coords[i_ext_coord+1]))
                            if isect is not None:
                                isects.append(isect)

                        if len(isects) == 0:
                            raise RuntimeError('far point must intersect with surrounding geometry from `geom`')
                        elif len(isects) == 1:
                            far_pt = isects[0]
                        else:
                            closest_isect_idx = np.argmin(np.linalg.norm(midpoint - isects, axis=1))
                            far_pt = isects[closest_isect_idx]

                        if (far_pt - finite_pt)[0] / direction[0] > 0:   # only if in same direction
                            p_vert_farpoints.add(tuple(far_pt))

            # create the Voronoi region polygon as convex hull of the ridge vertices and far points (Voronoi regions
            # are convex)

            p_pts = vor.vertices[np.asarray(list(p_vert_indices))]
            # adding the point itself prevents generating invalid hull (otherwise sometimes the hull becomes a line
            # if the vertices are almost colinear)
            p_pts = np.vstack((p_pts, vor.points[pt_indices]))
            if p_vert_farpoints:
                p_pts = np.vstack((p_pts, np.asarray(list(p_vert_farpoints))))

            p = MultiPoint(p_pts).convex_hull

        if not isinstance(p, Polygon):
            raise RuntimeError('generated convex hull is not a polygon')

        if not p.is_valid or not p.is_simple or p.is_empty:
            raise RuntimeError('generated polygon is not valid, not simple or empty')

        if not geom.contains(p):
            p = p.intersection(geom)

            if not p.is_valid or not p.is_simple or p.is_empty:
                raise RuntimeError('generated polygon is not valid, not simple or empty after intersection with the'
                                   ' surrounding geometry `geom`')

        covered_area += p.area
        region_polys[i_reg] = p

    uncovered_area_portion = (geom.area - covered_area) / geom.area
    polys_iter = iter(region_polys.items())
    while not np.isclose(uncovered_area_portion, 0) and uncovered_area_portion > 0:
        try:
            i_reg, p = next(polys_iter)
        except StopIteration:
            break
        logger.debug('will fill up %f%% uncovered area' % (uncovered_area_portion * 100))

        union_other_regions = cascaded_union([other_poly
                                              for i_other, other_poly in region_polys.items()
                                              if i_reg != i_other])
        diff = geom.difference(union_other_regions)
        if isinstance(diff, MultiPolygon):
            diff = diff.geoms
        else:
            diff = [diff]

        neighbor_regions = [vor.point_region[pt] for pt in region_neighbor_pts[i_reg]]
        add = []
        for diff_part in diff:
            if diff_part.is_valid and not diff_part.is_empty:
                isect = p.intersection(diff_part)
                if isinstance(isect, Polygon) and not isect.is_empty:
                    center_to_center = LineString([p.centroid, diff_part.centroid])
                    if not any(center_to_center.crosses(region_polys[i_neighb]) for i_neighb in neighbor_regions):
                        add.append(diff_part)
                        covered_area += (diff_part.area - p.area)

        if add:
            region_polys[i_reg] = cascaded_union([p] + add)

        uncovered_area_portion = (geom.area - covered_area) / geom.area

    logger.debug('%f%% uncovered area left' % (uncovered_area_portion * 100))

    if return_point_assignments:
        return region_polys, region_pts
    else:
        return region_polys


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
