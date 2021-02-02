"""
Functions to create Voronoi regions from points inside a geographic area.

"Shapely" refers to the [Shapely Python package for computational geometry](http://toblerity.org/shapely/index.html).

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.qhull import QhullError
from shapely.geometry import box, LineString, asPoint, MultiPoint, Polygon, MultiPolygon
from shapely.errors import TopologicalError
from shapely.ops import cascaded_union

from ._geom import line_segment_intersection


logger = logging.getLogger('geovoronoi')
logger.addHandler(logging.NullHandler())


def coords_to_points(coords):
    """
    Convert a NumPy array of 2D coordinates `coords` to a list of Shapely Point objects.

    This is the inverse of `points_to_coords()`.

    :param coords: NumPy array of shape (N,2) with N coordinates in 2D space
    :return: list of length N with Shapely Point objects
    """
    return list(map(asPoint, coords))


def points_to_coords(pts):
    """
    Convert a list of Shapely Point objects to a NumPy array of 2D coordinates `coords`.

    This is the inverse of `coords_to_points()`.

    :param pts: list of length N with Shapely Point objects
    :return: NumPy array of shape (N,2) with N coordinates in 2D space
    """
    return np.array([p.coords[0] for p in pts])


def voronoi_regions_from_coords(coords, geo_shape, per_geom=True, return_unassigned_points=False,
                                results_per_geom=False, **kwargs):
    """
    Generate Voronoi regions from NumPy array of 2D coordinates or list of Shapely Point objects in `coord`. These
    points must lie within a shape `geo_shape` which must be a valid Shapely Polygon or MultiPolygon object. If
    `geo_shape` is a MultiPolygon, each of its sub-geometries will be either treated separately during Voronoi
    region generation when `per_geom` is True or otherwise the whole MultiPolygon is treated as one object. In the
    former case, Voronoi regions may not span from one sub-geometry to another (e.g. from one island to another
    island) which also means that sub-geometries may remain empty (e.g. when there are no points on an island). In the
    latter case Voronoi regions from one sub-geometry may span to another sub-geometry, hence all sub-geometries
    should be covered by a Voronoi region as a result.

    This function returns at least two dicts in a tuple, called `region_polys` and `region_pts`. The first contains a
    dict that maps unique Voronoi region IDs to the generated Voronoi region geometries as Shapely Polygon/MultiPolygon
    objects. The second contains a dict that maps the region IDs to a list of point indices of `coords`. This dict
    describes which Voronoi region contains which points. By definition a Voronoi region surrounds only a single point.
    However, if you have duplicate coordinates in `coords`, these duplicate points will be surrounded by the same
    Voronoi region.

    The structure of the returned dicts depends on `results_per_geom`. If `results_per_geom` is False, there is a direct
    mapping from Voronoi region ID to region geometry or assigned points respectively. If `results_per_geom` is True,
    both dicts map a sub-geometry ID from `geo_shape` (the index of the sub-geometry in `geo_shape.geoms`) to the
    respective dict which in turn map a Voronoi region ID to its region geometry or assigned points.

    To conclude with N coordinates in `coords` and `results_per_geom` is False (default):

    ```
    region_polys =
    {
        0: <Shapely Polygon/MultiPolygon>,
        1: <Shapely Polygon/MultiPolygon>,
        ...
        N-1: <Shapely Polygon/MultiPolygon>
    }
    region_pts =
    {
        0: [5],
        1: [7, 2],   # coords[7] and coords[2] are duplicates
        ...
        N-1: [3]
    }
    ```

    And if `results_per_geom` is True and there are M sub-geometries in `geo_shape`:

    ```
    region_polys =
    {
        0: {   # first sub-geometry in `geom_shape` with Voronoi regions inside
            4: <Shapely Polygon/MultiPolygon>,
            2: <Shapely Polygon/MultiPolygon>,
            ...
        },
        ...
        M-1: { # last sub-geometry in `geom_shape` with Voronoi regions inside
            9: <Shapely Polygon/MultiPolygon>,
            12: <Shapely Polygon/MultiPolygon>,
            ...
        },
    }

    region_pts = (similar to above)
    ```

    Setting `results_per_geom` to True only makes sense when `per_geom` is True so that Voronoi region generation is
    done separately for each sub-geometry in `geo_shape`.

    :param coords: NumPy array of 2D coordinates as shape (N,2) for N points or list of Shapely Point objects; should
                   contain at least two points
    :param geo_shape: Shapely Polygon or MultiPolygon object that defines the restricting area of the Voronoi regions;
                      all points in `coords` should be within `geo_shape`; make sure that `geo_shape` is a valid
                      geometry
    :param per_geom: if True, treat sub-geometries in `geo_shape` separately during Voronoi region generation
    :param return_unassigned_points: If True, additionally return set of point indices which could not be assigned to
                                     any Voronoi region (usually because there were too few points inside a sub-geometry
                                     to generate Voronoi regions)
    :param results_per_geom: partition the result dicts by sub-geometry index
    :param kwargs: parameters passed to `region_polygons_from_voronoi()`
    :return tuple of length two if return_unassigned_points is False, otherwise of length three with: (1) dict
            containing generated Voronoi region geometries as Shapely Polygon/MultiPolygon, (2) dict mapping Voronoi
            regions to point indices of `coords`, (3 - optionally) set of point indices which could not be assigned to
            any Voronoi region
    """

    logger.info('running Voronoi tesselation for %d points / treating geoms separately: %s' % (len(coords), per_geom))

    if len(coords) < 2:
        raise ValueError('insufficient number of points provided in `coords`')

    if isinstance(coords, np.ndarray):
        pts = coords_to_points(coords)
    else:
        pts = coords
        coords = points_to_coords(pts)

    if not isinstance(geo_shape, (Polygon, MultiPolygon)):
        raise ValueError('`geo_shape` must be a Polygon or MultiPolygon')

    if not geo_shape.is_valid:
        raise ValueError('`geo_shape` is not a valid shape; try applying `.buffer(b)` where `b` is zero '
                         'or a very small number')

    if not per_geom or isinstance(geo_shape, Polygon):
        geoms = [geo_shape]
    else:   # Multipolygon
        geoms = geo_shape.geoms

    pts_indices = set(range(len(pts)))
    geom_region_polys = {}
    geom_region_pts = {}
    unassigned_pts = set()

    for i_geom, geom in enumerate(geoms):
        pts_in_geom = [i for i in pts_indices if geom.contains(pts[i])]

        geom_region_polys[i_geom] = {}
        geom_region_pts[i_geom] = {}

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
        geom_polys, geom_pts = region_polygons_from_voronoi(vor, geom, return_point_assignments=True, **kwargs)

        # map back to original point indices
        pts_in_geom_arr = np.asarray(pts_in_geom)
        for i_reg, pt_indices in geom_pts.items():
            geom_pts[i_reg] = pts_in_geom_arr[pt_indices].tolist()

        geom_region_polys[i_geom] = geom_polys
        geom_region_pts[i_geom] = geom_pts

    logger.info('collecting Voronoi region results')

    if results_per_geom:
        region_polys = geom_region_polys
        region_pts = geom_region_pts
    else:
        region_polys = {}
        region_pts = {}

        for i_geom, geom_polys in geom_region_polys.items():
            geom_pts = geom_region_pts[i_geom]

            region_ids = range(len(region_polys), len(region_polys) + len(geom_polys))
            region_ids_mapping = dict(zip(region_ids, geom_polys.keys()))
            region_polys.update(dict(zip(region_ids, geom_polys.values())))

            region_pts.update({reg_id: geom_pts[old_reg_id] for reg_id, old_reg_id in region_ids_mapping.items()})

    if return_unassigned_points:
        return region_polys, region_pts, unassigned_pts
    else:
        return region_polys, region_pts


def region_polygons_from_voronoi(vor, geom, return_point_assignments=False,
                                 bounds_buf_factor=0.1,
                                 diff_topo_error_buf_factor=0.00000001):
    max_extend = max(geom.bounds[2] - geom.bounds[0], geom.bounds[3] - geom.bounds[1])
    bounds_buf = max_extend * bounds_buf_factor
    geom_bb = box(*geom.bounds).buffer(bounds_buf)
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

                        nonzero_coord = np.nonzero(direction)[0][0]
                        if (far_pt - finite_pt)[nonzero_coord] / direction[nonzero_coord] > 0:   # only if in same direction
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

        if not p.is_valid or p.is_empty:
            raise RuntimeError('generated polygon is not valid or is empty')

        if not geom.contains(p):
            p = p.intersection(geom)

            if not p.is_valid or p.is_empty:
                raise RuntimeError('generated polygon is not valid or is empty after intersection with the'
                                   ' surrounding geometry `geom`')

        covered_area += p.area
        region_polys[i_reg] = p

    uncovered_area_portion = (geom.area - covered_area) / geom.area
    polys_iter = iter(region_polys.items())
    pass_ = 1
    while not np.isclose(uncovered_area_portion, 0) and 0 < uncovered_area_portion <= 1:
        try:
            i_reg, p = next(polys_iter)
        except StopIteration:
            if pass_ == 1:   # restart w/ second pass
                polys_iter = iter(region_polys.items())
                i_reg, p = next(polys_iter)
                pass_ += 1
            else:
                break
        logger.debug('will fill up %f%% uncovered area' % (uncovered_area_portion * 100))

        union_other_regions = cascaded_union([other_poly
                                              for i_other, other_poly in region_polys.items()
                                              if i_reg != i_other])
        try:
            diff = geom.difference(union_other_regions)
        except TopologicalError:   # may happen in rare circumstances
            try:
                diff = geom.buffer(max_extend * diff_topo_error_buf_factor).difference(union_other_regions)
            except TopologicalError:
                raise RuntimeError('difference operation failed with TopologicalError; try setting a different '
                                   '`diff_topo_error_buf_factor`')

        if isinstance(diff, MultiPolygon):
            diff = diff.geoms
        else:
            diff = [diff]

        neighbor_regions = [vor.point_region[pt] for pt in region_neighbor_pts[i_reg]]
        add = []
        for diff_part in diff:
            if diff_part.is_valid and not diff_part.is_empty:
                if pass_ == 1:
                    isect = p.intersection(diff_part)
                    has_isect = isinstance(isect, (Polygon, MultiPolygon)) and not isect.is_empty
                else:
                    has_isect = True   # we don't need an intersection on second pass -- handle isolated features

                if has_isect:
                    center_to_center = LineString([p.centroid, diff_part.centroid])
                    if not any(center_to_center.crosses(region_polys[i_neighb]) for i_neighb in neighbor_regions):
                        add.append(diff_part)
                        if pass_ == 1:
                            covered_area += (diff_part.area - p.area)
                        else:
                            covered_area += diff_part.area

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
    Reverse Voronoi region polygon IDs to point ID assignments by returning a dict that maps each point ID to its
    Voronoi region polygon ID. All IDs should be integers.

    :param poly_to_pt_assignments: dict mapping Voronoi region polygon IDs to list of point IDs
    :return: dict mapping point ID to Voronoi region polygon ID
    """

    pt_to_poly = {}
    for poly, pts in poly_to_pt_assignments.items():
        for pt in pts:
            if pt in pt_to_poly.keys():
                raise ValueError('invalid assignments in `poly_to_pt_assignments`: '
                                 'point %d is assigned to several polygons' % pt)
            pt_to_poly[pt] = poly

    return pt_to_poly
