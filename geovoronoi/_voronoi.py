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

    # check input arguments

    if len(coords) < 2:
        raise ValueError('insufficient number of points provided in `coords`')

    # we need the points as NumPy coordinates array and as list of Shapely Point objects
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

    # get sub-geometries of `geo_shape` if they exist and if we want to treat sub-geometries separately
    if not per_geom or isinstance(geo_shape, Polygon):
        geoms = [geo_shape]
    else:   # Multipolygon has sub-geometries
        geoms = geo_shape.geoms

    # set up data containers
    pts_indices = set(range(len(pts)))   # point indices to operate on
    geom_region_polys = {}
    geom_region_pts = {}
    unassigned_pts = set()

    # iterate through sub-geometries in `geo_shape`
    for i_geom, geom in enumerate(geoms):
        # get point indices of points that lie within `geom`
        if len(geoms) == 1 or i_geom == len(geoms) - 1:
            pts_in_geom = list(pts_indices)     # no need to check if we only have one geom or only one geom left
        else:
            pts_in_geom = [i for i in pts_indices if geom.contains(pts[i])]

        # start with empty data for this sub-geometry
        geom_region_polys[i_geom] = {}
        geom_region_pts[i_geom] = {}

        logger.info('%d of %d points in geometry #%d of %d' % (len(pts_in_geom), len(pts), i_geom + 1, len(geoms)))

        if not pts_in_geom:   # no points within `geom` -> skip
            continue

        # remove the points that we're about to use (point - geometry assignment is bijective)
        if i_geom < len(geoms) - 1:   # no need to do this on last iteration
            pts_indices.difference_update(pts_in_geom)

        # generate the Voronoi regions using the SciPy Voronoi class
        logger.info('generating Voronoi regions')
        try:
            vor = Voronoi(coords[pts_in_geom])
        except QhullError as exc:   # handle error for insufficient number of input points by skipping this sub-geom.
            if exc.args and 'QH6214' in exc.args[0]:
                logger.error('not enough input points (%d) for Voronoi generation; original error message: %s' %
                             (len(pts_in_geom), exc.args[0]))
                unassigned_pts.update(pts_in_geom)
                continue
            raise exc  # otherwise re-raise the error

        # generate Voronoi region polygons and region -> points mapping
        logger.info('generating Voronoi region polygons')
        geom_polys, geom_pts = region_polygons_from_voronoi(vor, geom, return_point_assignments=True, **kwargs)

        # point indices in `geom_pts` refer to `coords[pts_in_geom]` -> map them back to original point indices
        pts_in_geom_arr = np.asarray(pts_in_geom)
        for i_reg, pt_indices in geom_pts.items():
            geom_pts[i_reg] = pts_in_geom_arr[pt_indices].tolist()

        # save data for this sub-geometry
        geom_region_polys[i_geom] = geom_polys
        geom_region_pts[i_geom] = geom_pts

    # collect results
    logger.info('collecting Voronoi region results')

    if results_per_geom:   # nothing to do here since we already have results per sub-geometry
        region_polys = geom_region_polys
        region_pts = geom_region_pts
    else:   # merge the results from the sub-geometries to dicts that map region IDs to polygons / point indices
            # across all sub-geometries
        region_polys = {}
        region_pts = {}

        for i_geom, geom_polys in geom_region_polys.items():
            geom_pts = geom_region_pts[i_geom]

            # assign new region IDs
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
    """
    Construct Shapely Polygon/Multipolygon objects for each Voronoi region in `vor` so that they cover the geometry
    `geom`.

    :param vor: SciPy Voronoi object
    :param geom: Shapely Polygon/MultiPolygon object that confines the constructed Voronoi regions into this shape
    :param return_point_assignments: if True, also return a dict that maps Voronoi region IDs to list of point indices
                                     inside that Voronoi region
    :param bounds_buf_factor: factor multiplied to largest extend of the `geom` bounding box when calculating the buffer
                              distance to enlarge the `geom` bounding box
    :param diff_topo_error_buf_factor: factor multiplied to largest extend of the `geom` bounding box when calculating
                                       the buffer distance to enlarge `geom` in the rare case of a TopologicalError
    :return: dict mapping Voronoi region IDs to Shapely Polygon/Multipolygon objects; if `return_point_assignments` is
             True, this function additionally returns a dict that maps Voronoi region IDs to list of point indices
             inside that Voronoi region
    """

    # plotting utilities useful for debugging:
    # from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
    # fig, ax = subplot_for_map(figsize=(12, 8))
    # plotfile_fmt = '/tmp/geovoronoi/debug-%s.png'

    # check input arguments

    if not isinstance(vor, Voronoi):
        raise ValueError('`vor` must be SciPy Voronoi object')

    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise ValueError('`geom` must be a Polygon or MultiPolygon')

    # construct geom bounding box with buffer
    max_extend = max(geom.bounds[2] - geom.bounds[0], geom.bounds[3] - geom.bounds[1])
    bounds_buf = max_extend * bounds_buf_factor
    geom_bb = box(*geom.bounds).buffer(bounds_buf)

    # center of all points
    center = np.array(MultiPoint(vor.points).convex_hull.centroid)

    # ridge vertice's coordinates
    ridge_vert = np.asarray(vor.ridge_vertices)

    # data containers for output
    region_pts = defaultdict(list)           # maps region ID to list of point indices
    region_neighbor_pts = defaultdict(set)   # maps region ID to set of direct neighbor point IDs
    region_polys = {}                        # maps region ID to Shapely Polygon/MultiPolygon object

    logger.debug('generating preliminary polygons')

    # this function has two phases:
    # - phase 1 generates preliminary polygons from the Voronoi region vertices in `vor`; these polygons may not cover
    #   the full area of `geom` yet
    # - phase 2 fills up the preliminary polygons to fully cover the area of the surrounding `geom`

    # --- phase 1: preliminary polygons ---

    covered_area = 0            # keeps track of area so far covered by generated Voronoi polygons
    border_regions = set()      # keeps track of regions that don't have fully finite bounds, i.e. that are at the edge
    inner_regions = set()       # keeps track of regions that have fully finite bounds, i.e. "inner" regions
    # iterate through regions; `i_reg` is the region ID, `reg_vert` contains vertex indices of this region into
    # `ridge_vert`
    for i_reg, reg_vert in enumerate(vor.regions):
        pt_indices, = np.nonzero(vor.point_region == i_reg)   # points within this region
        if len(pt_indices) == 0:  # skip regions w/o points in them
            continue

        # update the region -> point mappings
        region_pts[i_reg].extend(pt_indices.tolist())

        # construct the theoretical Polygon `p` of this region
        if np.all(np.array(reg_vert) >= 0):  # fully finite-bounded region
            inner_regions.add(i_reg)
            p = Polygon(vor.vertices[reg_vert])
        else:
            # this region has a least one infinite bound aka "loose ridge"; we go through each ridge and we will need to
            # calculate a finite end ("far point") for loose ridges
            border_regions.add(i_reg)

            p_vert_indices = set()    # collect unique indices of vertices into `ridge_vert`
            p_vert_farpoints = set()  # collect unique calculated finite far points

            # iterate through points inside this region (this is usually only one point, but we also consider
            # duplicates)
            for i_pt in pt_indices:
                # create a mask that selects ridge vertices for when this point lies on either side of the ridge
                enclosing_ridge_pts_mask = (vor.ridge_points[:, 0] == i_pt) | (vor.ridge_points[:, 1] == i_pt)

                # iterate through the point-to-point pairs (`ridge_points`) and the line segments that represent the
                # ridge given by ridge vertices
                for pointidx, vertidx in zip(vor.ridge_points[enclosing_ridge_pts_mask],
                                             ridge_vert[enclosing_ridge_pts_mask]):
                    # update the set of neighbor point indices for this region
                    region_neighbor_pts[i_reg].update([x for x in pointidx if x != i_pt])

                    if np.all(vertidx >= 0):   # both vertices of the ridge are finite points
                        p_vert_indices.update(vertidx)   # we can add both points
                    else:
                        # "loose ridge": contains infinite Voronoi vertex
                        # we calculate the far point, i.e. the point of intersection with the bounding box
                        i = vertidx[vertidx >= 0][0]  # finite end Voronoi vertex
                        finite_pt = vor.vertices[i]
                        p_vert_indices.add(i)

                        t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                        t /= np.linalg.norm(t)
                        n = np.array([-t[1], t[0]])  # normal

                        midpoint = vor.points[pointidx].mean(axis=0)   # point in the middle should be directly on the
                                                                       # ridge
                        direction = np.sign(np.dot(midpoint - center, n)) * n   # re-orient according to center

                        # find intersection(s) from ridge line going from `midpoint` towards `direction`
                        # eventually hitting a side of the geom bounding box
                        isects = []
                        for i_ext_coord in range(len(geom_bb.exterior.coords) - 1):
                            isect = line_segment_intersection(midpoint, direction,
                                                              np.array(geom_bb.exterior.coords[i_ext_coord]),
                                                              np.array(geom_bb.exterior.coords[i_ext_coord+1]))
                            if isect is not None:
                                isects.append(isect)

                        if len(isects) == 0:
                            raise RuntimeError('ridge line must intersect with surrounding geometry from `geom`')
                        elif len(isects) == 1:   # one intersection
                            far_pt = isects[0]
                        else:                    # multiple intersections - take closest intersection
                            closest_isect_idx = np.argmin(np.linalg.norm(midpoint - isects, axis=1))
                            far_pt = isects[closest_isect_idx]

                        # only use this far point if the found ridge points in the same direction
                        nonzero_coord = np.nonzero(direction)[0][0]
                        if (far_pt - finite_pt)[nonzero_coord] / direction[nonzero_coord] > 0:
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

        # check the generated Polygon

        if not isinstance(p, Polygon):
            raise RuntimeError('generated convex hull is not a polygon')

        if not p.is_valid or p.is_empty:
            raise RuntimeError('generated polygon is not valid or is empty')

        # if the generated preliminary polygon `p` is not completely inside `geom`, then cut it (i.e. use intersection)
        if not geom.contains(p):
            p = p.intersection(geom)

            if not p.is_valid or p.is_empty:
                raise RuntimeError('generated polygon is not valid or is empty after intersection with the'
                                   ' surrounding geometry `geom`')

        # update covered area and set the preliminary polygon for this region
        covered_area += p.area
        region_polys[i_reg] = p

    # --- phase 2: filling preliminary polygons at the edges ---

    logger.debug('filling %d preliminary polygons to fully cover the surrounding area' % len(border_regions))

    # plot_voronoi_polys_with_points_in_area(ax, geom, region_polys, vor.points, region_pts)
    # fig.savefig(plotfile_fmt % 'beforefill')

    uncovered_area_portion = (geom.area - covered_area) / geom.area     # initial portion of uncovered area
    polys_iter = iter(border_regions)
    # the loop has two passes: in the first pass, only areas are considered that intersect with the preliminary
    # Voronoi region polygon; in the second pass, also areas that don't intersect are considered, i.e. isolated features
    pass_ = 1
    # the loop stops when all area is covered or there are no more preliminary Voronoi region polygons left after both
    # passes
    inner_regions_poly = None
    while not np.isclose(uncovered_area_portion, 0) and 0 < uncovered_area_portion <= 1:
        try:
            i_reg = next(polys_iter)
        except StopIteration:
            if pass_ == 1:   # restart w/ second pass
                polys_iter = iter(border_regions)
                i_reg = next(polys_iter)
                pass_ += 1
            else:     # no more preliminary Voronoi region polygons left after both passes
                break
        logger.debug('pass %d / region %d: will fill up %f%% uncovered area'
                     % (pass_, i_reg, uncovered_area_portion * 100))

        p = region_polys[i_reg]

        # generate polygon from all inner regions *once* because this stays constant and is re-used in the loop
        if not inner_regions_poly and inner_regions:
            if len(inner_regions) == 1:
                inner_regions_poly = region_polys[next(iter(inner_regions))]
            else:
                inner_regions_poly = cascaded_union([region_polys[i_reg] for i_reg in inner_regions])

        # generate polygon from all other regions' polygons other than the current region `i_reg`
        other_regions_polys = [region_polys[i_other] for i_other in border_regions if i_reg != i_other]
        if inner_regions_poly:
            other_regions_polys.append(inner_regions_poly)

        union_other_regions = cascaded_union(other_regions_polys)

        # generate difference between geom and other regions' polygons -- what's left is the current region's area
        # plus any so far uncovered area
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

        # list of neighbor region IDs
        neighbor_regions = [vor.point_region[pt] for pt in region_neighbor_pts[i_reg]]
        add = []  # will hold shapes to be added to the current region's polygon
        for diff_part in diff:   # iterate through the geometries in the difference
            if diff_part.is_valid and not diff_part.is_empty:
                if pass_ == 1:
                    # for first pass, we want an intersection between the so far uncovered area and the current
                    # region's polygon, so that the uncovered area is directly connected with the region's polygon
                    isect = p.intersection(diff_part)
                    has_isect = isinstance(isect, (Polygon, MultiPolygon)) and not isect.is_empty \
                                and not np.isclose(abs(diff_part.area - p.area), 0)
                else:
                    has_isect = True   # we don't need an intersection on second pass -- handle isolated features

                if has_isect:
                    # we make sure that there's no neighboring region's polygon in between the current region's polygon
                    # and the area we want to add
                    center_to_center = LineString([p.centroid, diff_part.centroid])
                    if not any(center_to_center.crosses(region_polys[i_neighb]) for i_neighb in neighbor_regions):
                        add.append(diff_part)

        # add new areas as union
        if add:
            old_reg_area = region_polys[i_reg].area
            new_reg = cascaded_union([p] + add)
            area_diff = new_reg.area - old_reg_area

            region_polys[i_reg] = new_reg
            covered_area += area_diff

            # plot_voronoi_polys_with_points_in_area(ax, geom, region_polys, vor.points, region_pts)
            # fig.savefig(plotfile_fmt % ('fill-' + str(pass_) + '-' + str(i_reg)))

        # update the portion of uncovered area
        uncovered_area_portion = (geom.area - covered_area) / geom.area

    logger.debug('%f%% uncovered area left' % (uncovered_area_portion * 100))

    if return_point_assignments:
        return region_polys, region_pts
    else:
        return region_polys


def points_to_region(region_pts):
    """
    Reverse Voronoi region polygon IDs to point ID assignments by returning a dict that maps each point ID to its
    Voronoi region polygon ID. All IDs should be integers.

    :param region_pts: dict mapping Voronoi region polygon IDs to list of point IDs
    :return: dict mapping point ID to Voronoi region polygon ID
    """

    pts_region = {}
    for poly, pts in region_pts.items():
        for pt in pts:
            if pt in pts_region.keys():
                raise ValueError('invalid assignments in `region_pts`: point %d is assigned to several polygons' % pt)
            pts_region[pt] = poly

    return pts_region
