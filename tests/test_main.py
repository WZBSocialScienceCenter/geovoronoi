"""
Tests for the main module.

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

from ._testtools import coords_2d_array
from geovoronoi import (
    voronoi_regions_from_coords, coords_to_points, points_to_coords, calculate_polygon_areas,
    points_to_region
)
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area


#%% tests for individual functions

@given(coords=coords_2d_array())
def test_coords_to_points_and_points_to_coords(coords):
    # test for bijectivity of points_to_coords and coords_to_points
    assert np.array_equal(points_to_coords(coords_to_points(coords)), coords)

@pytest.mark.parametrize(
    'poly_to_pts, expected',
    [
        ({5: [1], 2: [3]},      {1: 5, 3: 2}),
        ({5: [], 2: [3]},       {3: 2}),
        ({5: [], 2: []},        {}),
        ({1: [1]},              {1: 1}),
        ({1: [4, 5]},           {4: 1, 5: 1}),
        ({5: [1], 2: [1]},      None)
    ]
)
def test_get_points_to_poly_assignments(poly_to_pts, expected):
    if expected is None:
        with pytest.raises(ValueError):
            points_to_region(poly_to_pts)
    else:
        assert points_to_region(poly_to_pts) == expected


@given(available_points=st.permutations(list(range(10))), n_poly=st.integers(0, 10))
def test_get_points_to_poly_assignments_hypothesis(available_points, n_poly):
    # generate poly to point assignments
    n_pts = len(available_points)
    if n_poly == 0:
        poly_to_pts = {}
    elif n_poly == 10:   # one to one assignment
        poly_to_pts = dict(zip(range(n_poly), [[x] for x in available_points]))
    else:   # one to N assignment (we have duplicate points)
        pts_per_poly = n_pts // n_poly
        poly_to_pts = {}
        n_assigned = 0
        # try to evenly distribute point IDs to polys
        for p in range(0, n_poly):
            poly_to_pts[p] = [available_points[i] for i in range(p * pts_per_poly, (p+1) * pts_per_poly)]
            n_assigned += pts_per_poly

        # fill up
        if n_assigned < n_pts:
            poly_to_pts[n_poly-1].extend([available_points[i] for i in range(n_assigned, n_pts)])

    if n_poly > 0:
        assert set(sum(list(poly_to_pts.values()), [])) == set(available_points)

    pts_to_poly = points_to_region(poly_to_pts)

    assert isinstance(pts_to_poly, dict)

    if n_poly == 0:
        assert len(pts_to_poly) == 0
    else:
        assert len(pts_to_poly) == n_pts
        assert set(list(pts_to_poly.keys())) == set(available_points)
        assert set(list(pts_to_poly.values())) == set(list(range(n_poly)))


@settings(deadline=10000)
@given(n_pts=st.integers(0, 200),
       per_geom=st.booleans(),
       return_unassigned_pts=st.booleans(),
       results_per_geom=st.booleans())
def test_voronoi_regions_from_coords_italy(n_pts, per_geom, return_unassigned_pts, results_per_geom):
    area_shape = _get_country_shape('Italy')
    n_geoms = len(area_shape.geoms)     # number of geometries (is 3 -- main land Italy plus Sardinia and Sicilia)
    # put random coordinates inside shape
    coords = _rand_coords_in_shape(area_shape, n_pts)
    n_pts = len(coords)   # number of random points inside shape

    if n_pts < 2:  # check ValueError when less than 2 points are submitted
        with pytest.raises(ValueError):
            voronoi_regions_from_coords(coords, area_shape,
                                        per_geom=per_geom,
                                        return_unassigned_points=return_unassigned_pts,
                                        results_per_geom=results_per_geom)

        return

    # generate Voronoi region polygons
    res = voronoi_regions_from_coords(coords, area_shape,
                                      per_geom=per_geom,
                                      return_unassigned_points=return_unassigned_pts,
                                      results_per_geom=results_per_geom)

    # in any case, this must return a tuple of results
    assert isinstance(res, tuple)

    if return_unassigned_pts:   # additionally expect set of unassigned points
        assert len(res) == 3
        region_polys, region_pts, unassigned_pts = res
        assert isinstance(unassigned_pts, set)
        assert all([pt in range(n_pts) for pt in unassigned_pts])
    else:
        assert len(res) == 2
        region_polys, region_pts = res
        unassigned_pts = None

    # check general result structure
    assert isinstance(region_polys, dict)
    assert isinstance(region_pts, dict)
    assert list(region_polys.keys()) == list(region_pts.keys())

    if results_per_geom:    # expect a dict that maps geom ID to results
        if not per_geom:    # if geoms are not treated separately, there's only one geom ID
            assert list(region_polys.keys()) == list(region_pts.keys()) == [0]

        # iterate through geoms
        for i_geom in region_polys.keys():
            # get Voronoi polygons
            assert 0 <= i_geom < n_geoms
            region_polys_in_geom = region_polys[i_geom]
            assert isinstance(region_polys_in_geom, dict)

            # get Voronoi region -> points assignments
            region_pts_in_geom = region_pts[i_geom]
            assert isinstance(region_pts_in_geom, dict)
            assert list(region_polys_in_geom.keys()) == list(region_pts_in_geom.keys())

            # check region polygons
            if region_pts_in_geom:
                if per_geom:
                    geom_area = area_shape.geoms[i_geom].area
                else:
                    geom_area = sum(g.area for g in area_shape.geoms)
                _check_region_polys(region_polys_in_geom.values(), region_pts_in_geom.values(), coords,
                                    expected_sum_area=geom_area)
            else:
                # no polys generated -> must be insufficient number of points in geom
                pass
    else:   # not results_per_geom
        # results are *not* given per geom ID
        assert len(region_polys) <= n_pts
        assert len(region_pts) == len(region_polys)

        # points to region assignments
        pts_region = points_to_region(region_pts)
        if unassigned_pts is not None:   # check that unassigned points are not in the result set
            assert set(range(n_pts)) - set(pts_region.keys()) == unassigned_pts

        # check result structure
        assert isinstance(region_polys, dict)
        assert isinstance(region_pts, dict)
        assert list(region_polys.keys()) == list(region_pts.keys())

        # check region polygons
        if region_polys:
            if per_geom:
                geom_area = None  # can't determine this here
            else:
                geom_area = sum(g.area for g in area_shape.geoms)
            _check_region_polys(region_polys.values(), region_pts.values(), coords, expected_sum_area=geom_area)
        else:
            # no polys generated -> must be insufficient number of points
            assert n_pts < 4

    # fig, ax = subplot_for_map(show_x_axis=True, show_y_axis=True)
    # plot_voronoi_polys_with_points_in_area(ax, area_shape, region_polys, coords, region_pts,
    #                                        point_labels=list(map(str, range(len(coords)))),
    #                                        voronoi_labels=list(map(str, region_polys.keys())))
    # fig.show()


# #%% realistic full tests with plotting


@pytest.mark.parametrize(
    'n_pts,per_geom', [
        (10, True), (10, False),
        (20, True), (20, False),
        (50, True), (50, False),
        (100, True), (100, False),
        (500, True), (500, False),
        (1000, True), (1000, False),
    ]
)
@pytest.mark.mpl_image_compare
def test_voronoi_italy_with_plot(n_pts, per_geom):
    area_shape = _get_country_shape('Italy')
    coords = _rand_coords_in_shape(area_shape, n_pts)

    # generate Voronoi regions
    region_polys, region_pts = voronoi_regions_from_coords(coords, area_shape, per_geom=per_geom)

    # full checks for voronoi_regions_from_coords() are done in test_voronoi_regions_from_coords_italy()

    assert isinstance(region_polys, dict)
    assert isinstance(region_pts, dict)
    assert len(region_polys) == len(region_pts)
    assert 0 < len(region_polys) <= n_pts

    # generate plot
    fig, ax = subplot_for_map(show_spines=True)
    plot_voronoi_polys_with_points_in_area(ax, area_shape, region_polys, coords, region_pts,
                                           point_labels=list(map(str, range(len(coords)))))

    return fig


@pytest.mark.mpl_image_compare
def test_voronoi_spain_area_with_plot():
    area_shape = _get_country_shape('Spain')
    coords = _rand_coords_in_shape(area_shape, 20)

    # generate Voronoi regions
    region_polys, region_pts = voronoi_regions_from_coords(coords, area_shape)

    # full checks for voronoi_regions_from_coords() are done in test_voronoi_regions_from_coords_italy()

    assert isinstance(region_polys, dict)
    assert isinstance(region_pts, dict)
    assert len(region_polys) == len(region_pts)
    assert 0 < len(region_polys) <= 20

    # generate covered area
    region_areas = calculate_polygon_areas(region_polys, m2_to_km2=True)  # converts m² to km²
    assert isinstance(region_areas, dict)
    assert set(region_areas.keys()) == set(region_polys.keys())

    # generate plot
    fig, ax = subplot_for_map(show_x_axis=True, show_y_axis=True)
    voronoi_labels = {k: '%d km²' % round(a) for k, a in region_areas.items()}
    plot_voronoi_polys_with_points_in_area(ax, area_shape, region_polys, coords, region_pts,
                                           voronoi_labels=voronoi_labels, voronoi_label_fontsize=7,
                                           voronoi_label_color='gray')

    return fig


@pytest.mark.mpl_image_compare
def test_voronoi_geopandas_with_plot():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

    # focus on South America, convert to World Mercator (unit: meters)
    south_am = world[world.continent == 'South America'].to_crs(epsg=3395)
    cities = cities.to_crs(south_am.crs)  # convert city coordinates to same CRS!

    # create the bounding shape as union of all South American countries' shapes
    south_am_shape = unary_union(south_am.geometry)
    south_am_cities = cities[cities.geometry.within(south_am_shape)]  # reduce to cities in South America

    # convert the pandas Series of Point objects to NumPy array of coordinates
    coords = points_to_coords(south_am_cities.geometry)

    # calculate the regions
    region_polys, region_pts = voronoi_regions_from_coords(coords, south_am_shape, per_geom=False)

    # full checks for voronoi_regions_from_coords() are done in test_voronoi_regions_from_coords_italy()

    assert isinstance(region_polys, dict)
    assert isinstance(region_pts, dict)
    assert len(region_polys) == len(region_pts) == len(coords)

    # generate plot
    fig, ax = subplot_for_map(show_spines=True)
    plot_voronoi_polys_with_points_in_area(ax, south_am_shape, region_polys, coords, region_pts)

    return fig


@pytest.mark.mpl_image_compare
def test_voronoi_sweden_duplicate_points_with_plot():
    area_shape = _get_country_shape('Sweden')
    coords = _rand_coords_in_shape(area_shape, 20)

    # duplicate a few points
    rand_dupl_ind = np.random.randint(len(coords), size=10)
    coords = np.concatenate((coords, coords[rand_dupl_ind]))
    n_pts = len(coords)

    # generate Voronoi regions
    region_polys, region_pts = voronoi_regions_from_coords(coords, area_shape)

    # full checks for voronoi_regions_from_coords() are done in test_voronoi_regions_from_coords_italy()

    assert isinstance(region_polys, dict)
    assert isinstance(region_pts, dict)
    assert 0 < len(region_polys) <= n_pts
    assert 0 < len(region_pts) <= n_pts

    assert all([0 < len(pts_in_region) <= 10 for pts_in_region in region_pts.values()])

    # make point labels: counts of duplicate assignments per points
    count_per_pt = {pt_indices[0]: len(pt_indices) for pt_indices in region_pts.values()}
    pt_labels = list(map(str, count_per_pt.values()))
    distinct_pt_coords = coords[np.asarray(list(count_per_pt.keys()))]

    # highlight voronoi regions with point duplicates
    vor_colors = {i_poly: (1, 0, 0) if len(pt_indices) > 1 else (0, 0, 1)
                  for i_poly, pt_indices in region_pts.items()}

    # generate plot
    fig, ax = subplot_for_map(show_spines=True)
    plot_voronoi_polys_with_points_in_area(ax, area_shape, region_polys, distinct_pt_coords,
                                           plot_voronoi_opts={'alpha': 0.2},
                                           plot_points_opts={'alpha': 0.4},
                                           voronoi_color=vor_colors,
                                           voronoi_edgecolor=(0, 0, 0, 1),
                                           point_labels=pt_labels,
                                           points_markersize=np.square(np.array(list(count_per_pt.values()))) * 10)

    return fig


#%% tests against fixed issues

def test_issue_7a():
    centroids = np.array([[537300, 213400], [538700, 213700], [536100, 213400]])
    n_pts = len(centroids)
    polygon = Polygon([[540000, 214100], [535500, 213700], [535500, 213000], [539000, 213200]])
    region_polys, region_pts = voronoi_regions_from_coords(centroids, polygon)

    assert isinstance(region_polys, dict)
    assert isinstance(region_pts, dict)
    assert len(region_polys) == len(region_pts) == n_pts

    assert all([len(pts_in_region) == 1 for pts_in_region in region_pts.values()])  # no duplicates


@pytest.mark.mpl_image_compare
def test_issue_7b():
    centroids = np.array([[496712, 232672], [497987, 235942], [496425, 230252], [497482, 234933],
                          [499331, 238351], [496081, 231033], [497090, 233846], [496755, 231645],
                          [498604, 237018]])
    n_pts = len(centroids)
    polygon = Polygon([[495555, 230875], [496938, 235438], [499405, 239403], [499676, 239474],
                       [499733, 237877], [498863, 237792], [499120, 237335], [498321, 235010],
                       [497295, 233185], [497237, 231359], [496696, 229620], [495982, 230047],
                       [496154, 230347], [496154, 230347], [495555, 230875]])

    region_polys, region_pts = voronoi_regions_from_coords(centroids, polygon)

    assert isinstance(region_polys, dict)
    assert isinstance(region_pts, dict)
    assert len(region_polys) == len(region_pts) == n_pts

    assert all([len(pts_in_region) == 1 for pts_in_region in region_pts.values()])  # no duplicates

    fig, ax = subplot_for_map(show_spines=True)
    plot_voronoi_polys_with_points_in_area(ax, polygon, region_polys, centroids, region_pts)

    return fig


#%% a few helper functions

def _get_country_shape(country):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    area = world[world.name == country]
    assert len(area) == 1
    area = area.to_crs(epsg=3395)  # convert to World Mercator CRS
    return area.iloc[0].geometry  # get the Polygon


def _rand_coords_in_shape(area_shape, n_points):
    np.random.seed(123)

    # generate some random points within the bounds
    minx, miny, maxx, maxy = area_shape.bounds

    randx = np.random.uniform(minx, maxx, n_points)
    randy = np.random.uniform(miny, maxy, n_points)
    coords = np.vstack((randx, randy)).T

    # use only the points inside the geographic area
    pts = [p for p in coords_to_points(coords) if p.within(area_shape)]  # converts to shapely Point
    return points_to_coords(pts)


def _check_region_polys(region_polys, region_pts, coords, expected_sum_area,
                        contains_check_tol=1, area_check_tol=0.01):
    # check validity of each region's polygon, check that all assigned points are inside this polygon and
    # check that sum of polygons' area matches `expected_sum_area`
    sum_area = 0
    for poly, pt_indices in zip(region_polys, region_pts):
        assert isinstance(poly, (Polygon, MultiPolygon)) and poly.is_valid and not poly.is_empty
        if contains_check_tol != 0:
            polybuf = poly.buffer(contains_check_tol)
            if polybuf.is_empty or not polybuf.is_valid:   # this may happen due to buffering
                polybuf = poly
        else:
            polybuf = poly
        assert all([polybuf.contains(Point(coords[i_pt])) for i_pt in pt_indices])

        sum_area += poly.area

    if expected_sum_area is not None:
        assert abs(1 - sum_area / expected_sum_area) <= area_check_tol
