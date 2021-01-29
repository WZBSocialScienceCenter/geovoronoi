import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union
import pytest
from hypothesis import given
import hypothesis.strategies as st

from ._testtools import coords_2d_array
from geovoronoi import (
    voronoi_regions_from_coords, coords_to_points, points_to_coords, calculate_polygon_areas,
    get_points_to_poly_assignments
)
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area

np.random.seed(123)


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
            get_points_to_poly_assignments(poly_to_pts)
    else:
        assert get_points_to_poly_assignments(poly_to_pts) == expected


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

    pts_to_poly = get_points_to_poly_assignments(poly_to_pts)

    assert isinstance(pts_to_poly, dict)

    if n_poly == 0:
        assert len(pts_to_poly) == 0
    else:
        assert len(pts_to_poly) == n_pts
        assert set(list(pts_to_poly.keys())) == set(available_points)
        assert set(list(pts_to_poly.values())) == set(list(range(n_poly)))


@pytest.mark.parametrize(
    'n_pts, per_geom, return_unassigned_pts',
    [
        (100, True, False),
    ]
)
@pytest.mark.mpl_image_compare
def test_voronoi_regions_from_coords_italy(n_pts, per_geom, return_unassigned_pts):
    area_shape = _get_country_shape('Italy')
    coords = _rand_coords_in_shape(area_shape, n_pts)
    n_pts = len(coords)   # number of random points inside shape
    poly_shapes, poly_to_pt_assignments = voronoi_regions_from_coords(coords, area_shape,
                                                                      per_geom=per_geom,
                                                                      return_unassigned_points=return_unassigned_pts)

    assert isinstance(poly_shapes, dict)
    assert len(poly_shapes) == n_pts
    assert all([isinstance(p, (Polygon, MultiPolygon)) for p in poly_shapes])

    assert np.array_equal(points_to_coords(pts), coords)

    assert isinstance(poly_to_pt_assignments, dict)
    assert len(poly_to_pt_assignments) == len(poly_shapes)
    assert all([isinstance(assign, list) for assign in poly_to_pt_assignments])
    assert all([len(assign) == 1 for assign in poly_to_pt_assignments])   # in this case there is a 1:1 correspondance

    fig, ax = subplot_for_map()
    plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, coords, poly_to_pt_assignments)

    return fig



#%% realistic full tests with plotting


@pytest.mark.parametrize(
    'n_pts', [5, 10, 20, 50, 100, 105, 1000]
)
@pytest.mark.mpl_image_compare
def test_voronoi_italy_with_plot(n_pts):
    area_shape = _get_country_shape('Italy')
    coords = _rand_coords_in_shape(area_shape, n_pts)
    poly_shapes, poly_to_pt_assignments = voronoi_regions_from_coords(coords, area_shape)

    assert isinstance(poly_shapes, list)
    assert 0 < len(poly_shapes) <= 100
    assert all([isinstance(p, (Polygon, MultiPolygon)) for p in poly_shapes])

    assert np.array_equal(points_to_coords(pts), coords)

    assert isinstance(poly_to_pt_assignments, list)
    assert len(poly_to_pt_assignments) == len(poly_shapes)
    assert all([isinstance(assign, list) for assign in poly_to_pt_assignments])
    assert all([len(assign) == 1 for assign in poly_to_pt_assignments])   # in this case there is a 1:1 correspondance

    fig, ax = subplot_for_map()
    plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, coords, poly_to_pt_assignments)

    return fig


@pytest.mark.mpl_image_compare
def test_voronoi_spain_area_with_plot():
    area_shape = _get_country_shape('Spain')
    coords = _rand_coords_in_shape(area_shape, 20)
    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, area_shape)

    assert isinstance(poly_shapes, list)
    assert 0 < len(poly_shapes) <= 20
    assert all([isinstance(p, (Polygon, MultiPolygon)) for p in poly_shapes])

    assert np.array_equal(points_to_coords(pts), coords)

    assert isinstance(poly_to_pt_assignments, list)
    assert len(poly_to_pt_assignments) == len(poly_shapes)
    assert all([isinstance(assign, list) for assign in poly_to_pt_assignments])
    assert all([len(assign) == 1 for assign in poly_to_pt_assignments])   # in this case there is a 1:1 correspondance

    poly_areas = calculate_polygon_areas(poly_shapes, m2_to_km2=True)  # converts m² to km²
    assert isinstance(poly_areas, np.ndarray)
    assert np.issubdtype(poly_areas.dtype, np.float_)
    assert len(poly_areas) == len(poly_shapes)
    assert np.all(poly_areas > 0)

    fig, ax = subplot_for_map(show_x_axis=True, show_y_axis=True)

    voronoi_labels = ['%d km²' % round(a) for a in poly_areas]
    plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, coords, poly_to_pt_assignments,
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
    south_am_shape = cascaded_union(south_am.geometry)
    south_am_cities = cities[cities.geometry.within(south_am_shape)]  # reduce to cities in South America

    # convert the pandas Series of Point objects to NumPy array of coordinates
    coords = points_to_coords(south_am_cities.geometry)

    # calculate the regions
    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, south_am_shape)

    assert isinstance(poly_shapes, list)
    assert 0 < len(poly_shapes) <= len(coords)
    assert all([isinstance(p, (Polygon, MultiPolygon)) for p in poly_shapes])

    assert np.array_equal(points_to_coords(pts), coords)

    assert isinstance(poly_to_pt_assignments, list)
    assert len(poly_to_pt_assignments) == len(poly_shapes)
    assert all([isinstance(assign, list) for assign in poly_to_pt_assignments])
    assert all([len(assign) == 1 for assign in poly_to_pt_assignments])   # in this case there is a 1:1 correspondance

    fig, ax = subplot_for_map()

    plot_voronoi_polys_with_points_in_area(ax, south_am_shape, poly_shapes, pts, poly_to_pt_assignments)

    return fig


@pytest.mark.mpl_image_compare
def test_voronoi_sweden_duplicate_points_with_plot():
    area_shape = _get_country_shape('Sweden')
    coords = _rand_coords_in_shape(area_shape, 20)

    # duplicate a few points
    rand_dupl_ind = np.random.randint(len(coords), size=10)
    coords = np.concatenate((coords, coords[rand_dupl_ind]))

    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, area_shape,
                                                                           accept_n_coord_duplicates=10)

    assert isinstance(poly_shapes, list)
    assert 0 < len(poly_shapes) <= 20
    assert all([isinstance(p, (Polygon, MultiPolygon)) for p in poly_shapes])

    assert np.array_equal(points_to_coords(pts), coords)

    assert isinstance(poly_to_pt_assignments, list)
    assert len(poly_to_pt_assignments) == len(poly_shapes)
    assert all([isinstance(assign, list) for assign in poly_to_pt_assignments])
    assert all([0 < len(assign) <= 10 for assign in poly_to_pt_assignments])   # in this case there is not
                                                                               # everywhere a 1:1 correspondance

    pts_to_poly_assignments = np.array(get_points_to_poly_assignments(poly_to_pt_assignments))

    # make point labels: counts of duplicates per points
    count_per_pt = [sum(pts_to_poly_assignments == i_poly) for i_poly in pts_to_poly_assignments]
    pt_labels = list(map(str, count_per_pt))

    # highlight voronoi regions with point duplicates
    count_per_poly = np.array(list(map(len, poly_to_pt_assignments)))
    vor_colors = np.repeat('blue', len(poly_shapes))   # default color
    vor_colors[count_per_poly > 1] = 'red'             # hightlight color

    fig, ax = subplot_for_map()

    plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, coords,
                                           plot_voronoi_opts={'alpha': 0.2},
                                           plot_points_opts={'alpha': 0.4},
                                           voronoi_color=list(vor_colors),
                                           point_labels=pt_labels,
                                           points_markersize=np.array(count_per_pt)*10)

    return fig


#%% tests against fixed issues

def test_issue_7a():
    centroids = np.array([[537300, 213400], [538700, 213700], [536100, 213400]])
    polygon = Polygon([[540000, 214100], [535500, 213700], [535500, 213000], [539000, 213200]])
    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(centroids, polygon)

    assert isinstance(poly_shapes, list)
    assert 0 < len(poly_shapes) <= len(centroids)
    assert all([isinstance(p, (Polygon, MultiPolygon)) for p in poly_shapes])

    assert np.array_equal(points_to_coords(pts), centroids)

    assert isinstance(poly_to_pt_assignments, list)
    assert len(poly_to_pt_assignments) == len(poly_shapes)
    assert all([isinstance(assign, list) for assign in poly_to_pt_assignments])
    assert all([len(assign) == 1 for assign in poly_to_pt_assignments])   # in this case there is a 1:1 correspondance


@pytest.mark.mpl_image_compare
def test_issue_7b():
    centroids = np.array([[496712, 232672], [497987, 235942], [496425, 230252], [497482, 234933],
                          [499331, 238351], [496081, 231033], [497090, 233846], [496755, 231645],
                          [498604, 237018]])
    polygon = Polygon([[495555, 230875], [496938, 235438], [499405, 239403], [499676, 239474],
                       [499733, 237877], [498863, 237792], [499120, 237335], [498321, 235010],
                       [497295, 233185], [497237, 231359], [496696, 229620], [495982, 230047],
                       [496154, 230347], [496154, 230347], [495555, 230875]])

    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(centroids, polygon)

    assert isinstance(poly_shapes, list)
    assert 0 < len(poly_shapes) <= len(centroids)
    assert all([isinstance(p, (Polygon, MultiPolygon)) for p in poly_shapes])

    assert np.array_equal(points_to_coords(pts), centroids)

    assert isinstance(poly_to_pt_assignments, list)
    assert len(poly_to_pt_assignments) == len(poly_shapes)
    assert all([isinstance(assign, list) for assign in poly_to_pt_assignments])
    assert all([len(assign) == 1 for assign in poly_to_pt_assignments])   # in this case there is a 1:1 correspondance

    fig, ax = subplot_for_map()
    plot_voronoi_polys_with_points_in_area(ax, polygon, poly_shapes, centroids, poly_to_pt_assignments)

    return fig


#%% a few helper functions

def _get_country_shape(country):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    area = world[world.name == country]
    assert len(area) == 1
    area = area.to_crs(epsg=3395)  # convert to World Mercator CRS
    return area.iloc[0].geometry  # get the Polygon


def _rand_coords_in_shape(area_shape, n_points):
    # generate some random points within the bounds
    minx, miny, maxx, maxy = area_shape.bounds

    randx = np.random.uniform(minx, maxx, n_points)
    randy = np.random.uniform(miny, maxy, n_points)
    coords = np.vstack((randx, randy)).T

    # use only the points inside the geographic area
    pts = [p for p in coords_to_points(coords) if p.within(area_shape)]  # converts to shapely Point
    return points_to_coords(pts)
