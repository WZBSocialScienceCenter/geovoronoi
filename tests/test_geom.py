from math import pi, isclose
from itertools import permutations

from hypothesis import given
import numpy as np

from ._testtools import real_coords_2d
from geovoronoi._geom import angle_between_pts, inner_angle_between_vecs, polygon_around_center, calculate_polygon_areas


@given(a=real_coords_2d(), b=real_coords_2d())
def test_inner_angle_between_vecs(a, b):
    a = np.array(a)
    b = np.array(b)
    origin = np.array((0, 0))

    ang = inner_angle_between_vecs(a, b)

    if np.allclose(a, origin, rtol=0) or np.allclose(b, origin, rtol=0):
        assert np.isnan(ang)
    else:
        assert 0 <= ang <= pi


@given(a=real_coords_2d(), b=real_coords_2d())
def test_angle_between_pts(a, b):
    a = np.array(a)
    b = np.array(b)

    ang = angle_between_pts(a, b)

    if np.allclose(a, b, rtol=0):
        assert np.isnan(ang)
    else:
        assert 0 <= ang <= 2*pi


def test_polygon_around_center():
    points = np.array([[0, 0], [1, 0], [1, 1], [1, 0], [1, -1]])

    for perm_ind in permutations(range(len(points))):
        # many of these permutations do not form a valid polygon in that order
        # `polygon_around_center` will make sure that the order of points is correct to create a polygon around the
        # center of these points
        poly = polygon_around_center(points[perm_ind,:])

        assert poly.is_simple and poly.is_valid


def test_polygon_around_center_given_center_is_one_of_points():
    points = np.array([[0, 0], [1, 0], [1, 1], [1, 0], [0.5, 0.5]])

    for perm_ind in permutations(range(len(points))):
        # many of these permutations do not form a valid polygon in that order
        # `polygon_around_center` will make sure that the order of points is correct to create a polygon around the
        # center of these points
        poly = polygon_around_center(points[perm_ind,:], center=(0.5, 0.5))

        assert poly.is_simple and poly.is_valid


def test_calculate_polygon_areas_empty():
    areas = calculate_polygon_areas([])
    assert len(areas) == 0


def test_calculate_polygon_areas_world():
    import geopandas as gpd

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[world.continent != 'Antarctica'].to_crs(epsg=3395)  # meters as unit!

    areas = calculate_polygon_areas(world.geometry)

    assert len(areas) == len(world)
    assert all(0 <= a < 9e13 for a in areas)

    areas_km2 = calculate_polygon_areas(world.geometry, m2_to_km2=True)
    assert len(areas_km2) == len(world)
    assert all(isclose(a_m, a_km * 1e6) for a_m, a_km in zip(areas, areas_km2))
