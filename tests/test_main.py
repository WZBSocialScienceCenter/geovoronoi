import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import pytest
from hypothesis import given

from ._testtools import coords_2d_array
from geovoronoi import voronoi_regions_from_coords, coords_to_points, points_to_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area

np.random.seed(123)


@given(coords=coords_2d_array())
def test_coords_to_points_and_points_to_coords(coords):
    assert np.array_equal(points_to_coords(coords_to_points(coords)), coords)


@pytest.mark.mpl_image_compare
def test_voronoi_italy_with_plot():
    area_shape = _get_country_shape('Italy')
    coords = _rand_coords_in_shape(area_shape, 100)
    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, area_shape)

    assert isinstance(poly_shapes, list)
    assert 0 < len(poly_shapes) < 100
    assert all([isinstance(p, (Polygon, MultiPolygon)) for p in poly_shapes])

    assert np.array_equal(points_to_coords(pts), coords)

    assert isinstance(poly_to_pt_assignments, list)
    assert len(poly_to_pt_assignments) == len(poly_shapes)
    assert all([isinstance(assign, list) for assign in poly_to_pt_assignments])
    assert all([len(assign) == 1 for assign in poly_to_pt_assignments])   # in this case there is a 1:1 correspondance

    fig, ax = subplot_for_map()
    plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, coords, poly_to_pt_assignments)

    return fig


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
