import pytest
import numpy as np

from geovoronoi._geom import calculate_polygon_areas, line_segment_intersection


@pytest.mark.parametrize(
    'l_off, l_dir, segm_a, segm_b, expected',
    [
        ([4, 7], [12, -4], [1, 1], [17, 5], [13, 4]),
        # overlapping
        ([1, 1], [2, 1], [2, 1.5], [4, 2.5], [2, 1.5]),
        ([1, 1], [2, 1], [1, 1], [3, 2], [1, 1]),
        ([2, 1.5], [2, 1], [2, 1.5], [4, 2.5], [2, 1.5]),
        ([3, 2], [2, 1], [2, 1.5], [4, 2.5], [4, 2.5]),
        ([3, 2], [2, 1], [4, 2.5], [2, 1.5], [4, 2.5]),
        ([4, 2.5], [2, 1], [2, 1.5], [4, 2.5], [4, 2.5]),
        ([5, 3], [2, 1], [2, 1.5], [4, 2.5], None),     # "wrong" direction
        ([5, 3], [-2, -1], [2, 1.5], [4, 2.5], [4, 2.5]),
        ([-1, 0], [-2, -1], [2, 1.5], [4, 2.5], None),  # "wrong" direction
        # parallel
        ([1, 3], [2, 1], [2, 1.5], [4, 2.5], None),
        ([1, 3], [-2, -1], [2, 1.5], [4, 2.5], None),
        # some edge-cases
        ([0, 0], [0, 1], [0, 0], [0, 1], [0, 0]),
        ([0, 0], [1, 0], [0, 0], [1, 0], [0, 0]),
    ]
)
def test_line_segment_intersection(l_off, l_dir, segm_a, segm_b, expected):
    res = line_segment_intersection(np.array(l_off),
                                    np.array(l_dir),
                                    np.array(segm_a),
                                    np.array(segm_b))
    if expected is None:
        assert res is None
    else:
        assert np.array_equal(res, np.array(expected))


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
    assert all(np.isclose(a_m, a_km * 1e6) for a_m, a_km in zip(areas, areas_km2))
