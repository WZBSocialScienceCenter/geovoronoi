import logging

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from geovoronoi import coords_to_points, points_to_coords, voronoi_regions_from_coords, calculate_polygon_areas
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area


logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('geovoronoi')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True

N_POINTS = 20
COUNTRY = 'Spain'

np.random.seed(123)

print('loading country `%s` from naturalearth_lowres' % COUNTRY)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
area = world[world.name == COUNTRY]
assert len(area) == 1

print('CRS:', area.crs)   # gives epsg:4326 -> WGS 84

area = area.to_crs(epsg=3395)    # convert to World Mercator CRS
area_shape = area.iloc[0].geometry   # get the Polygon

# generate some random points within the bounds
minx, miny, maxx, maxy = area_shape.bounds

randx = np.random.uniform(minx, maxx, N_POINTS)
randy = np.random.uniform(miny, maxy, N_POINTS)
coords = np.vstack((randx, randy)).T

# use only the points inside the geographic area
pts = [p for p in coords_to_points(coords) if p.within(area_shape)]  # converts to shapely Point

print('will use %d of %d randomly generated points that are inside geographic area' % (len(pts), N_POINTS))
coords = points_to_coords(pts)   # convert back to simple NumPy coordinate array

del pts

#
# calculate the Voronoi regions, cut them with the geographic area shape and assign the points to them
#

poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, area_shape, shapes_from_diff_with_min_area=1)

poly_areas = calculate_polygon_areas(poly_shapes, m2_to_km2=True)   # converts m² to km²

#
# plotting
#

fig, ax = subplot_for_map(show_x_axis=True, show_y_axis=True)

voronoi_labels = ['%d km²' % round(a) for a in poly_areas]
plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, coords, poly_to_pt_assignments,
                                       voronoi_labels=voronoi_labels, voronoi_label_fontsize=7,
                                       voronoi_label_color='gray')

ax.set_title('%d random points and their Voronoi regions in %s' % (len(pts), COUNTRY))

plt.tight_layout()
plt.savefig('random_points_and_area.png')
plt.show()

