import logging

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from geovoronoi import coords_to_points, points_to_coords, voronoi_regions_from_coords
from geovoronoi.plotting import plot_voronoi_polys_with_points_in_area


logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('geovoronoi')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True

N_POINTS = 20

np.random.seed(123)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

area = world[world.name == 'Germany']
assert len(area) == 1


print('area CRS:', area.crs)   # gives epsg:4326 -> WGS 84

area = area.to_crs(epsg=3395)    # convert to World Mercator CRS
area_shape = area.iloc[0].geometry

minx, miny, maxx, maxy = area_shape.bounds

randx = np.random.uniform(minx, maxx, N_POINTS)
randy = np.random.uniform(miny, maxy, N_POINTS)
coords = np.vstack((randx, randy)).T
pts = [p for p in coords_to_points(coords) if p.within(area_shape)]
print('will use %d of %d randomly generated points that are inside geographic area' % (len(pts), N_POINTS))
coords = points_to_coords(pts)

del pts

poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, area_shape)

#
# plotting
#

fig, ax = plt.subplots()
ax.set_aspect('equal')

plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, coords, poly_to_pt_assignments)

plt.show()
