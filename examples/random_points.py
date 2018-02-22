import logging

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from geovoronoi import coords_to_points, points_to_coords, voronoi_regions_from_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area


logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('geovoronoi')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True

N_POINTS = 80

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

import pandas as pd
coord_hashes = np.apply_along_axis(lambda pt: hash(tuple(pt)), 1, coords)

hashes_freq = pd.Series(index=coord_hashes).groupby(level=0).size()
n_dupl = sum(hashes_freq > 1)
if n_dupl > 0:
    dupl_hashes = hashes_freq[hashes_freq > 1].index.values
    dupl_ind = np.where(coord_hashes[:, None] == dupl_hashes)[0]

    print('%d Geokoord.-Duplikate vorhanden' % n_dupl)
    #print(geodaten.iloc[dupl_ind].sort_values(['ort', 'strasse']))


poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, area_shape)

#
# plotting
#

fig, ax = subplot_for_map()

plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, coords,
                                       voronoi_labels=list(map(str, poly_to_pt_assignments)),
                                       point_labels=list(map(str, range(len(coords)))),
                                       points_markersize=3,
                                       plot_voronoi_opts={'alpha': 0.4})
#plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, coords)

plt.show()
