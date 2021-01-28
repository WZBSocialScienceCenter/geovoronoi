"""
Artificial mini-example useful for debugging.

Author: Markus Konrad <markus.konrad@wzb.eu>
January 2021
"""


import logging

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Polygon

from geovoronoi import voronoi_regions_from_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area

logging.basicConfig(level=logging.INFO)
geovoronoi_log = logging.getLogger('geovoronoi')
geovoronoi_log.setLevel(logging.INFO)
geovoronoi_log.propagate = True

#%%

# points = np.array([[0, 0], [0, 1], [0, 2],
#                    [1, 0], [1, 1], [1, 2],
#                    [2, 0], [2, 1], [2, 2]])

points = np.array([[1, 1], [1.1, 0.9], [1.15, 0.8], [2.5, 0]])

# surrounding shape
shape = Polygon([[-1, -1], [3, -1], [3, 3], [-1, 3]])

#%% Voronoi region generation

region_polys, region_pts = voronoi_regions_from_coords(points, shape)

print('Voronoi region to point assignments:')
print(region_pts)

#%% plotting

fig, ax = subplot_for_map(show_x_axis=True, show_y_axis=True)

plot_voronoi_polys_with_points_in_area(ax, shape, region_polys, points, region_pts,
                                       point_labels=list(map(str, range(len(points)))),
                                       voronoi_labels=list(map(str, region_polys.keys())))

ax.set_title('toy example')

plt.tight_layout()
plt.show()
