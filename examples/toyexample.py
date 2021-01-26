import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Polygon

from geovoronoi import voronoi_regions_from_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area

#%%

# points = np.array([[0, 0], [0, 1], [0, 2],
#                    [1, 0], [1, 1], [1, 2],
#                    [2, 0], [2, 1], [2, 2]])

points = np.array([[1, 1], [2, 0], [0, 2], [0, 0]])

shape = Polygon([[-1, -1], [3, -1], [3, 3], [-1, 3]])

#%%

region_polys, region_pts = voronoi_regions_from_coords(points, shape)

print('Voronoi region to point assignments:')
print(region_pts)

#%%

fig, ax = subplot_for_map()

plot_voronoi_polys_with_points_in_area(ax, shape, region_polys, points, region_pts,
                                       point_labels=list(map(str, range(len(points)))),
                                       voronoi_labels=list(map(str, region_polys.keys())))

ax.set_title('toy example')

plt.tight_layout()
plt.show()
