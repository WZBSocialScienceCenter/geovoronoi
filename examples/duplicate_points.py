import logging

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from geovoronoi import coords_to_points, points_to_coords, voronoi_regions_from_coords, get_points_to_poly_assignments
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area


logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('geovoronoi')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True

N_POINTS = 20
N_DUPL = 10
COUNTRY = 'Sweden'

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

# duplicate a few points

rand_dupl_ind = np.random.randint(len(coords), size=N_DUPL)
coords = np.concatenate((coords, coords[rand_dupl_ind]))

print('duplicated %d random points -> we have %d coordinates now' % (N_DUPL, len(coords)))

# if we didn't know in advance how many duplicates we have (and which points they are), we could find out like this:
# unique_coords, unique_ind, dupl_counts = np.unique(coords, axis=0, return_index=True, return_counts=True)
# n_dupl = len(coords) - len(unique_ind)
# n_dupl
# >>> 10

#
# calculate the Voronoi regions, cut them with the geographic area shape and assign the points to them
#

poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, area_shape,
                                                                       accept_n_coord_duplicates=N_DUPL)

# poly_to_pt_assignments is a nested list!

print('\n\nvoronoi region to points assignments:')
for i_poly, pt_indices in enumerate(poly_to_pt_assignments):
    print('> voronoi region', i_poly, '-> points', str(pt_indices))

print('\n\nvoronoi points to region assignments:')
pts_to_poly_assignments = np.array(get_points_to_poly_assignments(poly_to_pt_assignments))
for i_pt, i_poly in enumerate(pts_to_poly_assignments):
    print('> point ', i_pt, '-> voronoi region', i_poly)


#
# plotting
#

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

ax.set_title('%d random points (incl. %d duplicates)\nand their Voronoi regions in %s' % (len(pts), N_DUPL, COUNTRY))

plt.tight_layout()
plt.savefig('duplicate_points.png')
plt.show()
