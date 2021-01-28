"""
Example script to show how to handle duplicate points for which Voronoi regions should be generated.

Duplicate points, i.e. points with exactly the same coordinates will belong to the same Voronoi region.

Author: Markus Konrad <markus.konrad@wzb.eu>
January 2021
"""


import logging

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from geovoronoi import coords_to_points, points_to_coords, voronoi_regions_from_coords, get_points_to_poly_assignments
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area


logging.basicConfig(level=logging.INFO)
geovoronoi_log = logging.getLogger('geovoronoi')
geovoronoi_log.setLevel(logging.INFO)
geovoronoi_log.propagate = True


#%%

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


#%%

#
# calculate the Voronoi regions, cut them with the geographic area shape and assign the points to them
#
# the duplicate coordinates will belong to the same voronoi region
#

poly_shapes, poly_to_pt_assignments = voronoi_regions_from_coords(coords, area_shape)

# poly_to_pt_assignments is a nested list because a voronoi region might contain several (duplicate) points

print('\n\nvoronoi region to points assignments:')
for i_poly, pt_indices in poly_to_pt_assignments.items():
    print('> voronoi region', i_poly, '-> points', str(pt_indices))

print('\n\npoints to voronoi region assignments:')
pts_to_poly_assignments = get_points_to_poly_assignments(poly_to_pt_assignments)
for i_pt, i_poly in pts_to_poly_assignments.items():
    print('> point ', i_pt, '-> voronoi region', i_poly)


#%%

#
# plotting
#

# make point labels: counts of duplicate assignments per points
count_per_pt = {pt_indices[0]: len(pt_indices) for pt_indices in poly_to_pt_assignments.values()}
pt_labels = list(map(str, count_per_pt.values()))
distinct_pt_coords = coords[np.asarray(list(count_per_pt.keys()))]

# highlight voronoi regions with point duplicates
vor_colors = {i_poly: (1,0,0) if len(pt_indices) > 1 else (0,0,1)
              for i_poly, pt_indices in poly_to_pt_assignments.items()}

fig, ax = subplot_for_map()

plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, distinct_pt_coords,
                                       plot_voronoi_opts={'alpha': 0.2},
                                       plot_points_opts={'alpha': 0.4},
                                       voronoi_color=vor_colors,
                                       voronoi_edgecolor=(0,0,0,1),
                                       point_labels=pt_labels,
                                       points_markersize=np.square(np.array(list(count_per_pt.values())))*10)

ax.set_title('%d random points (incl. %d duplicates)\nand their Voronoi regions in %s' % (len(coords), N_DUPL, COUNTRY))

plt.tight_layout()
plt.savefig('duplicate_points.png')
plt.show()
