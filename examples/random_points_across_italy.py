"""
Example script that scatters random points across a country and generates the Voronoi regions for them. Both the regions
and their points will be plotted using the `plotting` sub-module of `geovoronoi`. This example will show the effect
of calculating the Voronoi regions separately per country sub-geometry (e.g. separately for islands) vs. calculating
the Voronoi regions for the whole country shape together.

Author: Markus Konrad <markus.konrad@wzb.eu>
January 2021
"""


import logging
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from geovoronoi import coords_to_points, voronoi_regions_from_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area

logging.basicConfig(level=logging.INFO)
geovoronoi_log = logging.getLogger('geovoronoi')
geovoronoi_log.setLevel(logging.INFO)
geovoronoi_log.propagate = True

#%%

N_POINTS = 105
COUNTRY = 'Italy'

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
del coords   # not used any more

print('will use %d of %d randomly generated points that are inside geographic area' % (len(pts), N_POINTS))

#%%

#
# Calculate the Voronoi regions, cut them with the geographic area shape and assign the points to them.
# Note how in Sardinia there's only one point which is not assigned to any Voronoi region.
# Since by default all sub-geometries (i.e. islands or other isolated features) in the geographic shape
# are treated separately (set `per_geom=False` to change this) and you need more than one point to generate
# a Voronoi region, this point is left unassigned. By setting `return_unassigned_points=True`, we can get a
# set of unassigned point indices:
#

region_polys, region_pts, unassigned_pts = voronoi_regions_from_coords(pts, area_shape,
                                                                       return_unassigned_points=True,
                                                                       per_geom=True)  # this is the default

print('Voronoi region to point assignments:')
pprint(region_pts)

print('Unassigned points:')
for i_pt in unassigned_pts:
    print('#%d: %.2f, %.2f' % (i_pt, pts[i_pt].x, pts[i_pt].y))

#%% plotting

fig, ax = subplot_for_map()

plot_voronoi_polys_with_points_in_area(ax, area_shape, region_polys, pts, region_pts,
                                       point_labels=list(map(str, range(len(pts)))))

ax.set_title('%d random points and their Voronoi regions in %s' % (len(pts), COUNTRY))

plt.tight_layout()
plt.savefig('random_points_across_italy.png')
plt.show()


#%%

# Now we change `per_geom` to False. Since all geometries of the country are treated as one during Voronoi region
# generation, also the single point on Sardinia gets a Voronoi region. Note however, that these regions now can
# span over all geometries, e.g. regions from an island can span over to the mainland or another island and vice versa.
# You can see this effect for point #39 as its region spans from Sicilia to the "tiptoe" of Italy's mainland.

region_polys2, region_pts2 = voronoi_regions_from_coords(pts, area_shape, per_geom=False)

print('Voronoi region to point assignments:')
pprint(region_pts)


#%% plotting

fig, ax = subplot_for_map()

plot_voronoi_polys_with_points_in_area(ax, area_shape, region_polys2, pts, region_pts2,
                                       point_labels=list(map(str, range(len(pts)))))

ax.set_title('%d random points and their Voronoi regions in %s (with per_geom=False)' % (len(pts), COUNTRY))

plt.tight_layout()
plt.savefig('random_points_across_italy_per_geom_false.png')
plt.show()
