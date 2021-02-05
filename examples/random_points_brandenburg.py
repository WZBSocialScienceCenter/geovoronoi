"""
Example script that scatters random points across Brandenburg and generates the Voronoi regions for them.
The boundary shape of Brandenburg contains a hole (Berlin) and when loaded is regarded as "invalid" shape.

Author: Markus Konrad <markus.konrad@wzb.eu>
January 2021
"""


import logging
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import fiona
import pyproj
from shapely.geometry import shape
from shapely.ops import transform

from geovoronoi import coords_to_points, voronoi_regions_from_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area

logging.basicConfig(level=logging.INFO)
geovoronoi_log = logging.getLogger('geovoronoi')
geovoronoi_log.setLevel(logging.INFO)
geovoronoi_log.propagate = True

#%%

INPUT_FILE = 'brandenburg.json'
N_POINTS = 20

np.random.seed(20210129)

print('loading %s' % INPUT_FILE)

with fiona.open(INPUT_FILE) as f:
    src_crs = pyproj.CRS.from_dict(f.meta['crs'])
    target_crs = pyproj.CRS.from_epsg(3395)         # World Mercator CRS
    print('source CRS:', src_crs)
    print('target CRS:', target_crs)
    crs_transformer = pyproj.Transformer.from_crs(src_crs, target_crs)
    brandenburg = shape(f[0]['geometry'])
    # note that we also apply ".buffer(0)", otherwise the shape is not valid
    # see https://shapely.readthedocs.io/en/stable/manual.html#object.buffer on the "buffer(0)" trick
    brandenburg = transform(crs_transformer.transform, brandenburg).buffer(0)

#%%

# generate some random points within the bounds
minx, miny, maxx, maxy = brandenburg.bounds

randx = np.random.uniform(minx, maxx, N_POINTS)
randy = np.random.uniform(miny, maxy, N_POINTS)
coords = np.vstack((randx, randy)).T

# use only the points inside the geographic area
pts = [p for p in coords_to_points(coords) if p.within(brandenburg)]  # converts to shapely Point
del coords   # not used any more

print('will use %d of %d randomly generated points that are inside geographic area' % (len(pts), N_POINTS))

#%%

#
# calculate the Voronoi regions, cut them with the geographic area shape and assign the points to them
#

region_polys, region_pts = voronoi_regions_from_coords(pts, brandenburg)

print('Voronoi region to point assignments:')
pprint(region_pts)

#%% plotting

fig, ax = subplot_for_map()

plot_voronoi_polys_with_points_in_area(ax, brandenburg, region_polys, pts, region_pts,
                                       point_labels=list(map(str, range(len(pts)))))

ax.set_title('%d random points and their Voronoi regions in Brandenburg' % len(pts))

plt.tight_layout()
plt.savefig('random_points_brandenburg.png')
plt.show()
