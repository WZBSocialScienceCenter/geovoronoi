from collections import defaultdict
from matplotlib import pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import LineString, asPoint, MultiPoint, Polygon
from shapely.ops import polygonize

import numpy as np

from examples.figures import SIZE, set_limits, plot_coords, plot_bounds, plot_line
from geovoronoi._geom import polygon_around_center

#%%

geo_shape = Polygon([[-1, -1], [3, -1], [3, 3], [-1, 3], [-1, -1]])

fig, ax = plt.subplots()
ax.set_aspect('equal')

plot_coords(ax, geo_shape.exterior)
plot_line(ax, geo_shape.exterior)

points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                   [2, 0], [2, 1], [2, 2]])

plot_coords(ax, LineString(points), color='r')

fig.show()

#%%

vor = Voronoi(points)

fig = voronoi_plot_2d(vor)
plt.show()

vor.vertices
vor.regions
vor.point_region
vor.ridge_vertices
vor.ridge_points
vor.ridge_dict

ridge_vert = np.array(vor.ridge_vertices)

#%%

farpoints_max_extend_factor = 10
max_dim_extend = vor.points.ptp(axis=0).max() * farpoints_max_extend_factor
center = np.array(MultiPoint(vor.points).convex_hull.centroid)

region_pts = defaultdict(list)
region_polys = {}
for i_reg, reg_vert in enumerate(vor.regions):
    pt_indices = np.where(vor.point_region == i_reg)[0]
    if len(pt_indices) == 0:   # skip regions w/o points in them
        continue

    region_pts[i_reg].extend(pt_indices)
    if np.all(np.array(reg_vert) >= 0):  # fully finite-bounded region
        print('finite')
        print(vor.vertices[reg_vert])
        p = Polygon(vor.vertices[reg_vert])
        region_polys[i_reg] = p
    else:
        print('not finite')
        print(reg_vert)

        for i_pt in pt_indices:
            enclosing_ridge_pts_mask = (vor.ridge_points[:, 0] == i_pt) | (vor.ridge_points[:, 1] == i_pt)
            for pointidx, simplex in zip(vor.ridge_points[enclosing_ridge_pts_mask],
                                         ridge_vert[enclosing_ridge_pts_mask]):
                assert np.any(simplex < 0)
                assert np.any(simplex >= 0)
                # "loose ridge": contains infinite Voronoi vertex
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                direction = direction / np.linalg.norm(direction)  # to unit vector

                far_point = vor.vertices[i] + direction * max_dim_extend * farpoints_max_extend_factor


#%%

farpoints_max_extend_factor = 10
max_dim_extend = vor.points.ptp(axis=0).max() * farpoints_max_extend_factor
center = np.array(MultiPoint(vor.points).convex_hull.centroid)

# generate lists of full polygon lines, loose ridges and far points of loose ridges from scipy Voronoi result object
poly_lines = []
loose_ridges = []
far_points = []
for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):  # full finite polygon line
        print('full finite:')
        print(pointidx)
        poly_lines.append(LineString(vor.vertices[simplex]))
    else:  # "loose ridge": contains infinite Voronoi vertex
        i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

        t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]])  # normal

        midpoint = vor.points[pointidx].mean(axis=0)
        direction = np.sign(np.dot(midpoint - center, n)) * n
        direction = direction / np.linalg.norm(direction)  # to unit vector

        far_point = vor.vertices[i] + direction * max_dim_extend * farpoints_max_extend_factor

        loose_ridges.append(LineString(np.vstack((vor.vertices[i], far_point))))
        far_points.append(far_point)

#
# confine the infinite Voronoi regions by constructing a "hull" from loose ridge far points around the centroid of
# the geographic area
#

# first append the loose ridges themselves
for l in loose_ridges:
    poly_lines.append(l)

# now create the "hull" of far points: `far_points_hull`
far_points = np.array(far_points)
far_points_hull = polygon_around_center(far_points, center)

if far_points_hull is None:
    raise RuntimeError('no polygonal hull of far points could be created')

# sometimes, this hull does not completely encompass the geographic area `geo_shape`
if not far_points_hull.contains(geo_shape):  # if that's the case, merge it by taking the union
    far_points_hull = far_points_hull.union(geo_shape)

if not isinstance(far_points_hull, Polygon):
    raise RuntimeError('hull of far points is not Polygon as it should be; try increasing '
                       '`farpoints_max_extend_factor`')

# now add the lines that make up `far_points_hull` to the final `poly_lines` list
far_points_hull_coords = far_points_hull.exterior.coords
for i, pt in list(enumerate(far_points_hull_coords))[1:]:
    poly_lines.append(LineString((far_points_hull_coords[i - 1], pt)))
poly_lines.append(LineString((far_points_hull_coords[-1], far_points_hull_coords[0])))

#%%

poly_shapes = []
for p in polygonize(poly_lines):
    if geo_shape is not None and not geo_shape.contains(p):  # if `geo_shape` contains polygon `p`,
        p = p.intersection(geo_shape)  # intersect it with `geo_shape` (i.e. "cut" it)

    if not p.is_empty:
        poly_shapes.append(p)

for p in poly_shapes:
    print(list(p.exterior.coords))

len(poly_shapes)

#%%

