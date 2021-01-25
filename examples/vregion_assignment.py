from collections import defaultdict
from matplotlib import pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import LineString, LinearRing, asPoint, MultiPoint, Polygon
from shapely.ops import polygonize

import numpy as np

from geovoronoi._geom import polygon_around_center
from geovoronoi.plotting import subplot_for_map, plot_points, plot_lines, plot_polygon

#%%

geo_shape = Polygon([[-1, -1], [3, -1], [3, 3], [-1, 3], [-1, -1]])

points = np.array([[0, 0], [1, 0], [2, 0],
                   [0, 1], [1, 1], [2, 1],
                   [0, 2], [1, 2], [2, 2]])

fig, ax = subplot_for_map(show_x_axis=True, show_y_axis=True)

plot_polygon(ax, geo_shape, facecolor=(0, 0, 0, 0.0), edgecolor='r')
plot_points(ax, points, color='r', markersize=1, labels=list(map(str, range(len(points)))), label_color='r')

fig.show()


#%%

vor = Voronoi(points)

vorfig = voronoi_plot_2d(vor)
vorfig.show()
del vorfig

vor.vertices
vor.regions
vor.point_region
vor.ridge_vertices
vor.ridge_points
vor.ridge_dict

ridge_vert = np.array(vor.ridge_vertices)

#%%

farpoints_max_extend_factor = 1
max_dim_extend = vor.points.ptp(axis=0).max() * farpoints_max_extend_factor
center = np.array(MultiPoint(vor.points).convex_hull.centroid)

region_pts = defaultdict(list)
region_polys = {}
for i_reg, reg_vert in enumerate(vor.regions):
    pt_indices = np.where(vor.point_region == i_reg)[0]
    if len(pt_indices) == 0:   # skip regions w/o points in them
        print(i_reg, 'empty')
        continue

    print(i_reg)
    print('> vertice indices: ', reg_vert)
    print('> point indices: ', pt_indices)

    region_pts[i_reg].extend(pt_indices)

    if np.all(np.array(reg_vert) >= 0):  # fully finite-bounded region
        print('> finite')
        p = Polygon(vor.vertices[reg_vert])
    else:
        print('> not finite')

        p_vertices = []
        i_pt = pt_indices[0]     # only consider one point, not a duplicate
        print('>> point', i_pt)
        enclosing_ridge_pts_mask = (vor.ridge_points[:, 0] == i_pt) | (vor.ridge_points[:, 1] == i_pt)
        for pointidx, simplex in zip(vor.ridge_points[enclosing_ridge_pts_mask],
                                     ridge_vert[enclosing_ridge_pts_mask]):
            print('>> pointidx', pointidx)
            print('>> simplex indices', simplex)

            if np.all(simplex >= 0):
                p_vertices.extend(vor.vertices[simplex])
            else:
                # "loose ridge": contains infinite Voronoi vertex
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
                finite_pt = vor.vertices[i]

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                direction = direction / np.linalg.norm(direction)  # to unit vector

                far_point = finite_pt + direction * max_dim_extend * farpoints_max_extend_factor

                print(f'>> finite point {i} is {finite_pt}')
                print(f'>> far point is {far_point}')

                p_vertices.extend([finite_pt, far_point])

        print(f'> polygon vertices: {p_vertices}')
        p = MultiPoint(p_vertices).convex_hull    # Voronoi regions are convex

    assert p.is_valid and p.is_simple, 'generated polygon is valid and simple'
    region_polys[i_reg] = p


#%%

list(region_polys[1].exterior.coords)

#%%

fig, ax = subplot_for_map(show_x_axis=True, show_y_axis=True)

plot_polygon(ax, geo_shape, facecolor=(0, 0, 0, 0.0), edgecolor='r')
plot_points(ax, points, color='r', markersize=1, labels=list(map(str, range(len(points)))), label_color='r')

for i_p, p in region_polys.items():
    plot_polygon(ax, p, edgecolor='k', facecolor=(0, 0, 0, 0.1), linestyle='dashed', label=str(i_p))


fig.show()

region_pts

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
    print(f'pointidx: {pointidx}, simplex: {simplex}')
    if np.all(simplex >= 0):  # full finite polygon line
        print('> finite')
        poly_lines.append(LineString(vor.vertices[simplex]))
    else:  # "loose ridge": contains infinite Voronoi vertex
        print('> loose')
        i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
        t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]])  # normal

        midpoint = vor.points[pointidx].mean(axis=0)
        direction = np.sign(np.dot(midpoint - center, n)) * n
        direction = direction / np.linalg.norm(direction)  # to unit vector

        far_point = vor.vertices[i] + direction * max_dim_extend * farpoints_max_extend_factor
        print(f'> finite vertex {i}: {vor.vertices[i]}')
        print(f'> far point: {far_point}')
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

for p in poly_shapes:
    plot_polygon(ax, p, edgecolor='k', facecolor=(0, 0, 0, 0.1), linestyle='dashed')

fig.show()