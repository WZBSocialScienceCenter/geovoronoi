"""
Functions for plotting Voronoi regions.

Most functions rely on [geopandas](http://geopandas.org/) plotting functions. So to use them, this package must be
installed.

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import numpy as np
import matplotlib.pyplot as plt
from geopandas.plotting import plot_polygon_collection

from ._voronoi import points_to_coords


def generate_n_colors(n, cmap_name='tab20'):
    pt_region_colormap = plt.get_cmap(cmap_name)
    max_i = len(pt_region_colormap.colors)
    return [pt_region_colormap(i % max_i) for i in range(n)]


def colors_for_voronoi_polys_and_points(poly_shapes, poly_to_pt_assignments, cmap_name='tab20'):
    vor_colors = generate_n_colors(len(poly_shapes), cmap_name=cmap_name)
    pt_colors = [vor_colors[poly_to_pt_assignments.index(i_pt)] for i_pt in range(max(poly_to_pt_assignments)+1)]

    return vor_colors, pt_colors


def plot_voronoi_polys(ax, poly_shapes, color=None, edgecolor=None, **kwargs):
    plot_polygon_collection(ax, poly_shapes, color=color, edgecolor=edgecolor, **kwargs)


def plot_points(ax, points, markersize, marker='o', color=None, **kwargs):
    if not isinstance(points, np.ndarray):
        coords = points_to_coords(points)
    else:
        coords = points

    ax.scatter(coords[:, 0], coords[:, 1], s=markersize, marker=marker, color=color, **kwargs)


def plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, points, poly_to_pt_assignments=None,
                                           area_color='white', area_edgecolor='black',
                                           voronoi_polys_and_points_cmap='tab20',
                                           voronoi_polys_color=None, voronoi_polys_edgecolor=None,
                                           points_color=None, points_markersize=5, points_marker='o',
                                           plot_area_opts=None,
                                           plot_voronoi_polys_opts=None,
                                           plot_points_opts=None):
    plot_area_opts = plot_area_opts or {}
    plot_voronoi_polys_opts = plot_voronoi_polys_opts or {'alpha': 0.5}
    plot_points_opts = plot_points_opts or {}

    plot_polygon_collection(ax, [area_shape], color=area_color, edgecolor=area_edgecolor, **plot_area_opts)

    if voronoi_polys_and_points_cmap and poly_to_pt_assignments and \
            not any(map(bool, (voronoi_polys_color, voronoi_polys_edgecolor, points_color))):
        voronoi_polys_color, points_color = colors_for_voronoi_polys_and_points(poly_shapes, poly_to_pt_assignments,
                                                                                cmap_name=voronoi_polys_and_points_cmap)

    plot_voronoi_polys(ax, poly_shapes, color=voronoi_polys_color, edgecolor=voronoi_polys_edgecolor,
                       **plot_voronoi_polys_opts)

    plot_points(ax, points, points_markersize, points_marker, color=points_color, **plot_points_opts)
