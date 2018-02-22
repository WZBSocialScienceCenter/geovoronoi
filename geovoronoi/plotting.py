"""
Functions for plotting Voronoi regions.

Most functions rely on [geopandas](http://geopandas.org/) plotting functions. So to use them, this package must be
installed.

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import numpy as np
import matplotlib.pyplot as plt
from geopandas.plotting import _flatten_multi_geoms

from ._voronoi import points_to_coords


def subplot_for_map(show_x_axis=False, show_y_axis=False, aspect='equal', **kwargs):
    fig, ax = plt.subplots(**kwargs)
    ax.set_aspect(aspect)

    ax.get_xaxis().set_visible(show_x_axis)
    ax.get_yaxis().set_visible(show_y_axis)

    if show_x_axis:
        fig.autofmt_xdate()

    return fig, ax


def generate_n_colors(n, cmap_name='tab20'):
    pt_region_colormap = plt.get_cmap(cmap_name)
    max_i = len(pt_region_colormap.colors)
    return [pt_region_colormap(i % max_i) for i in range(n)]


def colors_for_voronoi_polys_and_points(poly_shapes, poly_to_pt_assignments, cmap_name='tab20'):
    vor_colors = generate_n_colors(len(poly_shapes), cmap_name=cmap_name)
    pt_colors = [vor_colors[poly_to_pt_assignments.index(i_pt)] for i_pt in range(max(poly_to_pt_assignments)+1)]

    assert len(vor_colors) == len(pt_colors)

    return vor_colors, pt_colors


def plot_voronoi_polys(ax, poly_shapes, color=None, edgecolor=None, labels=None, label_fontsize=10, label_color=None,
                       **kwargs):
    _plot_polygon_collection_with_color(ax, poly_shapes, color=color, edgecolor=edgecolor, **kwargs)

    if labels:
        n_labels = len(labels)
        n_features = len(poly_shapes)
        if n_labels != n_features:
            raise ValueError('number of labels (%d) must match number of Voronoi polygons (%d)'
                             % (n_labels, n_features))

        for i, (p, lbl) in enumerate(zip(poly_shapes, labels)):
            tx, ty = p.centroid.coords[0]
            ax.text(tx, ty, lbl, fontsize=label_fontsize, color=_color_for_labels(label_color, color, i))


def plot_points(ax, points, markersize, marker='o', color=None, labels=None, label_fontsize=7, label_color=None,
                **kwargs):
    if not isinstance(points, np.ndarray):
        coords = points_to_coords(points)
    else:
        coords = points

    ax.scatter(coords[:, 0], coords[:, 1], s=markersize, marker=marker, color=color, **kwargs)

    if labels:
        n_labels = len(labels)
        n_features = len(coords)
        if n_labels != n_features:
            raise ValueError('number of labels (%d) must match number of points (%d)'
                             % (n_labels, n_features))

        for i, ((tx, ty), lbl) in enumerate(zip(coords, labels)):
            ax.text(tx, ty, lbl, fontsize=label_fontsize, color=_color_for_labels(label_color, color, i))


def plot_voronoi_polys_with_points_in_area(ax, area_shape, poly_shapes, points, poly_to_pt_assignments=None,
                                           area_color='white', area_edgecolor='black',
                                           voronoi_and_points_cmap='tab20',
                                           voronoi_color=None, voronoi_edgecolor=None,
                                           points_color=None, points_markersize=5, points_marker='o',
                                           voronoi_labels=None, voronoi_label_fontsize=10, voronoi_label_color=None,
                                           point_labels=None, point_label_fontsize=7, point_label_color=None,
                                           plot_area_opts=None,
                                           plot_voronoi_opts=None,
                                           plot_points_opts=None):
    plot_area_opts = plot_area_opts or {}
    plot_voronoi_opts = plot_voronoi_opts or {'alpha': 0.5}
    plot_points_opts = plot_points_opts or {}

    _plot_polygon_collection_with_color(ax, [area_shape], color=area_color, edgecolor=area_edgecolor, **plot_area_opts)

    if voronoi_and_points_cmap and poly_to_pt_assignments and \
            not all(map(bool, (voronoi_color, voronoi_edgecolor, points_color))):
        voronoi_color, points_color = colors_for_voronoi_polys_and_points(poly_shapes, poly_to_pt_assignments,
                                                                          cmap_name=voronoi_and_points_cmap)

    if voronoi_color is None and voronoi_edgecolor is None:
        voronoi_edgecolor = 'black'   # better visible default value

    plot_voronoi_polys(ax, poly_shapes, color=voronoi_color, edgecolor=voronoi_edgecolor,
                       labels=voronoi_labels, label_fontsize=voronoi_label_fontsize, label_color=voronoi_label_color,
                       **plot_voronoi_opts)

    plot_points(ax, points, points_markersize, points_marker, color=points_color,
                labels=point_labels, label_fontsize=point_label_fontsize, label_color=point_label_color,
                **plot_points_opts)


def _color_for_labels(label_color, default_color, seq_index):
    if label_color is None:
        if hasattr(default_color, '__getitem__'):
            c = default_color[seq_index]
        else:
            c = default_color
    else:
        c = label_color

    return c or 'black'


def _plot_polygon_collection_with_color(ax, geoms, color=None, **kwargs):
    """
    This is a hacked version of geopanda's `plot_polygon_collection` function that also accepts a sequences of colors
    passed as `color` for each polygon in `geoms` and *uses them correctly even when `geoms` contains MultiPolygon
    objects*.

    Plots a collection of Polygon and MultiPolygon geometries to `ax`

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        where shapes will be plotted

    geoms : a sequence of `N` Polygons and/or MultiPolygons (can be mixed)

    color : a sequence of `N` colors (optional) or None for default color or a single color for all shapes

    edgecolor : single color or sequence of `N` colors
        Color for the edge of the polygons

    **kwargs
        Additional keyword arguments passed to the collection

    Returns
    -------

    collection : matplotlib.collections.Collection that was plotted
    """
    from descartes.patch import PolygonPatch
    from matplotlib.collections import PatchCollection

    if type(color) in (list, tuple):
        geoms, color = _flatten_multi_geoms(geoms, color)
    else:
        geoms, _ = _flatten_multi_geoms(geoms)

    # PatchCollection does not accept some kwargs.
    if 'markersize' in kwargs:
        del kwargs['markersize']

    collection = PatchCollection([PolygonPatch(poly) for poly, c in zip(geoms, color)],
                                 color=color, **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection
