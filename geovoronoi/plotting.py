"""
Functions for plotting Voronoi regions.

Author: Markus Konrad <markus.konrad@wzb.eu>
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes.patch import PolygonPatch
from geopandas.plotting import _flatten_multi_geoms
from geopandas import GeoSeries

from ._voronoi import points_to_coords, points_to_region


def subplot_for_map(show_x_axis=False, show_y_axis=False, show_spines=None, aspect='equal', **kwargs):
    """
    Helper function to generate a matplotlib subplot Axes object suitable for plotting geographic data, i.e. axis
    labels are not shown and aspect ratio is set to 'equal' by default.

    :param show_x_axis: show x axis labels
    :param show_y_axis: show y axis labels
    :param show_spines: controls display of frame around plot; if set to None, this is "auto" mode, meaning
                        that the frame is removed when `show_x_axis` and `show_y_axis` are both set to False;
                        if set to True/False, the frame is always shown/removed
    :param aspect: aspect ratio
    :param kwargs: additional parameters passed to `plt.subplots()`
    :return: tuple with (matplotlib Figure, matplotlib Axes)
    """
    fig, ax = plt.subplots(**kwargs)
    ax.set_aspect(aspect)

    ax.get_xaxis().set_visible(show_x_axis)
    ax.get_yaxis().set_visible(show_y_axis)

    if show_spines is None:
        show_spines = show_x_axis or show_y_axis

    for sp in ax.spines.values():
        sp.set_visible(show_spines)

    if show_x_axis:
        fig.autofmt_xdate()

    return fig, ax


def generate_n_colors(n, cmap_name='tab20'):
    """
    Get a list of `n` numbers from matplotlib color map `cmap_name`. If `n` is larger than the number of colors in the
    color map, the colors will be recycled, i.e. they are not unique in this case.

    :param n: number of colors to generate
    :param cmap_name: matplotlib color map name
    :return: list of `n` colors
    """
    pt_region_colormap = plt.get_cmap(cmap_name)
    max_i = len(pt_region_colormap.colors)
    return [pt_region_colormap(i % max_i) for i in range(n)]


def colors_for_voronoi_polys_and_points(region_polys, region_pts, point_indices=None, cmap_name='tab20',
                                        unassigned_pts_color=(0,0,0,1)):
    """
    Generate colors for the shapes and points in `region_polys` and `region_pts` using matplotlib color
    map `cmap_name`.
    
    :param region_polys: dict mapping region IDs to Voronoi region geometries
    :param region_pts: dict mapping Voronoi region IDs to point indices
    :param point_indices: optional list of point indices; used to set color to `unassigned_pts_color` for unused point
                          indices
    :param cmap_name: matplotlib color map name
    :return: tuple with (dict mapping Voronoi region ID to color, list of point colors)
    """
    vor_colors = {p_id: col
                  for p_id, col in zip(region_polys.keys(), generate_n_colors(len(region_polys), cmap_name=cmap_name))}

    pt_to_poly = points_to_region(region_pts)

    if point_indices is not None:
        pt_to_poly.update({i_pt: None for i_pt in point_indices if i_pt not in pt_to_poly.keys()})

    pt_to_poly = sorted(pt_to_poly.items(), key=lambda x: x[0])

    pt_colors = [unassigned_pts_color if i_vor is None else vor_colors[i_vor] for _, i_vor in pt_to_poly]

    assert len(vor_colors) <= len(pt_colors)

    return vor_colors, pt_colors


def xy_from_points(points):
    """
    Helper function to get x and y coordinates from NumPy array or list of Shapely Points `points` especially used
    for plotting.

    :param points: NumPy array or list of Shapely Points with length N
    :return: tuple with (NumPy array of N x-coordinates, NumPy array of N y-coordinates)
    """
    if hasattr(points, 'xy'):
        return points.xy
    else:
        if not isinstance(points, np.ndarray):
            coords = points_to_coords(points)
        else:
            coords = points
        return coords[:, 0], coords[:, 1]


def plot_voronoi_polys(ax, region_polys, color=None, edgecolor=None, labels=None, label_fontsize=10, label_color=None,
                       **kwargs):
    """
    Plot Voronoi region polygons in `poly_shapes` on matplotlib Axes object `ax`. Use fill color `color`, edge color
    `edgecolor`. Optionally supply a list of labels `labels` that will be displayed on the respective Voronoi region
    using the styling options `label_fontsize` and `label_color`. All color parameters can also be a dict or sequence
    mapped to the Voronoi region accordingly.

    Additional parameters can be passed to `plot_polygon_collection_with_color()` function as `kwargs`.

    :param ax: matplotlib Axes object to plot on
    :param region_polys: dict mapping region IDs to Voronoi region geometries
    :param color: region polygons' fill color
    :param edgecolor: region polygons' edge color
    :param labels: region labels
    :param label_fontsize: region labels font size
    :param label_color: region labels color
    :param kwargs: additional parameters passed to `plot_polygon_collection_with_color()` function
    :return: None
    """

    plot_polygon_collection_with_color(ax, region_polys, color=color, edgecolor=edgecolor, **kwargs)

    if labels:
        # plot labels using matplotlib's text()
        n_labels = len(labels)
        n_features = len(region_polys)
        if n_labels != n_features:
            raise ValueError('number of labels (%d) must match number of Voronoi polygons (%d)'
                             % (n_labels, n_features))

        for i, p in region_polys.items():
            tx, ty = p.centroid.coords[0]
            ax.text(tx, ty, labels[i], fontsize=label_fontsize, color=_color_for_labels(label_color, color, i))


def plot_points(ax, points, markersize=1, marker='o', color=None, labels=None, label_fontsize=7, label_color=None,
                label_draw_duplicates=False, **kwargs):
    """
    Plot points `points` (either list of Point objects or NumPy coordinate array) on matplotlib Axes object `ax` with
    marker size `markersize`. Define marker style with parameters `marker` and `color`. Optionally supply a list of
    labels `labels` that will be displayed next to the respective point using the styling options `label_fontsize` and
    `label_color`. All color parameters can also be a sequence.

    Additional parameters can be passed to matplotlib's `scatter` function as `kwargs`.

    :param ax: matplotlib Axes object to plot on
    :param points: NumPy array or list of Shapely Points
    :param markersize: marker size
    :param marker: marker type
    :param color: marker color(s)
    :param labels: point labels
    :param label_fontsize:  point labels font size
    :param label_color:  point labels color
    :param label_draw_duplicates: if False, suppress drawing labels on duplicate coordinates
    :param kwargs: additional parameters passed to matplotlib's `scatter` function
    :return: return value from `ax.scatter()`
    """
    x, y = xy_from_points(points)

    scat = ax.scatter(x, y, s=markersize, marker=marker, color=color, **kwargs)

    if labels:
        # plot labels using matplotlib's text()
        n_labels = len(labels)
        n_features = len(x)
        if n_labels != n_features:
            raise ValueError('number of labels (%d) must match number of points (%d)'
                             % (n_labels, n_features))

        drawn_coords = set()
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            pos = (x_i, y_i)  # make hashable
            if label_draw_duplicates or pos not in drawn_coords:
                ax.text(x_i, y_i, labels[i], fontsize=label_fontsize, color=_color_for_labels(label_color, color, i))
                drawn_coords.add(pos)

    return scat


def plot_line(ax, points, linewidth=1, color=None, **kwargs):
    """
    Plot sequence of points `points` (either list of Point objects or NumPy coordinate array) on matplotlib Axes object
    `ax` as line.

    :param ax: matplotlib Axes object to plot on
    :param points: NumPy array or list of Shapely Point objects
    :param linewidth: line width
    :param color: line color
    :param kwargs: additional parameters passed to matplotlib's `plot` function
    :return: return value from `ax.plot()`
    """
    x, y = xy_from_points(points)

    return ax.plot(x, y, linewidth=linewidth, color=color, **kwargs)


def plot_polygon(ax, polygon, facecolor=None, edgecolor=None, linewidth=1, linestyle='solid',
                 label=None, label_fontsize=7, label_color='k', **kwargs):
    """
    Plot a Shapely polygon on matplotlib Axes object `ax`.

    :param ax: matplotlib Axes object to plot on
    :param polygon: Shapely Polygon/Multipolygon object
    :param facecolor: fill color of the polygon
    :param edgecolor: edge color of the polygon
    :param linewidth: line width
    :param linestyle: line style
    :param label: optional label to show at polygon's centroid
    :param label_fontsize: label font size
    :param label_color: label color
    :param kwargs: additonal parameters passed to matplotlib `PatchCollection` object
    :return: return value from `ax.add_collection()`
    """
    coll = ax.add_collection(PatchCollection([PolygonPatch(polygon)],
                                             facecolor=facecolor, edgecolor=edgecolor,
                                             linewidth=linewidth, linestyle=linestyle,
                                             **kwargs))
    if label:
        ax.text(polygon.centroid.x, polygon.centroid.y, label, fontsize=label_fontsize, color=label_color)

    return coll


def plot_voronoi_polys_with_points_in_area(ax, area_shape, region_polys, points, region_pts=None,
                                           area_color=(1,1,1,1), area_edgecolor=(0,0,0,1),
                                           voronoi_and_points_cmap='tab20',
                                           voronoi_color=None, voronoi_edgecolor=None,
                                           points_color=None, points_markersize=5, points_marker='o',
                                           voronoi_labels=None, voronoi_label_fontsize=10, voronoi_label_color=None,
                                           point_labels=None, point_label_fontsize=7, point_label_color=None,
                                           plot_area_opts=None,
                                           plot_voronoi_opts=None,
                                           plot_points_opts=None):
    """
    All-in-one function to plot Voronoi region polygons `region_polys` and the respective points `points` inside a
    geographic area `area_shape` on a matplotlib Axes object `ax`.

    By default, the regions will be blue and the points black. Optionally pass `region_pts` to show Voronoi regions and
    their respective points with the same color (which is randomly drawn from color map `voronoi_and_points_cmap`).
    Labels for Voronoi regions can be passed as `voronoi_labels`. Labels for points can be passed as `point_labels`.
    Use style options to customize the plot. Pass additional (matplotlib) parameters to the individual plotting steps
    as `plot_area_opts`, `plot_voronoi_opts` or `plot_points_opts` respectively.

    :param ax: matplotlib Axes object to plot on
    :param area_shape: geographic shape surrounding the Voronoi regions; can be None to disable plotting of geogr. shape
    :param region_polys: dict mapping region IDs to Voronoi region geometries
    :param points: NumPy array or list of Shapely Point objects
    :param region_pts: dict mapping Voronoi region IDs to point indices of `points`
    :param area_color: fill color of `area_shape`
    :param area_edgecolor: edge color of `area_shape`
    :param voronoi_and_points_cmap: matplotlib color map name used for Voronoi regions and points colors when colors are
                                    not given by `voronoi_color`
    :param voronoi_color: dict mapping Voronoi region ID to fill color or None to use `voronoi_and_points_cmap`
    :param voronoi_edgecolor: Voronoi region polygon edge colors
    :param points_color: points color
    :param points_markersize: points marker size
    :param points_marker: points marker type
    :param voronoi_labels: Voronoi region labels displayed at centroid of Voronoi region polygon
    :param voronoi_label_fontsize: Voronoi region labels font size
    :param voronoi_label_color: Voronoi region labels color
    :param point_labels: point labels
    :param point_label_fontsize: point labels font size
    :param point_label_color: point labels color
    :param plot_area_opts: options passed to function for plotting the geographic shape
    :param plot_voronoi_opts: options passed to function for plotting the Voronoi regions
    :param plot_points_opts: options passed to function for plotting the points
    :return: None
    """
    plot_area_opts = plot_area_opts or {}
    plot_voronoi_opts = plot_voronoi_opts or {'alpha': 0.5}
    plot_points_opts = plot_points_opts or {}

    if area_shape is not None:
        plot_polygon_collection_with_color(ax, [area_shape], color=area_color, edgecolor=area_edgecolor,
                                           **plot_area_opts)

    if voronoi_and_points_cmap and region_pts and (voronoi_color is None or points_color is None):
        voronoi_color, points_color = colors_for_voronoi_polys_and_points(region_polys, region_pts,
                                                                          point_indices=list(range(len(points))),
                                                                          cmap_name=voronoi_and_points_cmap)

    if voronoi_color is None and voronoi_edgecolor is None:
        voronoi_edgecolor = (0,0,0,1)   # better visible default value

    plot_voronoi_polys(ax, region_polys, color=voronoi_color, edgecolor=voronoi_edgecolor,
                       labels=voronoi_labels, label_fontsize=voronoi_label_fontsize, label_color=voronoi_label_color,
                       **plot_voronoi_opts)

    plot_points(ax, points, points_markersize, points_marker, color=points_color,
                labels=point_labels, label_fontsize=point_label_fontsize, label_color=point_label_color,
                **plot_points_opts)


def plot_polygon_collection_with_color(ax, geoms, color=None, **kwargs):
    """
    This is a modified version of geopanda's `_plot_polygon_collection` function that also accepts a sequence or dict
    of colors passed as `color` for each polygon in `geoms` and *uses them correctly even when `geoms` contains
    MultiPolygon objects*.

    Plots a collection of Polygon and MultiPolygon geometries to `ax`.

    :param ax: matplotlib.axes.Axes object where shapes will be plotted
    :param geoms: a sequence or dict of `N` Polygons and/or MultiPolygons (can be mixed)
    :param color: a sequence or dict of `N` colors (optional) or None for default color or a single color for all shapes
    :param kwargs: additional keyword arguments passed to the collection
    :return: matplotlib.collections.Collection that was plotted
    """

    color_values = color
    color_indices = None

    if not isinstance(geoms, GeoSeries):
        if isinstance(geoms, dict):
            geoms_indices = np.array(list(geoms.keys()))
            geoms_values = list(geoms.values())

            if isinstance(color, dict):
                color_indices = np.array(list(color.keys()))
                color_values = np.array(list(color.values()))
        else:
            geoms_indices = np.arange(len(geoms))
            geoms_values = geoms
        geoms = GeoSeries(geoms_values)
    else:
        geoms_indices = geoms.index.to_numpy()

    if isinstance(color, list):
        color_values = np.array(color)
        color_indices = np.arange(len(color))
    elif isinstance(color, np.ndarray):
        color_values = color
        color_indices = np.arange(len(color))

    geoms, multi_indices = _flatten_multi_geoms(geoms)

    if color_indices is not None:  # retain correct color indices
        color_values = color_values[np.nonzero(geoms_indices[multi_indices][..., np.newaxis] == color_indices)[1]]

    # PatchCollection does not accept some kwargs.
    if 'markersize' in kwargs:
        del kwargs['markersize']

    collection = PatchCollection([PolygonPatch(poly) for poly in geoms],
                                 color=color_values, **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def _color_for_labels(label_color, default_color, seq_index):
    """Helper function to get a color for a label with index `seq_index`."""
    if label_color is None:
        if hasattr(default_color, '__getitem__'):
            c = default_color[seq_index]
        else:
            c = default_color
    else:
        c = label_color

    return c or (0,0,0,1)
