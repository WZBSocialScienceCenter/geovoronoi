"""
geovoronoi – main module

Imports all necessary functions to calculate Voronoi regions from a set of coordinates on a geographic shape.
Addtionally imports some helper funcitons.

Author: Markus Konrad <markus.konrad@wzb.eu>
"""


from ._voronoi import (coords_to_points, points_to_coords, voronoi_regions_from_coords, polygon_lines_from_voronoi,
                       polygon_shapes_from_voronoi_lines, assign_points_to_voronoi_polygons,
                       get_points_to_poly_assignments)
from ._geom import calculate_polygon_areas


__title__ = 'geovoronoi'
__version__ = '0.2.0'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'
