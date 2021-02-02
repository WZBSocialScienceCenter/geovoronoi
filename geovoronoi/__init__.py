"""
geovoronoi – main module

Imports all necessary functions to calculate Voronoi regions from a set of coordinates inside a geographic shape.
Addtionally imports some helper functions.

Author: Markus Konrad <markus.konrad@wzb.eu>
"""


from ._voronoi import (coords_to_points, points_to_coords, voronoi_regions_from_coords,
                       get_points_to_poly_assignments)
from ._geom import calculate_polygon_areas, line_segment_intersection


__title__ = 'geovoronoi'
__version__ = '0.3.0dev'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'
