# Changes

## 0.4.0

- better exception message for *"ridge line must intersect ..."* RuntimeError
- added Python 3.10 support
- update dependencies
- fixed warnings for updated dependencies

## 0.3.0

**This is a major update that changes the default behavior and accepted arguments of `voronoi_regions_from_coords()`.**
You will most likely need to update your scripts. See the examples in this repository and the function documentation.

* complete rewrite of Voronoi region polygon construction
  * usually quicker and more memory efficient
  * **much** quicker when dealing with duplicate points
  * adds possibility to treat the geometries in a geographic area separately, if that area is a MultiPolygon shape;
    this is actually the new default behavior; to reset to the former behavior, set `per_geom=False` when using
    `voronoi_regions_from_coords()`
  * adds possiblity to also retrieve results separately for each geometry in a geographic area
  * `voronoi_regions_from_coords()` now returns dicts instead of list that map region IDs to region geometries and
    assigned points
* updated, better function documentation
* added and updated tests
* new and updated examples
* dependency updates
* dropped official Python 3.5 support (may still work, though), added Python 3.9 support 

## 0.2.0

* fix [issue #7](https://github.com/WZBSocialScienceCenter/geovoronoi/issues/7) (thanks [@AtelierLibre](https://github.com/AtelierLibre))
* update dependencies / make compatible with GeoPandas 0.7.0
* introduce more and better tests
* use *tox* for testing

## 0.1.2

* fix bug if centroid of boundary area is outside convex hull of generator set (thanks [@mjziebarth](https://github.com/mjziebarth))

## 0.1.1

* made compatible with GeoPandas 0.5.0
* added missing requirement descartes in setup.py

## 0.1.0

* initial release
