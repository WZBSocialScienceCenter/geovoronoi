"""
tmtoolkit setuptools based setup module
"""

from setuptools import setup

import geovoronoi


setup(
    name=geovoronoi.__title__,
    version=geovoronoi.__version__,
    description='a package to create and plot Voronoi regions in geographic areas',
    long_description="""geovoronoi is a small Python 3 package that uses SciPy to generate Voronoi regions for a given
set of points in a given geographic area. It then allows to intersect the geographic area with these Voronoi regions so
that they are limited to the geographic area. Furthermore, functions to visualize the results are implemented.""",
    url='https://github.com/WZBSocialScienceCenter/geovoronoi',

    author='Markus Konrad',
    author_email='markus.konrad@wzb.eu',

    license='Apache 2.0',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',

        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],

    keywords='voronoi tesselation gis geographic area',

    packages=['geovoronoi'],
    # include_package_data=True,
    python_requires='>=3.4',
    install_requires=['numpy>=1.11.0', 'scipy>=0.12.0', 'shapely>=1.6.0'],
    extras_require={
        'plotting': ['matplotlib>=2.1.0', 'geopandas>=0.3.0'],
    }
)
