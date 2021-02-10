"""
geovoronoi setuptools based setup module
"""

import os
from setuptools import setup

GITHUB_URL = 'https://github.com/WZBSocialScienceCenter/geovoronoi'

__title__ = 'geovoronoi'
__version__ = '0.3.0'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'

here = os.path.abspath(os.path.dirname(__file__))

DEPS_BASE = ['numpy>=1.19.0,<2', 'scipy>=1.5.0,<1.7', 'shapely>=1.7.0,<1.8']

DEPS_EXTRA = {
    'plotting': ['matplotlib>=3.3.0,<3.4', 'geopandas>=0.8.0,<0.9', 'descartes>=1.1.0,<1.2'],
    'test': ['pytest>=6.2.0,<6.3', 'pytest-mpl>=0.12,<0.13', 'hypothesis>=6.0.0,<6.1', 'tox>=3.21.0,<3.22'],
    'develop': ['ipython>=7.19.0', 'twine>=3.3.0'],
}

DEPS_EXTRA['all'] = []
for k, deps in DEPS_EXTRA.items():
    if k != 'all':
        DEPS_EXTRA['all'].extend(deps)

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=__title__,
    version=__version__,
    description='a package to create and plot Voronoi regions in geographic areas',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=GITHUB_URL,
    project_urls={
        'Source': GITHUB_URL,
        'Tracker': 'https://github.com/WZBSocialScienceCenter/geovoronoi' + '/issues',
    },
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],

    keywords='voronoi tesselation gis geographic area visualization plotting',

    packages=['geovoronoi'],
    # include_package_data=True,
    python_requires='>=3.6',
    install_requires=DEPS_BASE,
    extras_require=DEPS_EXTRA
)
