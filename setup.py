"""
geovoronoi setuptools based setup module
"""

import os
from setuptools import setup

import geovoronoi

GITHUB_URL = 'https://github.com/WZBSocialScienceCenter/geovoronoi'


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=geovoronoi.__title__,
    version=geovoronoi.__version__,
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],

    keywords='voronoi tesselation gis geographic area visualization plotting',

    packages=['geovoronoi'],
    # include_package_data=True,
    python_requires='>=3.4',
    install_requires=['numpy>=1.11.0', 'scipy>=0.12.0', 'shapely>=1.6.0'],
    extras_require={
        'plotting': ['matplotlib>=2.1.0', 'geopandas>=0.5.0', 'descartes>=1.1.0'],
    }
)
