#from setuptools import setup
from distutils.core import setup, Extension

module1 = Extension(
        'superfastcode',
        sources=['tinerator/test.cpp'],
        #include_dirs=[],
        language='c++',
        extra_compile_args = ['-std=c++11','-stdlib=libc++'],
        extra_link_args=["-stdlib=libc++", "-mmacosx-version-min=10.9"],
    )

setup(
    name = "tinerator",
    packages = ["tinerator"],
    ext_modules = [module1],
    version = "0.3.1",
    description = "DEM -> Refined TIN Generation",
    license = 'BSD-3',
    author = "Daniel Livingston, David Moulton, Terry Ann Miller, Zexuan Xu, Ilhan Ozgen",
    author_email = "livingston@lanl.gov",
    url = "http://www.github.com/lanl/tinerator",
    keywords = ["gis", "dem", "tin", "amanzi", "lagrit", "ats", "meshing"],
    install_requires=[
        'richdem',
        'matplotlib',
        'pylagrit',
        'numpy',
        'scipy',
        'rasterio',
        'fiona',
        'elevation',
        'scikit-fmm',
        'panel',
        'meshio'],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    long_description = """\
Digital Elevation Map to Refined and Extruded TIN
-------------------------------------

This library:
- Downloads a DEM from lat/long coordinates or from a shapefile
- Performs watershed deliniation to determine areas of refinement
- Triangles DEM with refinement along watershed features
- Extrudes DEM to configurable layers
"""
)