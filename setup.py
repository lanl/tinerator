"""

"""
import os
import platform
import sys
import warnings

from setuptools import setup

def get_version(package_root):
    sys.path.append(package_root)
    import _version
    return _version.__version__

package_name = 'tinerator'

filepath = os.path.dirname(__file__)
readme_file = os.path.join(filepath, 'README.md')
__version__ = get_version(os.path.join(filepath,package_name))
print('version = ',__version__)

# Update __version__ string from tinerator/_version.py


# pre-compiled vtk available for python3

setup(
    name=package_name,
    packages=[package_name, 'tinerator.meshing', 'tinerator.gis'],
    version=__version__,
    description='Powerful geological modeling',
    author='Daniel Livingston',
    author_email='livingston@lanl.gov',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    url='https://github.com/tinerator/tinerator-core',
    keywords='gis tin geophysical subsurface 3d 2.5d meshing',
    python_requires='>=3.5.*',
    install_requires=['numpy','pyvista','meshio>=4'],
)

#from setuptools import setup
#from distutils.core import setup
#
#setup(
#    name = "tinerator",
#    packages = ["tinerator"],
#    version = "0.3.1",
#    description = "DEM -> Refined TIN Generation",
#    license = 'BSD-3',
#    author = "Daniel Livingston, David Moulton, Terry Ann Miller, Zexuan Xu, Ilhan Ozgen",
#    author_email = "livingston@lanl.gov",
#    url = "http://www.github.com/lanl/tinerator",
#    keywords = ["gis", "dem", "tin", "amanzi", "lagrit", "ats"],
#    install_requires=[
#        'richdem',
#        'matplotlib',
#        'pylagrit',
#        'numpy',
#        'scipy',
#        'rasterio',
#        'fiona',
#        'elevation',
#        'scikit-fmm',
#        'panel'],
#    classifiers = [
#        "Programming Language :: Python",
#        "Programming Language :: Python :: 3",
#        "Development Status :: 4 - Beta",
#        "Environment :: Other Environment",
#        "Intended Audience :: Developers",
#        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
#        "Operating System :: OS Independent",
#        "Topic :: Software Development :: Libraries :: Python Modules",
#        ],
#    long_description = """\
#Digital Elevation Map to Refined and Extruded TIN
#-------------------------------------
#
#This library:
#- Downloads a DEM from lat/long coordinates or from a shapefile
#- Performs watershed deliniation to determine areas of refinement
#- Triangles DEM with refinement along watershed features
#- Extrudes DEM to configurable layers
#
#This version requires Python 3 or later; a Python 2 version is available separately.
#"""
#)