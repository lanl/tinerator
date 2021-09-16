"""
Copyright (c) 2020. Triad National Security, LLC. All rights reserved.
 
This program was produced under U.S. Government contract 89233218CNA000001
for Los Alamos National Laboratory (LANL), which is operated by Triad National
Security, LLC for the U.S. Department of Energy/National Nuclear Security
Administration.
 
All rights in the program are reserved by Triad National Security, LLC,
and the U.S. Department of Energy/National Nuclear Security Administration.
The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to
reproduce, prepare derivative works, distribute copies to the public,
perform publicly and display publicly, and to permit others to do so.
"""

import os
from tinerator._version import __version__
from tinerator.logging import set_logging_verbosity, debug_mode, LogLevel
import tinerator.meshing as meshing
import tinerator.gis as gis
import tinerator.examples as examples

# import tinerator.visualize as visualize
from tinerator.visualize import plot2d, plot3d, mapbox_styles

from tinerator.meshing import Mesh
from tinerator.meshing import SideSet, PointSet
from tinerator.gis import Geometry, Raster


def configure(**kwargs):
    """
    Sets TINerator global configuration.

    Args:
        debug (bool, optional): Toggle debug mode on/off.
        plot_backend (str, optional): one of: "jupyter", "window", "nothing"
        plot_server_host (str, optional): Plotting server host. Defaults to "127.0.0.1".
        plot_server_port (Union[int, str], optional): Plotting server port. Defaults to "8050".
    """
    from tinerator.visualize import set_server_settings

    if "debug" in kwargs:
        if kwargs["debug"] == True:
            debug_mode()

    if "plot_backend" in kwargs:
        set_server_settings(mode=kwargs["plot_backend"])

    if "plot_server_host" in kwargs:
        set_server_settings(host=kwargs["plot_server_host"])

    if "plot_server_port" in kwargs:
        set_server_settings(port=kwargs["plot_server_port"])
