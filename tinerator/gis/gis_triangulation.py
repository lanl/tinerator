import pyproj
import shapefile
import numpy as np
import os
#from ..meshing import Mesh, ElementType
from ..logging import log, warn, debug

def save_triangulation_to_shapefile(outfile: str, mesh):
    """
    Saves triangulated surface as an ESRI Shapefile.
    """

    # assert mesh.element_type == ElementType.TRIANGLE

    crs = mesh.crs

    elems = mesh.elements - 1
    elems = np.vstack([elems.T, elems[:,0]]).T
    points = mesh.nodes[elems].tolist()

    crs_outfile = os.path.splitext(outfile)[0] + ".prj"

    with shapefile.Writer(outfile) as w:
        debug(f"Attempting to save vector object to {outfile}")
        w.field("name", "C")
        w.poly(points)
        w.record("polygon1")

    log(f"Shapefile data written to {outfile}")

    # CRS info must be written out manually. See the reader.
    with open(crs_outfile, "w") as f:
        f.write(crs.to_wkt(version=pyproj.enums.WktVersion.WKT1_ESRI))
    
    log(f"CRS information written to {crs_outfile}")