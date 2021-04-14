import pyproj
import shapefile
import numpy as np
import os

# from ..meshing import Mesh, ElementType
from ..logging import log, warn, debug


def save_triangulation_to_shapefile(outfile: str, mesh):
    """
    Saves triangulated surface as an ESRI Shapefile.
    Contains additional fields of:

        - "elementID" - int - the integer ID of the triangle
        - "elevation" - float - triangle centroid Z value
    """

    # assert mesh.element_type == ElementType.TRIANGLE
    assert mesh.elements.shape[1] == 3

    crs = mesh.crs

    elems = mesh.elements - 1
    elems = np.vstack([elems.T, elems[:, 0]]).T
    triangles = mesh.nodes[elems].tolist()

    crs_outfile = os.path.splitext(outfile)[0] + ".prj"

    with shapefile.Writer(outfile) as w:
        debug(f"Attempting to save vector object to {outfile}")

        w.field("elementID", "N")
        w.field("elevation", "N", decimal=10)

        z_values = mesh.get_cell_centroids()[:,2]

        for (i, triangle) in enumerate(triangles):
            w.record(i+1, z_values[i])
            w.poly([triangle])

        w.balance()

    log(f"Shapefile data written to {outfile}")

    # CRS info must be written out manually. See the reader.
    with open(crs_outfile, "w") as f:
        f.write(crs.to_wkt(version=pyproj.enums.WktVersion.WKT1_ESRI))

    log(f"CRS information written to {crs_outfile}")