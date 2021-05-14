from shapely.geometry import Polygon
from .geometry import Geometry
from ..logging import log, warn, debug


def vectorize_triangulation(mesh):
    """
    Saves triangulated surface as an ESRI Shapefile.
    Contains additional fields of:

    - "elementID" - `int` - the integer ID of the triangle
    - "materialID" - `int` - the material ID of each triangle
    - "elevation" - `float` - triangle centroid Z value

    Useful for importing into QGIS / ArcGIS to validate
    the triangulation relative to other GIS objects.

    Args
    ----
        outfile (str): The path to save the shapefile.
        mesh (tinerator.meshing.Mesh): The triangulated surface to save.

    Note
    ----
        Only triangulations will work with this function. Prism meshes
        and other mesh types will not.

    Examples
    --------
        >>> surface_mesh = tin.meshing.triangulate(dem, min_edge_length=0.1)
        >>> triang_geom = tin.gis.vectorize_triangulation(surface_mesh)
        >>> triang_geom.save("triangulation.shp")
    """

    assert mesh.elements.shape[1] == 3

    crs = mesh.crs

    elems = mesh.elements - 1
    triangles = mesh.nodes[elems].tolist()
    shapes = [Polygon(tri) for tri in triangles]

    element_ids = list(range(1, len(shapes) + 1))
    material_ids = mesh.material_id
    elevations = mesh.get_cell_centroids()[:, 2]

    triang = Geometry(shapes=shapes, crs=mesh.crs)
    triang.add_property("elementID", element_ids, type="int")
    triang.add_property("materialID", material_ids, type="int")
    triang.add_property("elevation", elevations, type="float")

    return triang
