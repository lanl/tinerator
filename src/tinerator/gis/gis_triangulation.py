import numpy as np
from shapely.geometry import Polygon
from .geometry import Geometry
from ..logging import log, warn, debug


def exodus_cell_remapping(
    cells: np.ndarray, block_id: np.ndarray, nodes: np.ndarray
) -> np.ndarray:
    """
    The Exodus API requires mesh cells to be sorted
    by, in successive order:
    - Block ID (material ID)
    - Cell centroid -> Z coord
    - Cell centroid -> Y coord
    - Cell centroid -> X coord

    This returns a mapping for cells into this format.
    """
    import operator

    mapping = np.array(list(range(cells.shape[0])))
    cell_centroids = np.mean(nodes[cells], axis=1)

    # Sort by: matID, z_med, y_med, x_med
    key = np.hstack(
        [mapping[:, None], block_id[:, None], np.fliplr(cell_centroids)]
    ).tolist()
    key.sort(key=operator.itemgetter(1, 2, 3, 4))

    return np.array(key)[:, 0].astype(int)


def vectorize_stacked_mesh(mesh, at_layer: tuple = (1, 1), exodus_mapping=True):
    elements = mesh.elements
    elements_mask = mesh.get_cells_at_sublayer(at_layer, return_mask=True)
    element_ids = np.arange(mesh.n_elements)

    elems = elements[elements_mask]
    elems = elems[:, :3] - 1
    triangles = mesh.nodes[elems]

    element_ids = element_ids[elements_mask]
    material_ids = mesh.material_id[elements_mask]
    elevations = mesh.get_cell_centroids()[elements_mask][:, 2]

    shapes = [Polygon(tri) for tri in triangles.tolist()]

    triang = Geometry(shapes=shapes, crs=mesh.crs)
    triang.add_property("elementID", element_ids.tolist(), type="int")
    triang.add_property("materialID", material_ids, type="int")
    triang.add_property("elevation", elevations, type="float")

    if exodus_mapping:
        mapping = exodus_cell_remapping(elements - 1, mesh.material_id, mesh.nodes)
        set_mapping = np.array(
            sorted(list(range(len(mapping))), key=lambda x: mapping[x])
        )
        set_mapping = set_mapping[elements_mask]
        triang.add_property("exoElemID", set_mapping, type="int")

    return triang


def vectorize_triangulation(mesh, exodus_mapping: bool = True):
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

    element_type = str(mesh.element_type).split(".")[-1].lower()

    if element_type == "prism":
        return vectorize_stacked_mesh(mesh, exodus_mapping=exodus_mapping)
    elif element_type != "triangle":
        raise NotImplementedError(element_type)

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
