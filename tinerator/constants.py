DEFAULT_NO_DATA_VALUE = -9999.0
DEFAULT_PROJECTION = "EPSG:32601"  # Default projection if a CRS can't be parsed
PLOTLY_PROJECTION = "WGS84"  # All objects are projected to this for plotting
JUPYTER_BACKEND_DEFAULT = "panel"  # PyVista Jupyter backend default: https://docs.pyvista.org/user-guide/jupyter/index.html
PYVISTA_XVFB_STARTED = False  # toggle for _init_pyvista_framebuffer
DEFAULT_CMAP_VTK = None  # Default colormap for rendering meshes
VTK_COLORS_SETS = ("red", "blue", "yellow", "green", "pink", "purple")


class MeshingConstants:
    MATERIAL_ID_NAME = "material_id"  # The internal name for Material ID
    TOP_LAYER = -1  # The "top layer" of a stacked mesh
    BOTTOM_LAYER = -2  # The "botom layer" of a stacked mesh
    INTERIOR_LAYER = 0  # The interior layers of a stacked mesh


def is_tinerator_object(obj, name: str):
    """
    Alternate method for ``isinstance()``, but with
    TINerator objects.
    """
    if isinstance(name, (tuple, list)):
        return any([is_tinerator_object(x) for x in name])

    try:
        return (
            obj.__module__.split(".")[0] == "tinerator"
            and type(obj).__name__.lower() == name.lower()
        )
    except:
        return False
