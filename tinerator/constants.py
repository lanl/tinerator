DEFAULT_NO_DATA_VALUE = -9999.0
DEFAULT_PROJECTION = "EPSG:32601"  # Default projection if a CRS can't be parsed
PLOTLY_PROJECTION = "WGS84"  # All objects are projected to this for plotting
JUPYTER_BACKEND_DEFAULT = "panel"  # PyVista Jupyter backend default: https://docs.pyvista.org/user-guide/jupyter/index.html


class MeshingConstants:
    MATERIAL_ID_NAME = "material_id"  # The internal name for Material ID
    TOP_LAYER = -1  # The "top layer" of a stacked mesh
    BOTTOM_LAYER = -2  # The "botom layer" of a stacked mesh
    INTERIOR_LAYER = 0  # The interior layers of a stacked mesh


def _in_notebook():
    # https://stackoverflow.com/a/39662359/5150303
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def _in_docker_container():
    raise NotImplementedError()
    # import pyvista; pyvista.start_xvfb()
