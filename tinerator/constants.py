import subprocess

DEFAULT_NO_DATA_VALUE = -9999.0
DEFAULT_PROJECTION = "EPSG:32601"  # Default projection if a CRS can't be parsed
PLOTLY_PROJECTION = "WGS84"  # All objects are projected to this for plotting
JUPYTER_BACKEND_DEFAULT = "panel"  # PyVista Jupyter backend default: https://docs.pyvista.org/user-guide/jupyter/index.html
PYVISTA_XVFB_STARTED = False  # toggle for _init_pyvista_framebuffer


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


def _in_docker_container():
    """
    Checks if we are living in a Docker container or not.
    """
    # Ref: https://stackoverflow.com/a/23575107/5150303

    cmd = "awk -F/ '$2 == \"docker\"' /proc/self/cgroup"
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
        return "docker" in result.decode().lower()
    except subprocess.CalledProcessError:
        return False


def _init_pyvista_framebuffer(force: bool = False):
    """
    Initializes a headless framebuffer for 3D rendering.
    Used in Docker container.
    """
    global PYVISTA_XVFB_STARTED

    if force or (not PYVISTA_XVFB_STARTED):
        import pyvista as pv

        pv.start_xvfb()
        PYVISTA_XVFB_STARTED = True
