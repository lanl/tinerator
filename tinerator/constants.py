DEFAULT_NO_DATA_VALUE = -9999.0
DEFAULT_PROJECTION = "EPSG:32601"  # Default projection if a CRS can't be parsed
PLOTLY_PROJECTION = "WGS84"  # All objects are projected to this for plotting
JUPYTER_BACKEND_DEFAULT = "panel" # PyVista Jupyter backend default: https://docs.pyvista.org/user-guide/jupyter/index.html

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
