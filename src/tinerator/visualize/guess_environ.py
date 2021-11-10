import subprocess
import os
import platform
import sys
from enum import Enum, auto


class ExecEnvironment(Enum):
    JUPYTER = auto()  # Jupyter Notebook/Lab
    IPYTHON = auto()  # IPython REPL
    DOCKER = auto()  # Docker container
    COLAB = auto()  # Google Colab
    SCRIPT = auto()  # Standard execution


def in_notebook():
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


def in_docker_container():
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


def in_google_colab():
    return "google.colab" in sys.modules


def get_environment():
    env = []
    ee = ExecEnvironment

    if in_google_colab():
        env.append(ee.COLAB)

    if in_docker_container():
        env.append(ee.DOCKER)

    if in_notebook():
        env.append(ee.JUPYTER)

    return env


def open_file(filepath):
    if platform.system() == "Darwin":  # macOS
        subprocess.call(("open", filepath))
    elif platform.system() == "Windows":  # Windows
        os.startfile(filepath)
    else:  # linux variants
        subprocess.call(("xdg-open", filepath))


def init_pyvista_framebuffer(force: bool = False):
    """
    Initializes a headless framebuffer for 3D rendering.
    Used in Docker container.
    """
    global PYVISTA_XVFB_STARTED

    if force or (not PYVISTA_XVFB_STARTED):
        import pyvista as pv

        pv.start_xvfb()
        PYVISTA_XVFB_STARTED = True
