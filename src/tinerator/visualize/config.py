from enum import Enum, auto
from os import wait

#from distributed.core import Server
from tinerator.visualize.guess_environ import get_environment, open_file
from typing import Union, List
#import dash_bootstrap_components as dbc
from ..logging import log, warn, debug, error


class ServerTypes(Enum):
    DEFAULT = auto()
    JUPYTER = auto()
    WINDOWED = auto()


def guess_mode():
    from .guess_environ import get_environment
    from .guess_environ import ExecEnvironment as ee

    env = get_environment()

    # If we're in Jupyter, we want inline rendering
    if ee.JUPYTER in env:
        return ServerTypes.JUPYTER

    # If pywebview isn't install, just run as default
    try:
        import webview

        return ServerTypes.WINDOWED
        #return ServerTypes.DEFAULT
    except ModuleNotFoundError as e:
        return ServerTypes.DEFAULT


class ServerSettings:
    port = "8050"
    host = "127.0.0.1"
    mode = guess_mode()
    css = []#[dbc.themes.BOOTSTRAP]


def set_server_settings(
    host: str = None,
    port: Union[int, str] = None,
    css: List[str] = None,
    mode: Union[ServerTypes, str] = None,
):
    if host:
        ServerSettings.host = host

    if port:
        ServerSettings.port = f"{port}"

    if css:
        ServerSettings.css = css

    if mode:
        if isinstance(mode, str):
            mode = mode.strip().lower()

            if mode in ["window", "windowed", "app"]:
                mode = ServerTypes.WINDOWED
            elif mode in ["notebook", "jupyter", "colab"]:
                mode = ServerTypes.JUPYTER
            else:
                mode = ServerTypes.DEFAULT

        ServerSettings.mode = mode

    debug(f"{vars(ServerSettings)=}")


def run_server_default(
    layout, host=ServerSettings.host, port: str = ServerSettings.port, **kwargs
):
    from .dash_tin import DashTIN, find_open_port
    from .guess_environ import open_file

    # We want all Dash servers to exist in this mode
    # So, find an open port
    port = find_open_port(host, port)

    app = DashTIN(__name__, external_stylesheets=ServerSettings.css)
    app.layout = layout
    app.run_server(host=host, port=port, **kwargs)
    url = app.getServerURL()
    debug(f"Serving at {url=}")

    if not url.lower().startswith("http"):
        url = f"http://{url}"

    open_file(url)
    # debug(f"Shutting down server...")
    # app.shutdown()


def run_server_windowed(
    layout,
    host=ServerSettings.host,
    port: str = ServerSettings.port,
    width: int = 1200,
    height: int = 900,
    **kwargs,
):
    from .dash_tin import DashTIN
    from .guiwebview import run_web_app

    app = DashTIN(__name__, external_stylesheets=ServerSettings.css)
    app.layout = layout
    app.run_server(host=host, port=port, **kwargs)
    url = app.getServerURL()
    debug(f"Serving at {url=}")

    exit_code = run_web_app(url, width=width, height=height)

    if exit_code != 0:
        warn(f"GUI view exited with {exit_code=}")


def run_server_jupyter(layout, **kwargs):
    from jupyter_dash import JupyterDash

    app = JupyterDash(__name__, external_stylesheets=ServerSettings.css)
    app.layout = layout
    app.run_server(mode="inline", **kwargs)
    print(app.status())


def run_server(layout, mode: ServerTypes = None, **kwargs):

    from .guess_environ import get_environment
    from .guess_environ import ExecEnvironment as ee

    env = get_environment()

    if ee.DOCKER in env:
        ServerSettings.host = '0.0.0.0'

    if mode is None:
        mode = ServerSettings.mode

    debug(f"Server mode: {mode=}")

    if mode in [ServerTypes.DEFAULT, None]:
        run_server_default(layout, **kwargs)
    elif mode == ServerTypes.JUPYTER:
        run_server_jupyter(layout, **kwargs)
    elif mode == ServerTypes.WINDOWED:
        run_server_windowed(layout, **kwargs)
    else:
        raise ValueError(f"Server mode {mode} unknown")

    debug("Finished visualizing")
