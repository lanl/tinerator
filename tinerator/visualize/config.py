from enum import Enum, auto
from typing import Union, List
import dash_bootstrap_components as dbc
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

    # If we're in Docker and *not* Jupyter, all we can
    # (safely) do is run as a web service - 
    # because port forwarding is easier than X Server setup
    if ee.DOCKER in env:
        return ServerTypes.DEFAULT
    
    # If PyQt5 *and* QtWebEngine aren't available,
    # just run as a web service
    try:
        from PyQt5.QtCore import QUrl
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        from PyQt5.QtWidgets import QApplication
    except ModuleNotFoundError as e:
        return ServerTypes.DEFAULT

    # Looks like we're running locally with a proper Qt setup
    return ServerTypes.WINDOWED

class ServerSettings:
    port = '8050'
    host = '127.0.0.1'
    mode = guess_mode()
    css = [dbc.themes.BOOTSTRAP]

def set_server_settings(host: str = None, port: Union[int, str] = None, css: List[str] = None, mode: Union[ServerTypes, str] = None):
    if host:
        ServerSettings.host = host
    
    if port:
        ServerSettings.port = f"{port}"
    
    if css:
        ServerSettings.css = css
    
    if mode:
        if isinstance(mode, str):
            mode = mode.strip().lower()

            if mode in ['window', 'windowed', 'app']:
                mode = ServerTypes.WINDOWED
            elif mode in ['notebook', 'jupyter', 'colab']:
                mode = ServerTypes.JUPYTER
            else:
                mode = ServerTypes.DEFAULT
        
        ServerSettings.mode = mode

def run_server_default(layout, host = ServerSettings.host, port: str = ServerSettings.port, **kwargs):
    from .dash_tin import DashTIN

    app = DashTIN(__name__, external_stylesheets=ServerSettings.css)
    app.layout = layout
    app.run_server(host=host, port=port, **kwargs)
    url = app.getServerURL()
    log(f"Serving at {url=}")

    # Anything to do here?

    log(f"Shutting down server...")
    app.shutdown()

def run_server_windowed(layout, host = ServerSettings.host, port: str = ServerSettings.port, width: int = 1200, height: int = 900, allow_resize: bool = True, **kwargs):
    from .dash_tin import DashTIN
    from .qt_app import run_web_app

    app = DashTIN(__name__, external_stylesheets=ServerSettings.css)
    app.layout = layout
    app.run_server(host=host, port=port, **kwargs)
    url = app.getServerURL()
    log(f"Serving at {url=}")

    exit_code = run_web_app(url, width=width, height=height, allow_resize=allow_resize)

    if exit_code != 0:
        warn(f"QtApplication exited with {exit_code=}")
    
    log(f"Shutting down server...")
    app.shutdown()

def run_server_jupyter(layout, **kwargs):
    from jupyter_dash import JupyterDash

    app = JupyterDash(__name__, external_stylesheets=ServerSettings.css)
    app.layout = layout
    app.run_server(mode='inline', **kwargs)

def run_server(layout, mode: ServerTypes = ServerSettings.mode, **kwargs):
    log(f"Server mode: {mode=}")

    if mode in [ServerTypes.DEFAULT, None]:
        run_server_default(layout, **kwargs)
    elif mode == ServerTypes.JUPYTER:
        run_server_jupyter(layout, **kwargs)
    elif mode == ServerTypes.WINDOWED:
        run_server_windowed(layout, **kwargs)
    else:
        raise ValueError(f"Server mode {mode} unknown")
    
    log("Finished visualizing")