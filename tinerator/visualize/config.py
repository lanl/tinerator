from enum import Enum, auto
from dash import Dash
import dash_bootstrap_components as dbc

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

def run_server_default(layout, **kwargs):
    app = Dash(__name__, external_stylesheets=ServerSettings.css)
    app.layout = layout
    app.run_server()

def run_server_windowed(layout, host, port, **kwargs):
    from threading import Thread
    from retrying import retry

    from PyQt5.QtCore import QUrl
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtWidgets import QApplication

    url = f"{host}:${port}/"

    #if not url.lower().strip().startswith('http'):
    #    url = f"http://{url}"

    app = Dash(__name__, external_stylesheets=ServerSettings.css)
    app.layout = layout

    qt_app = QApplication(["TINerator",])
    webview = QWebEngineView()

    @retry(
        stop_max_attempt_number=15,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000
    )
    def run():
        app.run_server(host=host, port=port, **kwargs)

    thread = Thread(target=run)
    thread.setDaemon(True)
    thread.start()

    # Override shutdown to kill when the _shutdown_{token} endpoint is reached
    """
        # Register route to shut down server
        @self.server.route('/_shutdown_' + JupyterDash._token, methods=['GET'])
        def shutdown():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                raise RuntimeError('Not running with the Werkzeug Server')
            func()
            return 'Server shutting down...'

        # Register route that we can use to poll to see when server is running
        @self.server.route('/_alive_' + JupyterDash._token, methods=['GET'])
        def alive():
            return 'Alive'
    """

    # Kill other servers running
    """
        # Terminate any existing server using this port
        self._terminate_server_for_port(host, port)

    def _terminate_server_for_port(cls, host, port):
        shutdown_url = "http://{host}:{port}/_shutdown_{token}".format(
            host=host, port=port, token=JupyterDash._token
        )
        try:
            response = requests.get(shutdown_url)
        except Exception as e:
            pass
    """

    # Checks the _alive_{token} endpoint to see if the server is still running
    # TODO: should probably be done as a general thing
    """
    # Wait for server to start up
    alive_url = "http://{host}:{port}/_alive_{token}".format(
        host=host, port=port, token=JupyterDash._token
    )

    # Wait for app to respond to _alive endpoint
    @retry(
        stop_max_attempt_number=15,
        wait_exponential_multiplier=10,
        wait_exponential_max=1000
    )
    def wait_for_app():
        res = requests.get(alive_url).content.decode()
        if res != "Alive":
            url = "http://{host}:{port}".format(
                host=host, port=port, token=JupyterDash._token
            )
            raise OSError(
                "Address '{url}' already in use.\n"
                "    Try passing a different port to run_server.".format(
                    url=url
                )
            )

    wait_for_app()
    """

    webview.load(QUrl(url))
    webview.show()
    qt_app.exec_()

    if thread.isAlive:
        thread.join()

def run_server_jupyter(layout, **kwargs):
    from jupyter_dash import JupyterDash

    app = JupyterDash(__name__, external_stylesheets=ServerSettings.css)
    app.layout = layout
    app.run_server(mode='inline')

def run_server(layout, **kwargs):
    if ServerSettings.mode in [ServerTypes.DEFAULT, None]:
        run_server_default(layout, **kwargs)
    elif ServerSettings.mode == ServerTypes.JUPYTER:
        run_server_jupyter(layout, **kwargs)
    elif ServerSettings.mode == ServerTypes.WINDOWED:
        run_server_windowed(layout, **kwargs)
    else:
        raise ValueError(f"Server mode {ServerSettings.mode} unknown")
    