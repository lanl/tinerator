import dash
import os
import requests
from flask import request
from threading import Thread
from retrying import retry
import uuid


class DashTIN(dash.Dash):
    _token = str(uuid.uuid4())

    def __init__(
        self, name="TINerator", title: str = "TINerator: Data Visualization", **kwargs
    ):
        """
        This is based heavily off of the JupyterDash subclass.
        It is needed to do async serving + being able to kill the server
        at any time.

        Args:
            name (str, optional): Name. Defaults to None.
        """
        super(DashTIN, self).__init__(name=name, title=title, **kwargs)

        # Register route to shut down server
        @self.server.route("/_shutdown_" + DashTIN._token, methods=["GET"])
        def shutdown():
            func = request.environ.get("werkzeug.server.shutdown")
            if func is None:
                raise RuntimeError("Not running with the Werkzeug Server")
            func()
            return "Server shutting down..."

        # Register route that we can use to poll to see when server is running
        @self.server.route("/_alive_" + DashTIN._token, methods=["GET"])
        def alive():
            return "Alive"

        self.server.logger.disabled = False

        # @self.server.route("/_isloaded_" + DashTIN._token, methods=["GET"])
        # def loaded():
        #    return "Loaded"

    def run_server(self, **kwargs):
        super_run_server = super(DashTIN, self).run_server

        # Get host and port
        host = kwargs.get("host", os.getenv("HOST", "127.0.0.1"))
        port = kwargs.get("port", os.getenv("PORT", "8050"))

        kwargs["host"] = host
        kwargs["port"] = port

        # Terminate any existing server using this port
        self._terminate_server_for_port(host, port)

        # Compute server_url url
        self.server_url = "http://{host}:{port}".format(host=host, port=port)

        @retry(
            stop_max_attempt_number=15,
            wait_exponential_multiplier=100,
            wait_exponential_max=1000,
        )
        def run():
            super_run_server(**kwargs)

        thread = Thread(target=run)
        thread.setDaemon(True)
        thread.start()

        # Wait for server to start up
        alive_url = "http://{host}:{port}/_alive_{token}".format(
            host=host, port=port, token=DashTIN._token
        )

        # Wait for app to respond to _alive endpoint
        @retry(
            stop_max_attempt_number=15,
            wait_exponential_multiplier=10,
            wait_exponential_max=1000,
        )
        def wait_for_app():
            res = requests.get(alive_url).content.decode()
            if res != "Alive":
                url = "http://{host}:{port}".format(
                    host=host, port=port, token=DashTIN._token
                )
                raise OSError(
                    "Address '{url}' already in use.\n"
                    "    Try passing a different port to run_server.".format(url=url)
                )

        wait_for_app()

    def getServerURL(self):
        return self.server_url

    def shutdown(self):
        shutdown_url = f"{self.getServerURL()}/_shutdown_{DashTIN._token}"

        try:
            response = requests.get(shutdown_url)
        except Exception:
            pass
    
    def status(self):
        return request.get_json()

    @classmethod
    def _terminate_server_for_port(cls, host, port):
        shutdown_url = "http://{host}:{port}/_shutdown_{token}".format(
            host=host, port=port, token=DashTIN._token
        )
        try:
            response = requests.get(shutdown_url)
        except Exception as e:
            pass


def find_open_port(host: str, port_start: int, max_iter: int = 300) -> int:
    port = int(port_start)

    for _ in range(max_iter):
        url = "http://{host}:{port}/_alive_{token}".format(
            host=host, port=port, token=DashTIN._token
        )
        try:
            res = requests.get(url).content.decode()
            if res != "Alive":
                return port
        except Exception:
            return port

        port += 1

    raise ValueError(
        "Could not find an open port (checked ports in range: [{port_start}, {port}]"
    )
