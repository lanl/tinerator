from .docker import init as init_docker


def init() -> None:
    """
    Initializes TINerator in response
    to the host OS/environment.
    """

    _ = init_docker()
