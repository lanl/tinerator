"""
Helper functions for configuring
TINerator for proper Docker functionality.
"""
import os
from ..constants import DOCKER_PARAM_PYVISTA_BACKEND, DOCKER_PARAM_ENVIRONMENT_CHECK


def in_docker() -> bool:
    """
    Returns `True` if TINerator is being
    executed within the "official" Docker container
    and `False` otherwise.

    The official Docker container defines a 'special'
    variable that this function checks for.

    That variable is reflected in the
    ``DOCKER_PARAM_ENVIRONMENT_CHECK`` Python variable
    from ``tinerator.util.constants``.
    """

    # This dictionary only contains one environment
    # variable to check. It could be extended to check
    # for more, as distribution complexity increases.
    for item in DOCKER_PARAM_ENVIRONMENT_CHECK.items():
        try:
            if os.environ[item[0]] == item[1]:
                return True
        except KeyError:
            continue

    return False


def configure_pyvista_for_docker() -> None:
    """
    Configures PyVista to work in a
    headless, remote host - like Docker.
    """
    import pyvista

    pyvista.set_jupyter_backend(DOCKER_PARAM_PYVISTA_BACKEND)
    pyvista.start_xvfb()


def init() -> bool:
    """
    Configures TINerator for use in Docker.

    Returns
    -------
        success (bool): if configuration was successful
    """

    if in_docker():
        configure_pyvista()

    return True
