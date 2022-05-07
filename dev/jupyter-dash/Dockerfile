FROM ubuntu:20.04
LABEL vendor="Los Alamos National Laboratory" \
      version="0.0.1-beta" \
      release-date="2022-05-07" \
      description="JupyterDash Docker sanity test"

# ==== BUILD ARGUMENTS ============================================ #
# Default container user account
ARG container_user=tin_user
# Port that JupyterLab will use
ARG PORT_JUPYTER=8899
# Port that JupyterDash will use
ARG PORT_DASH=8050
# Conda build architecture (x86_64 or aarch64)
ARG CONDA_ARCH=x86_64

# ==== SYSTEM CONFIGURATION ======================================= #
COPY . /tmp/docker/

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    http_proxy=$http_proxy \
    https_proxy=$https_proxy \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        curl \
        sudo \
        # - tini is installed as a helpful container entrypoint that reaps zombie
        #   processes and such of the actual executable we want to start, see
        #   https://github.com/krallin/tini#why-tini for details.
        tini \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ==== USER CONFIGURATION ========================================= #
# Add an unprivileged user and group
RUN groupadd -r $container_user && \
    useradd -r -m -g $container_user $container_user

USER $container_user

# ==== MINICONDA CONFIGURATION ==================================== #
# Download & install Miniconda
RUN curl -o /tmp/miniconda.sh -sSf "https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-${CONDA_ARCH}.sh" && \
    bash /tmp/miniconda.sh -b -p /home/$container_user/miniconda && \
    # The '-b' flag doesn't modify ~/.bashrc: do that now
    /home/$container_user/miniconda/bin/conda init bash && \
    # Activate conda
    . /home/$container_user/.bashrc && \
    # Create the conda environment with the copied YAML file
    conda env create --file environment.yml && \
    # Change the on-launch environment
    echo "conda activate tinerator" >> /home/$container_user/.bashrc

# ==== CONTAINER LAUNCH CONFIGURATION ============================= #
EXPOSE ${PORT_JUPYTER} ${PORT_DASH}

# HEALTHCHECK documentation: https://docs.docker.com/engine/reference/builder/#healthcheck
# This healtcheck works well for `lab`, `notebook`, `nbclassic`, `server` and `retro` jupyter commands
# https://github.com/jupyter/docker-stacks/issues/915#issuecomment-1068528799
HEALTHCHECK  --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -O- --no-verbose --tries=1 http://localhost:${PORT_JUPYTER}{JUPYTERHUB_SERVICE_PREFIX:-/}api || exit 1

ENTRYPOINT [ "tini", "-g", "--", \
             "jupyter", "lab", \
                "--ip=0.0.0.0", \
                "-y", \
                "--log-level='INFO'", \
                "--port=${PORT_JUPYTER}" ]
WORKDIR "${HOME}"
