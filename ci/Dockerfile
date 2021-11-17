FROM ubuntu:20.04
LABEL Description="TINerator Docker container"

EXPOSE 8050-8100
EXPOSE 8899
ARG container_user=feynman

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    PYTHONDONTWRITEBYTECODE=true \
    https_proxy=$https_proxy \
    http_proxy=$http_proxy \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install --no-install-recommends -y \

    # Shared
    build-essential \
    apt-utils \
    ssh-client \
    patch \
    g++ \
    gfortran \
    git \
    make \
    cmake \
    emacs \
    vim \
    less \
    curl \
    wget \
    libz-dev \
    openssl \
    m4 \
    bzip2 \
    ca-certificates \

    # TINerator/Watershed Workflow
    unzip \
    bison \
    libgl1-mesa-glx \
    sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =================================================== #
# Build Miniconda =================================== #
# =================================================== #
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]

# Add an unprivileged user and group
RUN groupadd -r $container_user && \
    useradd -r -m -g $container_user $container_user

RUN /bin/bash -c 'set -ex && \
    ARCH=`uname -m` && \
    MINICONDA_URL="" && \
    if [[ "$ARCH" == "x86_64" ]]; then \
        echo "Downloading miniconda: x86_64" && \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh"; \
    elif [[ "$ARCH" == "arm64" ]]; then \
        echo "Downloading miniconda: aarch64" && \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-aarch64.sh"; \
    else \
        echo "unknown arch: default x86" && \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh"; \
    fi && \
    wget --quiet "$MINICONDA_URL" -O ~/miniconda.sh'

RUN /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    chown -R $container_user /opt/ && \
    chgrp -R $container_user /opt/

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

USER $container_user
ENV HOME=/home/$container_user

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda update -n base -c defaults conda && \
    conda install -y pip && \
    conda install -c conda-forge gdal && \
    conda clean -afy

# =================================================== #
# Build TINerator =================================== #
# =================================================== #
RUN git clone https://github.com/lanl/tinerator.git $HOME/tinerator && \
    cd $HOME/tinerator/ && \
    poetry shell && \
    poetry install
    #python -m pip install $HOME/tinerator/

RUN python -c "import tinerator"

#RUN conda install -c conda-forge tinerator

# Configure Jupyter Notebook/Jupyter Lab settings
RUN mkdir ~/.jupyter && \
    jupyter_cfg=~/.jupyter/jupyter_notebook_config.py && \
    echo "c.JupyterApp.config_file = ''" >> $jupyter_cfg && \
    echo "c.NotebookApp.allow_root = True" >> $jupyter_cfg && \
    echo "c.NotebookApp.allow_remote_access = True" >> $jupyter_cfg && \
    echo "c.NotebookApp.ip = '*'" >> $jupyter_cfg && \
    echo "c.NotebookApp.terminado_settings = { \"shell_command\": [\"/usr/bin/bash\"] }" >> $jupyter_cfg && \
    echo "c.NotebookApp.token = u''" >> $jupyter_cfg

RUN jupyter-lab build

#CMD ["jupyter", "lab", "--port=8899"]


CMD ["bash"]