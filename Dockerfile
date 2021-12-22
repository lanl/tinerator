FROM ubuntu:20.04
LABEL Description="TINerator"

ARG container_user=feynman

EXPOSE 8899

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    PYTHONDONTWRITEBYTECODE=true \
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

# Build Tini process manager ======================== #
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]
# =================================================== #

# === Add an unprivileged user and group ============ #
RUN groupadd -r $container_user && \
    useradd -r -m -g $container_user $container_user
# =================================================== #

# === Install & configure Miniconda ================= #
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
# =================================================== #
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    chown -R $container_user /opt/ && \
    chgrp -R $container_user /opt/
# =================================================== #
USER $container_user
ENV HOME=/home/$container_user
# =================================================== #
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda config --add channels conda-forge && \
    conda update -n base -c defaults conda && \
    conda install -y \
        pip \
        gdal \
        "nodejs>=12.0" && \
    conda clean -afy

# === Configure Jupyter Lab settings ================ #
RUN mkdir ~/.jupyter && \
    jupyter_cfg=~/.jupyter/jupyter_notebook_config.py && \
    echo "c.JupyterApp.config_file = ''" >> $jupyter_cfg && \
    echo "c.NotebookApp.allow_root = True" >> $jupyter_cfg && \
    echo "c.NotebookApp.allow_remote_access = True" >> $jupyter_cfg && \
    echo "c.NotebookApp.ip = '*'" >> $jupyter_cfg && \
    echo "c.NotebookApp.terminado_settings = { \"shell_command\": [\"/usr/bin/bash\"] }" >> $jupyter_cfg && \
    echo "c.NotebookApp.token = u''" >> $jupyter_cfg

RUN conda install -y jupyterlab && \
    jupyter-lab build
# =================================================== #

WORKDIR $HOME/tinerator
COPY . .
RUN python -m pip install .
RUN cd util/tpls && ./build-tpls.sh -e -M && . ~/.bashrc

CMD ["jupyter", "lab", "--port=8899"]