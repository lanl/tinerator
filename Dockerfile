FROM ubuntu:20.04
LABEL Description="TINerator Docker container"

EXPOSE 8050-8100
EXPOSE 8899
ARG container_user=docker_user

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
    apt-utils \
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

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    chown -R $container_user /opt/ && \
    chgrp -R $container_user /opt/

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
#RUN pip install tinerator
RUN git clone git@github.com:lanl/tinerator.git "$(pwd)/tinerator/" && \
    python -m pip install "$(pwd)/tinerator/"

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

CMD ["jupyter", "lab", "--port=8899"]
