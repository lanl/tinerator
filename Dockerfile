# https://betterprogramming.pub/super-slim-docker-containers-fdaddc47e560


FROM ubuntu:20.04
LABEL maintainer="Daniel Livingston <livingston@lanl.gov>"
WORKDIR /tinerator

# Expose the Jupyter notebook port
EXPOSE 8888

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

COPY . .

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y \
    build-essential openssl vim gfortran cmake git \
    wget libz-dev m4 bison r-base  \
    software-properties-common curl \
    python3 python3-pip python3-setuptools && \
    \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    \
    add-apt-repository -y ppa:ubuntugis/ppa && apt-get update -y && \
    apt-get install -y gdal-bin libgdal-dev && \ 
    export CPLUS_INCLUDE_PATH=/usr/include/gdal && export C_INCLUDE_PATH=/usr/include/gdal && \
    pip install GDAL==`gdal-config --version`

# Build Python packages
RUN pip install -r requirements.txt

# Configure Jupyter Notebook/Jupyter Lab
RUN mkdir ~/.jupyter && \
    jupyter_cfg=~/.jupyter/jupyter_notebook_config.py && \
    echo "c.JupyterApp.config_file = ''" >> $jupyter_cfg && \
    echo "c.NotebookApp.allow_root = True" >> $jupyter_cfg && \
    echo "c.NotebookApp.allow_remote_access = True" >> $jupyter_cfg && \
    echo "c.NotebookApp.ip = '*'" >> $jupyter_cfg && \
    echo "c.NotebookApp.token = u''" >> $jupyter_cfg

# Build all TPLs
RUN ./tpls/build-tpls.sh -A -M

# Test TINerator
#RUN cd /tinerator/tests && \
#    pytest

WORKDIR /tinerator/playground/

# Launch Jupyter Lab on start
CMD jupyter lab --port=8888