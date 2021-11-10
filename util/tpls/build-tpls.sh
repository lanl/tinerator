#!/usr/bin/env bash

# Gets the directory of this script
_script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

TPL_ROOT_DIR="${TPL_ROOT_DIR:-$_script_dir}"

# Basic logging functions - could be extended to log to file
_log() { echo "[$1 - $(date +%H:%M.%S)] $2"; }
info() { _log "INFO" "$1"; }
warn() { _log "WARNING" "$1"; }
debug() { _log "DEBUG" "$1"; }

build_pylagrit() {
    _cwd=$(pwd)
    LAGRIT_SRC_DIR=$1/LaGriT
    LAGRIT_EXE=${LAGRIT_SRC_DIR}/src/lagrit

    info "Building LaGriT: ${LAGRIT_SRC_DIR}"

    git clone --depth 1 https://github.com/lanl/LaGriT.git ${LAGRIT_SRC_DIR}
    cd ${LAGRIT_SRC_DIR}
    make release WITH_EXODUS=0

    # Reduce the repo size significantly
    # TODO: this should actually be changed upstream 
    #   in the LaGriT repo
    rm -rf test/ && rm -rf doc/ && rm -rf .git/

    cd PyLaGriT/ && python setup.py install
    cd $_cwd
}

build_jigsaw() {
    _cwd=$(pwd)
    JIGSAW_SRC_DIR=$1/jigsaw-python
    info "Building JIGSAW: ${JIGSAW_SRC_DIR}"

    git clone https://github.com/dengwirda/jigsaw-python.git ${JIGSAW_SRC_DIR}
    cd ${JIGSAW_SRC_DIR}

    python3 setup.py build_external
    python3 setup.py install

    cd $_cwd
}

build_exodus() {
    _cwd=$(pwd)
    info "Building EXODUSII: ${EXODUS_SRC_DIR}"

    CC="${CC:-$(which gcc)}"
    CXX="${CXX:-$(which g++)}"
    FC="${FC:-$(which gfortran)}"
    EXO_GIT_HASH="${EXO_GIT_HASH:-d55c8ff06dd6875bae5bcbf86db7ea2e190ca901}"

    SEACAS_SRC_DIR=$1/seacas
    SEACAS_BUILD_DIR=${SEACAS_SRC_DIR}/build
    SEACAS_DIR=${SEACAS_SRC_DIR}/install

    TPL_ROOT_DIR="${TPL_ROOT_DIR:-$_script_dir}"

    PYTHON_PREFIX="$(cd $(dirname $(which python))/../; pwd)"
    CONDA_PREFIX="${CONDA_PREFIX:-$PYTHON_PREFIX}"

    info "Building EXODUSII: ${SEACAS_SRC_DIR}"

    git clone https://github.com/gsjaardema/seacas.git ${SEACAS_SRC_DIR}
    cd ${SEACAS_SRC_DIR} && git checkout ${EXO_GIT_HASH}
    mkdir -p $SEACAS_BUILD_DIR && mkdir -p $SEACAS_DIR && cd $SEACAS_BUILD_DIR

    cmake  \
        -D SEACASProj_ENABLE_ALL_PACKAGES:BOOL=OFF \
        -D SEACASProj_ENABLE_SEACASExodus:BOOL=ON \
        -D CMAKE_INSTALL_PREFIX:PATH=${SEACAS_DIR} \
        -D CMAKE_BUILD_TYPE=Debug \
        -D BUILD_SHARED_LIBS:BOOL=ON \
        \
        -D CMAKE_CXX_COMPILER:FILEPATH=${CXX} \
        -D CMAKE_C_COMPILER:FILEPATH=${CC} \
        -D CMAKE_Fortran_COMPILER:FILEPATH=${FC} \
        -D SEACASProj_SKIP_FORTRANCINTERFACE_VERIFY_TEST:BOOL=ON \
        -D TPL_ENABLE_Netcdf:BOOL=ON \
        -D TPL_ENABLE_HDF5:BOOL=ON \
        -D TPL_ENABLE_Matio:BOOL=OFF \
        -D TPL_ENABLE_MPI=OFF \
        -D TPL_ENABLE_CGNS:BOOL=OFF \
        \
        -D Netcdf_LIBRARY_DIRS:PATH=${CONDA_PREFIX}/lib \
        -D Netcdf_INCLUDE_DIRS:PATH=${CONDA_PREFIX}/include \
        -D HDF5_ROOT:PATH=${CONDA_PREFIX} \
        -D HDF5_NO_SYSTEM_PATHS=ON \
    ${SEACAS_SRC_DIR}

    make && make install

    export PYTHONPATH=${SEACAS_SRC_DIR}/install/lib:${PYTHONPATH}

    cd $_cwd
}

help() {
    echo -e "------------------------------"
    echo -e "TINerator TPL bootstrap script"
    echo -e "------------------------------"
    echo -e "usage: ./build-tpls.sh [-h] [-A | -p | -j | -e] [-d path/] [-M]"
    echo -e ""
    echo -e "-h\tShows this help screen"
    echo -e ""
    echo -e "-A\tBuild all TPLs (PyLaGriT, JIGSAW, ExodusII)"
    echo -e "-p\tBuild only PyLaGriT"
    echo -e "-j\tBuild only JIGSAW"
    echo -e "-e\tBuild only ExodusII"
    echo -e ""
    echo -e "-d\tThe path where the TPLs will be built (currently set to: ${TPL_ROOT_DIR})"
    echo -e ""
    echo -e "-M\tWill append lines to ~/.bashrc and ~/.pylagritrc with package environment variables"
    exit 0
}

_should_build_all=false
_should_build_pylagrit=false
_should_build_jigsaw=false
_should_build_exodus=false
_should_modify_files=false


while getopts hApjed:M flag
do
    case "${flag}" in
        h) help;;
        A) _should_build_all=true;;
        p) _should_build_pylagrit=true;;
        j) _should_build_jigsaw=true;;
        e) _should_build_exodus=true;;
        d) TPL_ROOT_DIR=${OPTARG};;
        M) _should_modify_files=true;;
        [?]) help;;
    esac
done

if [ "$_should_build_all" = true ]
then
    _should_build_pylagrit=false
    _should_build_jigsaw=false
    _should_build_exodus=true
fi

if [[ "$_should_build_pylagrit" = false && "$_should_build_jigsaw" = false && "$_should_build_exodus" = false ]]
then
    echo ""
    warn "Not enough flags passed. Nothing will be built!"
    echo ""
    help
fi

info "TINERATOR TPL BOOTSTRAP SCRIPT"
info "=============================="
info "TPL_ROOT_DIR: ${TPL_ROOT_DIR}"
info "Building PyLaGriT? ${_should_build_pylagrit}"
info "Building JIGSAW? ${_should_build_jigsaw}"
info "Building ExodusII? ${_should_build_exodus}"

if [ "$_should_build_pylagrit" = true ]
then
    build_pylagrit ${TPL_ROOT_DIR}

    _rc_line="lagrit_exe : '${TPL_ROOT_DIR}/LaGriT/src/lagrit'"

    if [ "$_should_modify_files" = true ]
    then
        echo "$_rc_line" >> ~/.pylagritrc
    else
        info "Add the following line to your ~/.pylagritrc:"
        info "echo \"$_rc_line\" >> ~/.pylagritrc"
    fi
fi

if [ "$_should_build_jigsaw" = true ]
then
    build_jigsaw ${TPL_ROOT_DIR}
fi

if [ "$_should_build_exodus" = true ]
then
    build_exodus ${TPL_ROOT_DIR}

    _rc_line="${TPL_ROOT_DIR}/seacas/install/lib/"
    _rc_line="export PYTHONPATH=$_rc_line"':${PYTHONPATH}'

    if [ "$_should_modify_files" = true ]
    then
        echo "$_rc_line" >> ~/.bashrc
    else
        info "Add the following line to your ~/.bashrc:"
        info "echo \"$_rc_line\" >> ~/.bashrc"
    fi
fi
