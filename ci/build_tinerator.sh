echo "Current directory: $(pwd)"
echo "Python information: $(python --version); $(which python)"

#conda install -y \
#  pip \
#  gdal \
#  fiona \
#  shapely

# Install TINerator
python -m pip install .

# Install TPLs
cd util/tpls/ && ./build-tpls.sh -e -M && source ~/.bashrc