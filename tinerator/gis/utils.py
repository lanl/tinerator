import numpy as np
from scipy import interpolate

def map_elevation(dem, nodes: np.ndarray) -> np.ndarray:
    '''
    Maps elevation from a DEM raster to mesh nodes.
    '''

    array = dem.masked_data()

    # --- BEGIN INTERPOLATING DEM DATA ---- #
    # This is done to keep the z_value indexing from landing on 
    # NaNs.
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    data = interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                             method='nearest')

    # --- END INTERPOLATING DEM DATA ---- #

    n_nodes = nodes.shape[0]
    z_array = np.zeros((n_nodes,),dtype=float)

    indices = unproject_vector(nodes, dem)

    x_idx = indices[:,0]
    y_idx = indices[:,1]

    for i in range(n_nodes):
        z_array[i] = data[y_idx[i]][x_idx[i]]

    return z_array

def unproject_vector(vector: np.ndarray, raster) -> np.ndarray:
    '''
    Converts a vector of (x,y) point in a particular raster's CRS back into
    [row, col] indices relative to that raster.
    '''

    # TODO: verify that `vector == unproject_vector(project_vector(vector))`

    nNodes = vector.shape[0]
    xllCorner = raster.xll_corner
    yllCorner = raster.yll_corner
    cellSize = raster.cell_size
    nRows = raster.nrows

    map_x = lambda x: (cellSize + 2. * float(x) - 2. * xllCorner) / (2. * cellSize)
    map_y = lambda y: ((yllCorner - y) / cellSize + nRows + 1./2.)
    
    x_arr = np.reshape(list(map(map_x, vector[:,0])), (nNodes, 1))
    y_arr = np.reshape(list(map(map_y, vector[:,1])), (nNodes, 1))

    return np.hstack((np.round(x_arr), np.round(y_arr))).astype(int) - 1

def project_vector(vector: np.ndarray, raster) -> np.ndarray:
    '''
    Because every raster has a CRS projection, associated indices
    in that raster can be projected into that coordinate space.

    For example, imagine a DEM. The pixel at index [0,0] corresponds to
    (xll_corner, yll_corner).
    '''

    # TODO: something is (slightly) wrong with this calculation

    nNodes = vector.shape[0]
    xllCorner = raster.xll_corner
    yllCorner = raster.yll_corner
    cellSize = raster.cell_size
    nRows = raster.nrows

    map_x = lambda x: (xllCorner + (float(x) * cellSize) - (cellSize / 2.))
    map_y = lambda y: (yllCorner + (float(0. - y + float(nRows)) * cellSize) - (cellSize / 2.))
    
    x_arr = np.reshape(list(map(map_x, vector[:,0])), (nNodes, 1))
    y_arr = np.reshape(list(map(map_y, vector[:,1])), (nNodes, 1))

    if vector.shape[1] > 2:
        return np.hstack((x_arr, y_arr, np.reshape(vector[:,2], (nNodes, 1))))
    
    return np.hstack((x_arr, y_arr))
