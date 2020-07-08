import numpy as np
from scipy.spatial.distance import cdist
from .utils import project_vector
from .raster import Raster

def get_distance_map_from_points(points, raster):
    unraveled = []

    for x in range(1, raster.ncols + 1):
        for y in range(1, raster.nrows + 1):
            unraveled.append([x,y])

    projected = project_vector(np.array(unraveled), raster)
    distance_map = cdist(points, projected).min(axis=0).reshape(raster.ncols, raster.nrows)

    return np.flipud(np.rot90(distance_map))

'''
class DistanceMap(Raster):
    def __init__(self, points, parent):
'''