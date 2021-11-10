import numpy as np
import os
from scipy.spatial.distance import cdist
from .geoutils import project_vector, rasterize_shapefile_like, get_feature_trace
from .raster import Raster


class DistanceMap(Raster):
    def __init__(self, parent_raster, feature_vector: np.ndarray):
        super().__init__(parent_raster.filename)
        self.feature = feature_vector
        self.data = np.zeros(parent_raster.shape)
        # self.__compute_distancemap()

    def __compute_distancemap(self):
        print("in fnc")
        unraveled = []

        print("beginner iter")
        for x in range(1, self.ncols + 1):
            for y in range(1, self.nrows + 1):
                unraveled.append([x, y])

        print("projecting")
        projected = project_vector(np.array(unraveled), self)
        print("running cdist")
        distance_map = (
            cdist(self.feature, projected).min(axis=0).reshape(self.ncols, self.nrows)
        )
        print("done")

        self.data = np.flipud(np.rot90(distance_map))


def import_refinement_features(parent_raster: Raster, shp_paths: str) -> DistanceMap:
    """
    Imports one or more shapefiles and creates a distance map, relative
    to the DEM `parent_raster`. This distance map can then be used for
    creating a refined surface mesh.
    """

    if isinstance(shp_paths, str):
        shp_paths = [shp_paths]

    master_arr = None

    for shp_path in shp_paths:
        if not os.path.exists(shp_path):
            raise FileNotFoundError(f'Shapefile doesn\'t exist at path "{shp_path}"')

        arr = rasterize_shapefile_like(shp_path, parent_raster.filename)

        if master_arr is None:
            master_arr = arr
        else:
            master_arr[arr == True] = True

    feature = project_vector(
        get_feature_trace(master_arr, feature_threshold=0.5).astype(float),
        parent_raster,
    )

    return DistanceMap(parent_raster, feature)
