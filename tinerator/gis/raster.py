import richdem as rd
import numpy as np
from pyproj import CRS
from pyproj.crs import CRSError


class Raster:
    def __init__(self, raster_path: str, no_data: float = None):
        self.data = rd.LoadGDAL(raster_path, no_data=no_data)
        self.no_data_value = self.data.no_data
        self.cell_size = self.data.geotransform[1]
        self.xll_corner = self.data.geotransform[0]
        self.yll_corner = (
            self.data.geotransform[3] - self.nrows * self.cell_size
        )

        try:
            self.crs = CRS.from_wkt(self.data.projection)
        except CRSError:
            self.crs = CRS.from_epsg(32601)

    def __repr__(self):
        # DEM information
        display = "CRS\t\t: %s\n" % self.crs.name
        display += "Extent\t\t: %s\n" % repr(self.extent)
        display += "Cell size\t: %s\n" % repr(self.cell_size)
        display += "Units\t\t: %s\n" % self.crs.axis_info[0].unit_name
        display += "Dimensions\t: %s\n" % repr((self.nrows, self.ncols))
        display += "Value range\t: %s\n" % repr((-1,-1))

        return display

    @property
    def mask(self):
        return self.data[self.data == self.no_data_value]

    @property
    def origin(self):
        return (self.xll_corner, self.yll_corner)
    
    @origin.setter
    def origin(self, v):
        self.xll_corner = v[0]
        self.yll_corner = v[1]

    @property
    def ncols(self):
        return self.data.shape[1]

    @property
    def nrows(self):
        return self.data.shape[0]

    @property
    def extent(self):
        return (
            self.xll_corner,
            self.yll_corner,
            self.ncols * self.cell_size + self.xll_corner,
            self.nrows * self.cell_size + self.yll_corner,
        )
