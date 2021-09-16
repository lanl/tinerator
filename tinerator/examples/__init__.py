import os

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


class new_mexico:
    dem = os.path.join(
        dir_path, "new_mexico", "rasters", "USGS_NED_13_n37w108_Clipped.tif"
    )
    flowline = os.path.join(
        dir_path, "new_mexico", "shapefiles", "NHDFlowline", "NHDFlowline.shp"
    )
    boundary = os.path.join(
        dir_path, "new_mexico", "shapefiles", "WBDHU12", "WBDHU12.shp"
    )


class borden:
    dem_50cm = os.path.join(dir_path, "borden", "dem0.5m.dat")
    dem_100cm = os.path.join(dir_path, "borden", "dem1m.dat")
