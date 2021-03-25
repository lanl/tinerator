import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "example-data")

class ExampleData:
    class Simple:
        '''A very simple collection of meshes, used primarily for testing.'''
        surface_mesh = os.path.join(DATA_DIR, "meshes", "Surfaces", "simple_surface.inp")
        volume_mesh = os.path.join(DATA_DIR, "meshes", "Surfaces", "simple_volume.inp")
        exodus_mesh = os.path.join(DATA_DIR, "meshes", "Surfaces", "simple_volume_with_facesets.exo")
    class NewMexico:
        '''GIS data for a HUC10 in New Mexico.'''
        root_dir = os.path.join(DATA_DIR, "GIS", "NewMexico-HU8-14080103")
        dem = os.path.join(DATA_DIR, "GIS", "NewMexico-HU8-14080103", "rasters", "USGS_NED_13_n37w108_Clipped.tif")
        watershed_boundary = os.path.join(DATA_DIR, "GIS", "NewMexico-HU8-14080103", "shapefiles", "WBDHU12", "WBDHU12.shp")
        flowline = os.path.join(DATA_DIR, "GIS", "NewMexico-HU8-14080103", "shapefiles", "NHDFlowline", "NHDFlowline.shp")