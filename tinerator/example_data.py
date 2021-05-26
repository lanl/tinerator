import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "example-data")


class ExampleData:
    class Simple:
        """A very simple collection of meshes, used primarily for testing."""

        root_dir = os.path.join(DATA_DIR, "meshes", "Surfaces")
        surface_mesh = os.path.join(
            DATA_DIR, "meshes", "Surfaces", "simple_surface.inp"
        )
        volume_mesh = os.path.join(DATA_DIR, "meshes", "Surfaces", "simple_volume.inp")
        exodus_mesh = os.path.join(
            DATA_DIR, "meshes", "Surfaces", "simple_volume_with_facesets.exo"
        )

    class BordonDEM:
        """Simple DEM example - Bordon dataset."""
        root_dir = os.path.join(DATA_DIR, "BordenDEM")
        dem_1m = os.path.join(root_dir, "dem1m.dat")
        dem_05m = os.path.join(root_dir, "dem0.5m.dat")

    class RadialMesh:
        """A simple radial surface mesh."""

        root_dir = os.path.join(DATA_DIR, "radial_mesh")

        # layers =

    class NewMexico:
        """GIS data for a HUC10 in New Mexico."""

        root_dir = os.path.join(DATA_DIR, "GIS", "NewMexico-HU8-14080103")
        dem = os.path.join(
            DATA_DIR,
            "GIS",
            "NewMexico-HU8-14080103",
            "rasters",
            "USGS_NED_13_n37w108_Clipped.tif",
        )
        watershed_boundary = os.path.join(
            DATA_DIR,
            "GIS",
            "NewMexico-HU8-14080103",
            "shapefiles",
            "WBDHU12",
            "WBDHU12.shp",
        )
        flowline = os.path.join(
            DATA_DIR,
            "GIS",
            "NewMexico-HU8-14080103",
            "shapefiles",
            "NHDFlowline",
            "NHDFlowline.shp",
        )
