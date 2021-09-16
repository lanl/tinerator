import tinerator as tin

data = tin.ExampleData.NewMexico

dem = tin.gis.load_raster(data.dem)
flowline = tin.gis.load_shapefile(data.flowline)
boundary = tin.gis.load_shapefile(data.watershed_boundary)

# tin.plot2d([dem, flowline, boundary], mapbox_style="stamen-watercolor")

dem = tin.gis.clip_raster(dem, boundary)

mesh = tin.meshing.triangulate(
    dem,
    min_edge_length=0.01,
    max_edge_length=0.05,
    refinement_feature=flowline,
    method="jigsaw",
    scaling_type="relative",
)

mesh.plot()
