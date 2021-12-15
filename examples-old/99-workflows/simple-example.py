import tinerator as tin

# data = tin.examples.new_mexico

# dem = tin.gis.load_raster(data.dem)
# flowline = tin.gis.load_shapefile(data.flowline)
# boundary = tin.gis.load_shapefile(data.boundary)

nm = tin.examples.NewMexico()

# tin.plot2d([dem, flowline, boundary], mapbox_style="stamen-watercolor")
tin.plot2d([nm.flowline, nm.boundary], mapbox_style="stamen-watercolor")

dem = tin.gis.clip_raster(nm.dem, nm.boundary)
dem.plot()

mesh = tin.meshing.triangulate(
    dem,
    min_edge_length=0.01,
    max_edge_length=0.05,
    refinement_feature=nm.flowline,
    method="jigsaw",
    scaling_type="relative",
)

mesh.plot()
