import tinerator as tin

data = tin.examples.new_mexico

dem = tin.gis.load_raster(data.dem)
flowline = tin.gis.load_shapefile(data.flowline)
boundary = tin.gis.load_shapefile(data.boundary)

#tin.plot2d([dem, flowline, boundary], mapbox_style="stamen-watercolor")
tin.plot2d([flowline, boundary], mapbox_style="stamen-watercolor")

dem = tin.gis.clip_raster(dem, boundary)
dem.plot()

mesh = tin.meshing.triangulate(
    dem,
    min_edge_length=0.01,
    max_edge_length=0.05,
    refinement_feature=flowline,
    method="jigsaw",
    scaling_type="relative",
)

mesh.plot()
