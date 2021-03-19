import tinerator as tin
from tinerator import ExampleData

NM = ExampleData.NewMexico

dem = tin.gis.load_raster(NM.dem)
#dem.plot()

boundary = tin.gis.load_shapefile(NM.watershed_boundary)
#boundary.plot(layers=[dem])

dem = tin.gis.clip_raster(dem, boundary)
#dem.plot()

flowlines = tin.gis.load_shapefile(NM.flowline)
flowlines.plot(layers=[dem])

wsd = tin.gis.watershed_delineation(dem)
#wsd.plot(layers=[dem])

wsd = tin.gis.watershed_delineation(dem, threshold=200.)
wsd.plot(layers=[dem])