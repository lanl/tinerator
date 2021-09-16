import tinerator as tin
from tinerator import examples

dem_filename = examples.new_mexico.dem
print(f"Loading DEM from filename: {dem_filename}")

# Pass in the filename to read the raster
dem = tin.gis.load_raster(dem_filename)
print(dem)

# Available class methods for a Raster object:
print(vars(dem))

# Visualize it using one of:
# - dem.plot()
# - tin.plot2d(dem)
dem.plot()