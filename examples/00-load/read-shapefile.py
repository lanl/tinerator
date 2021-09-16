import tinerator as tin
from tinerator import examples

watershed_boundary = examples.new_mexico.boundary
print(f"Loading shapefile from filename: {watershed_boundary}")

# Pass in the filename to read the raster
boundary = tin.gis.load_shapefile(watershed_boundary)

print("Shapefile properties:")
print(boundary)

# Available class methods for a Raster object:
print(vars(boundary))

# Visualize it using one of:
# - boundary.plot()
# - tin.plot2d(boundary)
boundary.plot()