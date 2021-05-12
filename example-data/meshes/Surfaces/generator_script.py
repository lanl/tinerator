import os
import tinerator as tin

print(tin.VERSION)

os.chdir("/home/jovyan/work/")
print(os.listdir("."))
my_dem = tin.load.from_file("/home/jovyan/examples/data/dem.asc")
my_dem.set_verbosity(tin.cfg.DEBUG)
my_dem._surface_mesh = my_dem.lg.read("simple_surface.inp")
my_dem.lg.sendline("cmo/setatt/%s/itetclr/1,0,0/1" % my_dem._surface_mesh.name)
my_dem.lg.sendline("dump/avs/simple_surface_fixed.inp/%s" % my_dem._surface_mesh.name)

depths = [0.1, 0.3, 0.2, 0.1, 0.4]

my_dem.build_layered_mesh(depths, matids=[1, 2, 3, 3, 4])
my_dem._stacked_mesh.dump("simple_volume.inp")

fs_basic = tin.facesets.basic(has_top=True, has_bottom=True, has_sides=True)
tin.dump.to_exodus(my_dem, "simple_volume_with_facesets.exo", facesets=fs_basic)
