from copy import deepcopy
from .lagrit_helper import *
from .mesh import read_avs
from ..gis import Raster, DistanceMap



def build_refined_triplane(
    dem_raster: Raster,
    distance_map: DistanceMap,
    min_edge_length: float,
    max_edge_length: float,
    delta: float = 0.75,
    slope: float = 2.0,
    refine_dist: float = 0.5,
    verbose: bool = False,
):
    #boundary:np.ndarray,
    #feature:np.ndarray,
    #h:float,
    #connectivity:bool=None,
    #delta:float=0.75,
    #slope:float=2.,
    #refine_dist:float=0.5,
    #outfile:str=None):
    '''
    Constructs a triplane mesh refined around a feature using LaGriT
    as a backend.
    
    Requires an Nx2 np.ndarray as a boundary input, and an Nx2 np.ndarray
    as a feature input.
    '''

    slope = 2
    refine_dist = 0.5
    smooth_boundary = False
    boundary_distance = None
    connectivity = None

    from pylagrit import PyLaGriT

    lg = PyLaGriT()
    boundary_distance = max_edge_length
    #self._generate_boundary(boundary_distance,rectangular=rectangular_boundary)
    a = min_edge_length*1.75
    h = min_edge_length*1.75
    #feature = deepcopy(self.feature)
    #feature = util.filter_points(feature,a)

    boundary, _ = dem_raster.get_boundary(distance=boundary_distance)
    feature = filter_points(deepcopy(distance_map.feature), a)

    print('--> h = ', h)
    print('--> delta = ', delta)
    print('--> slope = ', slope)
    print('--> refine_dist = ', refine_dist)

    # Define initial parameters
    counterclockwise = False
    h_eps = h*10**-7
    PARAM_A = slope
    PARAM_B = h*(1-slope*refine_dist)
    PARAM_A2 = 0.5*slope
    PARAM_B2 = h*(1 - 0.5*slope*refine_dist)

    if connectivity is None:
        connectivity = line_connectivity(boundary)

    #cfg.log.debug('Writing boundary to poly_1.inp')
    write_line(boundary,"poly_1.inp",connections=connectivity)

    #cfg.log.debug('Writing feature to intersections_1.inp')
    write_line(feature,"intersections_1.inp")

    # Write massage macros
    with open('user_function.lgi','w') as f:
        f.write(Infiles.distance_field)

    with open('user_function2.lgi','w') as f:
        f.write(Infiles.distance_field_2)

    #cfg.log.info('Preparing feature and boundary')

    # Read boundary and feature
    mo_poly_work = lg.read('poly_1.inp',name='mo_poly_work')
    mo_line_work = lg.read('intersections_1.inp',name='mo_line_work')

    # Triangulate Fracture without point addition
    mo_pts = mo_poly_work.copypts(elem_type='triplane')
    mo_pts.select()

    #cfg.log.info('First pass triangulation')

    if counterclockwise:
        mo_pts.triangulate(order='counterclockwise')
    else:
        mo_pts.triangulate(order='clockwise')

    # Set element attributes for later use
    mo_pts.setatt('imt',1,stride=(1,0,0))
    mo_pts.setatt('itetclr',1,stride=(1,0,0))
    mo_pts.resetpts_itp()
    
    mo_pts.select()

    # Refine at increasingly smaller distances, approaching h
    for (i,ln) in enumerate([8,16,32,64][::-1]):
        #cfg.log.info('Refining at length %s' % str(ln))

        h_scale = ln*h
        perturb = h_scale*0.05

        mo_pts.massage(h_scale,h_eps,h_eps)
        
        # Do a bit of smoothing on the first pass
        if (i == 0):
            for _ in range(3):
                mo_pts.recon(0)
                mo_pts.smooth()
            mo_pts.recon(0)

        mo_pts.resetpts_itp()

        #p_move = mo_pts.pset_attribute('itp',0,comparison='eq',stride=(1,0,0),name='p_move')
        #p_move.perturb(perturb,perturb.format(ln),0.0)

        # Smooth and reconnect
        for _ in range(6):
            mo_pts.recon(0)
            mo_pts.smooth()
        mo_pts.recon(0)

    # Define attributes to be used for massage functions
    mo_pts.addatt('x_four',vtype='vdouble',rank='scalar',length='nnodes')
    mo_pts.addatt('fac_n', vtype='vdouble',rank='scalar',length='nnodes')

    # Define internal variables for user_functions
    lg.define(mo_pts=mo_pts.name,
              PARAM_A=PARAM_A,
              PARAM_A2=PARAM_A2,
              PARAM_B=PARAM_B,
              PARAM_B2=PARAM_B2)

    #cfg.log.info('Smoothing mesh (1/2)')

    # Run massage2
    print(" i am massaging2")
    mo_pts.dump("surface_lg.inp")
    print('0.8 * h = ', 0.8 * h)
    print('PARAM_A = ',PARAM_A)
    print('PARAM_A2 = ',PARAM_A2)
    print('PARAM_B = ',PARAM_B)
    print('PARAM_B2 = ',PARAM_B2)
    mo_pts.massage2('user_function2.lgi',
                     0.8*h,
                     'fac_n',
                     0.00001,
                     0.00001,
                     stride=(1,0,0),
                     strictmergelength=True)

    lg.sendline('assign///maxiter_sm/1')

    for _ in range(3):
        mo_pts.smooth()
        mo_pts.recon(0)

    #cfg.log.info('Smoothing mesh (2/2)')

    # Massage once more, cleanup, and return
    lg.sendline('assign///maxiter_sm/10')
    mo_pts.massage2('user_function.lgi',
                     0.8*h,
                     'fac_n',
                     0.00001,
                     0.00001,
                     stride=(1,0,0),
                     strictmergelength=True)

    mo_pts.delatt('rf_field_name')

    mo_line_work.delete()
    mo_poly_work.delete()

    if outfile is not None:
        mo_pts.dump(outfile)

    #util.cleanup(['user_function.lgi','user_function2.lgi'])

    return mo_pts


def bbbuild_refined_triplane(
    dem_raster: Raster,
    distance_map: DistanceMap,
    min_edge_length: float,
    max_edge_length: float,
    delta: float = 0.75,
    slope: float = 2.0,
    refine_dist: float = 0.5,
    verbose: bool = False,
):
    """
    Constructs a triplane mesh refined around a feature using LaGriT
    as a backend.
    
    Requires an Nx2 np.ndarray as a boundary input, and an Nx2 np.ndarray
    as a feature input.
    """
    from pylagrit import PyLaGriT

    lg = PyLaGriT(verbose=verbose)
    h = min_edge_length * 1.75

    print('filtering self.feature at ', h)
    print('refining the surface mesh at: ', min_edge_length*1.75)
    print('with refine dist ', refine_dist, '; slope ', slope)
    print('--> h = ', h)
    print('--> delta = ', delta)
    print('--> slope = ', slope)
    print('--> refine_dist = ', refine_dist)

    boundary, _ = dem_raster.get_boundary(distance=max_edge_length)
    feature = filter_points(deepcopy(distance_map.feature), h)

    # Define initial parameters
    counterclockwise = False
    h_eps = h * 10 ** -7
    PARAM_A = slope
    PARAM_B = h * (1 - slope * refine_dist)
    PARAM_A2 = 0.5 * slope
    PARAM_B2 = h * (1 - 0.5 * slope * refine_dist)

    # cfg.log.debug('Writing boundary to poly_1.inp')
    # import pdb; pdb.set_trace()
    write_line(boundary, "poly_1.inp", connections=line_connectivity(boundary))

    # cfg.log.debug('Writing feature to intersections_1.inp')
    write_line(feature, "intersections_1.inp")

    # Write massage macros
    with open("user_function.lgi", "w") as f:
        f.write(Infiles.distance_field)

    with open("user_function2.lgi", "w") as f:
        f.write(Infiles.distance_field_2)

    # cfg.log.info('Preparing feature and boundary')

    # Read boundary and feature
    mo_poly_work = lg.read("poly_1.inp", name="mo_poly_work")
    mo_line_work = lg.read("intersections_1.inp", name="mo_line_work")

    # Triangulate Fracture without point addition
    mo_pts = mo_poly_work.copypts(elem_type="triplane")
    mo_pts.select()

    # cfg.log.info('First pass triangulation')

    if counterclockwise:
        mo_pts.triangulate(order="counterclockwise")
    else:
        mo_pts.triangulate(order="clockwise")

    # Set element attributes for later use
    mo_pts.setatt("imt", 1, stride=(1, 0, 0))
    mo_pts.setatt("itetclr", 1, stride=(1, 0, 0))
    mo_pts.resetpts_itp()

    mo_pts.select()

    # Refine at increasingly smaller distances, approaching h
    for (i, ln) in enumerate([8, 16, 32, 64][::-1]):
        # cfg.log.info('Refining at length %s' % str(ln))
        print("in loop")

        h_scale = ln * h
        perturb = h_scale * 0.05

        print("massage")
        print(h_scale, h_eps, h_eps)
        mo_pts.massage(h_scale, h_eps, h_eps)
        print("done")

        # Do a bit of smoothing on the first pass
        print("first pass smooth")
        if i == 0:
            for _ in range(3):
                mo_pts.recon(0)
                mo_pts.smooth()
            mo_pts.recon(0)
        print("done")
        mo_pts.resetpts_itp()

        # Smooth and reconnect
        for _ in range(6):
            mo_pts.recon(0)
            mo_pts.smooth()
        mo_pts.recon(0)

    # Define attributes to be used for massage functions
    mo_pts.addatt("x_four", vtype="vdouble", rank="scalar", length="nnodes")
    mo_pts.addatt("fac_n", vtype="vdouble", rank="scalar", length="nnodes")

    # Define internal variables for user_functions
    print(" i am defiing")
    lg.define(
        mo_pts=mo_pts.name,
        PARAM_A=PARAM_A,
        PARAM_A2=PARAM_A2,
        PARAM_B=PARAM_B,
        PARAM_B2=PARAM_B2,
    )
    print(" i am done defining")

    # cfg.log.info('Smoothing mesh (1/2)')

    # Run massage2
    print(" i am massaging2")
    mo_pts.dump("surface_lg.inp")
    print('0.8 * h = ', 0.8 * h)
    print('PARAM_A = ',PARAM_A)
    print('PARAM_A2 = ',PARAM_A2)
    print('PARAM_B = ',PARAM_B)
    print('PARAM_B2 = ',PARAM_B2)
    mo_pts.massage2('user_function2.lgi',
                     0.8*h,
                     'fac_n',
                     0.00001,
                     0.00001,
                     stride=(1,0,0),
                     strictmergelength=True)
    print(" done massaging and smoothing")
    lg.sendline("assign///maxiter_sm/1")

    for _ in range(3):
        mo_pts.smooth()
        mo_pts.recon(0)

    # cfg.log.info('Smoothing mesh (2/2)')

    # Massage once more, cleanup, and return
    lg.sendline("assign///maxiter_sm/10")
    mo_pts.massage2(
        "user_function.lgi",
        0.8 * h,
        "fac_n",
        0.00001,
        0.00001,
        stride=(1, 0, 0),
        strictmergelength=True,
    )

    mo_pts.delatt("rf_field_name")

    mo_line_work.delete()
    mo_poly_work.delete()

    mo_pts.dump("surface_lg.inp")

    # TODO: change to Exodus!
    mo = read_avs(
        "surface_lg.inp",
        keep_material_id=False,
        keep_node_attributes=False,
        keep_cell_attributes=False,
    )

    cleanup(
        [
            "user_function.lgi",
            "user_function2.lgi",
            "poly_1.inp",
            "intersections_1.inp",
            "surface_lg.inp",
            "lagrit.log",
            "lagrit.out",
        ]
    )

    # TODO: interpolate elevation

    return mo
