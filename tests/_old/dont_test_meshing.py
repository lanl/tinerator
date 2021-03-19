def test_elevation():
    # test that mesh Z elevations == DEM data

def test_triangulation_uniform():

    dem = None

    for edge_length in [10., 100., 1000.]:
        for method in ["lagrit", "jigsaw", "poisson-disc"]:
            surf = tin.meshing.triangulate(dem, edge_length, method=method)
            mean_el = tin.meshing.mesh_quality(surf)['edge length']['mean']

            assert np.abs(mean_el - edge_length) < 0.2 * edge_length