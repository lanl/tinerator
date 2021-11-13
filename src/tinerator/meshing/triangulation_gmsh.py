# pip install gmsh
# pip install pygmsh

def triangulation_gmsh(algorithm: str = "delaunay", verbosity_level: int = 0):
    import pygmsh


    algorithms = {
        "meshadapt": 1,
        "automatic": 2,
        "initial_mesh_only": 3,
        "delaunay": 5,
        "frontal_delaunay": 6,
        "bamg": 7,
    }

    verbose = False if verbosity_level <= 0 else True
    mesh_size = 0.05 # ?????
    mesh_size_min = mesh_size / 10 # ????
    boundary = list(None)

    # Algorithm:
    # http://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eAlgorithm
    # 1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 
    # 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 
    # 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms

    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            boundary,
            mesh_size=mesh_size,
        )

        # Distance field: 
        # 1. https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorial/t10.geo
        # 2. https://github.com/nschloe/pygmsh/tree/e40e57b38888c5666e9a16b51824671c1e527c71#mesh-refinementboundary-layers

        if refinement_objects:
            fields = []
            for obj in refinement_objects:

                nodes_list = None
                edges_list = None

                fields.append(
                    geom.add_boundary_layer(
                        mesh_size_min,
                        mesh_size,
                        min_edge_length,
                        max_edge_length,
                        edges_list=edges_list,
                        nodes_list=nodes_list
                    )
                )
            geom.set_background_mesh(fields, operator="Min")

        #geom.set_mesh_size_callback(
        #    lambda dim, tag, x, y, z: abs(sqrt(x ** 2 + y ** 2 + z ** 2) - 0.5) + 0.1
        #)
        mesh = geom.generate_mesh(
            algorithm=algorithms[algorithm.lower()],
            dim=2,
            verbose=verbose
        )