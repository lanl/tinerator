import numpy as np
from matplotlib import pyplot as plt

def plot_mesh_quality(mesh):
    elens = edge_lengths(mesh)
    
    f, ax = plt.subplots(nrows=2,ncols=2)
    f.suptitle("Mesh Quality Metrics")
    ax[0,0].hist(elens, bins=20)
    ax[0,0].set_title("Edge Lengths")
    plt.show()

def mesh_quality(mesh):
    elens = edge_lengths(mesh)

    return {
        'edge length': {
            'mean': np.mean(elens), 
            'stdev': np.std(elens)
        },
    }

def edge_lengths(mesh) -> np.ndarray:
    '''
    Returns an array with the Euclidean length of each edge
    in `mesh.edges`.
    '''

    pts = mesh.nodes
    edges = mesh.edges - 1
    return np.linalg.norm(pts[edges[:,0]] - pts[edges[:,1]], axis=1)
