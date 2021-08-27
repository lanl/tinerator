from .rendering import init_window

def test_plot_vtk(mesh):
    from .vtk_layout import create_app
    init_window(create_app(mesh))