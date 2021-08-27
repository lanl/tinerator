import vtk
import os
import dash
import dash_vtk
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_vtk.utils import to_mesh_state
import plotly.express as px

class Viz:
    def __init__(self):
        self.main = self.load("open-book.vtk", id="main")
        self.sets = {
            "down": self.load("open-book-down.vtk", id="down"),
            "east": self.load("open-book-east.vtk", id="east"),
            "north": self.load("open-book-north.vtk", id="north"),
            "south": self.load("open-book-south.vtk", id="south"),
            "top": self.load("open-book-TopFaces.vtk", id="top"),
            "west": self.load("open-book-west.vtk", id="west"),
        }
    
    def load(self, f, id=None):
        d = "/Users/livingston/dev/lanl/plotly-port/notebooks/meshes"
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(os.path.join(d, f))
        reader.Update()
        mesh_vtk = reader.GetOutput()
        return dash_vtk.Mesh("mesh-" + id, state=to_mesh_state(mesh_vtk))

# -----------------------------------------------------------------------------
# Control UI
# -----------------------------------------------------------------------------

def get_controls(mesh):
    high = 0.6
    low = -0.55

    return [
        dbc.Card(
            [
                dbc.CardHeader("Seeds"),
                dbc.CardBody(
                    [
                        html.P("Line seed position (from bottom):"),
                        dcc.Slider(
                            id="point-1",
                            min=low,
                            max=high,
                            step=0.05,
                            value=0,
                            marks={low: str(low), high: str(high)},
                        ),
                        html.Br(),
                        html.P("Line seed position (from top):"),
                        dcc.Slider(
                            id="point-2",
                            min=low,
                            max=high,
                            step=0.05,
                            value=0,
                            marks={low: str(low), high: str(high)},
                        ),
                        html.Br(),
                        html.P("Line resolution:"),
                        dcc.Slider(
                            id="seed-resolution",
                            min=5,
                            max=50,
                            step=1,
                            value=10,
                            marks={5: "5", 50: "50"},
                        ),
                    ]
                ),
            ]
        ),
        html.Br(),
        dbc.Card(
            [
                dbc.CardHeader("Color By"),
                dbc.CardBody(
                    [
                        html.P("Field name"),
                        dcc.Dropdown(
                            id="color-by",
                            options=[
                                {"label": "p", "value": "p"},
                                {"label": "Rotation", "value": "Rotation"},
                                {"label": "U", "value": "U"},
                                {"label": "Vorticity", "value": "Vorticity"},
                                {"label": "k", "value": "k"},
                            ],
                            value="p",
                        ),
                        html.Br(),
                        html.P("Color Preset"),
                        dcc.Dropdown(
                            id="preset",
                            #options=preset_as_options,
                            value="erdc_rainbow_bright",
                        ),
                    ]
                ),
            ]
        ),
    ]


# - be able to read in exodus meshes for viz

# -----------------------------------------------------------------------------
# 3D Viz
# -----------------------------------------------------------------------------
def vtk_view(mesh):

    POINTS = 0
    WIREFRAME = 1
    SURFACE = 2

    viz = Viz()

    children = []

    for s in viz.sets.keys():
        children.append(dash_vtk.GeometryRepresentation(
                id=s,
                children=[viz.sets[s]],
                property={
                    "edgeVisibility": True,
                    #"color": (1, 0, 0),
                    "opacity": 1,
                    "representation": SURFACE,
                },
            )
        )

    #children.append(
    #    dash_vtk.GeometryRepresentation(
    #        id="main",
    #        children=[viz.main],
    #        property={"representation": WIREFRAME}
    #    )
    #)

    return dash_vtk.View(
        id="vtk-view",
        children=children,
    )

# -----------------------------------------------------------------------------
# App UI
# -----------------------------------------------------------------------------

def create_layout(mesh, **kwargs):
    return dbc.Container(
        fluid=True,
        style={"marginTop": "15px", "height": "calc(100vh - 30px)"},
        children=[
            dbc.Row(
                [
                    dbc.Col(width=4, children=get_controls(mesh)),
                    dbc.Col(
                        width=8,
                        children=[
                            html.Div(vtk_view(mesh), style={"height": "100%", "width": "100%"})
                        ],
                    ),
                ],
                style={"height": "100%"},
            ),
        ],
    )

def create_app(mesh, **kwargs):
    app = dash.Dash(
        __name__,
        #meta_tags=[{"name": "viewport", "content": "width=device-width"}],
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    app.title = "TINerator"

    server = app.server
    app.layout = create_layout(mesh, **kwargs)
    # demo_callbacks(app)

    return app