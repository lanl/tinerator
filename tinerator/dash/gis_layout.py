import plotly.graph_objects as go
import numpy as np

# x.imshow(elevation, color_continuous_scale=bamako)


def add_lines(fig, x, y):
    fig.add_trace(
        go.Scattermapbox(
            mode="lines+markers",
            lon=x,
            lat=y,
        )
    )


def add_polygon(fig, x, y):
    fig.add_trace(
        go.Scattermapbox(
            mode="lines",
            fill="toself",
            lon=x,
            lat=y,
        )
    )


def add_points(fig, x, y):
    fig.add_trace(
        go.Scattermapbox(
            mode="markers",
            lon=x,
            lat=y,
        )
    )


def add_raster(fig, array):
    # fig.add_trace(
    #    go.Image(data=array)
    # )
    pass


def init_figure():
    fig = go.Figure()

    add_lines(fig, [10, 20, 30], [10, 20, 30])
    add_lines(fig, [5, 15, 7], [5, 20, 3])

    add_polygon(fig, [0, 1, 4, 2, 0], [0, 3, 6, 8, 0])
    add_points(fig, [3, 8, 10], [3, 8, 10])

    add_raster(fig, np.random.rand(50, 50))
    img = np.random.rand(50, 50)
    coordinates = [
        [0, 0],  # min x, min y
        [5, 0],  # max x, min y
        [5, 5],  # max x, max y
        [0, 5],  # min x, max y
    ]

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        # margin ={'l':0,'t':0,'b':0,'r':0},
        # mapbox = {
        #    'center': {'lon': 10, 'lat': 10},
        #    #'style': "stamen-terrain",
        #    'style': 'open-street-map',
        #    'center': {'lon': -20, 'lat': -20},
        #    'zoom': 1},
        mapbox_layers=[
            {"sourcetype": "image", "source": img, "coordinates": coordinates}
        ],
    )

    return fig


def create():
    import dash
    import dash_core_components as dcc
    import dash_html_components as html

    app = dash.Dash()
    fig = init_figure()
    app.layout = html.Div([dcc.Graph(figure=fig)])
    return app
