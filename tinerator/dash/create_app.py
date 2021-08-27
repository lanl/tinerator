def create_app(**kwargs):
    app = dash.Dash(
        __name__,
        #meta_tags=[{"name": "viewport", "content": "width=device-width"}],
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    app.title = "TINerator"

    server = app.server
    app.layout = create_layout(app, **kwargs)
    # demo_callbacks(app)

    return app