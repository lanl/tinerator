def save(fig, filename: str, **kwargs):
    if filename.lower().endswith("html"):
        fig.write_html(filename, **kwargs)
    else:
        fig.write_image(filename, **kwargs)


def render(fig, context: str = "ipython"):
    fig.show()
    # if context == 'ipython':
    #    import plotly.express as px
    #    img_bytes = fig.to_image(format="png")
    #    t_fig = px.imshow(img_bytes, binary_format="png")#, binary_compression_level=0)
    #    t_fig.show()
    #    from IPython.display import Image, display
    #    display(Image(img_bytes))
