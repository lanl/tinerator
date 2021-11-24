def save(fig, filename: str, **kwargs):
    if filename.lower().endswith('html'):
        fig.write_html(filename, **kwargs)
    else:
        fig.write_image(filename, **kwargs)

def render(fig, context: str = 'ipython'):
    if context == 'ipython':
        img_bytes = fig.to_image(format="png")
        from IPython.display import Image, display
        display(Image(img_bytes))