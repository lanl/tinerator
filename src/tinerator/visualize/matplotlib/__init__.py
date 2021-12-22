class PyPlotKernel:
    def can_import(cls):
        try:
            from matplotlib import pyplot as plt
            return True
        except Exception as e:
            return False

    def metadata(cls):
        return {
            "name": "PyPlot",
            "type": "2D",
            "exports": ["pdf", "png", "jpg", "svg"]
        }
    
    def plot(cls, *args, **kwargs):
        from .mpl_plot import plot
        return plot(self, *args, **kwargs)
    
    def save(cls, fig, outfile: str):
        fig.save(outfile)
    
    def view(cls, fig):
        fig.show()
    