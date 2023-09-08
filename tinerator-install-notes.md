# TINerator: local  conda installation

This needs some sort of clean-up/formatting decision.  I was thinking it 
might be best as an installation script, but for now I put it in as 
markdown file.

cartopy seems to cause a lot of conflicts - moving it to pip install

Everything, including gdal, seems to work from conda-forge.


# Create conda environment for tinerator


Some packages seem to push us to python 3.11 so let's try that from the start

```
conda create -n tinerator-v1 -c conda-forge python=3.11
conda activate tinerator-v1
```

# Install packages

The install steps are broken down to avoid long iterations on dependency checks and to reveal when certain packages mess with dependencies and may cause previously installed dependencies to be downgraded.

## Pandas

```
conda install -c conda-forge pandas
```

Installs pandas 2.1.0, and numpy 1.25.2 (among other things).


## GDAL

This is one of the core libraries used by other packages, and has been broken in the past, but seems to work now

```
conda install -y -c conda-forge gdal
```

To test, run python and import

```python
import osgeo.gdal
from osgeo import ogr, gdal, gdal_array
```

If these commands seem to work, very encouraging, you can keep going.

## RichDEM

This is an important package, and has a lot of dependencies that seem to cause problems.  However, it seems clean this time:

```
conda install -y -c conda-forge richdem
```

At least it seems to install without adding any additional packages or downgrading an previously installed packages.

## Meshing stuff
 
``` 
conda install -c conda-forge meshio meshpy
conda install -c conda-forge pyepsg descartes
conda install -c conda-forge geopandas
conda install -c conda-forge sortedcontainers xarray datashader
```

We had a good run, but one package gets downgraded on this last install,

  * numpy gets downgraded 1.25.2-py311hb8f3215_0 --> 1.24.4-py311hb8f3215_0

Not sure if it was xarray, maybe we should put xarray earlier in the list.

# Visualization stuff

There are a few visualization packages, plotly and dash for the current master branch and newer stuff, pyvista for the regroup branch and working container.

```
conda install -c conda-forge plotly dash dash-renderer dash-bootstrap-components jupyter-dash
```

It looks like this may be simplified, need to dig into details a bit.  What I found so far indicates:
  * dash > 2.11 is needed (and has jupyter-dash capability built in), 
  * looks like we got dash 2.13 so that seems good.
  * probably could/should drop jupyter-dash 


# Jupyter Lab

At the end of the day we want all this to work from within jupyter lab.  

```
conda install -y -c conda-forge jupyter jupyterlab
```

To update extensions in jupyter lab we need nodejs.  Previously I had trouble getting the one from conda-forge to work, and I'm pretty sure I installed it through macports.  The version matters, I just don't remember how I got that info from jupyter.

For now trying the newest conda-forge version

```
conda install -c conda-forge nodejs
```

Which is 20.6.0.

# PIP stuff (vtk, snowy, cartopy)

Previously, the conda-forge vtk packages didn't work.  Not sure if this has been fixed, but assuming my pip workaround is still good.

```
pip install vtk pyvista dash-vtk
pip install snowy
pip install cartopy
```

All seem to install cleanly and seem to work.  At this point you should be able to go ahead and start testing TINerator.


# TPL library build/install

The last step for full functionality would be the build and installation of tge Jigsaw mesh geneartor and ExodusII libraries.  It's possible to condsider adding LaGriT as a mesh geneartor as well.  

Currently, we have a build problem with ExodusII, probably need to patch seacas for newest gcc.

