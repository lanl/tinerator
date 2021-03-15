---
title: 'TINerator: Extruded and refined mesh generation for earth science applications'
tags:
  - Python
  - hydrology
  - gis
  - mesh generation
  - multiresolution meshing
authors:
  - name: Daniel R. Livingston
    affiliation: 1
  - name: Ilhan Özgen-Xian
    affiliation: 2
  - name: David J. Moulton
    affiliation: 1
affiliations:
  - name: Los Alamos National Laboratory, New Mexico, USA
    index: 1
  - name: Lawrence Berkeley National Laboratory, California, USA
    index: 2
date: Feb 4th, 2021
bibliography: paper.bib
---

# Summary

TINerator is a tool for the fast creation of extruded and refined
meshes from digital elevation model (DEM) and geographic information
system (GIS) data to aid earth science simulations.  TINerator
provides a complete workflow to generate surface or volume meshes from
a bounding box, a shapefile, or a local DEM.  TINerator further
provides a host of two- and three-dimensional visualization functions
to inpsect the state of the mesh at every step in the workflow.  In
addition, tools to add cell- and node-based attributes to the mesh, as
well as tools to modify and analyze mesh geometry are provided.

# Statement of need

The often prohibitive computational cost of physically-based earth
science simulations can be approached by using multiresolution meshes.
Such meshes are locally refined in areas that need to be resolved with
higher fidelity, and coarsened elsewhere.  A foreseeable strategy is
to analyse DEM and GIS data to determine the areas to be refined (and
coarsened), see for example [Caviedes-Voullieme:2012, @Ferraro:2020,
Ozgen-Xian:2020].  However, generating quality multiresolution meshes
earth science simulations is not trivial, both from a methodological
and technical point of view.  

The challenge in methodology is to design an approach that robustly
predicts refinement levels in the domain.  Common approaches are to
refine around the river network, at steep slopes, or at large
curvatures.

The technical challenge is the design of a workflow that generates
multiresolution meshes from input data.  It often requires a toolchain
that consists at least of a GIS software to preprocess the DEM and a
mesh generator.  Depending on the interoperability of these programs,
the workflow may require additional components that convert input and
output data into formats that can be exchanged.  

TINerator addresses the technical challenge by providing a fast and
efficient workflow to generate multiresolution unstructured meshes
from DEM and GIS data in a unified framework.

# Overview

TINerator is a Python-based mesh generation tool for generating
unstructured meshes for earth science simulations.  Users can perform
watershed delineation to determine catchment boundaries and
preferential flow paths through RichDEM [@Barnes:2016], choose to
refine around these preferential flow paths, and map spatial datasets
to cells or nodes of the generated mesh.  The users can choose the
mesh generator from JIGSAW [@Engwirda:2018], Triangle
[@Shewchuk:1996], and LaGriT (via PyLaGriT) [@LosAlamos:2020].
TINerator exposes all data to the user, which allows working on the
data using the full Python ecosystem, for example to filter and
interpolate for specific use cases.

# Use cases

## Mesh generation

## Integrated hydrology simulations using the Amanzi/ATS solver

# References

# Acknowledgements

