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

Generating quality meshes from DEM and GIS data for physically-based
earth science simulations often requires a toolchain that consists at
least of a GIS software, for example QGIS or GRASS GIS, to preprocess
the DEM and a mesh generator, for example Gmsh or LaGriT.  Depending
on the interoperability of these programs, the workflow may require
additional components that convert input and output data into formats
that can be exchanged.

TINerator is a Python-based mesh generation tool for generating
unstructured meshes for earth science simulations.  TINerator provides
a complete set of tools for DEM and GIS data analysis and mesh
generation in a unified framework.

# Use cases

## Mesh generation

## Integrated hydrology simulations using the Amanzi/ATS solver

# References

# Acknowledgements

