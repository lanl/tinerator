"""
Meshing kernels.
"""

_registered = {
    "triangle": None,
    "quad": None,
    "polygon": None,
} # Registered kernels

def register(
    kernel_name,
    kernel_desc,
    kernel_fnc,
    kernel_kwargs,
    kernel_supported_elements,
    kernel_supported_refinement
):
    pass