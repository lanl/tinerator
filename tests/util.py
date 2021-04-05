import meshio
import numpy as np
import os
import sys


def meshes_equal(test_fname, gold_fname) -> bool:
    mo_test = meshio.read(test_fname, file_format="avsucd")
    mo_gold = meshio.read(gold_fname, file_format="avsucd")

    # TODO: what if points are in different order?
    assert np.allclose(mo_test.points, mo_gold.points)
    assert len(mo_test.cells) == len(mo_gold.cells)

    # TODO: what if cells are in different order?
    for i in range(len(mo_gold.cells)):
        assert np.array_equal(mo_test.cells[i].data, mo_gold.cells[i].data)

    assert np.array_equal(
        list(mo_test.point_data.keys()), list(mo_gold.point_data.keys())
    )
    assert np.array_equal(
        list(mo_test.cell_data.keys()), list(mo_gold.cell_data.keys())
    )

    for key in mo_gold.point_data.keys():
        assert np.allclose(mo_test.point_data[key], mo_gold.point_data[key])

    for key in mo_gold.cell_data.keys():
        assert np.allclose(mo_test.cell_data[key], mo_gold.cell_data[key])

    return True
