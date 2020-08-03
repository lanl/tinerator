import pytest
import helper
import sys
sys.path.insert(0,'/Users/livingston/playground/tinerator/tinerator-core')
import tinerator as tin

def func(x):
    return x + 1

def test_answer():
    assert func(3) == 5

def test_bob():
    surf_mesh = helper.init_surf_mesh_tri()
    assert func(4) == 5