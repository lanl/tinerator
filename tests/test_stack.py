import pytest

def func(x):
    return x + 1

def test_answer():
    assert func(3) == 5

def test_bob():
    assert func(4) == 5