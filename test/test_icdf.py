from cslib.numeric import inverse_cdf_table
from numpy import (sin, cos, exp, pi, arccos)
import numpy as np


def test_sine_distribution():
    icdf_table = np.array(list(inverse_cdf_table(sin, 0, pi, 33)))
    x = icdf_table[:, 0]
    y = icdf_table[:, 1]
    error = sum((y - arccos(1 - 2*x))**2)
    assert error < 1e-16

