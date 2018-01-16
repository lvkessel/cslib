from functools import reduce
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from . import units as ur


class DCS(object):
    """Differential cross-section: energy, q, cs.

    The energy and angle are 1d array quantities, with dimensions
    `N` and `M` respectively and having units with dimensionality
    of energy and angle. Note that the energy should always be given
    as a column vector.

    The cross-section is a 2d array of shape [N, M], having dimensionality
    of area."""
    def __init__(self, energy, q, cs):
        if len(energy.shape) == 1:
            energy = energy.reshape([energy.size, 1])

        self.energy = energy
        self.q = q
        self.cs = cs

        assert energy.shape == (energy.size, 1), \
            "Energy should be column vector."
        assert q.shape == (q.size,), \
            "Dependent quantity should be row vector."

        assert energy.dimensionality == ur.J.dimensionality, \
            "Energy units check."
        assert cs.dimensionality in (             \
            (ur.m**2 / q.units).dimensionality,   \
            (ur.m**-1 / q.units).dimensionality), \
            "Cross-section units check."
        assert cs.shape == (energy.size, q.size), \
            "Array dimensions do not match."

        self.interpolate_fn = RegularGridInterpolator((
            np.log(self.energy.magnitude.flat),
            self.q.magnitude.flat),
            self.cs.magnitude,
            bounds_error = False, fill_value = 0)

    def __rmul__(self, other):
        return DCS(self.energy, self.q, self.cs * other)

    def __add__(self, other):
        assert isinstance(other, DCS)
        assert np.array_equal(self.energy, other.energy), \
            "to add DCS, axes should match"
        assert np.array_equal(self.q, other.q), \
            "to add DCS, axes should match"
        return DCS(self.energy, self.q, self.cs + other.cs)

    def __call__(self, E, q):
        return self.interpolate_fn((
            np.log(E.to(self.energy.units).magnitude), \
            q.to(self.q.units).magnitude)) * self.cs.units
