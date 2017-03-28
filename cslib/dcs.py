import numpy as np
from scipy.interpolate import interp2d
# from .units import units


class DCS(object):
    """Differential cross-section: energy, q, cs.

    The energy and angle are 1d array quantities, with dimensions
    `N` and `M` respectively and having units with dimensionality
    of energy and angle. Note that the energy should always be given
    as a column vector.

    The cross-section is a 2d array of shape [N, M], having dimensionality
    of area."""
    def __init__(self, x, y, z, log='y',
                 x_units=None, y_units=None, z_units=None):
        assert y.shape == (y.size, 1), \
            "Energy should be column vector."
        assert x.shape == (x.size,), \
            "Dependent quantity should be row vector."

        x_units = x_units or x.units
        y_units = y_units or y.units
        z_units = z_units or z.units

        self.x = x.to(x_units)
        self.y = y.to(y_units)
        self.z = z.to(z_units)
        self.log = log

        y = np.log(self.y.magnitude.flat) if 'y' in log \
            else self.y.magnitude.flat
        x = np.log(self.x.magnitude) if 'x' in log \
            else self.x.magnitude

        self.f = interp2d(x, y, self.z.magnitude, kind='linear')

    def save_gnuplot(self, filename):
        xlen, ylen = self.z.shape
        gp_bin = np.zeros(dtype='float32', shape=[xlen+1, ylen+1])
        gp_bin[0, 0] = xlen
        gp_bin[1:, 0:1] = self.y
        gp_bin[0, 1:] = self.x
        gp_bin[1:, 1:] = self.z
        gp_bin.transpose().tofile(filename)

    def __rmul__(self, other):
        return DCS(self.x, self.y, self.z * other, self.log)

    def __add__(self, other):
        assert isinstance(other, DCS)
        assert np.array_equal(self.x, other.x), \
            "to add DCS, axes should match"
        assert np.array_equal(self.y, other.y), \
            "to add DCS, axes should match"
        return DCS(self.x, self.y, self.z + other.z, self.log)

    def unsafe(self, xx, yy):
        return self.f(np.log(xx) if 'x' in self.log else xx,
                      np.log(yy) if 'y' in self.log else yy)

    def __call__(self, x, y):
        xx = x.to(self.x.units).magnitude
        yy = y.to(self.y.units).magnitude.flat
        return self.unsafe(xx, yy) * self.z.units
