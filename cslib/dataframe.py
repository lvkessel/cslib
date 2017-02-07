from collections import OrderedDict
from functools import reduce

import io
import numpy as np
from scipy.interpolate import interp2d

from . import units as ur


class DataFrame(object):
    """A class like the Pandas DataFrame; this one supports physical units
    using the `pint` module, and storage to HDF5 files.

    This is a wrapper around a higher dimensional Numpy array.

    The class supports both column and
    row based access; however, it is optimised to handle entire columns
    of data more efficiently.

    Rows start counting at 0. Every column must have a unit."""
    def __init__(self, data, units=None, comments=None):
        self.data = data
        if units:
            self.units = [ur.parse_units(u) if isinstance(u, str)
                          else u for u in units]
        else:
            self.units = units
        self.comments = comments
        self.unit_dict = OrderedDict(zip(self.data.dtype.names, self.units))

    def __getitem__(self, x):
        s = self.data[x]
        if isinstance(x, str):
            return s * self.unit_dict[x]
        elif isinstance(x, tuple) and isinstance(x[1], int):
            return s * self.units[x[1]]
        elif isinstance(x, tuple) and isinstance(x[1], slice):
            return DataFrame(s, self.units[x[1]])

    def __len__(self):
        return len(self.data)

    def __str__(self):
        of = io.BytesIO()
        np.savetxt(of, self.data, fmt='%.4e')
        return '# ' + ', '.join(
            '{0} ({1:~})'.format(n, u)
            for n, u in self.unit_dict.items()) + \
            '\n' + of.getvalue().decode()


class TCS(object):
    """Total cross-section: energy, cs.

    This can be obtained by integrating the DCS over angle."""
    def __init__(self, energy, cs):
        assert energy.shape == cs.shape, \
            "Array shapes should match."
        assert energy.dimensionality == ur.J.dimensionality, \
            "Energy units check."
        assert cs.dimensionality == (ur.m**2).dimensionality, \
            "Cross-section units check."

        self.energy = energy
        self.cs = cs
        self._E_log_steps = np.log(self.energy[1:]/self.energy[:-1])

    def __call__(self, E):
        """Interpolates the tabulated values, logarithmic in energy."""
        E_idx = np.searchsorted(self.energy.to('eV').flat,
                                E.to('eV').flat)[:, None]
        mE_idx = np.ma.array(
            E_idx - 1,
            mask=np.logical_or(E_idx == 0,
                               E_idx == self.energy.size))
        # compute the weight factor
        E_w = np.log(E / np.ma.take(self.energy, mE_idx) / E.units) \
            / np.ma.take(self._E_log_steps, mE_idx)

        # take elements from a masked NdArray
        def take(a, *ix):
            i = np.meshgrid(*ix[::-1])[::-1]
            m = reduce(np.logical_or, [j.mask for j in i])
            return np.ma.array(a[[j.filled(0) for j in i]], mask=m)

        new_cs = (1 - E_w) * take(self.cs, mE_idx) \
            + E_w * take(self.cs, mE_idx + 1)

        return new_cs.filled(0.0) * self.cs.units


class DCS(object):
    """Differential cross-section: energy, q, cs.

    The energy and angle are 1d array quantities, with dimensions
    `N` and `M` respectively and having units with dimensionality
    of energy and angle. Note that the energy should always be given
    as a column vector.

    The cross-section is a 2d array of shape [N, M], having dimensionality
    of area."""
    def __init__(self, x, y, cs, log='y'):
        assert y.shape == (y.size, 1), \
            "Energy should be column vector."
        assert x.shape == (x.size,), \
            "Dependent quantity should be row vector."

        # assert energy.dimensionality == ur.J.dimensionality, \
        #     "Energy units check."
        # assert cs.dimensionality == (ur.m**2 / q.units).dimensionality, \
        #     "Cross-section units check."
        # assert cs.shape == (energy.size, q.size), \
        #     "Array dimensions do not match."

        self.x = x
        self.y = y
        self.cs = cs
        self.log = log

        # self._E_log_steps = np.log(self.energy[1:]/self.energy[:-1])
        # self._q_steps = self.q[1:] - self.q[:-1]

        y = np.log(self.y.magnitude.flat) if 'y' in log \
            else self.y.magnitude.flat
        x = np.log(self.x.magnitude) if 'x' in log \
            else self.x.magnitude
        self.f = interp2d(x, y, self.cs.magnitude, kind='linear')

    @staticmethod
    def from_function(f, cs_u, E, E_u, a, a_u, **kwargs):
        return DCS(E, E_u, a, a_u, f(E, a), cs_u, **kwargs)

    def save_gnuplot(self, filename):
        xlen, ylen = self.cs.shape
        gp_bin = np.zeros(dtype='float32', shape=[xlen+1, ylen+1])
        gp_bin[0, 0] = xlen
        gp_bin[1:, 0:1] = self.y.magnitude
        gp_bin[0, 1:] = self.x.magnitude
        gp_bin[1:, 1:] = self.cs.magnitude
        gp_bin.transpose().tofile(filename)

    def unsafe(self, xx, yy):
        return self.f(np.log(xx) if 'x' in self.log else xx,
                      np.log(yy) if 'y' in self.log else yy)

    def __call__(self, x, y):
        xx = x.to(self.x.units).magnitude
        yy = y.to(self.y.units).magnitude.flat
        return self.unsafe(xx, yy)
