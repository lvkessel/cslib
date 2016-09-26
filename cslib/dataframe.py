from collections import OrderedDict
from functools import reduce

import io
import numpy as np

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
    def __init__(self, energy, q, cs):
        if len(energy.shape) == 1:
            energy = energy.reshape([energy.size, 1])

        self.energy = energy
        self.q = q
        self.cs = cs

        self._E_log_steps = np.log(self.energy[1:]/self.energy[:-1])
        self._q_steps = self.q[1:] - self.q[:-1]

        assert energy.shape == (energy.size, 1), \
            "Energy should be column vector."
        assert q.shape == (q.size,), \
            "Dependent quantity should be row vector."

        assert energy.dimensionality == ur.J.dimensionality, \
            "Energy units check."
        assert cs.dimensionality == (ur.m**2 / q.units).dimensionality, \
            "Cross-section units check."
        assert cs.shape == (energy.size, q.size), \
            "Array dimensions do not match."

    @property
    def angle(self):
        assert self.q.dimensionality == ur.rad.dimensionality
        return self.q

    @staticmethod
    def from_function(f, E, a):
        return DCS(E, a, f(E, a))

    def save_gnuplot(self, filename):
        xlen, ylen = self.cs.shape
        gp_bin = np.zeros(dtype='float32', shape=[xlen+1, ylen+1])
        gp_bin[0, 0] = xlen
        gp_bin[1:, 0:1] = self.energy.to('eV')
        gp_bin[0, 1:] = self.q
        gp_bin[1:, 1:] = self.cs.to(ur('cm²') / self.q.units)
        gp_bin.transpose().tofile(filename)

    def __rmul__(self, other):
        return DCS(self.energy, self.q, self.cs * other)

    def __add__(self, other):
        assert isinstance(other, DCS)
        assert np.array_equal(self.energy, other.energy), \
            "to add DCS, axes should match"
        assert np.array_equal(self.q, other.q), \
            "to add DCS, axes should match"
        return DCS(self.energy, self.q, self.cs + other.cs)

    def __call__(self, E, a):
        """Multi-linear interpolation on this DCS table.
        The interpolation is log-linear on energy and linear on angle."""
        # get the nearest grid locations for the energy -> masked array
        E_idx = np.searchsorted(self.energy.to('eV').flat,
                                E.to('eV').flat)[:, None]
        mE_idx = np.ma.array(
            E_idx - 1,
            mask=np.logical_or(E_idx == 0,
                               E_idx == self.energy.size))
        # compute the weight factor
        E_w = np.log(E / np.ma.take(self.energy, mE_idx) / E.units) \
            / np.ma.take(self._E_log_steps, mE_idx)

        # get the nearest grid locations for the angle -> masked array
        qu = self.q.units
        search_a = ur.wraps(None, [qu, qu])(np.searchsorted)
        a_idx = search_a(self.q, a)
        ma_idx = np.ma.array(
            a_idx - 1,
            mask=np.logical_or(a_idx == 0,
                               a_idx == self.q.size))
        # compute the weight factor
        a_w = (a - np.ma.take(self.q, ma_idx)) \
            / np.ma.take(self._q_steps, ma_idx)

        # take elements from a masked NdArray
        def take(a, *ix):
            i = np.meshgrid(*ix[::-1])[::-1]
            m = reduce(np.logical_or, [j.mask for j in i])
            return np.ma.array(a[[j.filled(0) for j in i]], mask=m)

        new_cs = (1 - E_w) * (1 - a_w) * take(self.cs, mE_idx, ma_idx) \
            + E_w * (1 - a_w) * take(self.cs, mE_idx + 1, ma_idx) \
            + (1 - E_w) * a_w * take(self.cs, mE_idx, ma_idx + 1) \
            + E_w * a_w * take(self.cs, mE_idx + 1, ma_idx + 1)

        # set values outside the range to zero
        return new_cs.filled(0.0) * self.cs.units
