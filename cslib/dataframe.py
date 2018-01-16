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

