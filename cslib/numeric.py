from functools import (
    reduce)

import numpy as np
from numpy import (log)
from scipy.integrate import (romberg)
from scipy.interpolate import (interp1d)
from scipy.optimize import (brentq)


def identity(x):
    return x


def compose(*f):
    def compose_2(f, g):
        return lambda x: f(g(x))
    return reduce(compose_2, f, identity)


def interpolate_f(f1, f2, h, a, b):
    """Interpolate two functions `f1` and `f2` using interpolation
    function `h`, which maps [0,1] to [0,1] one-to-one."""
    def g(x):
        y1 = f1(x)
        y2 = f2(x)

        # Convert units and strip, to keep pint happy
        units = y1.units
        y2 = y2.to(units).magnitude
        y1 = y1.magnitude

        u = np.clip((x - a) / (b - a), 0.0, 1.0)
        w = h(u)
        ym = (1 - w) * y1 + w * y2

        return np.where(
            x < a, y1, np.where(
                x > b, y2, ym)) * units

    return g


def log_interpolate_f(f1, f2, h, a, b):
    """Interpolate two functions `f1` and `f2` using interpolation
    function `h`, which maps [0,1] to [0,1] one-to-one."""
    assert callable(f1)
    assert callable(f2)
    assert callable(h)

    def weight(x):
        return np.clip(log(x / a) / log(b / a), 0.0, 1.0)

    def g(x):
        y1 = f1(x)
        y2 = f2(x)

        # Convert units and strip, to keep pint happy
        units = y1.units
        y2 = y2.to(units).magnitude
        y1 = y1.magnitude

        u = np.clip(log(x / a) / log(b / a), 0.0, 1.0)
        w = h(u)
        ym = (1 - w) * y1 + w * y2

        return np.where(
            x < a, y1, np.where(
                x > b, y2, ym)) * units
    return g


def loglog_interpolate(x, y, bounds_error=None, fill_value='extrapolate'):
    """Interpolate a function (given by arrays of data points x, y) on a
    log-log scale.

    If fill_value is equal to 'extrapolate', out-of-bounds accesses are
    extrapolated on a log-log scale. Otherwise, if bounds_error is False,
    out-of-bounds accesses are filled with fill_value. If bounds_error is
    True, an error is raised for out-of-bounds accesses."""

    interp_function = interp1d(np.log(x.magnitude), np.log(y.magnitude),
        bounds_error = bounds_error, fill_value = fill_value)

    def g(x_points):
        return np.exp(interp_function(
            np.log(x_points.to(x.units).magnitude))) * y.units

    return g
