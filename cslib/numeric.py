from functools import (
    reduce)

import numpy as np
from numpy import (log)
from scipy.integrate import (romberg)
from scipy.optimize import (brentq)


def identity(x):
    return x


def compose(*f):
    def compose_2(f, g):
        return lambda x: f(g(x))
    return reduce(compose_2, f, identity)


def interpolate(f1, f2, h, a, b):
    """Interpolate two functions `f1` and `f2` using interpolation
    function `h`, which maps [0,1] to [0,1] one-to-one."""
    def g(x):
        y1 = f1(x)
        y2 = f2(x)
        u = (x - a) / (b - a)
        w = h(u)
        ym = (1 - w) * y1 + w * y2

        return np.where(
            x < a, y1, np.where(
                x > b, y2, ym))

    return g


def linear_interpolate(f1, f2, h, a, b):
    ya = f1(a)
    yb = f2(b)

    def fm(x):
        n = h((x - a) / (b - a))
        return (1 - n) * ya + n * yb

    def g(x):
        y1 = f1(x)
        y2 = f2(x)
        ym = fm(x)

        return np.where(
            x < a, y1, np.where(
                x > b, y2, ym))

    return g


def log_interpolate(f1, f2, h, a, b):
    """Interpolate two functions `f1` and `f2` using interpolation
    function `h`, which maps [0,1] to [0,1] one-to-one."""
    def weight(x):
        return np.clip(log(x / a) / log(b / a), 0.0, 1.0)

    def g(x):
        w = h(weight(x))
        return (1 - w) * f1(x) + w * f2(x)

    return g


def inverse_cdf_table(f, a, b, ys, fprime=None):
    """Takes a PDF function (doesn't have to be normalised), and
    returns a linearly spaced table of `n` elements giving the
    inverse cumulative distribution function. The CDF⁻¹ is suitable
    for generating random numbers following the PDF.

    :param f:
        PDF
    :param a:
        lower limit
    :param b:
        upper limit
    :param n:
        length of output table"""

    def F(x):
        return romberg(f, a, x, vec_func=True)

    A = 1 / F(b)
    # ys = np.linspace(0.0, 1.0, n)
    x = a

    for y in ys:
        x = brentq(lambda x: A*F(x) - y, x, b)
        yield x
