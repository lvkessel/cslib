"""This module contains an instance of :py:class:`pint.UnitRegistry`
called :py:data:`units`. By sharing this instance between different
packages that may use CSLib, these packages can compare units for
validity. CSLib uses `the Pint module`_.

Pint is very handy for converting between units.

.. doctest::

    >>> from cslib import units
    >>> from cslib.units import approx   # for testing only
    >>> units('41 Å').to('nm')
    <Quantity(4.1, 'nanometer')>
    >>> print("{}".format(units('2500 kcal').dimensionality))
    [length] ** 2 * [mass] / [time] ** 2

A few quantities have been added to the standard set of units in Pint.

    >>> x = units('T_room')
    >>> x.dimensionality == units.K.dimensionality
    True
    >>> approx(units('T_room')) == 297 * units.K
    True
    >>> approx(units('a_0')) == 5.2917721067e-11 * units.m
    True

.. _the Pint module: https://pint.readthedocs.io/en/0.7.2/"""

from pint import (UnitRegistry)
import pint


class MyUnitRegistry(UnitRegistry):
    def __getattr__(self, name):
        if name[0] == '_':
            try:
                value = super(MyUnitRegistry, self).__getattr__(name)
                return value
            except pint.errors.UndefinedUnitError as e:
                raise AttributeError()
        else:
            return super(MyUnitRegistry, self).__getattr__(name)


units = UnitRegistry()
units.__test__ = False

units.define("ε_0 = epsilon_0")
units.define("bohr_radius = 4 π ħ**2 ε_0 / (m_e e**2) = a0 = a_0")
units.define("G = gauss")
units.define("T_room = 297 K = room_temperature")


class Approx:
    def __init__(self, value, abs_err):
        self.value = value
        self.abs_err = abs_err

    def __eq__(self, other):
        if self.value.dimensionality != other.dimensionality:
            return False

        return abs(self.value - other) < self.abs_err


def approx(value, abs_err=None, rel_err=1e-6):
    """Creates an approximate value, similar to :py:func:`pytest.approx`. This
    version works with units. If an absolute error is not given, it is computed
    from the relative error which is set to `1e-6` by default."""
    abs_err = abs_err or value * rel_err
    return Approx(value, abs_err)
