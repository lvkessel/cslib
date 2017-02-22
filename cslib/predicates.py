import numbers
import collections
import os

from .units import (units)


class Predicate:
    def __init__(self, f, description=""):
        assert callable(f)
        self.f = f
        self.description = description

    def __call__(self, v):
        return self.f(v)

    def __and__(self, other):
        return Predicate(lambda v: self.f(v) and other.f(v),
                         self.description + " & " + other.description)

    def __or__(self, other):
        return Predicate(lambda v: self.f(v) or other.f(v),
                         self.description + " | " + other.description)

    def __invert__(self):
        return Predicate(lambda v: not self.f(v),
                         "!" + self.description)

    def display(self):
        return self.description


def predicate(description=""):
    def _predicate(f):
        return Predicate(f, description)
    return _predicate


@predicate("Integer")
def is_integer(v):
    return isinstance(v, int)


@predicate("String")
def is_string(v):
    return isinstance(v, str)


@predicate("Number")
def is_number(v):
    return isinstance(v, numbers.Number)


@predicate("None")
def is_none(v):
    return v is None


def has_units(u):
    if isinstance(u, str):
        u = units.parse_units(u)

    @predicate("{:~P} (e.g. {:~P})".format(u.dimensionality, u))
    def _has_units(v):
        return type(v).__name__ == 'Quantity' and \
            v.dimensionality == u.dimensionality

    return _has_units


is_energy = has_units(units.J)
is_length = has_units(units.m)
is_volume = has_units(units.m**3)


def in_range(a, b):
    @predicate("In [{}, {}>".format(a, b))
    def _in_range(v):
        return v >= a and v < b

    return _in_range


def is_(a):
    @predicate("{}".format(a))
    def _is_(v):
        return v == a

    return _is_


@predicate("Seq")
def is_iterable(v):
    return isinstance(v, collections.Iterable)


def is_list_of(p):
    @predicate("List[" + p.description + "]")
    def _is_list_of(v):
        if not isinstance(v, list):
            return False

        for e in v:
            if not p(e):
                return False

        return True

    return _is_list_of


@predicate("File")
def file_exists(path: str):
    abspath = os.path.abspath(path)
    return os.path.exists(abspath)
