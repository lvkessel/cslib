"""This module implements a system for parsing, validating and generating
:py:class:`Settings` from a generic description in a :py:class:`Model`.
We'll work through a small example. Suppose we have a code that computes
the trajectory of a cannon ball. We need to give the ball a starting position,
mass, and starting velocity. Let's start with the position, it has three
dimensions and should have units of length.

.. doctest::

    >>> import numpy as np
    >>> from cslib import (units)
    >>> from cslib.predicates import (predicate, has_units)
    >>> def array_dim(*n):
    ...     @predicate("array_dim({})".format(', '.join(map(str, n))))
    ...     def _array_dim(a):
    ...         if not isinstance(a.magnitude, np.ndarray):
    ...             return False
    ...         return a.shape == n
    ...     return _array_dim
    ...
    >>> position_check = array_dim(3) & has_units('m')
    >>> position_check.display()
    'array_dim(3) & [length] (e.g. m)'
    >>> position_check(np.r_[1, 2, 3] * units.mm)
    True
    >>> position_check(1 * units.m)               # not an array
    False
    >>> position_check(np.r_[4, 5] * units.km)    # wrong shape
    False
    >>> position_check(np.r_[6, 7, 8] * units.J)  # wrong unit
    False

Now we define the :py:class:`Type` that describes the entry in the
:py:class:`Model`. We need to define two functions, one that parses
an array with units, and one that formats an array with units to
a string::

    >>> def parse_array(s):
    ...     expr = '\[\s*(?P<numbers>(\S+\s*,\s*)*(\S+))\s*\]\s+(?P<unit>\S+)'
    ...     m = re.match(expr, s)
    ...     numbers, unit = m.group('numbers', 'unit')
    ...     return np.array([ast.literal_eval(n.strip())
    ...                     for n in numbers.split(',')]) * units(unit)
    ...
    >>> def format_array(a):
    ...     return "[{}] {:~P}".format(
    ...         ", ".join(map(str, a.magnitude)), a.units)
    ...
    >>> position_type = Type("The position of the cannon.",
    ...     check=position_check,
    ...     parser=parse_array, generator=format_array)
"""

from functools import (reduce)
from copy import (deepcopy)
from collections import OrderedDict
import textwrap

from ruamel import yaml

from .predicates import (Predicate, predicate)


class TemporaryEntry(object):
    """When a setting is assigned to it could very well be that the location
    assigned to, does not yet exist. Now, suppose a deep layered is_settings
    object, where we want to assign to a key within a key that does not exist.
    Once the bottom level key is assigned to, the mid-level settings objects
    need to be created. This is done by building up a path into this non-
    existing location on the fly and assigning at the end using the index
    syntax.

    For example, the following line::

        settings.a.b.c = 42

    gets translated into::

        settings['a.b.c'] = 42

    whenever `settings` does not contain an entry `a`.

    To effectuate this behaviour the `Settings` class generates a
    `TemporaryEntry` when an attribute is missing. An object of this type
    will just build the attribute path and act upon assignment. The
    expression::

        'a' in settings

    will still evaluate to `False`."""
    def __init__(self, settings, key):
        self._d = settings
        self._k = key

    def __getattr__(self, k):
        return TemporaryEntry(self._d, self._k + '.' + k)

    def __setattr__(self, k, v):
        if k[0] != '_':
            self._d[self._k + '.' + k] = v
        else:
            self.__dict__[k] = v

    def __bool__(self):
        return False

    def __contains__(self, k):
        return False


class Settings(OrderedDict):
    """A dictionary with additional additional accessability. Behaviour of a
    `Settings` object should mimic Javascript. The keys of the dictionary are
    restricted to be strings. The `Settings` object can be supplied with a
    `Model` to determine how certain elements should be serialised and also
    how to retrieve defaults for missing entries.

    :Basic interface:

    Accessing attributes of a `Settings` object is identical to getting an item
    from one:

    .. testsetup::

        from cslib import Settings

    .. doctest::

        >>> settings = Settings()
        >>> settings.a = "Omelet du fromage"
        >>> settings.a == settings['a']
        True

    It is possible to assign to an attribute path in a `Settings` object
    without first creating intermediate nested levels of settings::

        >>> settings = Settings()
        >>> 'x' in settings
        False
        >>> settings.x.y.z = 3
        >>> 'x' in settings
        True

    This means that referencing an attribute always returns a value, if the
    attribute does not exist, this will be a `TemporaryEntry`, which is falsy::

        >>> settings = Settings()
        >>> bool(settings.b)
        False

    :Adding a Model:

    A `Model` has to be supplied at construction time; we don't want to muddy
    the object interface with this feature.
    """
    def __init__(self, _data=None, _model=None, **kwargs):
        if _data:
            super(Settings, self).__init__(_data)
        else:
            super(Settings, self).__init__()
        for k, v in kwargs.items():
            self[k] = v

        self._model = _model

    def __setitem__(self, k, v):
        def get_or_create(d, k):
            if k not in d:
                super(Settings, d).__setitem__(k, Settings())
            return super(Settings, d).__getitem__(k)

        if isinstance(v, dict) and not isinstance(v, Settings):
            v = Settings(v)

        keys = k.split('.')
        lowest_obj = reduce(get_or_create, keys[:-1], self)
        OrderedDict.__setitem__(lowest_obj, keys[-1], v)

    def __getitem__(self, k):
        keys = k.split('.')
        obj = reduce(OrderedDict.__getitem__, keys, self)
        return obj

    def __missing__(self, k):
        err_str = "Key {} doesn't match any entry in settings.".format(k)

        if self._model and k in self._model:
            default = self._model[k].default

            if default is None:
                raise KeyError(err_str)

            if callable(default):
                value = default(self)
            else:
                value = default

            self[k] = value
            return value

        raise KeyError(err_str)

    def __dir__(self):
        return self.keys()

    def __getattr__(self, k):
        if k not in self and self._model and k in self._model:
            return self.__missing__(k)

        if k not in self:
            return TemporaryEntry(self, k)

        return self[k]

    def __deepcopy__(self, memo):
        return Settings(
            _data=[(k, deepcopy(v, memo)) for k, v in self.items()],
            _model=self._model)

    def __setstate__(self, rec):
        self._model = rec['_model']

    def __setattr__(self, k, v):
        if k[0] == '_':
            self.__dict__[k] = v
            return

        if isinstance(v, dict) and not isinstance(v, Settings):
            v = Settings(**v)

        self[k] = v


def identity(x):
    return x


class Type(object):
    """The type of a single setting.

    .. :py:attribute:: description
        (str) Describing the meaning of the setting.

    .. :py:attribute:: default
        (any) Default value of the setting.

    .. :py:attribute:: check
        (any -> bool) A function that checks the validity
        of a setting.

    .. :py:attribute:: obligatory
        (bool) Is the parameter obligatory?

    .. :py:attribute:: generator
        (any -> str) A function that transforms the value into a string
        suitable for this application. By default the `__str__` method
        (usual Python print method) is used.

    .. :py:attribute:: parser
        (str|number|dict -> any) Inverse of generator. Takes JSON
        compatible data.
    """
    def __init__(self, description, default=None, check=None,
                 obligatory=False, generator=identity,
                 parser=identity):
        self.description = description
        self.default = default
        self.check = check
        self.obligatory = obligatory
        self.generator = generator
        self.parser = parser

    def restructured_text(self, prefix=''):
        """Prints information in reStructured Text layout, suitable for
        inclusion in Sphinx doc."""
        if isinstance(self.check, Predicate):
            check_str = self.check.display()
        else:
            check_str = self.check.__name__

        if callable(self.default):
            default_str = '<computed from other settings>'
        else:
            default_str = str(self.default)

        return textwrap.indent('\n'.join(textwrap.wrap(
                self.description, width=66 - len(prefix))), prefix+'⋮ ') + \
            '\n' + prefix + '' + \
            '\n' + prefix + '(default) ' + default_str + \
            '\n' + prefix + '(type)    ' + check_str

    def display(self, prefix=''):
        if isinstance(self.check, Predicate):
            check_str = self.check.display()
        else:
            check_str = self.check.__name__

        if callable(self.default):
            default_str = '<computed from other settings>'
        else:
            default_str = str(self.default)

        return textwrap.indent('\n'.join(textwrap.wrap(
                self.description, width=66 - len(prefix))), prefix+'⋮ ') + \
            '\n' + prefix + '' + \
            '\n' + prefix + '(default) ' + default_str + \
            '\n' + prefix + '(type)    ' + check_str


class Model(Settings):
    """Settings can be matched against a template to check correctness of
    data types and structure etc. At the same time the template can act as
    a way to create default settings.

    A :py:class:`Model` can only contain objects of class :py:class:`Type`
    and :py:class:`Model`. There is a special :py:class:`ModelType` that
    automates validation of nested :py:class:`Model` instances."""
    def __init__(self, _data=None, **kwargs):
        super(Model, self).__init__(**kwargs)
        if _data is not None:
            for k, v in _data:
                self[k] = v

    def __setitem__(self, k, v):
        if isinstance(v, Model):
            return super(Model, self).__setitem__(k, v)

        if isinstance(v, Type):
            return super(Model, self).__setitem__(k, v)

        if isinstance(v, dict):
            return super(Model, self).__setitem__(k, Model(**v))

        raise TypeError(
            "Model object can only contain Model "
            "or Type: got {}".format(v))


def parse_to_model(model, data):
    """Takes a `Model` and generic `dict` like data. Returns a `Settings`
    object where the items in the dictionary have been parsed following
    the parsers specified in the model."""
    s = Settings(_model=model)
    for k, v in data.items():
        if k not in model:
            raise KeyError("Key {k} not in model.".format(k=k))
        s[k] = model[k].parser(v)
    return s


def generate_settings(settings):
    """Create a `CommentedMap` from a `Settings` object. If the settings have
    an underlying `Model`, that model is used to generate output, otherwise
    `str` is called on the values. This function can be considered to be the
    inverse of `parse_to_model`."""
    if hasattr(settings, '_model'):
        return yaml.comments.CommentedMap(
                (k, settings._model[k].generator(v))
                for k, v in settings.items())
    else:
        return yaml.comments.CommentedMap(
                (k, str(v))
                for k, v in settings.items())


class ModelType(Type):
    """Specialisation of the Type class, in the case of nested settings.
    Since a Model is a subclass  of Settings we cannot really attach
    metadata to the object, but we do want to automate type checking for
    nested settings, since it is a common thing to do.

    This allows giving a model a name and adding a description."""
    def __init__(self, m: Model, name: str, description: str, check=None,
                 obligatory=False):
        if check is not None:
            check = conforms(m, name) & check
        else:
            check = conforms(m, name)

        super(ModelType, self).__init__(
                description,
                check=check, obligatory=obligatory,
                parser=lambda d: parse_to_model(m, d),
                generator=lambda d: generate_settings(d))
        self.model = m

    def display(self, prefix=''):
        return textwrap.indent('\n'.join(textwrap.wrap(
                self.description, width=66 - len(prefix))), prefix+'⋮ ') + \
            '\n' + prefix + '\n' + \
            ('\n' + prefix + '\n').join(
                prefix + '+ ' + k + '\n' +
                self.model[k].display(prefix=prefix+'|   ')
                for k in self.model.keys())


def check_settings(s: Settings, d: Model):
    for k, v in s.items():
        if not d[k].check(v):
            raise TypeError(
                    "Type-check for setting `{}` failed: {}".format(k, v))
    return True


@predicate("Settings")
def is_settings(obj):
    return isinstance(obj, Settings)


def conforms(m: Model, description=""):
    """Returns a `Predicate` that checks if a value conforms a certain
    `Model`."""
    @predicate("Model <{}>".format(description))
    def _conforms(s: Settings):
        if not is_settings(s):
            return False
        return check_settings(s, m)

    return _conforms


def each_value_conforms(m: Model, description=""):
    """Returns a `Predicate` that checks if a value is a `Settings` object, of
    which each value conforms the given `Model`."""
    @predicate("{{key: Model <{}>}}".format(description))
    def _each_value_conforms(s: Settings):
        if not is_settings(s):
            return False

        for v in s.values():
            if not check_settings(v, m):
                return False
        else:
            return True

    return _each_value_conforms
