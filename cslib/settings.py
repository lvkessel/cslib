from functools import (reduce)
from copy import (copy, deepcopy)
from collections import OrderedDict

from .predicates import (Predicate, predicate)

from ruamel import yaml
import textwrap


class TemporaryEntry(object):
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
    """A dictionary with additional additional accessability.
    Behaviour of a `Settings` object should mimic Javascript.
    The keys of the dictionary are restricted to be strings.
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

    # def __getstate__(self):
    #    return {'data': super(Settings, self), 'model': self._model}

    def __setstate__(self, rec):
        self._model = rec['_model']
        # self._model = rec['model']
        # for k, v in rec['data']:
        #    self[k] = v

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

    .. :py:attribute:: transformer
        (any -> str) A function that transforms the value into a string
        suitable for this application. By default the `__str__` method
        (usual Python print method) is used.

    .. :py:attribute:: parser
        (str|number|dict -> any) Inverse of transformer. Takes JSON
        compatible data.
    """
    def __init__(self, description, default=None, check=None,
                 obligatory=False, transformer=identity,
                 parser=identity):
        self.description = description
        self.default = default
        self.check = check
        self.obligatory = obligatory
        self.transformer = transformer
        self.parser = parser

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
    a way to create default settings."""
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
    s = Settings(_model=model)
    for k, v in data.items():
        if k not in model:
            raise KeyError("Key {k} not in model.".format(k=k))
        s[k] = model[k].parser(v)
    return s


def transform_settings(model, settings):
    return yaml.comments.CommentedMap(
            (k, model[k].transformer(v))
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
                transformer=lambda d: transform_settings(m, d))
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
    @predicate("Model <{}>".format(description))
    def _conforms(s: Settings):
        if not is_settings(s):
            return False
        return check_settings(s, m)

    return _conforms


def each_value_conforms(m: Model, description=""):
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


def apply_defaults_and_check(s: Settings, d: Model):
    s = Settings(**s)

    for k in d:
        if k not in s:
            if isinstance(d[k], Model):
                s[k] = Settings()
            elif d[k].obligatory:
                raise Exception("Setting `{}` is obligatory but was not"
                                " given.".format(k))
            else:
                s[k] = copy(d[k].default)

        if isinstance(d[k], Model):
            if not isinstance(s[k], Settings):
                raise TypeError(
                    "Sub-folder {} of settings should be a collection."
                    .format(k))
            s[k] = apply_defaults_and_check(s[k], d[k])

        if not d[k].check(s[k]):
            raise TypeError(
                    "Type-check for setting `{}` failed: {}".format(k, s[k]))

    return s
