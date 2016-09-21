from noodles.serial import (Serialiser, Registry, base)
from noodles.serial.numpy import registry as numpy_registry

from .dataframe import DataFrame, DCS
from .units import units


class SerQuantity(Serialiser):
    def encode(self, obj, make_rec):
        return make_rec(obj.to_tuple())

    def decode(self, cls, data):
        return units.Quantity.from_tuple(data)


class SerUnit(Serialiser):
    def encode(self, obj, make_rec):
        return make_rec(str(obj))

    def decode(self, cls, data):
        return units(data)


class SerStandardObject(Serialiser):
    def __init__(self, cls, items):
        super(SerStandardObject, self).__init__(cls)
        self.items = items

    def encode(self, obj, make_rec):
        return make_rec({k: getattr(obj, k) for k in self.items})

    def decode(self, cls, data):
        return cls(**data)


def quantity_hook(obj):
    if isinstance(obj, units.Quantity):
        return '<quantity>'

    if type(obj).__name__ == 'Quantity':
        return '<quantity>'

    if isinstance(obj, units.Unit):
        return '<unit>'

    return None


def registry():
    return Registry(
        parent=base() + numpy_registry(),
        types={
            DataFrame: SerStandardObject(
                DataFrame, ['data', 'units', 'comments']),
            DCS: SerStandardObject(
                DCS, ['energy', 'angle', 'cs'])
        },
        hooks={
            '<quantity>': SerQuantity('<quantity>'),
            '<unit>': SerUnit('<unit>')
        },
        hook_fn=quantity_hook)
