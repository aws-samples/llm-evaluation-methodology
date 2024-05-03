# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for de/serializing Python objects to/from JSON"""
# Python Built-Ins:
from datetime import date, time
from enum import Enum
from json import dump, dumps


def default_serializer(obj):
    """Handler for objects not serializable by default Python json.dump(s)

    Usage
    -----
    >>> import json
    >>> json.dumps(my_object, default=default_serializer)
    """

    if isinstance(obj, date) or isinstance(obj, time):
        # ISO-able date/time objects:
        return obj.isoformat()

    if isinstance(obj, Enum):
        # Enum instances just take their underlying value, ignoring their 'name':
        return obj.value

    try:
        # Arbitrary object-like classes:
        return obj.__dict__
    except AttributeError:
        return None


def json_dump(obj, fp, **kwargs) -> None:
    """Wrapper around json.dump() that handles custom object serialization

    Parameters
    ----------
    obj :
        The object to be serialized
    fp :
        The file to serialize to
    **kwargs :
        As per standard `json.dump()`, but the `default` argument is already set"""
    return dump(obj, fp, default=default_serializer, **kwargs)


def json_dumps(obj, **kwargs) -> str:
    """Wrapper around json.dumps() that handles custom object serialization

    Parameters
    ----------
    obj :
        The object to be serialized
    **kwargs :
        As per standard `json.dump()`, but the `default` argument is already set"""
    return dumps(obj, default=default_serializer, **kwargs)
