import inspect
from typing import Callable, TypeVar

T = TypeVar("T")


def bind_skip_invalid(signature: inspect.Signature, *args, **kwargs):
    bound = signature.bind_partial()
    for a in args:
        try:
            bound = signature.bind_partial(*bound.args, a)
        except TypeError:
            ...
    for n, a in kwargs.items():
        try:
            bound = signature.bind_partial(*bound.args, **bound.kwargs, **{n: a})
        except TypeError:
            ...
    return bound


def call_with_needed_arguments(_function: Callable[..., T], *args, **kwargs) -> T:
    b = bind_skip_invalid(inspect.signature(_function), *args, **kwargs)
    b.apply_defaults()
    return _function(*b.args, **b.kwargs)
