from pathlib import Path
from typing import TypeVar, Optional, Union

PathLike = Union[str, Path]

_T = TypeVar("_T")


def cast_none(x: Optional[_T]) -> _T:
    """removes None type hint. use when you certain it's not None"""
    return x  # type: ignore