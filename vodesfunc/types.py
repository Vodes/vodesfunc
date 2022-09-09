from pathlib import Path
from typing import TypeVar, Union

__all__: list[str] = [
    'PathLike', 'Paths',
    'Trim',
    'Zone',
]

PathLike = TypeVar("PathLike", str, Path)
Trim = tuple[int | None, int | None]
Zone = tuple[int, int, float | str, str | None]

Paths = Union[PathLike, list[PathLike]]
