from enum import IntEnum
from pathlib import Path
from typing import TypeVar, Union, Optional
from datetime import timedelta

__all__: list[str] = [
    'PathLike', 'Paths',
    'Trim',
    'Zone',
    'TrackType',
]

PathLike = TypeVar("PathLike", str, Path)
Trim = tuple[int | None, int | None]
Zone = tuple[int, int, float | str, str | None]

Paths = Union[PathLike, list[PathLike]]

# Timedelta (or frame, which will be converted internally), Optional Name
Chapter = tuple[timedelta | int, Optional[str]]

class TrackType(IntEnum):
    VIDEO = 1
    AUDIO = 2
    SUB = 3
    ATTACHMENT = 4
    CHAPTERS = 5
    MKV = 6