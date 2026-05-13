"""
Oh god no
"""

__version__: str
__version_tuple__: tuple[int | str, ...]

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0+unknown"
    __version_tuple__ = (0, 0, 0, "+unknown")

# flake8: noqa

from . import misc, noise, scale, denoise, rescale, util, spikefinder, rescale_ext
from .misc import *
from .noise import *
from .scale import *
from .rescale import *
from .denoise import *
from .util import *
from .spikefinder import *

from .rescale_ext.mixed_rescale import *
