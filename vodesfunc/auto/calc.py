from math import trunc
from decimal import ROUND_HALF_DOWN, Decimal
from fractions import Fraction
from datetime import timedelta

__all__: list[str] = [
    'mpls_timestamp_to_timedelta',
    'timedelta_to_frame',
    'frame_to_timedelta',
    'format_timedelta'
]

def _fraction_to_decimal(f: Fraction) -> Decimal:
    return Decimal(f.numerator) / Decimal(f.denominator)

def mpls_timestamp_to_timedelta(timestamp: int) -> timedelta:
    seconds = Decimal(timestamp) / Decimal(45000)
    return timedelta(seconds = float(seconds))

def timedelta_to_frame(time: timedelta, fps: Fraction = Fraction(24000, 1001)) -> int:
    s = Decimal(time.total_seconds())
    fps_dec = _fraction_to_decimal(fps)
    return round((s * fps_dec))

def frame_to_timedelta(f: int, fps: Fraction = Fraction(24000, 1001)) -> timedelta:
    fps_dec = _fraction_to_decimal(fps)
    seconds = Decimal(f) / fps_dec
    return timedelta(seconds = float(seconds))

def format_timedelta(time: timedelta, precision: int = 3) -> str:
    dec = Decimal(time.total_seconds())
    pattern = "." + ''.join(["0"] * (precision - 1)) + "1"
    rounded = float(dec.quantize(Decimal(pattern), rounding=ROUND_HALF_DOWN))
    s = trunc(rounded)
    m = s // 60
    s %= 60
    h = m // 60
    m %= 60
    return f'{h:02d}:{m:02d}:{s:02d}.{str(rounded).split(".")[1].zfill(precision)}'

