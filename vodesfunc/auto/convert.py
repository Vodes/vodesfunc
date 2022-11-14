from math import trunc
from decimal import ROUND_HALF_DOWN, Decimal
from fractions import Fraction
from datetime import timedelta
import vapoursynth as vs

__all__: list[str] = [
    'mpls_timestamp_to_timedelta',
    'timedelta_to_frame',
    'frame_to_timedelta',
    'format_timedelta',
    'timedelta_from_formatted',
    'frames_to_samples'
]

def _fraction_to_decimal(f: Fraction) -> Decimal:
    return Decimal(f.numerator) / Decimal(f.denominator)

def mpls_timestamp_to_timedelta(timestamp: int) -> timedelta:
    """
        Converts a mpls timestamp (from BDMV Playlist files) to a timedelta.

        :param timestamp:       The mpls timestamp

        :return:                The resulting timedelta
    """
    seconds = Decimal(timestamp) / Decimal(45000)
    return timedelta(seconds = float(seconds))

def timedelta_to_frame(time: timedelta, fps: Fraction = Fraction(24000, 1001)) -> int:
    """
        Converts a timedelta to a frame number.

        :param time:    The timedelta
        :param fps:     A Fraction containing fps_num and fps_den

        :return:        The resulting frame number
    """

    s = Decimal(time.total_seconds())
    fps_dec = _fraction_to_decimal(fps)
    return round((s * fps_dec))

def frame_to_timedelta(f: int, fps: Fraction = Fraction(24000, 1001)) -> timedelta:
    """
        Converts a frame number to a timedelta.
        Mostly used in the conversion for manually defined chapters.

        :param f:       The frame number
        :param fps:     A Fraction containing fps_num and fps_den

        :return:        The resulting timedelta
    """
    fps_dec = _fraction_to_decimal(fps)
    seconds = Decimal(f) / fps_dec
    return timedelta(seconds = float(seconds))

def format_timedelta(time: timedelta, precision: int = 3) -> str:
    """
        Formats a timedelta to hh:mm:ss.s[*precision] and pads with 0 if there aren't more numbers to work with.
        Mostly to be used for ogm/xml files.

        :param time:        The timedelta
        :param precision:   3 = milliseconds, 6 = microseconds, 9 = nanoseconds

        :return:            The formatted string
    """
    dec = Decimal(time.total_seconds())
    pattern = "." + ''.join(["0"] * (precision - 1)) + "1"
    rounded = float(dec.quantize(Decimal(pattern), rounding=ROUND_HALF_DOWN))
    s = trunc(rounded)
    m = s // 60
    s %= 60
    h = m // 60
    m %= 60
    return f'{h:02d}:{m:02d}:{s:02d}.{str(rounded).split(".")[1].ljust(precision, "0")}'

def timedelta_from_formatted(formatted: str) -> timedelta:
    """
        Parses a string with the format of hh:mm:ss.sss
        Mostly to be used for ogm/xml files.

        :param formatted:       The timestamp string
        
        :return:                The parsed timedelta
    """
    # 00:05:25.534...
    seconds: float = 0.0
    split = formatted.split(':')
    seconds += float(split[0]) * 3600
    seconds += float(split[1]) * 60
    seconds += float(split[2])
    return timedelta(seconds=seconds)

def frames_to_samples(frame: int, sample_rate: vs.AudioNode | int = 48000, fps: vs.VideoNode | Fraction = Fraction(24000, 1001)) -> int:
    """
        Converts a frame number to a sample number

        :param frame:           The frame number
        :param sample_rate:     Can be a flat number like 48000 (=48 kHz) or an AudioNode to get the sample rate from
        :param fps:             Can be a Fraction or a VideoNode to get the fps from

        :return:                The sample number
    """
    sample_rate = sample_rate.sample_rate if isinstance(sample_rate, vs.AudioNode) else sample_rate
    fps = Fraction(fps.fps_num, fps.fps_den) if isinstance(fps, vs.VideoNode) else fps
    return int(sample_rate * (fps.denominator / fps.numerator) * frame)