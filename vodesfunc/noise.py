import vapoursynth as vs
core = vs.core

from typing import Sequence, Callable
from vstools import get_depth, scale_value, split, normalize_seq, get_neutral_value, get_peak_value, mod4
from vskernels import Scaler, Lanczos, BicubicDidee

__all__ = [
    'adaptive_grain', 'grain', 'ntype4'
]

ntype4 = {"type": 2, "scale": 0.7, "scaler": BicubicDidee()}

def adaptive_grain(clip: vs.VideoNode, strength: float | list[float] = [2.0, 0.5], size: float | list[float] = 3, 
    type: int = 3, static: bool = False, temporal_average: int = 25, luma_scaling: float = 6, seed: int = -1, temporal_radius: int = 3,
    scale: float = 1, scaler: Scaler = Lanczos(), post_grain: Callable[[vs.VideoNode], vs.VideoNode] | None = None,
    fade_edges: bool = True, tv_range: bool = True, lo: int | Sequence[int] | None = None, hi: int | Sequence[int] | None = None,
    protect_neutral: bool = True, **kwargs) -> vs.VideoNode:

    """
        Very frankenstein'd mix of setsu's and the original adptvgrnMod
        Only supports https://github.com/wwww-wwww/vs-noise and has some stuff I don't need stripped out.

        :param clip:                Input clip.
        :param strength:            Grainer strength. Use a list to specify [luma, chroma] graining.
                                    Default chroma grain is luma / 5.
        :param size:                Grain size. Will be passed as xsize and ysize. Can be adjusted individually with a list.
                                    This should not be confused with the resizing of adptvgrnMod. For something similar, use the `scale` param.
        :param type:                See vs-noise github for 0-3. Type 4 is type 2 with a 0.7 scale and using BicubicDidee as the scaler.
        :param static:              Static or dynamic grain.
        :param seed:                Grain seed for the grainer.
        :param temporal_average:    Reference frame weighting for temporal softening and grain consistency.
        :param temporal_radius:     How many frames the averaging will use.
        :param luma_scaling:        Luma scaling passed to the adaptivegrain mask. While use the absolute value on an inverted clip if a negative number is passed.
                                    Mainly useful for graining the bright parts of an image.
        :param scale:               Makes the grain bigger if > 1 and smaller if < 1 by graining a different sized blankclip and scaling to clip res after.
                                    Can be used to tweak sharpness/frequency considering vs-noise always keeps those the same no matter the size.
        :param scaler:              Scaler/Kernel used for down- or upscaling the grained blankclip.
        :param post_grain:          A callable function to run on the grained blankclip pre scaling. An example use would be to sharpen like I did for something.

        :param fade_edges:          Keeps grain from exceeding legal range.
                                    With this, values whiclip.height go towards the neutral point, but would generate
                                    illegal values if they pointed in the other direction are also limited.
                                    This is better at maintaining average values and prevents flickering pixels on OLEDs.
        :param tv_range:            TV or PC legal range.
        :param lo:                  Overwrite legal range's minimums. Value is scaled from 8-bit to clip depth.
        :param hi:                  Overwrite legal range's maximums. Value is scaled from 8-bit to clip depth.
        :param protect_neutral:     Disable chroma grain on neutral chroma.
        :param kwargs:              Kwargs passed to the grainer.
        
        :returns: Grained clip.
    """
    
    strength = strength if isinstance(strength, list) else [strength, 0.2 * strength]
    size = size if isinstance(size, list) else [size, size]

    if type > 4 or type < 0:
        raise ValueError('adaptive_grain: Type has to be a number between 0 and 4')

    if scale >= 2:
        raise ValueError('adaptive_grain: Scale has to be a number below 2. (Default is 1, to disable scaling)')

    mask = core.adg.Mask(clip.std.PlaneStats() if luma_scaling >= 0 else clip.std.Invert().std.PlaneStats(), abs(luma_scaling))
    ogdepth = get_depth(clip)

    def scale_val8x(value: int, chroma: bool = False) -> float:
        return scale_value(value, 8, ogdepth, scale_offsets=not tv_range, chroma=chroma)

    neutral = [get_neutral_value(clip), get_neutral_value(clip, True)]

    if not static and temporal_average > 0:
        length = clip.num_frames + temporal_radius - 1
    else:
        length = clip.num_frames

    width = clip.width - (clip.width * scale - clip.width)
    height = clip.height - (clip.height * scale - clip.height)

    if scale != 1:
        width = mod4(width)
        height = mod4(height)

    blank = clip.std.BlankClip(width, height, length=length, color=normalize_seq(neutral, clip.format.num_planes))
    grained = blank.noise.Add(strength[0], strength[1], type=type, xsize=size[0], ysize=size[1], seed=seed, constant=static, **kwargs)

    if callable(post_grain):
        grained = post_grain(grained)

    grained = scaler.scale(grained, clip.width, clip.height)

    if not static and temporal_average > 0:
        cut = (temporal_radius - 1) // 2
        grained = core.std.Merge(grained, core.std.AverageFrames(grained, weights=[1] * temporal_radius), weight=temporal_average / 100)
        grained = grained[cut:-cut]

    if fade_edges:
        if lo is None:
            lo = [scale_val8x(16), scale_val8x(16, True)]
        elif not isinstance(lo, list):
            lo = [scale_val8x(lo), scale_val8x(lo, True)]

        if hi is None:
            hi = [scale_val8x(235), scale_val8x(240, True)]
        elif not isinstance(hi, list):
            hi = [scale_val8x(hi), scale_val8x(hi, True)]

        limit_expr = "x y {0} - abs - {1} < x y {0} - abs + {2} > or x y {0} - x + ?"
        if clip.format.sample_type == vs.INTEGER:
            limit_expr = 2 * [limit_expr]
        else:
            limit_expr = [limit_expr, "x y abs + {2} > x abs y - {1} < or x x y + ?"]

        grained = core.std.Expr([clip, grained], [limit_expr[_].format(
            neutral[_], lo[_], hi[_]) for _ in range(0, clip.format.num_planes - 1)])

        if protect_neutral and strength[1] > 0 and clip.format.color_family == vs.YUV:
            format444 = core.query_video_format(vs.YUV, clip.format.sample_type, ogdepth, 0, 0)
            neutral_mask = clip.resize.Bicubic(format=format444)
            # disable grain if neutral chroma
            neutral_mask = core.std.Expr(split(neutral_mask), f"y {neutral[1]} = z {neutral[1]} = and {get_peak_value(clip)} 0 ?")
            grained = core.std.MaskedMerge(grained, clip, neutral_mask, planes=[1, 2])
    else:
        if clip.format.sample_type == vs.INTEGER:
            grained = core.std.MergeDiff(clip, grained)
        else:
            grained = core.std.Expr([clip, grained], [f"y {neutral[_]} - x +" for _ in range(clip.format.num_planes - 1)])

    return clip.std.MaskedMerge(grained, mask)

grain = adaptive_grain