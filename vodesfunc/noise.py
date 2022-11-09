import vapoursynth as vs
core = vs.core

from typing import Sequence
from vstools import get_depth, scale_value, split, normalize_seq, get_neutral_value, get_peak_value

__all__ = [
    'adaptive_grain', 'grain'
]

def adaptive_grain(clip: vs.VideoNode, strength: float | list[float] = [2.0, 0.5], size: float | list[float] = 3, 
    type: int = 3, static: bool = False, temporal_average: int = 25, luma_scaling = 6, seed: int = -1, temporal_radius: int = 3,
    fade_edges: bool = True, tv_range: bool = True, lo: int | Sequence[int] | None = None, hi: int | Sequence[int] | None = None,
    protect_neutral: bool = True, **kwargs) -> vs.VideoNode:

    """
        Very frankenstein'd mix of setsu's and the original adptvgrnMod
        Only supports https://github.com/wwww-wwww/vs-noise and has some stuff I don't need stripped out.

        :param clip:                Input clip.
        :param strength:            Grainer strength. Use a list to specify [luma, chroma] graining.
                                    Default chroma grain is luma / 2.
        :param size:                Grain size. Will be passed as xsize and ysize. Can be adjusted individually with a list.
        :param type:                See vs-noise github for 0-3. Type 4 is type 2 on a higher res clip and downscaled
        :param static:              Static or dynamic grain.
        :param fade_edges:          Keeps grain from exceeding legal range.
                                    With this, values whiclip.height go towards the neutral point, but would generate
                                    illegal values if they pointed in the other direction are also limited.
                                    This is better at maintaining average values and prevents flickering pixels on OLEDs.
        :param tv_range:            TV or PC legal range.
        :param lo:                  Overwrite legal range's minimums. Value is scaled from 8-bit to clip depth.
        :param hi:                  Overwrite legal range's maximums. Value is scaled from 8-bit to clip depth.
        :param protect_neutral:     Disable chroma grain on neutral chroma.
        :param seed:                Grain seed for the grainer.
        :param temporal_average:    Reference frame weighting for temporal softening and grain consistency.
        :param temporal_radius:     How many frames the averaging will use.
        :param kwargs:              Kwargs passed to the grainer.
        
        :returns: Grained clip.
    """
    
    strength = strength if isinstance(strength, list) else [strength, 0.5 * strength]
    size = size if isinstance(size, list) else [size, size]

    mask = core.adg.Mask(clip.std.PlaneStats(), luma_scaling)
    ogdepth = get_depth(clip)

    def scale_val8x(value: int, chroma: bool = False) -> float:
        return scale_value(value, 8, ogdepth, scale_offsets=not tv_range, chroma=chroma)

    neutral = [get_neutral_value(clip), get_neutral_value(clip, True)]

    if not static and temporal_average > 0:
        length = clip.num_frames + temporal_radius - 1
    else:
        length = clip.num_frames

    if type == 4:
        from vskernels import BicubicDidee
        blank = clip.std.BlankClip(clip.width * 1.3, clip.height * 1.3, length=length, color=normalize_seq(neutral, clip.format.num_planes))
        grained = blank.noise.Add(strength[0], strength[1], type=2, xsize=size[0] * 0.9, ysize=size[1] * 0.9, seed=seed, constant=static, **kwargs)
        grained = BicubicDidee().scale(grained, clip.width, clip.height)
    elif type > 4 or type < 0:
        raise ValueError('adaptive_grain: Type has to be a number between 0 and 4')
    else: 
        blank = clip.std.BlankClip(clip.width, clip.height, length=length, color=normalize_seq(neutral, clip.format.num_planes))
        grained = blank.noise.Add(strength[0], strength[1], type=type, xsize=size[0], ysize=size[1], seed=seed, constant=static, **kwargs)

    if not static and temporal_average > 0:
        grained = core.std.Merge(grained, core.std.AverageFrames(grained, weights=[1] * 3), weight=temporal_average / 100)
    
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