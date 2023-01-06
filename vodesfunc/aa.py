import vapoursynth as vs
core = vs.core

from .scale import mod_padding

from vsrgtools import unsharp_masked
from vstools import depth, get_y, join, plane

def pre_aa(clip: vs.VideoNode, radius: int = 1, strength: float = 100, opencl: bool = True, **nnedi3_args):
    """
        A prefilter to use in conjunction with an AA function.
        The idea is to fix the luminance uniformity on lineart and make AAing more effective.

        :param radius:      Radius used for the unsharp function
        :param strength:    Strength used for the unsharp function
        :param opencl:      Use nnedi3cl instead of znedi3
        :param nnedi3_args: Additional args passed to the respective nnedi function

        :return:            Processed clip
    """
    args = {"qual": 2, "nsize": 0, "nns": 4, "pscrn": 1}
    args.update(**nnedi3_args)
    clip_y = plane(clip, 0)
    
    if opencl:
        (left, right, top, bottom) = mod_padding(clip_y, 2, 2)
        width = clip.width + left + right
        height = clip.height + top + bottom
        clip_y = clip_y.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)

    for i in range(2):
        bob = clip_y.nnedi3cl.NNEDI3CL(field=3, **nnedi3_args) if opencl else \
            clip_y.znedi3.nnedi3(field=3, **nnedi3_args)
        sharp = unsharp_masked(clip_y, radius, strength)
        limit = core.std.Expr([sharp, clip_y, bob[::2], bob[1::2]], "x y z a max max min y z a min min max")
        clip_y = limit.std.Transpose()
    
    if opencl:
        clip_y = clip_y.std.Crop(left, right, top, bottom)
        clip_y = clip_y.std.CopyFrameProps(clip)

    return clip_y if clip.format.color_family == vs.GRAY else join(clip_y, clip)