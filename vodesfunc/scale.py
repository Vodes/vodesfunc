from typing import Any, Callable

import vapoursynth as vs
from vskernels import Catrom, Kernel, Scaler
from vstools import depth, get_depth, get_y, Matrix
from vsrgtools.sharp import unsharp_masked
from .types import PathLike
from abc import ABC, abstractmethod
core = vs.core


__all__: list[str] = [
    'NNEDI_Doubler',
    'Clamped_Doubler',
    'Shader_Doubler',
    'Waifu2x_Doubler',
    'vodes_rescale',
]

class Doubler(ABC):
    
    kwargs: dict[str, Any]
    """Arguments passed to the internal scale function"""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def double(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
            Returns doubled clip
        """
        pass

class NNEDI_Doubler(Doubler):
    ediargs: dict[str, Any]
    opencl: bool 

    def __init__(self, opencl: bool = True, nns: int = 4, nsize: int = 4, qual: int = 2, pscrn: int = 1, **kwargs) -> None:
        """
            Simple utility class for doubling a clip using znedi or nnedi3cl (also fixes the shift)

            :param opencl:          Will use nnedi3cl if True and znedi3 if False
        """
        self.ediargs = {"qual": qual, "nsize": nsize, "nns": nns, "pscrn": pscrn}
        self.ediargs.update(**kwargs)
        self.opencl = opencl

    def double(self, clip: vs.VideoNode, correct_shift: bool = True) -> vs.VideoNode:
        y = get_y(clip)
        # nnedi3cl needs padding, to avoid issues on edges (https://slow.pics/c/QcJef38u)
        if self.opencl:
            (left, right, top, bottom) = mod_padding(y, 2, 2)
            width = clip.width + left + right
            height = clip.height + top + bottom
            pad = y.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)
            doubled_y = pad.nnedi3cl.NNEDI3CL(dh=True, field=0, **self.ediargs).std.Transpose() \
                .nnedi3cl.NNEDI3CL(dh=True, field=0, **self.ediargs).std.Transpose()
            doubled_y = doubled_y.std.Crop(left * 2, right * 2, top * 2, bottom * 2)
        else:
            doubled_y = depth(y, 16).znedi3.nnedi3(dh=True, field=0, **self.ediargs).std.Transpose() \
                .znedi3.nnedi3(dh=True, field=0, **self.ediargs).std.Transpose()
            doubled_y = depth(doubled_y, get_depth(clip))
        
        if correct_shift:
            doubled_y = doubled_y.resize.Bicubic(src_top=.5, src_left=.5)

        return doubled_y.std.CopyFrameProps(y)

class Shader_Doubler(Doubler):
    shaderfile: str 

    def __init__(self, shaderfile: PathLike = r'C:\FSRCNNX_x2_56-16-4-1.glsl') -> None:
        """
            Simple utility class for doubling a clip using a glsl shader

            :param shaderfile:      The glsl shader used to double the resolution
        """
        self.shaderfile = shaderfile if isinstance(shaderfile, str) else str(shaderfile.resolve())

    def double(self, clip: vs.VideoNode) -> vs.VideoNode:
        y = depth(get_y(clip), 16)
        filler_chroma = core.std.BlankClip(y, format=vs.YUV444P16)
        doubled = core.std.ShufflePlanes([y, filler_chroma], [0, 1, 2], vs.YUV) \
            .placebo.Shader(self.shaderfile, filter='box', width=y.width*2, height=y.height*2)
        doubled_y = get_y(doubled)
        return depth(doubled_y, get_depth(clip))

class Waifu2x_Doubler(Doubler):
    backend: any
    kwargs: dict[str, Any]

    def __init__(self, cuda: bool | str = 'trt', fp16: bool = True, num_streams: int = 1, **kwargs) -> None:
        """
            Simple utility class for doubling a clip using Waifu2x

            :param cuda:            ORT-Cuda if True, NCNN-VK if False, TRT if some string
            :param fp16:            Uses 16 bit floating point internally if True
            :param num_streams:     Amount of streams to use for Waifu2x; Sacrifices a lot of vram for a speedup
            :param w2xargs:         Args that get passed to Waifu2x
        """
        from vsmlrt import Backend
        self.backend = Backend.ORT_CUDA(num_streams=num_streams, fp16=fp16) if cuda == True else \
            Backend.NCNN_VK(num_streams=num_streams, fp16=fp16) if cuda == False else Backend.TRT(num_streams=num_streams, fp16=fp16)
        self.kwargs = kwargs

    def double(self, clip: vs.VideoNode) -> vs.VideoNode:
        from vsmlrt import Waifu2x
        pre = depth(clip, 32).std.Limiter()

        (left, right, top, bottom) = mod_padding(pre)
        width = pre.width + left + right
        height = pre.height + top + bottom
        pad = pre.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)

        was_444 = pre.format.color_family == vs.YUV and pre.format.subsampling_w == 0 and pre.format.subsampling_h == 0

        if was_444:
            pad = Catrom().resample(pad, format=vs.RGBS, matrix=Matrix.RGB, matrix_in=Matrix.from_video(pre))
        else: 
            pad = get_y(pad).std.ShufflePlanes(0, vs.RGB)

        up = Waifu2x(pad, noise=-1, model=6, backend=self.backend, **self.kwargs)

        if was_444:
            up = Catrom().resample(up, format=vs.YUV444PS, matrix=Matrix.from_video(pre), matrix_in=Matrix.RGB)
        else:
            up = up.std.ShufflePlanes(0, vs.GRAY)

        up = up.std.Crop(left * 2, right * 2, top * 2, bottom * 2)
        up = up.std.Expr("x 0.5 255 / +")
        return depth(up, get_depth(clip)).std.CopyFrameProps(pre)
 
class Clamped_Doubler(Doubler):

    sharp_doubler: Doubler
    sharpen_smooth: bool | vs.VideoNode | Callable[[vs.VideoNode], vs.VideoNode] = None

    def __init__(self, sharpen_smooth: bool | vs.VideoNode | Callable[[vs.VideoNode], vs.VideoNode] = True, 
            sharp_doubler: Doubler | str = Shader_Doubler(), ratio: int = 100, **kwargs) -> None:
        """
            Simple utility class for doubling a clip using fsrcnnx / any shader clamped to nnedi.
            Using sharpen will be basically the same as the zastin profile in varde's fsrcnnx upscale.
            Not sharpening will on the other hand be the same as the slow profile.

            :param sharpen_smooth:  Sharpened "smooth upscale" clip or a sharpener function. Will use z4usm if True.
                                    Uses the other mode if False or None.
            :param sharp_doubler:   The doubler used for the sharp upscale. Defaults to Shader_Doubler (which defaults to fsrcnnx)
            :param ratio:           Does a weighted average of the clamped and nnedi clips. 
                                    The higher, the more of the clamped clip will be used.
            :param kwargs:          You can pass all kinds of stuff here, ranging from the default sharpener params to nnedi args.
                                    z4usm params: radius (default 2), strength (default 35)
                                    nnedi params: see `NNEDI_Doubler`
                                    overshoot, undershoot for non-sharpen mode (defaults to ratio / 100)
        """
        self.sharp_doubler = Shader_Doubler(sharp_doubler) if isinstance(sharp_doubler, str) else sharp_doubler
        self.sharpen_smooth = sharpen_smooth

        if ratio > 100 or ratio < 1:
            raise "Clamped_Doubler: ratio should be a value between 1 and 100"
        self.ratio = ratio
        self.kwargs = kwargs

    def double(self, clip: vs.VideoNode) -> vs.VideoNode:
        y = depth(get_y(clip), 16)

        overshoot = self.kwargs.pop("overshoot", self.ratio / 100)
        undershoot = self.kwargs.pop("undershoot", overshoot)
        radius = self.kwargs.pop("radius", 2)
        strength = self.kwargs.pop("strength", 35)

        smooth = NNEDI_Doubler(**self.kwargs).double(y)
        shader = self.sharp_doubler.double(y)

        if self.sharpen_smooth != None and self.sharpen_smooth != False:
            if isinstance(self.sharpen_smooth, vs.VideoNode):
                sharpened_smooth = self.sharpen_smooth
            elif isinstance(self.sharpen_smooth, Callable):
                sharpened_smooth = self.sharpen_smooth(smooth)
            elif self.sharpen_smooth == True:
                sharpened_smooth = unsharp_masked(smooth, radius, strength)
            
            clamped = core.std.Expr([smooth, shader, sharpened_smooth], 'x y z min max y z max min')
            if self.ratio != 100:
                clamped = core.std.Expr(
                    [clamped, smooth], f"{self.ratio / 100} x * {1 - (self.ratio / 100)} y * +")
        else:
            upscaled = core.std.Expr([shader, smooth], 'x {ratio} * y 1 {ratio} - * +'.format(ratio=self.ratio / 100))
            dark_limit = core.std.Minimum(smooth)
            bright_limit = core.std.Maximum(smooth)
            overshoot *= 2**8
            undershoot *= 2**8
            clamped = core.std.Expr(
                    [upscaled, bright_limit, dark_limit],
                    f'x y {overshoot} + > y {overshoot} + x ? z {undershoot} - < z {undershoot} - x y {overshoot} + > y {overshoot} + x ? ?'
            )
        return depth(clamped, get_depth(clip))

def vodes_rescale(
    src: vs.VideoNode, height: float = 0,
    descale_kernel: Kernel | list[vs.VideoNode] = Catrom(),
    doubler: Kernel | Scaler | Callable[[vs.VideoNode], vs.VideoNode] | Doubler = NNEDI_Doubler(),
    downscaler: Kernel | Scaler | str = Catrom(),
    line_mask: vs.VideoNode | bool = None, credit_mask: vs.VideoNode = None, mask_threshold: float = 0.04,
    use_baseheight_mask: bool = False, width: float = None, base_height: float = None, base_width: float = None,
    do_post_double: Callable[[vs.VideoNode], vs.VideoNode] = None
) -> list[vs.VideoNode | None]:
    """
    More or less just a compatibility layer that uses :py:class:`DescaleTarget` from this same package.
    
    Check that out to see what the params do.

    :return:                    List of rescaled, reference upscale, credit mask and line mask
    """
    clip = depth(src, 32)
    y = get_y(clip)
    from .descale import DescaleTarget
    desc = DescaleTarget(height, descale_kernel, doubler, downscaler, base_height, width, base_width, do_post_double, credit_mask, mask_threshold, use_baseheight_mask, line_mask, 0)
    desc.generate_clips(y)
    blank_mask = core.std.BlankClip(y)
    return [desc.get_upscaled(src),
            desc.rescale,
            desc.credit_mask if isinstance(desc.credit_mask, vs.VideoNode) else blank_mask,
            desc.line_mask if isinstance(desc.line_mask, vs.VideoNode) else blank_mask]

def mod_padding(clip: vs.VideoNode, mod: int = 4, min: int = 4):
    from math import floor
    width = clip.width + min * 2
    height = clip.height + min * 2
    ph = mod - ((width - 1) % mod + 1)
    pv = mod - ((height - 1) % mod + 1)

    left = floor(ph / 2)
    right = ph - left
    top = floor(pv / 2)
    bottom = pv - top
    return (left + min, right + min, top + min, bottom + min)
