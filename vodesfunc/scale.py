from typing import Any, Callable

import vapoursynth as vs
from vskernels import Catrom, Kernel
from vstools import depth, get_depth, get_y, iterate, ColorRange
from .types import PathLike
from abc import ABC, abstractmethod

core = vs.core


__all__: list[str] = [
    'nnedi_double',
    'double_nnedi', 'NNEDI_Doubler',
    'double_waifu2x', 'Waifu2x_Doubler',
    'double_shader', 'Shader_Doubler', 'Clamped_Doubler',
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
            y = y.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)
            doubled_y = y.nnedi3cl.NNEDI3CL(dh=True, field=0, **self.ediargs).std.Transpose() \
                .nnedi3cl.NNEDI3CL(dh=True, field=0, **self.ediargs).std.Transpose()
            doubled_y = doubled_y.std.Crop(left * 2, right * 2, top * 2, bottom * 2)
        else:
            doubled_y = depth(y, 16).znedi3.nnedi3(dh=True, field=0, **self.ediargs).std.Transpose() \
                .znedi3.nnedi3(dh=True, field=0, **self.ediargs).std.Transpose()
            doubled_y = depth(doubled_y, get_depth(clip))
        
        if correct_shift:
            doubled_y = doubled_y.resize.Bicubic(src_top=.5, src_left=.5)

        return doubled_y

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
    from vsmlrt import Backend
    backend: Backend
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
        y = depth(get_y(clip), 32)
    
        (left, right, top, bottom) = mod_padding(y)
        width = clip.width + left + right
        height = clip.height + top + bottom
        y = y.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)
        
        dsrgb = y.std.ShufflePlanes(0, vs.RGB)
        up = Waifu2x(dsrgb, noise=-1, model=6, backend=self.backend, **self.kwargs)
        up = up.std.ShufflePlanes(0, vs.GRAY)
        up = up.std.Crop(left * 2, right * 2, top * 2, bottom * 2)
        up = up.std.Expr("x 0.5 255 / +")
        return depth(up, get_depth(clip))
 
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
                try:
                    import vardefunc as vdf
                    sharpened_smooth = vdf.sharp.z4usm(smooth, radius, strength)
                except:
                    raise "Clamped_Doubler: Couldn't import vardefunc. Please use a different sharpener or none at all."
            
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
    doubler: Kernel | Callable[[vs.VideoNode], vs.VideoNode] | Doubler = NNEDI_Doubler(),
    downscaler: Kernel | str = Catrom(),
    line_mask: vs.VideoNode | bool = None, credit_mask: vs.VideoNode = None, mask_threshold: float = 0.04,
    width: float = None, do_post_double: Callable[[vs.VideoNode], vs.VideoNode] = None
) -> list[vs.VideoNode | None]:
    """
    Rescale function with masking for convenience etc.

    :param src:             Input clip
    :param height:          Height to be descaled to
    :param width:           Width to be descaled to; will be calculated if you don't pass any
    :param doubler:         A callable or kernel or Doubler class that will be used to upscale
    :param descale_kernel:  Kernel used for descaling, supports passing a list of descaled and ref clip instead
    :param downscaler:      Kernel used to downscale the doubled clip, uses vsscale.ssim_downsample if passed 'ssim'
    :param line_mask:       Linemask to only rescale lineart | Will generate one if None or skip masking if `False` is passed
    :param credit_mask:     Credit Masking | Will generate one if None or skip masking if *mask_threshold* is <= 0
    :param mask_threshold:  Threshold for the diff based credit mask | lower catches more
    :param do_post_double:  Pass your own function as a lambda if you want to manipulate the clip before the downscale happens
    :return:                List of rescaled, reference upscale, credit mask and line mask
    """
    wdepth = 16 if get_depth(src) < 16 else get_depth(src)
    clip = depth(src, wdepth)
    y = clip if clip.format.color_family == vs.GRAY else get_y(clip)

    if height is None or height < 1:
        raise ValueError('Rescale: Height may not be None or 0')

    if width is None or width < 1:
        aspect_ratio = src.width / src.height
        width = height * aspect_ratio

    descaled_y = descale_kernel.descale(y, width, height) if isinstance(descale_kernel, Kernel) else descale_kernel[0]
    ref_y = descale_kernel.scale(descaled_y, src.width, src.height) if isinstance(
        descale_kernel, Kernel) else descale_kernel[1]

    if isinstance(doubler, Doubler):
        doubled_y = doubler.double(descaled_y)
    elif isinstance(doubler, Kernel):
        doubled_y = doubler.scale(descaled_y, descaled_y.width * 2, descaled_y.height * 2)
    else: 
        doubled_y = doubler(descaled_y)

    if do_post_double is not None:
        doubled_y = do_post_double(doubled_y)

    if isinstance(downscaler, str) and downscaler.lower() == 'ssim':
        import vsscale as vss
        rescaled_y = vss.ssim_downsample(depth(doubled_y, 16), src.width, src.height)
        rescaled_y = depth(rescaled_y, wdepth)
    else:
        rescaled_y = downscaler.scale(doubled_y, src.width, src.height)

    if credit_mask is None and mask_threshold > 0:
        credit_mask = core.std.Expr([depth(y, 32), depth(ref_y, 32)], f"x y - abs {mask_threshold} < 0 1 ?")
        credit_mask = depth(credit_mask, wdepth, range_out=ColorRange.FULL, range_in=ColorRange.FULL)
        credit_mask = core.rgvs.RemoveGrain(credit_mask, mode=6)
        credit_mask = iterate(credit_mask, core.std.Maximum, 2)
        credit_mask = iterate(credit_mask, core.std.Inflate, 2 if do_post_double is None else 4)
        rescaled_y = core.std.MaskedMerge(rescaled_y, y, credit_mask)
    elif credit_mask is not None:
        rescaled_y = core.std.MaskedMerge(rescaled_y, y, credit_mask)

    if line_mask is None or (isinstance(line_mask, bool) and line_mask == True):
        from vsmask.edge import Kirsch
        line_mask = Kirsch().edgemask(y, lthr=80 << 8, hthr=150 << 8)
        if do_post_double is not None:
            line_mask = core.std.Inflate(line_mask)
        rescaled_y = core.std.MaskedMerge(y, rescaled_y, line_mask)
    elif isinstance(line_mask, vs.VideoNode):
        rescaled_y = core.std.MaskedMerge(y, rescaled_y, line_mask)

    out = rescaled_y if clip.format.color_family == vs.GRAY else \
        core.std.ShufflePlanes([rescaled_y, clip], [0, 1, 2], vs.YUV)
    ref_out = ref_y if clip.format.color_family == vs.GRAY else \
        core.std.ShufflePlanes([ref_y, clip], [0, 1, 2], vs.YUV)

    blank_mask = core.std.BlankClip(y)
    return [depth(out, get_depth(src)),
            depth(ref_out, get_depth(src)),
            credit_mask if isinstance(credit_mask, vs.VideoNode) else blank_mask,
            line_mask if isinstance(line_mask, vs.VideoNode) else blank_mask]

def double_nnedi(clip: vs.VideoNode, opencl: bool = True, correct_shift: bool = True,
                ediargs: dict[str, Any] = {"qual": 2, "nsize": 4, "nns": 4, "pscrn": 1}) -> vs.VideoNode:
    return NNEDI_Doubler(opencl, **ediargs).double(clip, correct_shift)

nnedi_double = double_nnedi

def double_shader(clip: vs.VideoNode, shaderfile: PathLike) -> vs.VideoNode:
    return Shader_Doubler(shaderfile).double(clip)

def double_waifu2x(clip: vs.VideoNode, cuda: bool | str = 'trt', protect_edges: bool = True, fix_tint: bool = True, 
        fp16: bool = True, num_streams: int = 1, **w2xargs) -> vs.VideoNode:
    return Waifu2x_Doubler(cuda, fp16, num_streams, w2xargs).double(clip)

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