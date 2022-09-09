from typing import Any, Callable

import vapoursynth as vs
from vskernels import Catrom, Kernel
from vsutil import Range, depth, get_depth, get_y, iterate

core = vs.core


__all__: list[str] = [
    'nnedi_double',
    'vodes_rescale',
]


def vodes_rescale(
    src: vs.VideoNode, height: float = 0, width: float = None,
    mode: int = 0, descale_kernel: Kernel | list[vs.VideoNode] = Catrom(),
    downscaler: Kernel | str = Catrom(), opencl: bool = True,
    modeargs: dict[str, Any] = {"radius": 2, "strength": 50},
    ediargs: dict[str, Any] = {"qual": 2, "nsize": 4, "nns": 4, "pscrn": 1},
    line_mask: vs.VideoNode | bool = None, credit_mask: vs.VideoNode = None, mask_threshold: float = 0.04,
    do_post_double: Callable[[vs.VideoNode], vs.VideoNode] = None, shaderfile: str = r'C:\FSRCNNX_x2_56-16-4-1.glsl'
) -> list[vs.VideoNode | None]:
    """
    Rescale function with masking for convenience etc.

    :param src:             Input clip
    :param height:          Height to be descaled to
    :param width:           Width to be descaled to; will be calculated if you don't pass any
    :param descale_kernel:  Kernel used for descaling, supports passing a list of descaled and ref clip instead
    :param downscaler:      Kernel used to downscale the doubled clip, uses vsscale.ssim_downsample if passed 'ssim'
    :param opencl:          Will use nnedi3cl if True and znedi3 if not
    :param mode:            0 (default) = nnedi doubling, 1 = fsrcnnx doubling (vardefunc zastin if available), 2 = wtf
    :param modeargs:        Dict mode related params (Mode 1 supports 'radius' and 'strength' for the sharpening and Mode 2 adds 'ratio' support)
    :param ediargs:         Other params you may want to pass to the doubler (with sane defaults)
    :param line_mask:       Linemask to only rescale lineart | Will generate one if None or skip masking if `False` is passed
    :param credit_mask:     Credit Masking | Will generate one if None or skip masking if *mask_threshold* is <= 0
    :param mask_threshold:  Threshold for the diff based credit mask | lower catches more
    :param do_post_double:  Pass your own function as a lambda if you want to manipulate the clip before the downscale happens
    :param shaderfile:      Pass your FSRCNNX Shaderfile (my path as default lol)
    :return:                List of rescaled, reference upscale, credit mask and line mask
    """
    clip = depth(src, 16)
    y = clip if clip.format.color_family == vs.GRAY else get_y(clip)

    if height is None or height < 1:
        raise ValueError('Rescale: Height may not be None or 0')

    if width is None or width < 1:
        aspect_ratio = src.width / src.height
        width = height * aspect_ratio

    descaled_y = descale_kernel.descale(y, width, height) if isinstance(descale_kernel, Kernel) else descale_kernel[0]
    ref_y = descale_kernel.scale(descaled_y, src.width, src.height) if isinstance(
        descale_kernel, Kernel) else descale_kernel[1]

    if mode == 0:
        doubled_y = nnedi_double(descaled_y, opencl, ediargs)
    elif mode == 1:
        try:
            import vardefunc as vdf
            doubled_y = vdf.fsrcnnx_upscale(descaled_y, height=descaled_y.height * 2, downscaler=None, profile='zastin', shader_file=shaderfile,
                                            sharpener=lambda clip: vdf.sharp.z4usm(clip, modeargs.get("radius", 2), modeargs.get("strength", 50)))
        except:
            trash = core.std.BlankClip(descaled_y, format=vs.YUV420P16)
            doubled = core.std.ShufflePlanes([descaled_y, trash], [0, 1, 2], vs.YUV) \
                .placebo.Shader(shader=shaderfile, filter='box', width=descaled_y.width*2, height=descaled_y.height*2)
            doubled_y = get_y(doubled)
    elif mode == 2:
        import vardefunc as vdf
        nnedi = nnedi_double(descaled_y, opencl, ediargs)
        fsrcnnx = vdf.fsrcnnx_upscale(descaled_y, height=descaled_y.height * 2, downscaler=None, profile='zastin', shader_file=shaderfile,
                                      upscaled_smooth=nnedi,
                                      sharpener=lambda clip: vdf.sharp.z4usm(clip, modeargs.get("radius", 2), modeargs.get("strength", 35)))

        if "ratio" in modeargs:
            doubled_y = core.std.Expr(
                [nnedi, fsrcnnx], f"{modeargs.get('ratio')} x * {1 - modeargs.get('ratio')} y * +")
        else:
            doubled_y = fsrcnnx

    if do_post_double is not None:
        doubled_y = do_post_double(doubled_y)

    if isinstance(downscaler, str) and downscaler.lower() == 'ssim':
        import vsscale as vss
        rescaled_y = vss.ssim_downsample(doubled_y, src.width, src.height)
    else:
        rescaled_y = downscaler.scale(doubled_y, src.width, src.height)

    if credit_mask is None and mask_threshold > 0:
        credit_mask = core.std.Expr([depth(y, 32), depth(ref_y, 32)], f"x y - abs {mask_threshold} < 0 1 ?")
        credit_mask = depth(credit_mask, 16, range=Range.FULL, range_in=Range.FULL)
        credit_mask = core.rgvs.RemoveGrain(credit_mask, mode=6)
        credit_mask = iterate(credit_mask, core.std.Maximum, 2)
        credit_mask = iterate(credit_mask, core.std.Inflate, 2)
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


def nnedi_double(clip: vs.VideoNode, opencl: bool = True,
                 ediargs: dict[str, Any] = {"qual": 2, "nsize": 4, "nns": 4, "pscrn": 1}) -> vs.VideoNode:
    """
    Simple utility function for doubling a clip using znedi or nnedi3cl (also fixes the shift)

    :param src:             Input clip
    :param opencl:          Will use nnedi3cl if True and znedi3 if False
    :param ediargs:         Other params you may want to pass to the doubler (with sane defaults)
    """
    y = get_y(clip)
    doubled_y = y.nnedi3cl.NNEDI3CL(dh=True, field=0, **ediargs).std.Transpose() \
        .nnedi3cl.NNEDI3CL(dh=True, field=0, **ediargs).std.Transpose() \
        if opencl else y.znedi3.nnedi3(dh=True, field=0, **ediargs).std.Transpose() \
        .znedi3.nnedi3(dh=True, field=0, **ediargs).std.Transpose()
    doubled_y = doubled_y.resize.Bicubic(src_top=.5, src_left=.5)
    return doubled_y
