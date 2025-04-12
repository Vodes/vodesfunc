from vstools import vs, core, get_y, get_u, get_v, depth, get_depth, join, KwargsT, get_var_infos, FunctionUtil
from vsrgtools import contrasharpening

from importlib.metadata import version as fetch_version
from packaging.version import Version

__all__ = ["VMDegrain", "schizo_denoise"]


def check_jetpack_version():
    jetpack_version = Version(fetch_version("vsjetpack"))
    if jetpack_version >= Version("0.3.0"):
        print(
            "VMDegrain: There's probably a good reason for it but note that the new mvtools wrapper is MUCH slower and I'm not sure how to roughly match the settings."
        )
        if jetpack_version < Version("0.3.2"):
            print("Please update vsjetpack to atleast 0.3.2 if you want to use 0.3.X. There are some necessary repair fixes on it.")


def VMDegrain(
    src: vs.VideoNode,
    thSAD: int = 60,
    prefilter: vs.VideoNode | int = 2,
    smooth: bool = True,
    block_size: int | None = None,
    overlap: int | None = None,
    refine: int | None = None,
    **kwargs: KwargsT,
) -> vs.VideoNode:
    """
    Just some convenience function for mvtools with a useable preset and temporal smoothing.\n
    Check the MVTools Docs for the params that aren't listed below.\n
    `block_size`, `overlap` and `refine` are using somewhat optimized defaults depending on the resolution if `None`.


    :param src:             Input to denoise
    :param smooth:          Run TTempsmooth on the denoised clip if True
    :return:                Denoised clip
    """
    check_jetpack_version()
    from vsdenoise import MVTools, SADMode, SearchMode, MotionMode, Prefilter

    if isinstance(prefilter, int):
        prefilter = Prefilter(prefilter)

    futil = FunctionUtil(src, VMDegrain, 0, vs.YUV, 16)

    if any([block_size, overlap, refine]) and not all([block_size, overlap, refine]):
        raise ValueError("VMDegrain: If you want to play around with blocksize, overlap or refine, you have to set all of them.")

    if not block_size or not overlap or not refine:
        refine = 3
        _, width, height = get_var_infos(src)
        if width <= 1024 and height <= 576:
            block_size = 32
            overlap = 16
        elif width <= 2048 and height <= 1536:
            block_size = 64
            overlap = 32
        else:
            block_size = 128
            overlap = 64

    try:
        from vsdenoise import (
            mc_degrain,
            RFilterMode,
            MVToolsPreset,
            prefilter_to_full_range,
            SuperArgs,
            AnalyzeArgs,
            RecalculateArgs,
            SharpMode,
        )

        analyze_recalc_args = dict(search=SearchMode.DIAMOND, dct=SADMode.ADAPTIVE_SPATIAL_MIXED, truemotion=MotionMode.SAD)
        preset = MVToolsPreset(
            search_clip=prefilter_to_full_range,
            pel=2,
            super_args=SuperArgs(sharp=SharpMode.WIENER, rfilter=RFilterMode.TRIANGLE),
            analyze_args=AnalyzeArgs(blksize=block_size, overlap=overlap, **analyze_recalc_args),
            recalculate_args=RecalculateArgs(blksize=int(block_size / 2), overlap=int(overlap / 2), **analyze_recalc_args),
        )

        out = mc_degrain(
            futil.work_clip, prefilter=prefilter, thsad=thSAD, blksize=block_size, refine=refine, rfilter=RFilterMode.TRIANGLE, preset=preset
        )
    except:  # noqa: E722
        from vsdenoise import PelType

        d_args = KwargsT(
            prefilter=prefilter,
            thSAD=thSAD,
            block_size=block_size,
            overlap=overlap,
            sad_mode=SADMode.SPATIAL.same_recalc,
            search=SearchMode.DIAMOND,
            motion=MotionMode.HIGH_SAD,
            pel_type=PelType.BICUBIC,
            refine=refine,
            rfilter=2,
            sharp=2,
        )
        d_args.update(**kwargs)
        out = MVTools.denoise(futil.work_clip, **d_args)

    if smooth:
        out = out.ttmpsm.TTempSmooth(maxr=1, thresh=1, mdiff=0, strength=1)

    return futil.return_clip(out)


def schizo_denoise(
    src: vs.VideoNode,
    sigma: float | list[float] = [0.8, 0.3],
    thSAD: int = 60,
    radius: int | list[int] = 2,
    nlm_a: int = 2,
    prefilter: vs.VideoNode | int = 2,
    cuda: bool | list[bool] = True,
    csharp: int | bool = False,
    **kwargs,
) -> vs.VideoNode:
    """
    Convenience function for (k)nlm on chroma and mvtools + bm3d(cuda) on luma.
    Mostly for personal scripts so please don't complain too much unless it's an actual issue.

    :param src:         Input to denoise
    :param sigma:       Essentially strength for NLMeans and BM3D.
                        Float or list of floats in this order [bm3d, nlm_uv] or [bm3d, nlm_u, nlm_v]
    :param thSAD:       Not exactly strength but something like that, for mvtools.
    :param radius:      Temporal Radius used for NLMeans and BM3D.
                        Int or list of ints in this order [bm3d, nlm]
    :param prefilter:   vsdenoise Prefilter or prefiltered clip to use for mvtools.
                        Defaults to MINBLUR3
    :param cuda:        Uses NlmCuda and BM3DCuda respectively if available. The latter prefers RTC if available.
                        Will fallback to BM3DHip if installed and no cuda available.
    :param csharp:      Apply contrasharpening after denoising. True defaults to 3 while False obviously disables it.
    :param kwargs:      Any parameters you might wanna pass to bm3d or mvtools.

    :return:            Denoised clip
    """
    if src.format.color_family != vs.YUV:  # type: ignore
        raise ValueError("schizo_denoise: This function expects a full YUV clip.")

    if not isinstance(radius, list):
        radius = [radius, radius]

    if not isinstance(sigma, list):
        sigma = [sigma, sigma]

    if not isinstance(cuda, list):
        cuda = [cuda, cuda]

    if isinstance(prefilter, int):
        from vsdenoise import Prefilter

        prefilter = Prefilter(prefilter)

    clip = depth(src, 16)

    nlmfunc = core.knlm.KNLMeansCL if not hasattr(core, "nlm_cuda") or not cuda[0] else core.nlm_cuda.NLMeans

    if len(sigma) == 3:
        clip_u = nlmfunc(clip, a=nlm_a, d=radius[1], h=sigma[1], channels="UV")
        clip_v = nlmfunc(clip, a=nlm_a, d=radius[1], h=sigma[2], channels="UV")
        nlm = join(get_y(clip), get_u(clip_u), get_v(clip_v))  # type: ignore
    else:
        clip_uv = nlmfunc(clip, a=nlm_a, d=radius[1], h=sigma[1], channels="UV")
        nlm = join(clip, clip_uv)  # type: ignore

    # 'Extract' possible bm3d args before passing kwargs to mvtools :)
    bm3dargs = dict(
        block_step=kwargs.pop("block_step", 8),
        bm_range=kwargs.pop("bm_range", 9),
        ps_num=kwargs.pop("ps_num", 2),
        ps_range=kwargs.pop("ps_range", 4),
        fast=kwargs.pop("fast", True),
    )

    y = get_y(clip)
    mv = VMDegrain(y, thSAD, prefilter, **kwargs)

    has_cuda = hasattr(core, "bm3dcuda") or hasattr(core, "bm3dcuda_rtc")
    has_hip = hasattr(core, "bm3dhip")

    if cuda[1] and (has_cuda or has_hip):
        if has_cuda:
            bm3dfunc = core.bm3dcuda if not hasattr(core, "bm3dcuda_rtc") else core.bm3dcuda_rtc
        else:
            bm3dfunc = core.bm3dhip
    else:
        bm3dargs.pop("fast")
        bm3dfunc = core.bm3dcpu

    bm3d = bm3dfunc.BM3Dv2(depth(y, 32), depth(mv, 32), sigma[0], radius=radius[0], **bm3dargs)

    out = join(depth(bm3d, 16), nlm)  # type: ignore
    out = depth(out, get_depth(src))
    if csharp != False:  # noqa: E712
        out = contrasharpening(out, src, mode=3 if csharp == True else csharp)  # noqa: E712
    return out.std.CopyFrameProps(src)
