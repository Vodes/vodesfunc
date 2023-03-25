from vstools import vs, core, get_y, get_u, get_v, depth, get_depth, join
from vsrgtools import contrasharpening

__all__ = ['VMDegrain', 'schizo_denoise']

def VMDegrain(src: vs.VideoNode, thSAD: int = 60, prefilter: vs.VideoNode | int = 2, 
    smooth: bool = True, block_size: int = 32, overlap: int = 16, **kwargs) -> vs.VideoNode:
    """
        Just some convenience function for mvtools with a useable preset and temporal smoothing.
        Check the MVTools Docs for the params that aren't listed below.


        :param src:             Input to denoise
        :param smooth:          Run TTempsmooth on the denoised clip if True
        :return:                Denoised clip              
    """
    from vsdenoise import MVTools, SADMode, SearchMode, MotionMode, PelType, Prefilter
    if isinstance(prefilter, int):
        prefilter = Prefilter(prefilter)
    y = depth(get_y(src), 16)
    d_args = dict(
        prefilter=prefilter, thSAD=thSAD, block_size=block_size, overlap=overlap,
        sad_mode=SADMode.SPATIAL.same_recalc, search=SearchMode.DIAMOND, 
        motion=MotionMode.HIGH_SAD, pel_type=PelType.BICUBIC, rfilter=2, sharp=2,
    )
    d_args.update(**kwargs)

    out = MVTools.denoise(y, **d_args)
    if smooth:
        out = out.ttmpsm.TTempSmooth(maxr=1, thresh=1, mdiff=0, strength=1)

    out = depth(out, get_depth(src))
    return out if src.format.color_family == vs.GRAY else join(out, src)


def schizo_denoise(src: vs.VideoNode, sigma: float | list[float] = [0.8, 0.3], thSAD: int = 60, 
    radius: int | list[int] = 2, nlm_a: int = 2, prefilter: vs.VideoNode | int = 2, 
    cuda: bool = True, csharp: int | bool = False, **kwargs) -> vs.VideoNode:
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
        :param csharp:      Apply contrasharpening after denoising. True defaults to 3 while False obviously disables it.
        :param kwargs:      Any parameters you might wanna pass to bm3d or mvtools.

        :return:            Denoised clip
    """
    if src.format.color_family != vs.YUV:
        raise ValueError("schizo_denoise: This function expects a full YUV clip.")

    if not isinstance(radius, list):
        radius = [radius, radius]

    if not isinstance(sigma, list):
        sigma = [sigma, sigma]

    if isinstance(prefilter, int):
        from vsdenoise import Prefilter
        prefilter = Prefilter(prefilter)

    clip = depth(src, 16)

    nlmfunc = core.knlm.KNLMeansCL if not hasattr(core, "nlm_cuda") or not cuda else core.nlm_cuda.NLMeans

    if sigma.count == 3:
        clip_u = nlmfunc(clip, a=nlm_a, d=radius[1], h=sigma[1], channels='U')
        clip_v = nlmfunc(clip, a=nlm_a, d=radius[1], h=sigma[2], channels='V')
        nlm = join(get_y(clip), get_u(clip_u), get_v(clip_v))
    else:
        clip_uv = nlmfunc(clip, a=nlm_a, d=radius[1], h=sigma[1], channels='UV')
        nlm = join(clip, clip_uv)

    # 'Extract' possible bm3d args before passing kwargs to mvtools :)
    bm3dargs = dict(block_step=kwargs.pop("block_step", 8), bm_range=kwargs.pop("bm_range", 9),
        ps_num=kwargs.pop("ps_num", 2), ps_range=kwargs.pop("ps_range", 4), fast=kwargs.pop("fast", True))

    y = get_y(clip)
    mv = VMDegrain(y, thSAD, prefilter, **kwargs)

    bm3dfunc = core.bm3dcpu
    if cuda:
        bm3dfunc = core.bm3dcuda if not hasattr(core, "bm3dcuda_rtc") else core.bm3dcuda_rtc

    bm3d = bm3dfunc.BM3Dv2(depth(y, 32), depth(mv, 32), sigma[0], radius=radius[0], **bm3dargs)

    out = join(depth(bm3d, 16), nlm)
    out = depth(out, get_depth(src))
    if csharp != False:
        out = contrasharpening(out, src, mode = 3 if csharp == True else csharp)
    return out