from enum import IntEnum
from typing import Sequence, Union
from vsaa import Antialiaser, Eedi3
from vskernels import Bicubic, Kernel, KernelT, Lanczos, Scaler, ScalerT
from vstools import FrameRangesN, KwargsT, mod2, vs, get_w, scale_mask


__all__ = ["cope_aa", "CopeMode"]


class CopeMode(IntEnum):
    UpDown = 1
    Descale = 2
    Inverse = 3


def cope_aa(
    clip: vs.VideoNode,
    multiplier: float | None = None,
    antialiaser: Antialiaser = Eedi3(0.125, 0.25, gamma=65, vthresh0=40, vthresh1=60, field=1, sclip_aa=None),
    scaler: KernelT | ScalerT | Sequence[Union[KernelT, ScalerT]] = Lanczos(),
    mode: CopeMode | int = CopeMode.Inverse,
    mask: bool | vs.VideoNode = True,
    no_aa_ranges: FrameRangesN = [],
    hybrid_cl: bool = False,
    **kwargs: KwargsT,
) -> vs.VideoNode:
    """
    Cope and lazy function to AA a doubled clip. Usually while rescaling.
    This is probably overall an awful idea.

    :param multiplier:              Basically rfactor. If you're doubling a 720p clip you'll only have a 1440p clip to AA. EEDI3 will fuck it.
                                    Defaults to 1.2 if the input clip is smaller than 1700p. 1 otherwise.
    :param antialiaser:             Antialiaser used for actually doing the stuff. Defaults to EEDI3 with some somewhat conservative settings kindof.
    :param scaler:                  Scaler(s) or rather kernel(s) in this case. Used to up- and downscale for the rfactor.
                                    If you specify a third one it will get used to scale the mask. (Otherwise Bilinear)
    :param mode:                    Method to return back to input res. Available are UpDown (simple downscale), Descale and Inverse (fmtc).
    :param mask:                    Mask for eedi3 possibly save some calculations. Can be a custom one, True for a Kirsch or False to disable.
    :param no_aa_ranges:            Ranges you might not wanna AA for one reason or another.
    :param hybrid_cl:               Use eedi3cl on one of the two interpolate calls.
                                    Not sure if this is useful or not. Just wrote it to test.
    """

    def fmtc_args(kernel: Kernel) -> KwargsT:
        if isinstance(kernel, Bicubic):
            return KwargsT(kernel="bicubic", a1=kernel.b, a2=kernel.c)
        else:
            return KwargsT(kernel=kernel.__class__.__name__.lower(), taps=kernel.taps if isinstance(kernel, Lanczos) else None)

    if not isinstance(scaler, Sequence):
        scaler = [scaler, scaler]
    scalers = [Kernel.ensure_obj(s) if mode != CopeMode.UpDown else Scaler.ensure_obj(s) for s in scaler]

    if mask is True:
        from vsmasktools import KirschTCanny

        mask = KirschTCanny.edgemask(clip, lthr=60 / 255, hthr=150 / 255, planes=0)

    if not multiplier:
        multiplier = 1.2 if clip.height < 1700 else 1.0
    height = mod2(clip.height * multiplier)
    width = get_w(height, mod=None)
    if mask:
        mask = mask.resize.Bilinear(width, height) if len(scalers) < 3 else scalers[2].scale(mask, width, height)
        mask = mask.std.Binarize(scale_mask(16, 8, mask))
    wclip = scalers[0].scale(clip, width, height)
    aa = wclip.std.Transpose()
    aa = antialiaser.interpolate(aa, False, sclip=aa, mclip=mask.std.Transpose() if mask else None, **kwargs).std.Transpose()
    if hybrid_cl and isinstance(antialiaser, Eedi3):
        from copy import deepcopy

        other_antialiaser = deepcopy(antialiaser)
        other_antialiaser.opencl = True
        aa = other_antialiaser.interpolate(aa, False, sclip=aa, **kwargs)
    else:
        aa = antialiaser.interpolate(aa, False, sclip=aa, mclip=mask if mask else None, **kwargs)
    aa = wclip.std.MaskedMerge(aa, mask)
    match mode:
        case CopeMode.Descale:
            aa = scalers[1].descale(aa, clip.width, clip.height)
        case CopeMode.Inverse:
            aa = aa.fmtc.resample(clip.width, clip.height, invks=True, **fmtc_args(scalers[1]))
        case _:
            aa = scalers[1].scale(aa, clip.width, clip.height)
    if not no_aa_ranges:
        return aa
    else:
        try:
            from jvsfunc import rfs

            return rfs(aa, clip, no_aa_ranges)
        except:
            from vstools import replace_ranges as rfs

            return rfs(aa, clip, no_aa_ranges)
