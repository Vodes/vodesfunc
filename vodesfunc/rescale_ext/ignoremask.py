from __future__ import annotations

from math import ceil

from vskernels import Kernel
from vsscale import ScalingArgs
from vskernels import BorderHandling
from vstools import scale_value, vs, core

from .base import descale_rescale

__all__ = ["border_clipping_mask"]


def border_clipping_mask(
    clip: vs.VideoNode,
    scaling_args: ScalingArgs,
    kernel: Kernel,
    border_handling: BorderHandling,
    dark_thr: int = 0,
    bright_thr: int = 235,
) -> vs.VideoNode:
    scale_factor = clip.width / scaling_args.width if scaling_args.mode == "w" else clip.height / scaling_args.height
    kernel_radius = ceil(kernel.kernel_radius * scale_factor) - 1

    size_args = dict(
        width=scaling_args.width if scaling_args.mode == "w" else clip.width,
        height=scaling_args.height if scaling_args.mode == "h" else clip.height,
    )
    neutral_native = core.std.BlankClip(clip, color=0.5, **size_args)
    neutral_rescaled = descale_rescale(
        neutral_native, kernel, width=clip.width, height=clip.height, border_handling=int(border_handling), **scaling_args.kwargs()
    )
    # Whatever you wanna do with it here I guess?

    mask = clip.akarin.Expr(
        f"x {scale_value(dark_thr, 8, clip)} < x {scale_value(bright_thr, 8, clip)} > or "
        f"{'height' if scaling_args.mode == 'h' else 'width'} "
        f"{'Y' if scaling_args.mode == 'h' else 'X'} - {kernel_radius} < "
        f"{'Y' if scaling_args.mode == 'h' else 'X'} {kernel_radius - 1} < or "
        "and 255 0 ?",
        format=vs.GRAY8,
    )
    return mask
