from __future__ import annotations

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
    dark_thr: float = 0.0,
    bright_thr: float = 1.0,
) -> vs.VideoNode:
    size_args = dict(
        width=scaling_args.width if scaling_args.mode == "w" else clip.width,
        height=scaling_args.height if scaling_args.mode == "h" else clip.height,
    )

    blank = descale_rescale(
        clip.std.BlankClip(length=1, color=0.5, keep=True, **size_args),
        kernel,
        width=clip.width,
        height=clip.height,
        border_handling=int(border_handling),
        **scaling_args.kwargs(),
    )

    return core.std.Expr(
        [clip, blank],
        f"y 0.5 - dup 0 = not swap x {bright_thr} >= 255 0 ? x {dark_thr} <= 255 0 ? ? 0 ?",
        format=vs.GRAY8,
    )
