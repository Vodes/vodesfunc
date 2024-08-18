from vstools import vs, core, depth, get_y, get_h, get_w
from typing import Any
from math import ceil
from functools import partial

from .rescale_ext import RescaleBase

__all__ = ["dirty_prop_set"]


def dirty_prop_set(
    clip: vs.VideoNode,
    threshold: int = 1100,
    luma_scaling: int = 24,
    prop_name: str | None = None,
    src_prop_val: Any | None = None,
    bbm_prop_val: Any | None = None,
    debug_output: bool = False,
) -> list[vs.VideoNode]:
    """
    Dirty-edge-based frameprop setting function using bbm, a brightness difference check and a brightness scaling
    (might be a very specific usecase)

    Returns both filtered clip and mask in a VideoNode List (0 = clip, 1 = mask)

    An example for this would be my tanya script:
        Only 720p frames have dirty edges so write a 720 prop if dirty edges are detected.

        dirty_prop_set(.., prop_name = 'Rescale', src_prop_val = 812, bbm_prop_val = 720)
    """

    def _select_frame(n: int, f: vs.VideoFrame, clip_a: vs.VideoNode, clip_b: vs.VideoNode) -> vs.VideoNode:
        plane_stats_average = f.props["PlaneStatsAverage"]
        # print(f"Frame {n}: {plane_stats_average:.20f}")
        return clip_b if plane_stats_average > 0.00010 else clip_a  # type: ignore

    def _get_mask(n: int, f: vs.VideoFrame, clip_a: vs.VideoNode, clip_b: vs.VideoNode) -> vs.VideoNode:
        brightness = f.props["PlaneStatsAverage"]
        weighted_thr = threshold * (1 - (1 - brightness) ** (brightness**2 * luma_scaling))  # type: ignore
        if debug_output:
            print(f"Frame {n}: Average Brightness - {brightness:.20f}, Weighted - {weighted_thr:.20f}")
        return core.std.Expr([clip_a, clip_b], [f"y x - {weighted_thr} > 65536 0 ?", ""])

    try:
        import awsmfunc as awf
    except:
        raise ModuleNotFoundError("awsmfunc not found!")

    clip = depth(clip, 16).std.PlaneStats()  # Wouldn't this be set way earlier?
    bbm = awf.bbmod(clip, 1, 1, 1, 1, thresh=50, blur=666)
    mask = get_y(core.std.FrameEval(clip, partial(_get_mask, clip_a=clip, clip_b=bbm), clip)).std.PlaneStats()

    if isinstance(src_prop_val, int) and isinstance(bbm_prop_val, int):
        bbm_prop, src_prop = [c.std.SetFrameProp(prop=prop_name, intval=i) for c, i in zip([bbm, clip], [bbm_prop_val, src_prop_val])]
    else:
        bbm_prop, src_prop = [c.std.SetFrameProp(prop=prop_name, data=i) for c, i in zip([bbm, clip], [str(bbm_prop_val), str(src_prop_val)])]

    return [core.std.FrameEval(clip, partial(_select_frame, clip_a=src_prop, clip_b=bbm_prop), prop_src=mask), mask]


# fmt: off
def get_border_crop(input_clip: vs.VideoNode, base: RescaleBase, override_window: int | None = None) -> tuple[int]:
    """Get the crops for the border handling masking."""

    kernel_window = override_window or base.kernel.kernel_radius

    if base.height == input_clip.height:
        vertical_crop = (0, 0)
    else:
        base_height = base.base_height or get_h(base.base_width, base.descaled) if base.base_width else base.height
        src_top = base.descale_func_args.get("src_top", 0)

        top = max(ceil(
            (-(base.height - 1) / 2 + kernel_window - src_top - 1)
            * input_clip.height / base.height + (input_clip.height - 1) / 2
        ), 0)

        bottom = max(ceil(
            (-(base.height - 1) / 2 + kernel_window - (base_height - base.height - src_top) - 1)
            * input_clip.height / base.height + (input_clip.height - 1) / 2
        ), 0)

        vertical_crop = (top, bottom)

    if base.width == input_clip.width:
        horizontal_crop = (0, 0)
    else:
        base_width = base.base_width or get_w(base.base_height, base.descaled) if base.base_height else base.width
        src_left = base.descale_func_args.get("src_left", 0)

        left = max(ceil(
            (-(base.width - 1) / 2 + kernel_window - src_left - 1)
            * input_clip.width / base.width + (input_clip.width - 1) / 2
        ), 0)

        right = max(ceil(
            (-(base.width - 1) / 2 + kernel_window - (base_width - base.width - src_left) - 1)
            * input_clip.width / base.width + (input_clip.width - 1) / 2
        ), 0)

        horizontal_crop = (left, right)

    return horizontal_crop + vertical_crop
# fmt: on
