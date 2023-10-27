from vstools import vs, core, depth, get_y
from functools import partial

__all__ = ["dirty_prop_set"]


def dirty_prop_set(
    clip: vs.VideoNode,
    threshold: int = 1100,
    luma_scaling: int = 24,
    prop_name: str = None,
    src_prop_val: any = None,
    bbm_prop_val: any = None,
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
        return clip_b if plane_stats_average > 0.00010 else clip_a

    def _get_mask(n: int, f: vs.VideoFrame, clip_a: vs.VideoNode, clip_b: vs.VideoNode) -> vs.VideoNode:
        brightness = f.props["PlaneStatsAverage"]
        weighted_thr = threshold * (1 - (1 - brightness) ** (brightness**2 * luma_scaling))
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
