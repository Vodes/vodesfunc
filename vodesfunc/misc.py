"""
A collection of functions to make me embrace my laziness
"""
import random
from functools import partial
from typing import Any

import awsmfunc as awf
import lvsfunc as lvf
import vapoursynth as vs
from vsutil import depth, get_y

core = vs.core


# Anything else


def dirty_prop_set(clip: vs.VideoNode, threshold: int = 1100, luma_scaling: int = 24, prop_name: str = None,
                   src_prop_val: Any = None, bbm_prop_val: any = None, debug_output: bool = False
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
        #print(f"Frame {n}: {plane_stats_average:.20f}")
        return clip_b if plane_stats_average > 0.00010 else clip_a

    def _get_mask(n: int, f: vs.VideoFrame, clip_a: vs.VideoNode, clip_b: vs.VideoNode) -> vs.VideoNode:
        brightness = f.props["PlaneStatsAverage"]
        weighted_thr = threshold * (1 - (1 - brightness)**(brightness ** 2 * luma_scaling))
        if debug_output:
            print(f"Frame {n}: Average Brightness - {brightness:.20f}, Weighted - {weighted_thr:.20f}")
        return core.std.Expr([clip_a, clip_b], [f'y x - {weighted_thr} > 65536 0 ?', ''])

    clip = depth(clip, 16).std.PlaneStats()  # Wouldn't this be set way earlier?
    bbm = awf.bbmod(clip, 1, 1, 1, 1, thresh=50, blur=666)
    mask = get_y(core.std.FrameEval(clip, partial(_get_mask, clip_a=clip, clip_b=bbm), clip)).std.PlaneStats()

    if(isinstance(src_prop_val, int) and isinstance(bbm_prop_val, int)):
        bbm_prop, src_prop = [c.std.SetFrameProp(prop=prop_name, intval=i)
                              for c, i in zip([bbm, clip], [bbm_prop_val, src_prop_val])]
    else:
        bbm_prop, src_prop = [c.std.SetFrameProp(prop=prop_name, data=i)
                              for c, i in zip([bbm, clip], [str(bbm_prop_val), str(src_prop_val)])]

    return [core.std.FrameEval(clip, partial(_select_frame, clip_a=src_prop, clip_b=bbm_prop), prop_src=mask), mask]


def lazylist(clip: vs.VideoNode, dark_frames: int = 8, light_frames: int = 4, seed: int = 20202020, diff_thr: int = 15):
    """
    Blame Sea for what this shits out

    A function for generating a list of frames for comparison purposes.
    Works by running `core.std.PlaneStats()` on the input clip,
    iterating over all frames, and sorting all frames into 2 lists
    based on the PlaneStatsAverage value of the frame.
    Randomly picks frames from both lists, 8 from `dark` and 4
    from `light` by default.
    :param clip:          Input clip
    :param dark_frame:    Number of dark frames
    :param light_frame:   Number of light frames
    :param seed:          seed for `random.sample()`
    :param diff_thr:      Minimum distance between each frames (In seconds)
    :return:              List of dark and light frames
    """

    dark = []
    light = []

    def checkclip(n, f, clip):

        avg = f.props["PlaneStatsAverage"]

        if 0.062746 <= avg <= 0.380000:
            dark.append(n)

        elif 0.450000 <= avg <= 0.800000:
            light.append(n)

        return clip

    s_clip = clip.std.PlaneStats()

    eval_frames = vs.core.std.FrameEval(
        clip, partial(checkclip, clip=s_clip), prop_src=s_clip
    )
    lvf.clip_async_render(eval_frames)

    dark.sort()
    light.sort()

    dark_dedupe = [dark[0]]
    light_dedupe = [light[0]]

    thr = round(clip.fps_num / clip.fps_den * diff_thr)
    lastvald = dark[0]
    lastvall = light[0]

    for i in range(1, len(dark)):

        checklist = dark[0:i]
        x = dark[i]

        for y in checklist:
            if x >= y + thr and x >= lastvald + thr:
                dark_dedupe.append(x)
                lastvald = x
                break

    for i in range(1, len(light)):

        checklist = light[0:i]
        x = light[i]

        for y in checklist:
            if x >= y + thr and x >= lastvall + thr:
                light_dedupe.append(x)
                lastvall = x
                break

    if len(dark_dedupe) > dark_frames:
        random.seed(seed)
        dark_dedupe = random.sample(dark_dedupe, dark_frames)

    if len(light_dedupe) > light_frames:
        random.seed(seed)
        light_dedupe = random.sample(light_dedupe, light_frames)

    return dark_dedupe + light_dedupe
