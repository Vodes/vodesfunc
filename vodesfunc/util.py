from jetpytools import KwargsT
from vstools import vs, core

from functools import partial
from vsmuxtools import src_file

from contextlib import suppress


__all__: list[str] = [
    "set_output",
    "out",
]


def set_output(
    clip: vs.VideoNode | src_file,
    name: str | None = None,
    frame_info: bool = False,
    allow_comp: bool = True,
    cache: bool | None = None,
    **kwargs: KwargsT,
) -> vs.VideoNode:
    """
    Outputs a clip. Less to type.
    If you're using vsview, cache is irrelevant and allow_comp is unfortunately not a thing right now.
    """
    if isinstance(clip, src_file):
        clip = clip.src_cut

    if frame_info and name:
        clip = _print_frameinfo(clip, name)

    is_preview_vsview = False
    with suppress(ImportError):
        from vsview import is_preview as vsv_is_preview

        is_preview_vsview = vsv_is_preview()

        if is_preview_vsview:
            from vsview import set_output as vsv_out

            vsv_out(clip, name, **kwargs)
            return clip

    is_preview_vspreview = False
    with suppress(ImportError):
        from vspreview.api import is_preview as vsp_is_preview

        is_preview_vspreview = vsp_is_preview()

        if is_preview_vspreview:
            from vspreview.api import set_output as vsp_out

            args = KwargsT(name=name, cache=cache or True, disable_comp=not allow_comp)
            if kwargs:
                args.update(**kwargs)
            vsp_out(clip, **args)
            return clip

    if not is_preview_vspreview and not is_preview_vsview:
        if name is not None:
            clip = clip.std.SetFrameProp("Name", data=name)

        clip.set_output(len(vs.get_outputs()))

    return clip


def _print_frameinfo(clip: vs.VideoNode, title: str = "") -> vs.VideoNode:
    style = "sans-serif,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1"

    def FrameProps(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
        if "_PictType" in f.props:
            info = f"Frame {n} of {clip.num_frames}\nPicture type: {f.props['_PictType'].decode()}"
        else:
            info = f"Frame {n} of {clip.num_frames}\nPicture type: N/A"

        clip = core.sub.Subtitle(clip, text=info, style=style)
        return clip

    clip = core.std.FrameEval(clip, partial(FrameProps, clip=clip), prop_src=clip)
    clip = core.sub.Subtitle(clip, text=["".join(["\n"] * 4) + title], style=style)
    return clip


out = set_output
