from vstools import vs, core, KwargsT

from functools import partial
from vsmuxtools import PathLike, ensure_path_exists, src as vmt_src


__all__: list[str] = [
    "set_output",
    "out",
    "src",
    "source",
]


def src(filePath: PathLike, force_lsmas: bool = False, delete_dgi_log: bool = True, **kwargs) -> vs.VideoNode:
    """
    Used dgindex as Source and requires dgindexnv in path to generate files if they don't exist.\n
    Now deprecated in favour of the vsmuxtools implementation.

    :param filepath:        Path to video or dgi file
    :param force_lsmas:     Skip dgsource entirely and use lsmas
    :param delete_dgi_log:  Delete the .log files dgindexnv creates
    :return:                Video Node
    """
    print("vodesfunc.src is deprecated and currently only calls vsmuxtools.src!")
    print("It might get removed in the next update or two.")
    return vmt_src(ensure_path_exists(filePath, src), force_lsmas, **kwargs)


def set_output(
    clip: vs.VideoNode, name: str | None = None, frame_info: bool = False, allow_comp: bool = True, cache: bool | None = None, **kwargs: KwargsT
) -> vs.VideoNode:
    """
    Outputs a clip. Less to type.
    Designed to be used with the good ol 'from vodesfunc import *' and the 'out' alias
    """
    if frame_info and name:
        clip = _print_frameinfo(clip, name)

    try:
        args = KwargsT(name=name, cache=cache, disable_comp=not allow_comp)
        if kwargs:
            args.update(**kwargs)
        from vspreview import is_preview, set_output as setsu_sucks

        if cache is None:
            cache = is_preview()
        setsu_sucks(clip, **args)
    except:
        if name is not None:
            clip = clip.std.SetFrameProp("Name", data=name)
        clip.set_output(len(vs.get_outputs()))
    return clip


def _print_frameinfo(clip: vs.VideoNode, title: str = "") -> vs.VideoNode:
    style = "sans-serif,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000," "0,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1"

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
source = src
