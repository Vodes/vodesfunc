from functools import partial

import vapoursynth as vs

from .automation import src_file

core = vs.core


__all__: list[str] = [
    'set_output_source', 'out_src',
    'set_output', 'out',
    'src', 'source',
]


def src(filePath: str = None, force_lsmas: bool = False) -> vs.VideoNode:
    """
        Uses dgindex as Source and requires dgindexnv in path
        to generate files if they don't exist.

        :param filepath:        Path to video or dgi file
        :param force_lsmas:     Skip dgsource entirely and use lsmas
        :return:                Video Node
    """
    if filePath.lower().endswith('.dgi'):
        return core.dgdecodenv.DGSource(filePath)

    import shutil as sh
    from pathlib import Path

    forceFallBack = sh.which('dgindexnv') is None

    # I don't want that to be a hard dependency :trollhd:
    try:
        import pymediainfo as pym
        parsed = pym.MediaInfo.parse(filePath, parse_speed=0.25)
        trackmeta = parsed.video_tracks[0].to_data()
        format = trackmeta.get('format')
        bitdepth = trackmeta.get('bit_depth')
        if (format is not None and bitdepth is not None):
            if (str(format).strip().lower() == 'avc' and int(bitdepth) > 8):
                forceFallBack = True
                print(f'Falling back to lsmas for Hi10 ({Path(filePath).name})')
            elif(str(format).strip().lower() == 'ffv1'):
                forceFallBack = True
                print(f'Falling back to lsmas for FFV1 ({Path(filePath).name})')
    except OSError:
        print('pymediainfo could not find the mediainfo library! (it needs to be in path)')
    except:
        print('Parsing mediainfo failed. (Do you have pymediainfo installed?)')

    if force_lsmas or forceFallBack:
        return core.lsmas.LWLibavSource(filePath)

    path = Path(filePath)
    dgiFile = path.with_suffix('.dgi')

    if dgiFile.exists():
        return core.dgdecodenv.DGSource(dgiFile.resolve(True))
    else:
        print("Generating dgi file...")
        import os
        import subprocess as sub
        sub.Popen(f"dgindexnv -i \"{path.resolve(True)}\" -h -o \"{dgiFile.resolve(False)}\" -e",
                  shell=True, stdout=sub.DEVNULL).wait()
        if path.with_suffix('.log').exists():
            os.remove(path.with_suffix('.log').resolve(True))
        return core.dgdecodenv.DGSource(dgiFile.resolve(True))


def set_output(clip: vs.VideoNode, name: str = None, frame_info: bool = False, allow_comp: bool = True) -> vs.VideoNode:
    """
    Outputs a clip. Less to type.
    Designed to be used with the good ol 'from vodesfunc import *' and the 'out' alias
    """
    if name is not None:
        clip = clip.std.SetFrameProp('Name', data=name)
    if not allow_comp:
        clip = clip.std.SetFrameProp('_VSPDisableComp', 1)
    if frame_info:
        output = _print_frameinfo(clip, name)
        output.set_output(len(vs.get_outputs()))
    else:
        clip.set_output(len(vs.get_outputs()))

    return clip


def set_output_source(filePath: str | src_file, clip: vs.VideoNode = None, frame_info: bool = False) -> vs.VideoNode:
    """
    Outputs your source clip while also outputting the audio for it
    so scenefiltering becomes less boring

    Also returns the clip in case you wanna use it at the start of your script
    """
    filePath = filePath if isinstance(filePath, str) else str(filePath.file.resolve())

    if clip is None:
        clip = src(filePath)

    clip = clip.std.SetFrameProp('Name', data='Source')
    if frame_info:
        output = _print_frameinfo(clip, 'Source')
        output.set_output(len(vs.get_outputs()))
    else:
        clip.set_output(len(vs.get_outputs()))

    audio = core.bs.AudioSource(filePath)
    audio.set_output(len(vs.get_outputs()) + 20)
    return clip


def _print_frameinfo(clip: vs.VideoNode, title: str = '') -> vs.VideoNode:
    style = ("sans-serif,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
             "0,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1")

    def FrameProps(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
        if "_PictType" in f.props:
            info = f"Frame {n} of {clip.num_frames}\nPicture type: {f.props['_PictType'].decode()}"
        else:
            info = f"Frame {n} of {clip.num_frames}\nPicture type: N/A"

        clip = core.sub.Subtitle(clip, text=info, style=style)
        return clip

    clip = core.std.FrameEval(clip, partial(FrameProps, clip=clip), prop_src=clip)
    clip = core.sub.Subtitle(clip, text=["".join(['\n'] * 4) + title], style=style)
    return clip


out = set_output
out_src = set_output_source
source = src
