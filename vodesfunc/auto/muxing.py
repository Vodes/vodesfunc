from enum import IntEnum
from pathlib import Path
from datetime import timedelta
import os
from fractions import Fraction

import ass
from .fonts import validate_and_save_fonts
from ..types import PathLike, Chapter, TrackType
from ..util import uniquify_path
from ..automation import get_workdir
from .convert import timedelta_to_frame, frame_to_timedelta

__all__: list[str] = [
    '_track',
    'Attachment',
    'AudioTrack', 'AT',
    'Chapter',
    'GlobSearch',
    'MkvTrack',
    'SubTrack', 'ST',
    'TrackType',
    'VideoTrack', 'VT',
]


class GlobSearch():

    paths: Path | list[Path] = None

    def __init__(self, pattern: str, allow_multiple: bool = False, dir: PathLike = None, recursive: bool = True) -> None:
        """
            Glob Pattern based search for files

            :param pattern:         Glob pattern
            :param allow_multiple:  Will return all file matches if True and only the first if False
            :param dir:             Directory to run the search in. Defaults to current working dir.
            :param recursive:       Search recursively
        """

        dir = Path(dir) if isinstance(dir, str) else dir
        if dir is None:
            dir = Path(os.getcwd()).resolve()

        search = dir.rglob(pattern) if recursive else dir.glob(pattern)
        # print(search)
        for f in search:
            if allow_multiple:
                if self.paths:
                    self.paths.append(f)
                else:
                    init: list[Path] = [f, ]
                    self.paths = init
            else:
                self.paths = f
                break


class _track():

    file: Path
    type: TrackType
    default: bool
    forced: bool
    name: str
    lang: str
    delay: int

    def __init__(self, file: PathLike, type: str | int | TrackType, name: str = '', lang: str = '', default: bool = True, forced: bool = False, delay: int = 0) -> None:
        """
            :param file:        Filepath as string or Path object
            :param type:        TrackType enum, or int or string (1 = 'video', 2 = 'audio', 3 = 'sub')
            :param name:        The track name in the resulting mkv file
            :param lang:        The language tag for the track
            :param default:     Default flag
            :param forced:      Forced flag
            :param delay:       Container delay of track in ms
        """
        self.file = file if isinstance(file, Path) else Path(file)
        self.default = default
        self.forced = forced
        self.name = name
        self.delay = delay
        # Maybe use https://pypi.org/project/pycountry/ to automatically convert iso-2 to iso-3
        # as ffmpeg expects 3 letter codes; I am not sure what mkvmerge wants or can work with
        self.lang = lang
        self.type = type if isinstance(type, TrackType) \
            else (TrackType(type) if isinstance(type, int) else TrackType[type.upper()])

    def mkvmerge_args(self) -> str:
        self.file = self.file if isinstance(self.file, Path) else Path(self.file)
        if self.type == TrackType.ATTACHMENT:
            is_font = self.file.suffix.lower() in ['.ttf', '.otf']
            if not is_font and not self.lang:
                raise ValueError(f'Please specify a mimetype for the attachments if they\'re not fonts!')
            if not is_font:
                return f' --attachment-mime-type {self.lang} --attach-file "{self.file.resolve()}"'
            else:
                return f' --attachment-mime-type {"font/ttf" if self.file.suffix.lower() == ".ttf" else "font/otf"} --attach-file "{self.file.resolve()}"'
        elif self.type == TrackType.MKV:
            return f' {self.name.strip()} "{self.file.resolve()}"'
        elif self.type == TrackType.CHAPTERS:
            return f' --chapters "{self.file.resolve()}"'
        name_args = f' --track-name 0:"{self.name}"' if self.name else ''
        lang_args = f' --language 0:{self.lang}' if self.lang else ''
        delay_args = f' --sync 0:{self.delay}' if self.delay != 0 else ''
        default_args = f' --default-track-flag 0:{"yes" if self.default else "no"}'
        forced_args = f' --forced-display-flag 0:{"yes" if self.forced else "no"}'
        timecode_args = ''
        if isinstance(self, VideoTrack) and self.timecode_file is not None:
            timecode_args = f' --timestamps 0:"{self.timecode_file.resolve()}"'
        return f'{timecode_args}{name_args}{lang_args}{default_args}{forced_args}{delay_args} "{self.file.resolve()}"'


class VideoTrack(_track):
    """
        _track object with VIDEO type preselected and japanese language default
    """

    timecode_file: PathLike | None = None

    def __init__(self, file: PathLike | GlobSearch, name: str = '', lang: str = 'ja', default: bool = True, forced: bool = False, delay: int = 0, timecode_file: PathLike | GlobSearch = None) -> None:
        if isinstance(file, GlobSearch):
            file = file.paths[0] if isinstance(file.paths, list) else file.paths
        if timecode_file is not None:
            if isinstance(timecode_file, GlobSearch):
                timecode_file = timecode_file.paths[0] if isinstance(timecode_file.paths, list) else timecode_file.paths
            self.timecode_file = timecode_file if isinstance(timecode_file, Path) else Path(timecode_file)
        super().__init__(file, TrackType.VIDEO, name, lang, default, forced, delay)


class AudioTrack(_track):
    """
        _track object with AUDIO type preselected and japanese language default
    """

    def __init__(self, file: PathLike | GlobSearch, name: str = '', lang: str = 'ja', default: bool = True, forced: bool = False, delay: int = 0) -> None:
        if isinstance(file, GlobSearch):
            file = file.paths[0] if isinstance(file.paths, list) else file.paths
        super().__init__(file, TrackType.AUDIO, name, lang, default, forced, delay)


class Attachment(_track):
    """
        pseudo _track object for attachments
    """

    def __init__(self, file: str | Path, mimetype: str = '') -> None:
        super().__init__(file, TrackType.ATTACHMENT, '', mimetype, False, False, 0)


class SubTrack(_track):
    """
        _track object with SUB type preselected and english language default

        Supports merging multiple files by passing a List of Path objects or filepath strings
        and of course also a GlobSearch
    """

    def __init__(self, file: PathLike | list[PathLike] | GlobSearch, name: str = '', lang: str = 'en',
                 default: bool = True, forced: bool = False, delay: int = 0) -> None:
        if isinstance(file, GlobSearch):
            file = file.paths

        # Merge if multiple sub files
        if isinstance(file, list):
            ffs_python = f'for track "{name}"'
            print(f'Merging subtitle files {ffs_python if name else ""}...')
            ass_documents: list[ass.Document] = []
            for ass_file in file:
                ass_file = ass_file if isinstance(ass_file, Path) else Path(ass_file)
                with open(ass_file, 'r', encoding='utf_8_sig') as read:
                    ass_documents.append(ass.parse(read))

            merged = ass_documents[0]
            existing_styles = [style.name for style in (merged.styles)]
            ass_documents.remove(merged)
            for doc in ass_documents:
                # Merges all the lines
                merged.events.extend(doc.events)
                # Check for dupe styles
                for style in doc.styles:
                    if style.name in existing_styles:
                        print(f'WARN: Ignoring style "{style.name}" due to preexisting style of the same name!')
                        continue
                    merged.styles.append(style)
                
            out_file = uniquify_path(Path(os.path.join(get_workdir(), f'{Path(file[0]).stem}-merged.ass')))
            with open(out_file, 'w', encoding='utf_8_sig') as merge_write:
                merged.dump_file(merge_write)

            file = Path(out_file)
            print('Done.\n')

        super().__init__(file, TrackType.SUB, name, lang, default, forced, delay)

    def collect_fonts(self, work_dir: Path = get_workdir(), font_sources: list[str | Path] = None,
                      debug_output: bool = False) -> list[Attachment]:
        """
            Validates and copies the fonts needed for this track into the specified `work_dir`.
            `font_sources` can be mkv files or directories.

            Returns a list of Attachment tracks you can feed into Mux()
        """
        out: list[Attachment] = []
        doc: ass.Document = None
        with open(self.file, 'r', encoding='utf_8_sig') as read:
            doc = ass.parse(read)
        validate_and_save_fonts([f'track "{self.name}"' if self.name else self.file.stem,
                                doc], work_dir, font_sources, debug_output)
        for f in os.listdir(work_dir):
            filepath = Path(os.path.join(work_dir, f))
            if filepath.suffix.lower() in ['.ttf', '.otf']:
                out.append(Attachment(filepath.resolve()))

        return out

    def autoswapper(self, allowed_styles: list[str] | None = ['Default', 'Main', 'Alt', 'Overlap', 'Flashback', 'Top', 'Italics'], print_swaps: bool = False) -> "SubTrack":
        """
            autoswapper does the swapping.
            Too lazy to explain

            :param allowed_styles:      List of allowed styles to do the swapping on
                                        Will run on every line if passed `None`
            :param print_swaps:         Prints the swaps
            
            :return:                    This SubTrack
        """
        import re
        with open(self.file, 'r', encoding='utf_8_sig') as f:
            doc = ass.parse(f)
        
        events = []

        for i, line in enumerate(doc.events):
            if not allowed_styles or line.style.lower() in (style.lower() for style in allowed_styles):
                to_swap: dict = {}
                # {*}This will be replaced{*With this}
                for match in re.finditer(re.compile(r'\{\*\}([^{]*)\{\*([^}*]+)\}'), line.text):
                    to_swap.update({
                        f"{match.group(0)}":
                        f"{{*}}{match.group(2)}{{*{match.group(1)}}}"
                    })
                
                # This sentence is no longer{** incomplete}
                for match in re.finditer(re.compile(r'\{\*\*([^}]+)\}'), line.text):
                    to_swap.update({
                        f"{match.group(0)}":
                        f"{{*}}{match.group(1)}{{*}}"
                    })
                
                # This sentence is no longer{*} incomplete{*} 
                for match in re.finditer(re.compile(r'\{\*\}([^{]*)\{\* *\}'), line.text):
                    to_swap.update({
                        f"{match.group(0)}":
                        f"{{**{match.group(1)}}}"
                    })
                #print(to_swap)
                for key, val in to_swap.items():
                    if print_swaps:
                        print(f'autoswapper: Swapped "{key}" for "{val}" on line {i}')
                    line.text = line.text.replace(key, val)
            
            if line.effect.strip() == "***" or line.name.strip() == "***":
                if isinstance(line, ass.Comment):
                    line.TYPE = 'Dialogue'
                elif isinstance(line, ass.Dialogue):
                    line.TYPE = 'Comment'

            events.append(line)
        
        doc.events = events
        out_file = Path(os.path.join(get_workdir(), Path(self.file).stem + "-swapped.ass"))
        with open(out_file, 'w', encoding='utf_8_sig') as f:
            doc.dump_file(f)
        
        self.file = out_file
        return self

    def syncpoint_merge(self, syncpoint: str, mergefile: PathLike | GlobSearch, use_actor_field: bool = False, 
        use_frames: bool = False, fps: Fraction = Fraction(24000, 1001), override_p1: int | timedelta = None) -> "SubTrack":
        """
            Merge other sub files (Opening/Ending kfx for example) with offsetting by syncpoints

            :param syncpoint:           The syncpoint to be used
            :param mergefile:           The file to be merged
            :param use_actor_field:     Search the actor field instead of the effect field for the syncpoint
            :param use_frames:          Uses frames to shift lines instead of direct timestamps
            :param fps:                 The fps to go off of for the conversion
            :param override_p1:         A manual override of the initial syncpoint
                                        Obviously either a frame number or timedelta

            :return:                    This SubTrack
        """
        if isinstance(mergefile, GlobSearch):
            mergefile = mergefile.paths[0] if isinstance(mergefile.paths, list) else mergefile.paths
        mergefile = mergefile if isinstance(mergefile, Path) else Path(mergefile)
        was_merged = False

        with open(self.file, 'r', encoding='utf_8_sig') as f:
            doc = ass.parse(f)
        with open(mergefile, 'r', encoding='utf_8_sig') as f:
            mergedoc = ass.parse(f)

        events = []
        tomerge = []
        existing_styles = [style.name for style in (doc.styles)]

        for line in doc.events:
            events.append(line)
            if was_merged:
                continue
            field = line.name if use_actor_field else line.effect
            if field.lower().strip() == syncpoint.lower().strip() or line.text.lower().strip() == syncpoint.lower().strip() or override_p1 is not None:
                was_merged = True
                start = line.start if override_p1 is None else override_p1
                offset: timedelta | int = None
                for l in mergedoc.events:
                    lfield = l.name if use_actor_field else l.effect
                    if lfield.lower().strip() == syncpoint.lower().strip() or l.text.lower().strip() == syncpoint.lower().strip():
                        mergedoc.events.remove(l)
                        if use_frames:
                            offset = timedelta_to_frame(start - l.start, fps)
                        else:
                            offset = start - l.start
                        break

                for l in sorted(mergedoc.events, key = lambda event: event.start):
                    if offset is None:
                        if use_frames:
                            offset = timedelta_to_frame(start - l.start, fps)
                        else:
                            offset = start - l.start
                        l.start = start
                        l.end = (l.end + (frame_to_timedelta(offset, fps) if use_frames else offset))
                    else:
                        l.start = (l.start + (frame_to_timedelta(offset, fps) if use_frames else offset))
                        l.end = (l.end + (frame_to_timedelta(offset, fps) if use_frames else offset))
                    tomerge.append(l)

        if was_merged:
            events.extend(tomerge)
            # Merge the styles in aswell
            for style in mergedoc.styles:
                if style.name in existing_styles:
                    continue
                doc.styles.append(style)

            doc.events = events
            out_file = uniquify_path(Path(os.path.join(get_workdir(), Path(self.file).stem + "-merge.ass")))
            with open(out_file, 'w', encoding='utf_8_sig') as f:
                doc.dump_file(f)

            self.file = Path(out_file)
        else:
            print(f'Syncpoint "{syncpoint}" was not found!')

        return self


class MkvTrack(_track):

    def __init__(self, file: PathLike | GlobSearch, mkvmerge_args: str = '') -> None:
        if isinstance(file, GlobSearch):
            file = file.paths[0] if isinstance(file.paths, list) else file.paths
        super().__init__(file, TrackType.MKV, mkvmerge_args, '', False, False, 0)

VT = VideoTrack
AT = AudioTrack
ST = SubTrack
