import os
import re
import shutil as sh
import subprocess
from configparser import ConfigParser
from fractions import Fraction
from pathlib import Path
from typing import Callable
from datetime import timedelta

import vapoursynth as vs
from vstools import Matrix, Primaries, Transfer, ColorRange, ChromaLocation

from .auto.download import get_executable
from .auto.convert import timedelta_to_frame, frame_to_timedelta, format_timedelta
from .auto.parsing import parse_ogm, parse_xml, parse_src_file
from .types import PathLike, Trim, Zone, Chapter, TrackType
from .util import src_file, uniquify_path, get_crc32, is_x264_zone

global setup
setup = None

__all__: list[str] = [
    'Chapters',
    'light_sucks',
    'Mux',
    'settings_builder_x265', 'sb', 'sb264', 'sb265',
    'Setup',
    'VT', 'AT', 'ST',
    'VideoTrack', 'AudioTrack', 'SubTrack',
    'Attachment', 'GlobSearch'
]

class Setup:
    """
        When you initiate this for the first time in a directory
        it will create a new config.ini. Set that up and have fun with all the other functions :)
    """

    bdmv_dir = "BDMV"
    show_name = "Nice Series"
    allow_binary_download = True
    clean_work_dirs = False
    out_dir = "premux"
    out_name = "$show$ - $ep$ (premux)"
    mkv_title_naming = r"$show$ - $ep$"

    episode: str = "01"
    work_dir: Path = None
    webhook_url: str = None

    def __init__(self, episode: str = "01", config_file: str = "config.ini"):
        """
            :param episode:         Episode identifier(?)
            :param config_file:     Path to config file (defaults to 'config.ini' in current working dir)
        """

        if config_file:
            config = ConfigParser()
            config_name = config_file

            if not os.path.exists(config_name):
                config['SETUP'] = {
                    'bdmv_dir': self.bdmv_dir,
                    'show_name': self.show_name,
                    'allow_binary_download': self.allow_binary_download,
                    'clean_work_dirs': self.clean_work_dirs,
                    'out_dir': self.out_dir,
                    'out_name': self.out_name,
                    'mkv_title_naming': self.mkv_title_naming,
                    # 'webhook_url': self.webhook_url
                }

                with open(config_name, 'w') as config_file:
                    config.write(config_file)

                raise SystemExit(f"Template config created at {Path(config_name).resolve()}.\nPlease set it up!")

            config.read(config_name)
            settings = config['SETUP']

            for key in settings:
                setattr(self, key, settings[key])

        self.episode = episode
        self.work_dir = Path(os.path.join(os.getcwd(), "_workdir", episode))
        self.work_dir.mkdir(parents=True, exist_ok=True)

        global setup
        setup = self
        return None

    def encode_video(self, clip: vs.VideoNode, settings: str = '', zones: Zone | list[Zone] = None, codec: str = 'x265',
                     save_csv_log: bool = True, generate_qpfile: bool = True, src: vs.VideoNode | src_file = None,
                     specify_props: bool = True, print_command: bool = False) -> str:
        """
            Encodes the clip you pass into it with your desired encoder

            :param clip:            Clip to be encoded
            :param settings:        Settings passed to the encoder. I recommend using the settings_builder function
            :param zones:           Zone(s) like these (start, end, bitrate_multiplier)
            :param codec:           x265, x264 and ffv1 are supported
            :param save_csv_log:    Saves the csv log file from x265
            :param generate_qpfile: Automatically generates a qpfile from your source clip (would not recommend running on the filtered clip)
            :param src:             Source Clip or `src_file` for the qpfile generation
            :param specify_props:   Specify color related props to the encoder if using x265 or x264
            :param print_command:   Prints the final encoder command before running it
            :return:                Absolute filepath for resulting video file
        """
        if codec.lower() not in ['x265', 'x264', 'ffv1']:
            raise ValueError('encode_video: codec has to be either x265, x264 or ffv1')
        encoder_exe = get_executable('ffmpeg' if codec.lower() == 'ffv1' else codec, self.allow_binary_download)
        args = settings

        if codec.lower() in ['x265', 'x264']:
            if zones:
                zones_settings: str = ''
                for i, zone in enumerate(zones):
                    if is_x264_zone(zone):
                        if codec.lower() == 'x265' and zone[2].lower() != 'q':
                            raise ValueError(f"Zone '{zone}' is invalid for x265. Please only use b or q.")
                        zones_settings += f'{zone[0]},{zone[1]},{zone[2]}={zone[3]}'
                    else:
                        zones_settings += f'{zone[0]},{zone[1]},b={zone[2]}'
                    if i != len(zones) - 1:
                        zones_settings += '/'
                args += f' --zones {zones_settings}'

            if save_csv_log and codec.lower() == 'x265':
                args += f' --csv "{Path(self.show_name + "_log_x265.csv").resolve()}"'
            if generate_qpfile:
                if isinstance(src, vs.VideoNode) or isinstance(src, src_file):
                    src = src if isinstance(src, vs.VideoNode) else src.src_cut
                    qpfile = self.generate_qp_file(src)
                    if qpfile:
                        args += f' --qpfile "{qpfile}"'
                else:
                    print("encode_video: No 'src' parameter passed, Skipping qpfile creation!")

            if specify_props:
                bits = clip.format.bits_per_sample
                c_range = ColorRange.from_video(clip).string if codec.lower() == 'x265' else \
                    ('tv' if ColorRange.from_video(clip) == ColorRange.LIMITED else 'pc')
                args += f' --input-depth {bits} --output-depth {bits} --colorprim {Primaries.from_video(clip).string}'
                args += f' --transfer {Transfer.from_video(clip).string} --colormatrix {Matrix.from_video(clip).string}'
                args += f' --chromaloc {int(ChromaLocation.from_video(clip))} --range {c_range}'

            outpath = self.work_dir.joinpath(self.episode + "." + codec.strip('x')).resolve()
            if codec.lower() == 'x265':
                encoder_command = f'"{encoder_exe}" -o "{outpath}" - --y4m ' + args.strip()
            else:
                encoder_command = f'"{encoder_exe}" -o "{outpath}" --demuxer y4m - ' + args.strip()
        else:
            if not args:
                args = f'-coder 1 -context 0 -g 1 -level 3 -threads 0 -slices 24 -slicecrc 1'
            outpath = self.work_dir.joinpath(self.episode + ".mkv").resolve()
            encoder_command = f'"{encoder_exe}" -f yuv4mpegpipe -i - -c:v ffv1 {args.strip()} "{outpath}"'
        if print_command:
            print(f'\nxEncoder Command:\n{encoder_command}\n')

        print(f"Encoding episode {self.episode} to {codec}...")
        if os.name != 'nt':
            process = subprocess.Popen(encoder_command, stdin=subprocess.PIPE, shell=True)
        else:
            process = subprocess.Popen(encoder_command, stdin=subprocess.PIPE, shell=False)
        clip.output(process.stdin, y4m=True, progress_update=lambda x, y: self._update_progress(x, y))
        process.communicate()

        print("\nDone encoding video.")
        return str(outpath.resolve())

    def encode_audio(self, file: PathLike | src_file, track: int = 0, codec: str = 'opus', q: int = 200,
                     encoder_settings: str = '', trim: Trim = None, clip: vs.VideoNode | src_file = None,# use_bs_trimming: bool = False,
                     dither_flac: bool = True, always_dither: bool = False, quiet: bool = True) -> str:
        """
            Encodes the audio

            :param file:                Either a string based filepath, a Path object or a `src_file`
            :param track:               Audio Track Number of your input file. 0-based
            :param codec:               Either flac, opus or aac. Uses ffmpeg, opusenc or qaac respectively.
                                        'pass' and 'passthrough' also exist and do what they say
            :param q:                   Quality. Basically just the bitrate when using opus and the tVBR/-V value for qaac
            :param encoder_settings:    Arguments directly passed to opusenc or qaac
            :param trim:                Tuple of frame numbers; Can be left empty if you passed a `src_file` with trims
            :param clip:                Vapoursynth VideoNode needed when trimming; Can be left empty if you passed a `src_file`
            :param dither_flac:         Will dither your FLAC output to 16bit and 48 kHz
            :param always_dither:       Dithers regardless of your final output
            :param quiet:               Will print the subprocess outputs if False
            :return:                    Absolute filepath for resulting audio file
        """
        encoder_settings = ' ' + encoder_settings.strip()
        
        if trim is not None:
            if isinstance(file, src_file):
                print("Warning: trims in src_file types will overwrite other trims passed!")
            else:
                if not isinstance(clip, vs.VideoNode) and not isinstance(clip, src_file):
                    raise "encode_audio: Trimming audio requires a clip input!"
                elif isinstance(clip, src_file):
                    clip = clip.src
                    fps = Fraction(clip.fps_num, clip.fps_den)
                else:
                    fps = Fraction(clip.fps_num, clip.fps_den)

        if isinstance(file, src_file):
            trim = file.trim
            clip = file.src
            fps = Fraction(clip.fps_num, clip.fps_den)

        file = file.file if isinstance(file, src_file) else file
        file = file if isinstance(file, Path) else Path(file)

        base_path = os.path.join(self.work_dir.resolve(), file.stem + "_" + str(track))

        def ffmpeg_header() -> str:
            ffmpeg_exe = get_executable('ffmpeg', self.allow_binary_download)
            return f'"{ffmpeg_exe}" -hide_banner{" -loglevel warning" if quiet else ""}'

        def ffmpeg_seekargs() -> str:
            args = ''
            if trim:
                if trim[0] is not None and trim[0] > 0:
                    args += f' -ss {format_timedelta(frame_to_timedelta(trim[0], fps))}'
                if trim[1] is not None and trim[1] != 0:
                    if trim[1] > 0:
                        args += f' -to {format_timedelta(frame_to_timedelta(trim[1], fps))}'
                    else:
                        end_frame = clip.num_frames - abs(trim[1])
                        args += f' -to {format_timedelta(frame_to_timedelta(end_frame, fps))}'
                if not quiet:
                    print(args)
            return args

        def toflac() -> str:
            is_intermediary = codec.lower() != 'flac'
            compression_level = "10" if not is_intermediary else "0"
            commandline = f'{ffmpeg_header()} -i "{file.resolve()}" -map_metadata -1 -map_chapters -1 -map 0:a:{track} {ffmpeg_seekargs()} -f flac -compression_level {compression_level}'
            if (dither_flac and codec.lower() == 'flac') or always_dither:
                commandline += ' -sample_fmt s16 -ar 48000 -resampler soxr -precision 28 -dither_method shibata'
            if codec.lower() != 'opus':
                _flac = base_path + ".flac"
                if not should_create_again(_flac):
                    return _flac
                commandline += f' "{_flac}"'
                print(f'Creating FLAC intermediary audio track {track} for EP{self.episode}...'
                    if is_intermediary else f'Encoding audio track {track} for EP{self.episode} to FLAC...')
                run_commandline(commandline, quiet, False)
                if not is_intermediary:
                    print('Done\n')
                return _flac
            else:
                # We can just use a cool pipe with opusenc
                return commandline + " - | "
            
        if codec.lower() == 'flac':
            return toflac()

        if codec.lower() in ['pass', 'passthrough']:
            out_file = base_path + ".mka"
            if not should_create_again(out_file):
                return out_file
            print(f'Trimming audio track {track} for EP{self.episode}...'
                if trim else f'Extracting audio track {track} for EP{self.episode}')
            commandline = f'{ffmpeg_header()} -i "{file.resolve()}" -map_metadata -1 -map_chapters -1 -map 0:a:{track} {ffmpeg_seekargs()} -c:a copy "{out_file}"'
            run_commandline(commandline, quiet, False)
            print('Done.\n')
            return out_file

        if codec.lower() == 'aac':
            if q > 127 or q < 0:
                raise ValueError(f'encode_audio: QAAC tvbr must be in the range of 0 - 127')
            flac = toflac()
            qaac =  get_executable('qaac', self.allow_binary_download)
            out_file = base_path + ".m4a"
            if not should_create_again(out_file):
                return out_file
            commandline = f'"{qaac}" -V {q} {encoder_settings} -o "{out_file}" "{flac}"'
            print(f'Encoding audio track {track} for EP{self.episode} to AAC...')
            run_commandline(commandline, quiet, False)
            print('Done.\n')
            Path(flac).unlink(missing_ok = True)
            return out_file
        
        if codec.lower() == 'opus':
            if q > 512 or q < 8:
                raise ValueError(f'encode_audio: Opus bitrate must be in the range of 8 - 512 (kbit/s)')
            commandline = toflac()
            opusenc = get_executable('opusenc', self.allow_binary_download)
            out_file = base_path + ".ogg"
            if not should_create_again(out_file):
                return out_file
            commandline += f'"{opusenc}" --bitrate {q} {encoder_settings} - "{out_file}"'
            print(f'Encoding audio track {track} for EP{self.episode} to Opus...')
            run_commandline(commandline, quiet, True)
            print('Done.\n')
            return out_file

    def generate_qp_file(self, clip: vs.VideoNode) -> str:
        filepath = os.path.join(self.work_dir, 'qpfile.txt')
        if not should_create_again(filepath, 100):
            print('Reusing existing QP File.')
            return str(Path(filepath).resolve())
        print('Generating QP File...')
        clip = clip.resize.Bicubic(640, 360, format=vs.YUV410P8)
        clip = clip.wwxd.WWXD()
        out = ""
        for i in range(1, clip.num_frames):
            if clip.get_frame(i).props.Scenechange == 1:
                out += f"{i} I -1\n"

        with open(filepath, 'w') as file:
            file.write(out)

        return str(Path(filepath).resolve()) if os.path.exists(filepath) else ""

    def from_mkv(self, mkv: PathLike, type: TrackType, track: int = -1) -> str:
        """
            Get various tracks from an existing mkv file

            :param mkv:         Path to file
            :param type:        TrackType to get
            :param track:       The *absolute* track number. No idea why they do this
                                but a specific video/audio/sub track is not a thing
                                so you're gonna have to pass the absolute number
            :return:            Path to resulting mkv or txt (if chapters)
        """
        mkv = mkv if isinstance(mkv, Path) else Path(mkv)
        mkvmerge_exe = get_executable('mkvmerge', self.allow_binary_download)
        mkvextract_exe = get_executable('mkvextract', self.allow_binary_download)
        out_file = f"{mkv.stem}_{TrackType(type).name}_{str(track)}.{'txt' if type == TrackType.CHAPTERS else 'mkv'}"
        out_path = os.path.join(self.work_dir.resolve(), out_file)

        if type == TrackType.CHAPTERS:
            commandline = f'"{mkvextract_exe}" "{mkv.resolve()}" chapters --simple "{out_path}"'
        else:
            commandline = f'"{mkvmerge_exe}" -o "{out_path} '
            if type != TrackType.ATTACHMENT and track < 0:
                raise ValueError(f'from_mkv: Please specify a track for anything but \'Attachment\'')
            match type:
                case TrackType.VIDEO:
                    commandline += f' -A -d {track} -S -B -T -M --no-chapters --no-global-tags'
                case TrackType.AUDIO:
                    commandline += f' -a {track} -D -S -B -T -M --no-chapters --no-global-tags'
                case TrackType.SUB:
                    commandline += f' -A -D -s {track} -B -T -M --no-chapters --no-global-tags'
                case TrackType.ATTACHMENT:
                    commandline += f' -A -D -S -B -T --no-chapters --no-global-tags'
            commandline += f' "{mkv.resolve()}"'

        p = subprocess.Popen(commandline, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        output, error = p.communicate()
        if p.returncode != 0:
            s = f"from_mkv: {str(output)} \n{str(error)}"
            if p.returncode == 1:
                print(f"WARN: {s}")
            else:
                raise ChildProcessError(s)

        return out_path

    def _update_progress(self, current_frame, total_frames):
        print(f"\rVapoursynth: {current_frame} / {total_frames} "
              f"({100 * current_frame // total_frames}%) || Encoder: ", end="")

    video = encode_video
    audio = encode_audio

def get_setup() -> Setup:
    global setup
    return setup

def get_workdir() -> Path:
    if get_setup() is None:
        return os.getcwd()
    return get_setup().work_dir

setup: Setup | None = None

from .auto import muxing
from .auto.muxing import VT, AT, ST, VideoTrack, AudioTrack, SubTrack, Attachment, GlobSearch, _track

class Chapters():

    chapters: list[Chapter] = []
    fps: Fraction

    def __init__(self, chapter_source: PathLike | GlobSearch | Chapter | list[Chapter] | src_file,
                 fps: Fraction = Fraction(24000, 1001), _print: bool = True) -> None:
        """
            Convenience class for chapters

            :param chapter_source:      Input either `vodesfunc.src_file` or (a list of) self defined chapters
            :param fps:                 Needed for timestamp convertion (Will be taken from your source clip
                                        if passed a `src_file`). Assumes 24000/1001 by default
            :param _print:              Prints chapters after parsing and after trimming
        """
        self.fps = fps
        if isinstance(chapter_source, tuple):
            self.chapters = [chapter_source]
        elif isinstance(chapter_source, list):
            self.chapters = chapter_source
        elif isinstance(chapter_source, src_file):
            self.chapters = parse_src_file(chapter_source, _print)
            self.fps = Fraction(chapter_source.src.fps_num, chapter_source.src.fps_den)
            if chapter_source.trim:
                self.trim(chapter_source.trim[0], chapter_source.trim[1], chapter_source)
                if _print:
                    print('After trim:')
                    self.print()
        else:
            # Handle both OGM .txt files and xml files
            if isinstance(chapter_source, GlobSearch):
                chapter_source = chapter_source.paths[0] if isinstance(chapter_source.paths, list) else chapter_source.paths
            chapter_source = chapter_source if isinstance(chapter_source, Path) else Path(chapter_source)

            self.chapters = parse_xml(chapter_source) if chapter_source.suffix.lower() == '.xml' else parse_ogm(chapter_source)
            if _print:
                self.print()
        
        # Convert all framenumbers to timedeltas
        chapters = []
        for ch in self.chapters:
            if isinstance(ch[0], int):
                current = list(ch)
                current[0] = frame_to_timedelta(current[0], self.fps)
                chapters.append(tuple(current))
            else:
                chapters.append(ch)
        self.chapters = chapters

    def trim(self, trim_start: int = 0, trim_end: int = 0, src: src_file = None):
        if trim_start > 0:
            chapters: list[Chapter] = []
            for chapter in self.chapters:
                if timedelta_to_frame(chapter[0]) == 0:
                    chapters.append(chapter)
                    continue
                if timedelta_to_frame(chapter[0]) - trim_start < 0:
                    continue
                current = list(chapter)
                current[0] = current[0] - frame_to_timedelta(trim_start, self.fps)
                if src:
                    if current[0] > frame_to_timedelta(src.src_cut.num_frames - 1, self.fps):
                        continue
                chapters.append(tuple(current))

            self.chapters = chapters
        if trim_end != 0:
            if trim_end > 0:
                chapters: list[Chapter] = []
                for chapter in self.chapters:
                    if timedelta_to_frame(chapter[0], self.fps) < trim_end:
                        chapters.append(chapter)
                self.chapters = chapters

        return self

    def set_names(self, names: list[str | None]) -> "Chapters":
        """
            Renames the chapters

            :param names:   List of names
        """
        old: list[str] = [chapter[1] for chapter in self.chapters]
        if len(names) > len(old):
            raise ValueError(f'Chapters: too many names!')
        if len(names) < len(old):
            names += [None] * (len(old) - len(names))

        chapters: list[Chapter] = []
        for i, name in enumerate(names):
            current = list(self.chapters[i])
            current[1] = name
            chapters.append(tuple(current))

        self.chapters = chapters
        return self

    def add(self, chapters: Chapter | list[Chapter], index: int = 0) -> "Chapters":
        if isinstance(chapters, tuple):
            chapters = [chapters]
        else:
            chapters = chapters
        
        converted = []
        for ch in chapters:
            if isinstance(ch[0], int):
                current = list(ch)
                current[0] = frame_to_timedelta(current[0], self.fps)
                converted.append(tuple(current))
            else:
                converted.append(ch)

        for ch in converted:
            self.chapters.insert(index, ch)
            index += 1
        return self

    def shift_chapter(self, chapter: int = 0, shift_amount: int = 0) -> "Chapters":
        """
            Used to shift a single chapter by x frames

            :param chapter:         Chapter number (starting at 0)
            :param shift_amount:    Frames to shift by
        """
        ch = list(self.chapters[chapter])
        shifted_frame = ch[0] + frame_to_timedelta(shift_amount, self.fps)
        if shifted_frame.total_seconds() > 0:
            ch[0] = shifted_frame
        else:
            ch[0] = timedelta(seconds=0)
        self.chapters[chapter] = tuple(ch)
        return self

    def print(self) -> "Chapters":
        """
            Prettier print for these because default timedelta formatting sucks
        """
        for (time, name) in self.chapters:
            print(f'{name}: {format_timedelta(time)} | {timedelta_to_frame(time, self.fps)}')
        print("", end='\n')
        return self

    def to_file(self, out: PathLike = Path(get_workdir())) -> str:
        """
            Outputs the chapters to an OGM file

            :param out:     Can be either a directory or a full file path
        """
        out = out.resolve() if isinstance(out, Path) else Path(out).resolve()
        if out.is_dir():
            out_file = os.path.join(out, 'chapters.txt')
        else:
            out_file = out
        with open(out_file, 'w', encoding='UTF-8') as f:
            f.writelines([f'CHAPTER{i:02d}={format_timedelta(chapter[0])}\nCHAPTER{i:02d}NAME='
                          f'{chapter[1] if chapter[1] else ""}\n' for i, chapter in enumerate(self.chapters)])
        return out_file


class Mux():

    outfile: str | Path
    commandline: str
    setup: Setup

    def __init__(self, setup: Setup, *tracks) -> None:
        """
            Initialize the commandline for muxing your track objects
            Call `this.run()` to actually start the process

            :param tracks:      However many track objects you want
        """
        filename = re.sub(r'\$show\$', setup.show_name, setup.out_name)
        filename = re.sub(r'\$ep\$', setup.episode, filename)
        filename = re.sub(r'\$crc32\$', "#crc32#", filename)

        mkvtitle = re.sub(r'\$show\$', setup.show_name, setup.mkv_title_naming)
        mkvtitle = re.sub(r'\$ep\$', setup.episode, mkvtitle)

        self.setup = setup
        mkvmerge = get_executable('mkvmerge', self.setup.allow_binary_download)

        self.outfile = Path(os.path.join(Path(setup.out_dir), filename + ".mkv"))
        self.commandline = f'"{mkvmerge}" -o "{self.outfile.resolve()}" --title "{mkvtitle}"'

        for track in tracks:
            if isinstance(track, _track):
                self.commandline += track.mkvmerge_args()
                continue
            elif isinstance(track, Chapters):
                chapterfile = track.to_file(setup.work_dir)
                self.commandline += f' --chapters "{chapterfile}"'
                continue
            elif isinstance(track, PathLike) or isinstance(track, GlobSearch):
                # Failsave for if someone passes Chapters().to_file() or a txt/xml file
                if isinstance(track, GlobSearch):
                    track = track.paths[0] if isinstance(track.paths, list) else track.paths
                track = track if isinstance(track, Path) else Path(track)
                if track.suffix.lower() in ['.txt', '.xml']:
                    self.commandline += f' --chapters "{track.resolve()}"'
                continue

            raise f'Mux: Only _track or Chapters types are supported as muxing input!'

    def run(self, print_command: bool = False) -> str:
        """
            Starts the muxing process

            :param print_command:   Prints final command if True
            :return:                Absolute path of resulting mux
        """
        print("Muxing episode...")
        if print_command:
            print(f'\n\n{self.commandline}\n\n')
        code = run_commandline(self.commandline, False)
        if self.setup.clean_work_dirs == True and code == 0:
            sh.rmtree(self.setup.work_dir)
        print("Done.")
        absolute = str(self.outfile.resolve())
        if r'#crc32#' in absolute:
            print("Generating CRC32 for muxed file...")
            self.outfile = self.outfile.rename(re.sub(r'#crc32#', get_crc32(self.outfile), absolute))
            print("Done.")
        return str(self.outfile.resolve())


def settings_builder_x265(
        preset: str | int = 'slow', crf: float = 14.5, qcomp: float = 0.75,
        psy_rd: float = 2.0, psy_rdoq: float = 2.0, aq_strength: float = 0.75, aq_mode: int = 3, rd: int = 4,
        rect: bool = True, amp: bool = False, chroma_qpoffsets: int = -2, tu_intra_depth: int = 2,
        tu_inter_depth: int = 2, rskip: bool | int = 0, tskip: bool = False, ref: int = 4, bframes: int = 16,
        cutree: bool = False, rc_lookahead: int = 60, subme: int = 5, me: int = 3, b_intra: bool = True,
        weightb: bool = True, deblock: list[int] | str = [-2, -2], sar: int | str = 1, append: str = "") -> str:

    # Simple insert values
    settings = f" --preset {preset} --crf {crf} --bframes {bframes} --ref {ref} --rc-lookahead {rc_lookahead} --subme {subme} --me {me}"
    settings += f" --aq-mode {aq_mode} --aq-strength {aq_strength} --qcomp {qcomp} --cbqpoffs {chroma_qpoffsets} --crqpoffs {chroma_qpoffsets}"
    settings += f" --rd {rd} --psy-rd {psy_rd} --psy-rdoq {psy_rdoq} --tu-intra-depth {tu_intra_depth} --tu-inter-depth {tu_inter_depth} --sar {sar}"

    # Less simple
    settings += f" --{'rect' if rect else 'no-rect'} --{'amp' if amp else 'no-amp'} --{'tskip' if tskip else 'no-tskip'}"
    settings += f" --{'b-intra' if b_intra else 'no-b-intra'} --{'weightb' if weightb else 'no-weightb'} --{'cutree' if cutree else 'no-cutree'}"
    settings += f" --rskip {int(rskip) if isinstance(rskip, bool) else rskip}"

    if isinstance(deblock, list):
        deblock = f"{str(deblock[0])}:{str(deblock[1])}"
    settings += f" --deblock={deblock}"

    # Don't need to change these lol
    settings += " --no-sao --no-sao-non-deblock --no-strong-intra-smoothing --no-open-gop"

    settings += (" " + append.strip()) if append.strip() else ""
    return settings

def settings_builder_x264(
        preset: str = 'placebo', crf: float = 13, qcomp: float = 0.7, psy_rd: float = 1.0, psy_trellis: float = 0.0, trellis: int = 0,
        aq_strength: float = 0.8, aq_mode: int = 3, ref: int = 16, bframes: int = 16, mbtree: bool = False, rc_lookahead: int = 250, me: str = "umh",
        subme: int = 11, threads: int = 6,
        merange: int = 32, deblock: list[int] | str = [-1, -1], dct_decimate: bool = False, sar: str = "1:1", append: str = "") -> str:

    # Simple insert values
    settings = f" --preset {preset} --crf {crf} --bframes {bframes} --ref {ref} --rc-lookahead {rc_lookahead} --me {me} --merange {merange}"
    settings += f" --aq-mode {aq_mode} --aq-strength {aq_strength} --qcomp {qcomp}"
    settings += f" --psy-rd {psy_rd}:{psy_trellis} --trellis {trellis} --subme {subme} --threads {threads} --sar {sar}"

    # Less simple
    settings += f" {'--no-mbtree' if not mbtree else ''} {'--no-dct-decimate' if not dct_decimate else ''}"

    if isinstance(deblock, list):
        deblock = f"{str(deblock[0])}:{str(deblock[1])}"
    settings += f" --deblock={deblock}"

    settings += (" " + append.strip()) if append.strip() else ""
    return settings


def light_sucks(**kwargs) -> str:
    return " --".join(f'{setting} {value}' for setting, value in kwargs.items()).strip()


def should_create_again(file: str | Path, min_bytes: int = 10000) -> bool:
    file = file if isinstance(file, Path) else Path(file)
    if file.exists() and file.stat().st_size < min_bytes:
        os.remove(file)
        return True
    elif not file.exists():
        return True
    else:
        return False

def run_commandline(command: str, quiet: bool = True, shell: bool = False) -> int:
    if os.name != 'nt':
        shell = True
    if quiet:
        p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=shell)
    else:
        p = subprocess.Popen(command, shell=shell)
    
    return p.wait()

sb = settings_builder_x265
sb265 = sb
sb264 = settings_builder_x264
