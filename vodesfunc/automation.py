import os
import re
import shutil as sh
import subprocess
from configparser import ConfigParser
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
import vapoursynth as vs
from pyparsebluray import mpls
from pytimeconv import Convert

from .auto import check, muxing
from .auto.muxing import TrackType
from .types import PathLike, Trim, Zone

_exPrefix = 'vodesfunc.automation.'


__all__: list[str] = [
    'Chapters',
    'get_chapters_from_srcfile',
    'light_sucks',
    'microsecond_duration',
    'Mux',
    'settings_builder', 'sb',
    'Setup',
    'should_create_again',
    'src_file', 'SRC_FILE',
]


class src_file:

    file: Path
    src: vs.VideoNode
    src_cut: vs.VideoNode
    trim: Trim = None

    def __init__(self, file: PathLike, trim_start: int = 0, trim_end: int = 0, idx: Callable[[str], vs.VideoNode] = None) -> None:
        """
            Custom `FileInfo` kind of thing for convenience

            :param file:            Either a string based filepath or a Path object
            :param trim_start:      At what frame the `src_cut` clip should start
            :param trim_end:        At what frame the `src_cut` clip should end
            :param idx:             Indexer for the input file. Pass a function that takes a string in and returns a vs.VideoNode.\nDefaults to `vodesfunc.src`
        """
        from vodesfunc import source
        self.file = file if isinstance(file, Path) else Path(file)
        self.src = idx(str(self.file.resolve())) if idx else source(str(self.file.resolve()))
        if trim_start != 0 or trim_end != 0:
            self.trim = (trim_start, trim_end)
            if trim_start != 0 and trim_end != 0:
                self.src_cut = self.src[trim_start: trim_end]
            else:
                if trim_start != 0:
                    self.src_cut = self.src[trim_start:]
                else:
                    self.src_cut = self.src[:trim_end]
        else:
            self.src_cut = self.src


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
            config_name = 'config.ini'

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
        return None

    def encode_video(self, clip: vs.VideoNode, settings: str, zones: Zone | list[Zone] = None,
                     save_csv_log: bool = True, generate_qpfile: bool = True, src: vs.VideoNode | src_file = None,
                     print_command: bool = False) -> str:
        """
            Encodes the clip you pass into it with x265

            :param clip:            Clip to be encoded
            :param settings:        Settings passed to x265. I recommend using the settings_builder function
            :param zones:           Zone(s) like these (start, end, bitrate_multiplier)
            :param save_csv_log:    Saves the csv log file from x265
            :param generate_qpfile: Automatically generates a qpfile from your source clip (would not recommend running on the filtered clip)
            :param src:             Source Clip or `src_file` for the qpfile generation
            :param print_command:   Prints the final x265 command before running it
            :return:                Absolute filepath for resulting video file
        """
        x265_exe = check.check_x265(self.allow_binary_download)
        args = settings
        # TODO: range, matrix, etc. parsing from the clip
        args += f' --output-depth {clip.format.bits_per_sample} --range limited --colorprim 1 --transfer 1 --colormatrix 1'

        if zones:
            zones_settings: str = ''
            for i, ((start, end, multiplier)) in enumerate(zones):
                zones_settings += f'{start},{end},b={multiplier}'
                if i != len(zones) - 1:
                    zones_settings += '/'
            args += f' --zones {zones_settings}'

        if save_csv_log:
            args += f' --csv "{Path(self.show_name + "_log_x265.csv").resolve()}"'
        if generate_qpfile:
            if isinstance(src, vs.VideoNode) or isinstance(src, src_file):
                src = src if isinstance(src, vs.VideoNode) else src.src_cut
                qpfile = self.generate_qp_file(src)
                if qpfile:
                    args += f' --qpfile "{qpfile}"'
            else:
                print(_exPrefix + "encode_video: No 'src' parameter passed, Skipping qpfile creation!")

        outpath = self.work_dir.joinpath(self.episode + ".265").resolve()
        x265_command = f'"{x265_exe}" -o "{outpath}" - --y4m ' + args.strip()

        if print_command:
            print(f'\nx265 Command:\n{x265_command}\n')

        print(f"Encoding episode {self.episode}...")
        process = subprocess.Popen(x265_command, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda x, y: self._update_progress(x, y))
        process.communicate()

        print("\nDone encoding video.")
        return str(outpath.resolve())

    def encode_audio(self, file: PathLike | src_file, track: int = 0, codec: str = 'opus', q: int = 200,
                     encoder_settings: str = '', trim: Trim = None, clip: vs.VideoNode | src_file = None,
                     dither_flac: bool = True) -> str:
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
            :param dither_flac:         Will dither your FLAC output to 16bit
            :return:                    Absolute filepath for resulting audio file
        """
        encoder_settings = ' ' + encoder_settings.strip()

        if trim is not None:
            if isinstance(file, src_file):
                print("Warning: trims in src_file types will overwrite other trims passed!")
            else:
                if not isinstance(clip, vs.VideoNode) and not isinstance(clip, src_file):
                    raise _exPrefix + ".encode_audio: Trimming audio requires a clip input!"
                elif isinstance(clip, src_file):
                    clip = clip.src

        if isinstance(file, src_file):
            trim = file.trim
            clip = file.src_cut

        file = file.file if isinstance(file, src_file) else file
        file = file if isinstance(file, Path) else Path(file)

        ffmpeg_exe = check.check_FFmpeg(self.allow_binary_download)
        flac = os.path.join(self.work_dir.resolve(), file.stem + "_" + str(track) + ".flac")
        commandline = f'"{ffmpeg_exe}" -i "{file.resolve()}" -map 0:a:{track}'
        if should_create_again(flac) and codec.lower() not in ['pass', 'passthrough']:
            if trim:
                if isinstance(trim[0], int) and trim[0] != 0:
                    commandline += f' -ss "{microsecond_duration(trim[0], clip.fps_num, clip.fps_den)}us"'
                if isinstance(trim[1], int) and trim[1] != 0:
                    if trim[1] < 0:
                        commandline += f' -to "{microsecond_duration(clip.num_frames, clip.fps_num, clip.fps_den) - microsecond_duration(abs(trim[1]), clip.fps_num, clip.fps_den)}us"'
                    else:
                        commandline += f' -to "{microsecond_duration(trim[1], clip.fps_num, clip.fps_den)}us"'

            commandline += f' -c flac -compression_level 10'
            if codec.lower() == 'flac' and dither_flac:
                commandline += f' -resampler soxr -sample_fmt s16 -ar 48000 -precision 28 -dither_method shibata'
            commandline += f' "{flac}"'
            print("Creating FLAC intermediary for actual target codec..."
                  if codec.lower() != 'flac' else f"Encoding FLAC Audio for EP{self.episode}...")
            subprocess.run(commandline, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if codec.lower() == 'flac':
                print('Done.\n')

        if codec.lower() == 'opus':
            if q > 512 or q < 8:
                raise ValueError(f'{_exPrefix}.encode_audio: OPUS bitrate must be in the range of 8 - 512 (kbit/s)')

            opusenc_exe = check.check_OpusEnc(self.allow_binary_download)
            out = os.path.join(self.work_dir.resolve(), file.stem + "_" + str(track) + ".ogg")
            if should_create_again(out):
                commandline = f'"{opusenc_exe}" --bitrate {q} {encoder_settings} "{flac}" "{out}"'
                print(f"Encoding OPUS Audio for EP{self.episode}...")
                subprocess.run(commandline, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print('Done.\n')
            return out
        elif codec.lower() == 'aac':
            if q > 127 or q < 0:
                raise ValueError(f'{_exPrefix}.encode_audio: QAAC tvbr must be in the range of 0 - 127')

            qaac_exe = check.check_QAAC(self.allow_binary_download)
            out = os.path.join(self.work_dir.resolve(), file.stem + "_" + str(track) + ".m4a")
            if should_create_again(out):
                commandline = f'"{qaac_exe}" -V {q} {encoder_settings} -o "{out}" "{flac}"'
                print(f"Encoding AAC Audio for EP{self.episode}...")
                subprocess.run(commandline, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print('Done.\n')
            return out
        elif codec.lower() in ['pass', 'passthrough']:
            if trim:
                if isinstance(trim[0], int) and trim[0] != 0:
                    commandline += f' -ss "{microsecond_duration(trim[0], clip.fps_num, clip.fps_den)}us"'
                if isinstance(trim[1], int) and trim[1] != 0:
                    if trim[1] < 0:
                        commandline += f' -to "{microsecond_duration(clip.num_frames, clip.fps_num, clip.fps_den) - microsecond_duration(abs(trim[1]), clip.fps_num, clip.fps_den)}us"'
                    else:
                        commandline += f' -to "{microsecond_duration(trim[1], clip.fps_num, clip.fps_den)}us"'

            out = os.path.join(self.work_dir.resolve(), file.stem + "_" + str(track) + ".mka")
            if should_create_again(out):
                print(f"Cutting audio without reencoding..." if trim else f"Extracting audio for EP{self.episode}...")
                commandline += f' -c:a copy -rf64 auto "{out}"'
                subprocess.run(
                    commandline,
                    # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print('Done.\n')
            return out
        else:
            return flac

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
        mkvmerge_exe = check.check_mkvmerge(self.allow_binary_download)
        mkvextract_exe = check.check_mkvextract(self.allow_binary_download)
        out_file = f"{mkv.stem}_{TrackType(type).name}_{str(track)}.{'txt' if type == TrackType.CHAPTERS else 'mkv'}"
        out_path = os.path.join(self.work_dir.resolve(), out_file)

        if type == TrackType.CHAPTERS:
            commandline = f'"{mkvextract_exe}" "{mkv.resolve()}" chapters --simple "{out_path}"'
        else:
            commandline = f'"{mkvmerge_exe}" -o "{out_path} '
            if type != TrackType.ATTACHMENT and track < 0:
                raise ValueError(f'{_exPrefix}.from_mkv: Please specify a track for anything but \'Attachment\'')
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
            s = f"{_exPrefix}.from_mkv: {str(output)} \n{str(error)}"
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


class Chapters():

    chapters: list[muxing.Chapter] = []
    fps: Fraction

    def __init__(self, chapter_source: PathLike | muxing.Chapter | list[muxing.Chapter] | src_file,
                 fps: Fraction = Fraction(24000, 1001)) -> None:
        """
            Convenience class for chapters

            :param chapter_source:      Input either `vodesfunc.src_file` or (a list of) self defined chapters
            :param fps:                 Needed for timestamp convertion (Will be taken from your source clip
                                        if passed a `src_file`). Assumes 24000/1001 by default
        """
        self.fps = fps
        if isinstance(chapter_source, tuple):
            self.chapters = [chapter_source]
        elif isinstance(chapter_source, list):
            self.chapters = chapter_source
        elif isinstance(chapter_source, src_file):
            self.chapters = get_chapters_from_srcfile(chapter_source)
            self.fps = Fraction(chapter_source.src.fps_num, chapter_source.src.fps_den)
            if chapter_source.trim:
                self.trim(chapter_source.trim[0], chapter_source.trim[1])
        elif isinstance(chapter_source, PathLike):
            # Handle both OGM .txt files and xml files maybe
            # Supposed to be used with something like setup.from_mkv()
            file = file if isinstance(chapter_source, Path) else Path(chapter_source)
            raise f'{_exPrefix}Chapters: No Chapterfile input supported (yet)'
        pass

    def trim(self, trim_start: int = 0, trim_end: int = 0):
        if trim_start > 0:
            chapters: list[muxing.Chapter] = []
            for chapter in self.chapters:
                if chapter[0] - trim_start < 0:
                    chapters.append(chapter)
                    continue
                current = list(chapter)
                current[0] = current[0] - trim_start
                chapters.append(tuple(current))

            self.chapters = chapters
        if trim_end != 0:
            if trim_end > 0:
                chapters: list[muxing.Chapter] = []
                for chapter in self.chapters:
                    if chapter[0] < trim_end:
                        chapters.append(chapter)
                self.chapters = chapters

        return self

    def set_names(self, names: list[str | None]):
        old: list[str] = [chapter[1] for chapter in self.chapters]
        if len(names) > len(old):
            raise ValueError(f'Chapters: too many names!')
        if len(names) < len(old):
            names += [None] * (len(old) - len(names))

        chapters: list[muxing.Chapter] = []
        for i, name in enumerate(names):
            current = list(self.chapters[i])
            current[1] = name
            chapters.append(tuple(current))

        self.chapters = chapters
        return self

    def to_file(self, work_dir: PathLike = Path(os.getcwd())) -> str:
        work_dir = work_dir.resolve() if isinstance(work_dir, Path) else Path(work_dir).resolve()
        out_file = os.path.join(work_dir, 'chapters.txt')
        with open(out_file, 'w', encoding='UTF-8') as f:
            f.writelines([f'CHAPTER{i:02d}={Convert.f2assts(chapter[0], self.fps)}\nCHAPTER{i:02d}NAME='
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
        filename = re.sub('\$show\$', setup.show_name, setup.out_name)
        filename = re.sub('\$ep\$', setup.episode, filename)

        mkvtitle = re.sub('\$show\$', setup.show_name, setup.mkv_title_naming)
        mkvtitle = re.sub('\$ep\$', setup.episode, mkvtitle)

        self.setup = setup
        mkvmerge = check.check_mkvmerge(self.setup.allow_binary_download)

        self.outfile = Path(os.path.join(Path(setup.out_dir), filename + ".mkv"))
        self.commandline = f'"{mkvmerge}" -o "{self.outfile.resolve()}" --title "{mkvtitle}"'

        for track in tracks:
            if isinstance(track, muxing._track):
                self.commandline += track.mkvmerge_args()
                continue
            elif isinstance(track, Chapters):
                chapterfile = track.to_file(setup.work_dir)
                self.commandline += f' --chapters "{chapterfile}"'
                continue
            elif isinstance(track, PathLike):
                # Failsave for if someone passes Chapters().to_file()
                chapterfile = track if isinstance(chapterfile, Path) else Path(chapterfile)
                if chapterfile.name.lower() == 'chapters.txt':
                    self.commandline += f' --chapters "{chapterfile}"'
                continue

            raise f'{_exPrefix}.Mux: Only _track or Chapters types are supported as muxing input!'

    def run(self, print_command: bool = False) -> str:
        """
            Starts the muxing process

            :param print_command:   Prints final command if True
            :return:                Absolute path of resulting mux
        """
        print("Muxing episode...")
        if print_command:
            print(f'\n\n{self.commandline}\n\n')
        code = subprocess.Popen(self.commandline).wait()
        if self.setup.clean_work_dirs and code == 0:
            sh.rmtree(self.setup.work_dir)
        print("Done.")
        return str(self.outfile.resolve())


def settings_builder(
        preset: str | int = 'slow', crf: float = 14.5, qcomp: float = 0.75,
        psy_rd: float = 2.0, psy_rdoq: float = 2.0, aq_strength: float = 0.75, aq_mode: int = 3, rd: int = 4,
        rect: bool = True, amp: bool = False, chroma_qpoffsets: int = -2, tu_intra_depth: int = 2,
        tu_inter_depth: int = 2, rskip: bool | int = 0, tskip: bool = False, ref: int = 4, bframes: int = 16,
        cutree: bool = False, rc_lookahead: int = 60, subme: int = 5, me: int = 3, b_intra: bool = True,
        weightb: bool = True, append: str = "") -> str:

    # Simple insert values
    settings = f" --preset {preset} --crf {crf} --bframes {bframes} --ref {ref} --rc-lookahead {rc_lookahead} --subme {subme} --me {me}"
    settings += f" --aq-mode {aq_mode} --aq-strength {aq_strength} --qcomp {qcomp} --cbqpoffs {chroma_qpoffsets} --crqpoffs {chroma_qpoffsets}"
    settings += f" --rd {rd} --psy-rd {psy_rd} --psy-rdoq {psy_rdoq} --tu-intra-depth {tu_intra_depth} --tu-inter-depth {tu_inter_depth}"

    # Less simple
    settings += f" --{'rect' if rect else 'no-rect'} --{'amp' if amp else 'no-amp'} --{'tskip' if tskip else 'no-tskip'}"
    settings += f" --{'b-intra' if b_intra else 'no-b-intra'} --{'weightb' if weightb else 'no-weightb'} --{'cutree' if cutree else 'no-cutree'}"
    settings += f" --rskip {int(rskip) if isinstance(rskip, bool) else rskip}"

    # Don't need to change these lol
    settings += " --deblock=-2:-2 --no-sao --no-sao-non-deblock --no-strong-intra-smoothing --no-open-gop"

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


def microsecond_duration(frame: int, fps_num: int = 24000, fps_den: int = 1001) -> int:
    framerate = np.divide(float(fps_num), float(fps_den), dtype=np.longdouble)
    frametime_microseconds = np.multiply(np.divide(float(1), framerate, dtype=np.longdouble),
                                         float(1000) * float(1000), dtype=np.longdouble)
    frametime_microseconds = np.multiply(frametime_microseconds, frame, dtype=np.longdouble)
    rounded_microseconds = np.round(frametime_microseconds)
    return int(rounded_microseconds)


def get_chapters_from_srcfile(src: src_file) -> list[muxing.Chapter]:
    stream_dir = src.file.resolve().parent
    if stream_dir.name.lower() != 'stream':
        print(f'Your source file is not in a default bdmv structure!\nWill skip chapters.')
        return None
    playlist_dir = Path(os.path.join(stream_dir.parent, "PLAYLIST"))
    if not playlist_dir.exists():
        print(f'PLAYLIST folder couldn\'t have been found!\nWill skip chapters.')
        return None

    chapters: list[muxing.Chapter] = []
    for f in playlist_dir.rglob("*"):
        if not os.path.isfile(f) or f.suffix.lower() != '.mpls':
            continue
        with f.open('rb') as file:
            header = mpls.load_movie_playlist(file)
            file.seek(header.playlist_start_address, os.SEEK_SET)
            playlist = mpls.load_playlist(file)
            if not playlist.play_items:
                continue

            file.seek(header.playlist_mark_start_address, os.SEEK_SET)
            playlist_mark = mpls.load_playlist_mark(file)
            if (plsmarks := playlist_mark.playlist_marks) is not None:
                marks = plsmarks
            else:
                raise ValueError(f'There is no playlist marks in this file!')

        for i, playitem in enumerate(playlist.play_items):
            if playitem.clip_information_filename == src.file.stem and \
                    playitem.clip_codec_identifier.lower() == src.file.suffix.lower().split('.')[1]:

                linked_marks = [mark for mark in marks if mark.ref_to_play_item_id == i]
                try:
                    assert playitem.intime
                    offset = min(playitem.intime, linked_marks[0].mark_timestamp)
                except IndexError:
                    continue
                if playitem.stn_table and playitem.stn_table.length != 0 and playitem.stn_table.prim_video_stream_entries \
                        and (fps_n := playitem.stn_table.prim_video_stream_entries[0][1].framerate):
                    try:
                        fps = mpls.FRAMERATE[fps_n]
                    except AttributeError as attr_err:
                        print('Couldn\'t parse fps from playlist! Will take fps from source clip.')
                        fps = Fraction(src.src_cut.fps_num, src.src_cut.fps_den)

                    for i, lmark in enumerate(linked_marks, start=1):
                        frame = Convert.seconds2f((lmark.mark_timestamp - offset) / 45000, fps)
                        chapters.append((frame, f'Chapter {i:02.0f}'))

    return chapters


sb = settings_builder
SRC_FILE = src_file
