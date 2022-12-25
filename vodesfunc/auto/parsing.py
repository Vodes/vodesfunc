import re
import os
from pathlib import Path
from fractions import Fraction
from pyparsebluray import mpls
from ..types import Chapter
from .convert import timedelta_from_formatted, timedelta_to_frame, \
        frame_to_timedelta, mpls_timestamp_to_timedelta, format_timedelta


__all__: list[str] = [
    'parse_ogm',
    'parse_xml',
    'parse_src_file',
    'parse_m2ts_path'
]

OGM_REGEX = r'(^CHAPTER(?P<num>\d+)=(?P<time>.*)\nCHAPTER\d\dNAME=(?P<name>.*))'
XML_REGEX = r'(\<ChapterAtom\>.*?\<ChapterTimeStart\>(?P<time>[^\<]*).*?\<ChapterString\>(?P<name>[^\<]*)\<\/ChapterString\>.*?\<\/ChapterAtom\>)'

def parse_ogm(file: Path) -> list[Chapter]:
    return _parse_chapters(file, OGM_REGEX, re.I | re.M) 

def parse_xml(file: Path) -> list[Chapter]:
    return _parse_chapters(file, XML_REGEX, re.I | re.M | re.S)

def _parse_chapters(file: Path, reg: str, flags: int = 0) -> list[Chapter]:
    chapters: list[Chapter] = []
    with file.open('r', encoding='utf-8') as f:
        for match in re.finditer(re.compile(reg, flags), f.read()):
            chapters.append(
                (timedelta_from_formatted(match.group('time')),
                match.group('name'))
            )

    return chapters

def parse_m2ts_path(dgiFile: Path) -> Path:
    with open(dgiFile, 'r') as fp:
        for i, line in enumerate(fp):
            for match in re.finditer(re.compile(r"^(.*\.m2ts) \d+$", re.IGNORECASE), line):
                m2tsFile = Path(match.group(1))
                if m2tsFile.exists():
                    return m2tsFile
    print("Warning!\nCould not resolve origin file path from the dgindex input!")
    return dgiFile

from ..util import src_file

def parse_src_file(src: src_file, _print: bool = False) -> list[Chapter]:
    stream_dir = src.file.resolve().parent
    if stream_dir.name.lower() != 'stream':
        print(f'Your source file is not in a default bdmv structure!\nWill skip chapters.')
        return None
    playlist_dir = Path(os.path.join(stream_dir.parent, "PLAYLIST"))
    if not playlist_dir.exists():
        print(f'PLAYLIST folder couldn\'t have been found!\nWill skip chapters.')
        return None

    chapters: list[Chapter] = []
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
                raise 'There is no playlist marks in this file!'

        for i, playitem in enumerate(playlist.play_items):
            if playitem.clip_information_filename == src.file.stem and \
                    playitem.clip_codec_identifier.lower() == src.file.suffix.lower().split('.')[1]:
                if _print:
                    print(f'Found chapters for "{src.file.name}" in "{f.name}":')
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
                    except:
                        print('Couldn\'t parse fps from playlist! Will take fps from source clip.')
                        fps = Fraction(src.src_cut.fps_num, src.src_cut.fps_den)

                    for i, lmark in enumerate(linked_marks, start=1):
                        time = mpls_timestamp_to_timedelta(lmark.mark_timestamp - offset)
                        if time > frame_to_timedelta(src.src.num_frames - 1, fps):
                            continue
                        chapters.append((time, f'Chapter {i:02.0f}'))
                    if chapters and _print:
                        for (time, name) in chapters:
                            print(f'{name}: {format_timedelta(time)} | {timedelta_to_frame(time, fps)}')

        if chapters:
            break

    return chapters