from vstools import (
    initialize_clip,
    finalize_clip,
    Keyframes,
    get_depth,
    vs,
    FrameRangesN,
    FrameRangeN,
)
from jetpytools import SoftRange, normalize_ranges_to_list, normalize_list_to_ranges
from muxtools import get_executable, PathLike, VideoFile, warn, make_output, ensure_path_exists, info, debug
from vsmuxtools.video.encoders import VideoEncoder
from dataclasses import dataclass
import shlex
import subprocess
import json
from fractions import Fraction
from pathlib import Path

__all__ = ["find_spikes"]


@dataclass
class NVENC_H265(VideoEncoder):
    """
    Uses ffmpeg to encode clip to a h265 stream via nvenc.
    (Should this be in vsmuxtools?)

    :param settings:        Can either be a string of your own settings or any of the 3 presets.
    :param ensure_props:    Calls initialize_clip on the clip to have at the very least guessed props
    """

    settings: str = ""
    ensure_props: bool = True

    def __post_init__(self):
        self.executable = get_executable("ffmpeg")

    def encode(self, clip: vs.VideoNode, outfile: PathLike | None = None) -> VideoFile:
        bits = get_depth(clip)
        if bits > 10:
            warn("This encoder does not support a bit depth over 10.\nClip will be dithered to 10 bit.", self, 2)
            clip = finalize_clip(clip, 10)
            bits = 10
        if self.ensure_props:
            clip = initialize_clip(clip, bits)
            clip = finalize_clip(clip, bits)

        out = make_output("encoded_nvenc", "mkv", user_passed=outfile)

        args = [self.executable, "-hide_banner", "-v", "quiet", "-stats", "-f", "yuv4mpegpipe", "-i", "-", "-c:v", "hevc_nvenc"]
        if self.settings:
            args.extend(shlex.split(self.settings))
        args.append(str(out))

        process = subprocess.Popen(args, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda x, y: self._update_progress(x, y))  # type: ignore
        process.communicate()
        return VideoFile(out)


@dataclass
class Framedata:
    frame: int
    frame_time: float
    frame_size: float


def fetch_frames_with_sizes(fileIn: Path, fps: Fraction) -> list[Framedata]:
    """
    Extracts frame info with ffprobe
    """
    bitrate_data = list[Framedata]()
    current_frame = 0
    command = [
        get_executable("ffprobe"),
        "-show_entries",
        "packet=size,duration_time,pts_time",
        "-select_streams",
        "v",
        "-print_format",
        "json=compact=1",
        str(fileIn),
    ]
    out = subprocess.run(command, capture_output=True, text=True, universal_newlines=True)
    output = out.stdout + out.stderr
    for line in output.splitlines():
        if len(line) == 0:
            break
        if len(line) > 0 and line[-1] == ",":
            line = line[:-1]

        if "pts_time" in line:
            try:
                decoded = json.loads(line)
            except:
                print(line)
                raise Exception

            frame_bitrate = (float(decoded.get("size")) * 8 / 1000) * fps
            frame_time = float(decoded.get("pts_time"))
            bitrate_data.append(Framedata(current_frame, frame_time, frame_bitrate))
            current_frame += 1

    return bitrate_data


def split_by_keyframes(data: list[Framedata], clip: vs.VideoNode) -> list[list[Framedata]]:
    """
    Search for scene changes and divide framedata into chunks that start with one.
    """
    keyframes = Keyframes.from_clip(clip)
    chunks = []
    current_chunk = []

    for item in data:
        if item.frame in keyframes:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [item]
        else:
            current_chunk.append(item)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def read_ranges_bookmarks(fileIn: Path) -> FrameRangesN:
    ranges = list[FrameRangeN]()
    with open(fileIn, "r", encoding="utf-8") as reader:
        line = reader.readline()
        ints = [int(it.strip()) for it in line.split(",")]
        for range_start in ints[0::2]:
            ranges.append((range_start, ints[ints.index(range_start) + 1]))

    return ranges


def find_spikes(
    clip: vs.VideoNode,
    threshold: int = 11500,
    nvenc_settings: str = "-preset 3 -rc vbr_hq -pix_fmt p010le -b:v 6M -maxrate:v 22M",
    print_ranges: bool = False,
    export_file: None | PathLike = None,
    ignore_existing: bool = False,
) -> FrameRangesN:
    """
    Encodes a clip with nvenc hevc and analyzes the bitrate averages between scene changes to find spikes.

    :param clip:            Clip to encode
    :param threshold:       Bitrate threshold to add to ranges (in kbps, I think)
    :param nvenc_settings:  Settings to use for the encoder
    :param print_ranges:    If you want to print the ranges with corresponding bitrates
    :param export_file:     Export the ranges to a bookmarks file with the given name. None to disable.
    :param ignore_existing: Run again and overwrite the exported file if it exists. By default it won't run again.
    """
    if export_file:
        out_file = make_output(export_file, "bookmarks", user_passed=export_file)
        if out_file.exists():
            if ignore_existing:
                out_file.unlink(True)
            else:
                return read_ranges_bookmarks(out_file)

    ranges: list[SoftRange] = []
    info("Encoding clip using nvenc...", find_spikes)
    temp_encode = NVENC_H265(nvenc_settings).encode(clip, "temp_nvenc")
    encoded_file = ensure_path_exists(temp_encode.file, find_spikes)
    info("Extracting frame data...", find_spikes)
    framedata = fetch_frames_with_sizes(encoded_file, Fraction(clip.fps_num, clip.fps_den))
    info("Finding scene changes...")
    chunks = split_by_keyframes(framedata, clip)
    encoded_file.unlink(True)

    for chunk in chunks:
        size_all: float = 0.0
        for data in chunk:
            size_all += data.frame_size
        avg = size_all / len(chunk)
        if avg > threshold:
            ranges.append((chunk[0].frame, chunk[-1].frame))
            if print_ranges:
                debug(f"Frames {chunk[0].frame} - {chunk[-1].frame}: {round(avg, 2)} kbps", find_spikes)

    # To make the ranges not have single frame outliers
    ranges_int = normalize_ranges_to_list(ranges)
    final_ranges = normalize_list_to_ranges(ranges_int)

    if export_file:
        with open(out_file, "w", encoding="utf-8") as writer:
            all_nums = list[int]()
            for start, end in final_ranges:
                all_nums.extend([start, end])
            writer.write(", ".join([str(it) for it in all_nums]))

    return final_ranges
