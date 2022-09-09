import os
import shutil as sh
from pathlib import Path

import py7zr as p7z
import wget

"""
Dependencies so far:
py7zr, wget,
"""

_exPrefix = 'vodesfunc.automation.download: '

__all__: list[str] = [
    'download_binary',
    'unpack_all',
]

types: dict = {
    'ffmpeg': 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z',
    'opusenc': 'https://archive.mozilla.org/pub/opus/win32/opus-tools-0.2-opus-1.3.zip',
    'qaac': 'https://github.com/nu774/qaac/releases/download/v2.76/qaac_2.76.zip',
    'x265': 'https://github.com/DJATOM/x265-aMod/releases/download/3.5+20/x265-x64-v3.5+20-aMod-gcc10.3.0-opt-znver3.7z',
    'mkvmerge': 'https://www.fosshub.com/MKVToolNix.html?dwl=mkvtoolnix-64-bit-70.0.0.7z',
    'mkvextract': 'https://www.fosshub.com/MKVToolNix.html?dwl=mkvtoolnix-64-bit-70.0.0.7z'
}


def download_binary(type: str) -> str:
    if os.name != 'nt':
        raise EnvironmentError(_exPrefix + 'Of course only Windows is supported for downloading of binaries!')

    binary_dir = Path(os.path.join(os.getcwd(), '_binaries'))
    binary_dir.mkdir(exist_ok=True)

    executable: Path = None
    executables = binary_dir.rglob(type.lower() + "*.exe")

    for exe in executables:
        if exe.is_file():
            return exe.resolve()

    print(f'Downloading {type.lower()} executables...')
    url = types.get(type.lower())
    wget.download(url, str(binary_dir.resolve()))
    print('')
    unpack_all(binary_dir)

    executables = binary_dir.rglob(type.lower() + "*.exe")

    for exe in executables:
        if exe.is_file():
            executable = exe

    return str(executable.resolve())


def unpack_all(dir: str | Path):
    dir = Path(dir) if isinstance(dir, str) else dir

    for file in dir.rglob('*.zip'):
        sh.unpack_archive(file, dir)
        os.remove(file)

    for file in dir.rglob('*.7z'):
        p7z.unpack_7zarchive(file, dir)
        os.remove(file)
