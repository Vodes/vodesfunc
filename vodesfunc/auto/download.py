import os
import shutil as sh
from pathlib import Path

import py7zr as p7z
import wget
import shutil as sh

__all__: list[str] = [
    'get_executable',
    'download_binary',
    'unpack_all',
]

types: dict = {
    'ffmpeg': 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z',
    'opusenc': 'https://archive.mozilla.org/pub/opus/win32/opus-tools-0.2-opus-1.3.zip',
    'qaac': 'https://github.com/nu774/qaac/releases/download/v2.76/qaac_2.76.zip',
    'x265': 'https://github.com/DJATOM/x265-aMod/releases/download/3.5+20/x265-x64-v3.5+20-aMod-gcc10.3.0-opt-znver3.7z',
    'x264': 'https://github.com/DJATOM/x264-aMod/releases/download/r3101%2B20/x264-aMod-x64-core164-r3101+20.7z',
    # Before anyone complains, this is the exact same file as on https://www.fosshub.com/MKVToolNix.html?dwl=mkvtoolnix-64-bit-72.0.0.7z
    # Feel free to compare the hashes; They changed their cloudflare bot settings and wget doesn't work... 
    'mkvmerge': 'https://files.catbox.moe/5efx9l.7z',
    'mkvextract': 'https://files.catbox.moe/5efx9l.7z'
}

def get_executable(type: str, can_download: bool = True) -> str:
    path = sh.which(type)
    if path is None:
        if not can_download or can_download == False:
            raise Exception(f"{type.upper()} executable not found in path!")
        else:
            path = download_binary(type.lower())

    return path

def download_binary(type: str) -> str:
    if os.name != 'nt':
        raise EnvironmentError('Of course only Windows is supported for downloading of binaries!')

    binary_dir = Path(os.path.join(os.getcwd(), '_binaries'))
    binary_dir.mkdir(exist_ok=True)

    executable: Path = None
    executables = binary_dir.rglob(type.lower() + "*.exe")

    for exe in sorted(executables):
        if exe.is_file():
            return exe.resolve()

    print(f'Downloading {type.lower()} executables...')
    url = types.get(type.lower())
    wget.download(url, str(binary_dir.resolve()))
    print('')
    unpack_all(binary_dir)

    executables = binary_dir.rglob(type.lower() + "*.exe")

    for exe in sorted(executables):
        if exe.is_file():
            executable = exe

    if executable is None:
        raise f"Binary for '{type}' could not have been found!"

    return str(executable.resolve())


def unpack_all(dir: str | Path):
    dir = Path(dir) if isinstance(dir, str) else dir

    for file in dir.rglob('*.zip'):
        out = Path(os.path.join(file.resolve(True).parent, file.stem))
        out.mkdir(exist_ok=True)
        sh.unpack_archive(file, out)
        os.remove(file)

    for file in dir.rglob('*.7z'):
        out = Path(os.path.join(file.resolve(True).parent, file.stem))
        out.mkdir(exist_ok=True)
        p7z.unpack_7zarchive(file, out)
        os.remove(file)
