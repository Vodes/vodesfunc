import os
import shutil as sh

from .download import *

_exPrefix = 'vodesfunc.automation.check.'

def check_QAAC(download: bool = False) -> bool | str:
    path = sh.which('qaac')
    if path is None:
        if not download:
            raise Exception(_exPrefix + "check_QAAC: QAAC executable not found in path!")
        else:
            path = download_binary('qaac')

    return path

def check_FFmpeg(download: bool = False) -> bool | str:
    path = sh.which('ffmpeg')
    if path is None:
        if not download:
            raise Exception(_exPrefix + "check_FFmpeg: FFmpeg executable not found in path!")
        else:
            path = download_binary('ffmpeg')
            
    return path

def check_OpusEnc(download: bool = False) -> bool | str:
    path = sh.which('opusenc')
    if path is None:
        if not download:
            raise Exception(_exPrefix + "check_OpusEnc: OpusEnc executable not found in path!")
        else:
            path = download_binary('opusenc')
            
    return path

def check_mkvmerge(download: bool = False) -> bool | str:
    path = sh.which('mkvmerge')
    if path is None:
        if not download:
            raise Exception(_exPrefix + "check_mkvmerge: mkvmerge executable not found in path!")
        else:
            path = download_binary('mkvmerge')
            
    return path

def check_mkvextract(download: bool = False) -> bool | str:
    path = sh.which('mkvextract')
    if path is None:
        if not download:
            raise Exception(_exPrefix + "check_mkvextract: mkvextract executable not found in path!")
        else:
            path = download_binary('mkvextract')
            
    return path

def check_x265(download: bool = False) -> bool | str:
    path = sh.which('x265')
    if path is None:
        if not download:
            raise Exception(_exPrefix + "check_x265: x265 executable not found in path!")
        else:
            path = download_binary('x265')
            
    return path