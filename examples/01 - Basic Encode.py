import vapoursynth as vs
core = vs.core

from vstools import depth

import vodesfunc as vof

# Initializes a setup for the episode number and generates a config.ini if not present already
setup = vof.Setup("01")

# Init a FileInfo-esque object
SOURCE = vof.SRC_FILE(f'{setup.bdmv_dir}/Vol.1/ANZX-15351/BDMV/STREAM/00000.m2ts')

# ... with trims
SOURCE = vof.SRC_FILE(f'{setup.bdmv_dir}/Vol.1/ANZX-15351/BDMV/STREAM/00000.m2ts', 24, -24)

# ... with your indexer of choice (lsmas in this case)
SOURCE = vof.SRC_FILE(f'{setup.bdmv_dir}/Vol.1/ANZX-15351/BDMV/STREAM/00000.m2ts', 
            idx = lambda file: core.lsmas.LWLibavSource(file))

src = SOURCE.src_cut
src = depth(src, 16)

# Just some filtering
denoise = core.knlm.KNLMeansCL(src, a=2, h=0.2, d=3, channels='UV')

# Dither back down to 10-bit for final output
out = depth(denoise, 10)

# This will run if you simply start this file in python
if __name__ == '__main__':
    settings = vof.settings_builder(crf=15, psy_rd=2.5)
    zones = [(0, 100, 0.8), (750, 900, 1.4)]
    video = setup.encode_video(out, settings, zones, generate_qpfile=True, src = SOURCE)
    audio = setup.encode_audio(SOURCE, track=0, codec='opus', q=128)
    vof.Mux(setup,
        # Alias for vof.VideoTrack
        vof.VT(video, name='Encode by some guy'),
        # Alias for vof.AudioTrack
        vof.AT(audio, name='Japanese 2.0 Opus'),
        # This grabs the chapters from the playlist in your bdmv folder and renames them
        # Will also, like everything else, respect the trims you set in your SRC_FILE
        vof.Chapters(SOURCE).set_names(['Prologue', 'Part A', 'Part B', 'Opening', 'Preview'])
    ).run()

# My set_output replacement with naming
vof.out(out, 'Example name')