import vodesfunc as vof

# This has 23 episodes
for ep in range(1, 24):
    # Format to have leading zeroes...
    ep = f"{ep:02d}"

    # As a reminder, this will also use the file & title pattern in the config.ini
    # for the resulting mux
    setup = vof.Setup(ep)

    r"""
        Imagine if you will, these files:
        D:\Muxing\86\dialogue eng\86 - 01 (Dialogue) [eng].ass
        and
        D:\Muxing\86\signs eng\86 - 01 (Signs) [eng].ass
        
        This will match both and merge them in the SubTrack call
    """
    eng_paths = vof.GlobSearch(f'86 - {setup.episode}*eng*.ass', allow_multiple=True, recursive=True, dir=r'D:\Muxing\86')
    eng_sub = vof.SubTrack(eng_paths, name='English Merged')

    # Same deal for german subs
    ger_paths = vof.GlobSearch(f'86 - {setup.episode}*ger*.ass', allow_multiple=True, dir=r'D:\Muxing\86')
    ger_sub = vof.SubTrack(ger_paths, name='German Merged', lang='ger', default=False)

    # Gets all the fonts needed for the subs from your system fonts (ofc you can also add your own additional paths)
    fonts = eng_sub.collect_fonts(setup.work_dir)
    # Not an issue to call twice, because it will just return all the fonts in the workdir
    fonts = ger_sub.collect_fonts(setup.work_dir)

    # As you can see, you can also use the globsearch for single files and other kinds of tracks
    audio_search = vof.GlobSearch(f'86 - {setup.episode}*ger*.aac', allow_multiple=False, dir=r'D:\Muxing\86')
    ger_audio = vof.AT(audio_search, 'German (CR)', 'ger', default=False)

    # This contains video, japanese audio and chapters (in this specific case)
    video = vof.muxing.MkvTrack(fr'D:\Muxing\86\video\86 - {setup.episode}.mkv')

    # the * is necessary to make python unpack the list of Attachments
    vof.Mux(setup, video, ger_audio, eng_sub, ger_sub, *fonts).run()