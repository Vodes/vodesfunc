import logging
import os
import re
import shutil
import sys
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import groupby
from pathlib import Path

import ass
import ebmlite
import fontTools
from fontTools.misc import encodingTools
from fontTools.ttLib import ttFont

__all__: list[str] = [
    '_HOME',
    'disable_logging',
    'FONT_MIMETYPES',
    'Font', 'FontCollection',
    'get_dicts',
    'get_element', 'get_elements',
    'get_fonts',
    'getFontDirs',
    'INT_PATTERN',
    'is_mkv',
    'LINE_PATTERN',
    'LinuxFontDirs', 'OSxFontDirs', 'WinFontDirs',
    'parse_int', 'parse_line', 'parse_tags', 'parse_text',
    'State',
    'strip_fontname',
    'TAG_PATTERN',
    'TEXT_WHITESPACE_PATTERN',
    'validate_and_save_fonts', 'validate_fonts',
]

TAG_PATTERN = re.compile(r"\\\s*([^(\\]+)(?<!\s)\s*(?:\(\s*([^)]+)(?<!\s)\s*)?")
INT_PATTERN = re.compile(r"^[+-]?\d+")
LINE_PATTERN = re.compile(r"(?:\{(?P<tags>[^}]*)\}?)?(?P<text>[^{]*)")
TEXT_WHITESPACE_PATTERN = re.compile(r"\\[nNh]")

State = namedtuple("State", ["font", "italic", "weight", "drawing"])


def parse_int(s):
    if match := INT_PATTERN.match(s):
        return int(match.group(0))
    else:
        return 0


def strip_fontname(s):
    if s.startswith('@'):
        return s[1:]
    else:
        return s


def parse_tags(s, state, line_style, styles):
    for match in TAG_PATTERN.finditer(s):
        value, paren = match.groups()

        def get_tag(name, *exclude):
            if value.startswith(name) and not any(value.startswith(ex) for ex in exclude):
                args = []
                if paren is not None:
                    args.append(paren)
                if len(stripped := value[len(name):].lstrip()) > 0:
                    args.append(stripped)
                return args
            else:
                return None

        if (args := get_tag("fn")) is not None:
            if len(args) == 0:
                font = line_style.font
            else:
                font = strip_fontname(args[0])
            state = state._replace(font=font)
        elif (args := get_tag("b", "blur", "be", "bord")) is not None:
            weight = None if len(args) == 0 else parse_int(args[0])
            if weight is None:
                transformed = None
            elif weight == 0:
                transformed = 400
            elif weight in (1, -1):
                transformed = 700
            elif 100 <= weight <= 900:
                transformed = weight
            else:
                transformed = None

            state = state._replace(weight=transformed or line_style.weight)
        elif (args := get_tag("i", "iclip")) is not None:
            slant = None if len(args) == 0 else parse_int(args[0])
            state = state._replace(italic=slant == 1 if slant in (0, 1) else line_style.italic)
        elif (args := get_tag("p", "pos", "pbo")) is not None:
            scale = 0 if len(args) == 0 else parse_int(args[0])
            state = state._replace(drawing=scale != 0)
        elif (args := get_tag("r")) is not None:
            if len(args) == 0:
                style = line_style
            else:
                if (style := styles.get(args[0])) is None:
                    print(rf"Warning: \r argument {args[0]} does not exist; defaulting to line style")
                    style = line_style
            state = state._replace(font=style.font, italic=style.italic, weight=style.weight)
        elif (args := get_tag("t")) is not None:
            if len(args) > 0:
                state = parse_tags(args[0], state, line_style, styles)

    return state


def parse_text(text):
    return TEXT_WHITESPACE_PATTERN.sub(' ', text)


def parse_line(line, line_style, styles):
    state = line_style
    for tags, text in LINE_PATTERN.findall(line):
        if len(tags) > 0:
            state = parse_tags(tags, state, line_style, styles)
        if len(text) > 0:
            yield state, parse_text(text)


class Font:
    def __init__(self, fontfile, font_number=0, debug=False):
        self.fontfile = fontfile
        self.font = ttFont.TTFont(fontfile, fontNumber=font_number)
        self.num_fonts = getattr(self.font.reader, "numFonts", 1)
        self.postscript = self.font.has_key("CFF ")
        self.glyphs = self.font.getGlyphSet()

        os2 = self.font["OS/2"]
        self.weight = os2.usWeightClass
        self.italic = os2.fsSelection & 0b1 > 0
        self.slant = self.italic * 110
        self.width = 100

        self.names = [name for name in self.font["name"].names
                      if name.platformID == 3 and name.platEncID in (0, 1)]
        self.family_names = [name.string.decode('utf_16_be')
                             for name in self.names if name.nameID == 1]
        self.full_names = [name.string.decode('utf_16_be')
                           for name in self.names if name.nameID == 4]
        self.postscript_name = ''

        for name in self.font["name"].names:
            if name.nameID == 6 and (encoding := encodingTools.getEncoding(
                    name.platformID, name.platEncID, name.langID)) is not None:
                self.postscript_name = name.string.decode(encoding).strip()

                # these are the two recommended formats, prioritize them
                if (name.platformID, name.platEncID, name.langID) in \
                        [(1, 0, 0), (3, 1, 0x409)]:
                    break

        exact_names = [self.postscript_name] if (self.postscript and self.postscript_name) else self.full_names
        self.exact_names = [name for name in exact_names
                            if all(name.lower() != family.lower() for family in self.family_names)]

        mac_italic = self.font["head"].macStyle & 0b10 > 0
        if mac_italic != self.italic and debug:
            print(f"warning: different italic values in macStyle and fsSelection for font {self.postscript_name}")

        # fail early if glyph tables can't be accessed
        self.missing_glyphs('', debug)

    def missing_glyphs(self, text, debug=False):
        if (uniTable := self.font.getBestCmap()):
            return [c for c in text
                    if ord(c) not in uniTable]
        elif (symbolTable := self.font["cmap"].getcmap(3, 0)):
            macTable = self.font["cmap"].getcmap(1, 0)
            encoding = encodingTools.getEncoding(1, 0, macTable.language) if macTable else 'mac_roman'
            missing = []
            for c in text:
                try:
                    if (c.encode(encoding)[0] + 0xf000) not in symbolTable.cmap:
                        missing.append(c)
                except UnicodeEncodeError:
                    missing.append(c)
            return missing
        else:
            if debug:
                print(f"warning: could not read glyphs for font {self}")

    def __repr__(self):
        return f"{self.postscript_name}(italic={self.italic}, weight={self.weight})"


class FontCollection:
    def __init__(self, fontfiles, debug: bool = False):
        self.fonts = []
        for name, f in fontfiles:
            try:
                font = Font(f, debug=debug)
                self.fonts.append(font)

                if font.num_fonts > 1:
                    for i in range(1, font.num_fonts):
                        self.fonts.append(Font(f, font_number=i))
            except Exception as e:
                print(f"Error reading {name}: {e}")

        self.cache = {}
        self.by_full = {name.lower(): font
                        for font in self.fonts
                        for name in font.exact_names}
        self.by_postscriptName = {name.lower(): font
                                  for font in self.fonts
                                  for name in [font.postscript_name]}
        self.by_family = {name.lower(): [font for (_, font) in fonts]
                          for name, fonts in groupby(
                              sorted([(family, font)
                                      for font in self.fonts
                                      for family in font.family_names],
                                     key=lambda x: x[0]),
                              key=lambda x: x[0])}

    def similarity(self, state, font):
        return abs(state.weight - font.weight) + abs(state.italic * 100 - font.slant)

    def _match(self, state):
        # if not os.path.exists('Test.txt'):
        #   with(open('Text.txt', 'w', encoding = 'utf-8') as t):
        #      t.write(str(self.by_postscriptName))
        if (exact := self.by_full.get(state.font)):
            return exact, True
        elif (family := self.by_family.get(state.font)):
            # print('Test')
            return min(family, key=lambda font: self.similarity(state, font)), False
        else:
            # print('None')
            return None, False

    def match(self, state):
        s = state._replace(font=state.font.lower(), drawing=False)
        try:
            return self.cache[s]
        except KeyError:
            font = self._match(s)
            self.cache[s] = font
            return font


def validate_fonts(doc, fonts, ignore_drawings=False, warn_on_exact=False, debug=False):
    report = {
        "should_copy": defaultdict(set),
        "missing_font": defaultdict(set),
        "missing_glyphs": defaultdict(set),
        "missing_glyphs_lines": defaultdict(set),
        "faux_bold": defaultdict(set),
        "faux_italic": defaultdict(set),
        "mismatch_bold": defaultdict(set),
        "mismatch_italic": defaultdict(set)
    }

    styles = {style.name: State(strip_fontname(style.fontname), style.italic, 700 if style.bold else 400, False)
              for style in doc.styles}
    for i, line in enumerate(doc.events):
        if isinstance(line, ass.Comment):
            continue
        nline = i + 1

        try:
            style = styles[line.style]
        except KeyError:
            print(f"Warning: Unknown style {line.style} on line {nline}; assuming default style")
            style = State("Arial", False, 400, False)

        for state, text in parse_line(line.text, style, styles):
            font, exact_match = fonts.match(state)

            if ignore_drawings and state.drawing:
                continue

            if font is None:
                report["missing_font"][state.font].add(nline)
                continue
            else:
                report["should_copy"][font].add(nline)

            if state.weight >= font.weight + 150:
                report["faux_bold"][state.font, state.weight, font.weight].add(nline)

            if state.weight <= font.weight - 150 and (not exact_match or warn_on_exact):
                report["mismatch_bold"][state.font, state.weight, font.weight].add(nline)

            if state.italic and not font.italic:
                report["faux_italic"][state.font].add(nline)

            if not state.italic and font.italic and (not exact_match or warn_on_exact):
                report["mismatch_italic"][state.font].add(nline)

            if not state.drawing:
                missing = font.missing_glyphs(text, debug)
                report["missing_glyphs"][state.font].update(missing)
                if len(missing) > 0:
                    report["missing_glyphs_lines"][state.font].add(nline)

    issues = 0

    def format_lines(lines, limit=3):
        sorted_lines = sorted(lines)
        if len(sorted_lines) > limit:
            sorted_lines = sorted_lines[:limit]
            sorted_lines.append("[...]")
        return ' '.join(map(str, sorted_lines))

    if len(report["missing_font"].items()) > 0:
        print("-------------- Missing --------------\n")

    for font, lines in sorted(report["missing_font"].items(), key=lambda x: x[0]):
        issues += 1
        print(f"- {font}\non line(s): {format_lines(lines)}\n")

    if len(report["missing_font"].items()) > 0:
        print("-------------------------------------\n")

    for (font, reqweight, realweight), lines in sorted(report["faux_bold"].items(), key=lambda x: x[0]):
        issues += 1
        print(f"- Faux bold used for font {font} (requested weight {reqweight}, got {realweight}) "
              f"on line(s): {format_lines(lines)}")

    for font, lines in sorted(report["faux_italic"].items(), key=lambda x: x[0]):
        issues += 1
        print(f"- Faux italic used for font {font} on line(s): {format_lines(lines)}")

    for (font, reqweight, realweight), lines in sorted(report["mismatch_bold"].items(), key=lambda x: x[0]):
        issues += 1
        print(f"- Requested weight {reqweight} but got {realweight} for font {font} "
              f"on line(s): {format_lines(lines)}")

    for font, lines in sorted(report["mismatch_italic"].items(), key=lambda x: x[0]):
        issues += 1
        print(f"- Requested non-italic but got italic for font {font} on line(s): "
              + format_lines(lines))

    for font, lines in sorted(report["missing_glyphs_lines"].items(), key=lambda x: x[0]):
        issues += 1
        missing = ' '.join(f'{g}(U+{ord(g):04X})' for g in sorted(report['missing_glyphs'][font]))
        print(f"- Font {font} is missing glyphs {missing} "
              f"on line(s): {format_lines(lines)}")

    print(f"{issues} issue(s) found")
    return issues > 0, report


def get_element(parent, element, id=False):
    return next(get_elements(parent, element, id=id))


def get_elements(parent, *element, id=False):
    if id:
        return filter(lambda x: x.id in element, parent)
    else:
        return filter(lambda x: x.name in element, parent)


def get_dicts(parent, element, id=False):
    return ({x.name: x for x in elem} for elem in get_elements(parent, element, id=id))


# from mpv
FONT_MIMETYPES = {
    b"application/x-truetype-font",
    b"application/vnd.ms-opentype",
    b"application/x-font-ttf",
    b"application/x-font",
    b"application/font-sfnt",
    b"font/collection",
    b"font/otf",
    b"font/sfnt",
    b"font/ttf"
}


def get_fonts(mkv):
    fonts = []

    for segment in get_elements(mkv, "Segment"):
        for attachments in get_elements(segment, "Attachments"):
            for attachment in get_dicts(attachments, "AttachedFile"):
                if Path(attachment['FileName'].value.lower()).suffix not in ('.otf', '.ttf'):
                    print(f"Ignoring non-font attachment {attachment['FileName'].value}")
                    continue

                fonts.append((attachment["FileName"].value,
                              BytesIO(attachment["FileData"].value)))

    return fonts


def is_mkv(filename):
    with open(filename, 'rb') as f:
        return f.read(4) == b'\x1a\x45\xdf\xa3'


try:
    _HOME = Path.home()
except Exception:
    _HOME = Path(os.devnull)

def getFontDirs() -> list[str]:
    if sys.platform == 'win32':
        fontpaths = [
            # System
            os.path.join(os.environ['WINDIR'], "Fonts"),
            # User
            os.path.join(os.getenv("LOCALAPPDATA"), "Microsoft", "Windows", "Fonts")
        ]
    else:
        LinuxFontDirs = [
            # old x11 dirs
            "/usr/X11R6/lib/X11/fonts/TTF/",
            "/usr/X11/lib/X11/fonts",
            # New standard loc apparently?
            "/usr/share/fonts/",
            # Kinda user
            "/usr/local/share/fonts/",
            # User
            os.path.join(os.environ.get('XDG_DATA_HOME') or os.path.join(_HOME, ".local", "share"), "fonts"),
            os.path.join(_HOME, ".fonts"),
        ]

        if sys.platform == 'darwin':
            OSxFontDirs = [
                # System
                "/Library/Fonts/",
                "/Network/Library/Fonts/",
                "/System/Library/Fonts/",
                "/opt/local/share/fonts",
                # User
                os.path.join(_HOME, "Library", "Fonts"),
            ]
            fontpaths = [*LinuxFontDirs, *OSxFontDirs]
        else:
            fontpaths = LinuxFontDirs
    return fontpaths


def disable_logging():
    logging.getLogger(fontTools.__name__).setLevel(logging.CRITICAL)


def validate_and_save_fonts(ass_doc: tuple[str, ass.Document], out_dir: str | Path,
                            font_sources: list[str | Path] | tuple[str | Path] = None,
                            debug: bool = False):
    if not debug:
        disable_logging()
    out_dir = out_dir if isinstance(out_dir, Path) else Path(out_dir)
    fontlist = []
    fontdirs = [os.getcwd()]
    if font_sources:
        for source in font_sources:
            fontdirs.append(str(source.resolve()) if isinstance(source, Path) else source)
    fontdirs.extend(getFontDirs())
    print(f'Parsing all available fonts...')

    for additional_fonts in fontdirs:
        path = Path(additional_fonts)
        if not path.exists():
            continue
        if path.is_dir():
            fontlist.extend((p.name, str(p)) for p in path.rglob(
                '*') if p.is_file() and p.suffix.lower() in ('.otf', '.ttf'))
        elif is_mkv(additional_fonts):
            schema = ebmlite.loadSchema("matroska.xml")
            fontmkv = schema.load(additional_fonts)
            fontlist.extend(get_fonts(fontmkv))
        else:
            fontlist.append((path.name, additional_fonts))

    fonts = FontCollection(fontlist, debug)
    print(f'Checking {ass_doc[0]} ...')
    validate = validate_fonts(ass_doc[1], fonts, False, False, debug)
    print('')

    for font, _ in validate[1]["should_copy"].items():
        current = Path(font.fontfile)
        future_name = font.postscript_name.strip() + current.suffix
        dest = os.path.join(out_dir, future_name)
        if not os.path.exists(dest):
            shutil.copyfile(current, dest)
            print(f'Copied font "{future_name}"')
