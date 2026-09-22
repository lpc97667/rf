import os
import warnings

import matplotlib
from matplotlib import font_manager as fm

FONT_DIR = ""
_FACES = [
    ("Times_New_Roman.ttf", "normal", "normal"),
    ("Times_New_Roman_Bold.ttf", "bold", "normal"),
    ("Times_New_Roman_Italic.ttf", "normal", "italic"),
    ("Times_New_Roman_Bold_Italic.ttf", "bold", "italic"),
]
_FAMILY = "Times New Roman"
_FALLBACKS = [_FAMILY, "Nimbus Roman", "Nimbus Roman No9 L",
              "Liberation Serif", "Tinos", "FreeSerif", "DejaVu Serif"]


def _register_bundled_ttfs():
    registered = False
    for fname, _weight, _style in _FACES:
        path = os.path.join(FONT_DIR, fname)
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            registered = True
    return registered


def _first_installed(candidates):
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def use_times():
    if _register_bundled_ttfs():
        family = _FAMILY
    else:
        family = _first_installed(_FALLBACKS) \
            or matplotlib.rcParamsDefault["font.serif"][0]

    matplotlib.rcParams.update({
        "font.family": family,
        "font.serif": [family],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return family
