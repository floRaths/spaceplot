from pathlib import Path

from matplotlib import font_manager as fm


def register_fonts(fonts_path):
    if not isinstance(fonts_path, Path):
        fonts_path = Path(fonts_path)

    for ttf_font in fonts_path.rglob('*.ttf'):
        fm.fontManager.addfont(ttf_font)
