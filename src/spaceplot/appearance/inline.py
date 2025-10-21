import matplotlib_inline.backend_inline as mpl_inline
from matplotlib import rcParams

rc_mapping = {
    'dpi': 'figure.dpi',
    'pad_inches': 'savefig.pad_inches',
    'facecolor': 'figure.facecolor',
    'bbox_inches': 'savefig.bbox',
}


def from_rc(key):
    return rcParams[rc_mapping[key]] if key in rc_mapping else None


def inline_config(
    retina: bool = None,
    facecolor: str = 'rc',
    dpi: int | str = 'rc',
    pad_inches: float | str = 'rc',
    bbox_inches: float | str = 'tight',
    transparent: bool = False,
    **kwargs,
):
    dpi = from_rc('dpi') if dpi == 'rc' else dpi
    pad_inches = from_rc('pad_inches') if pad_inches == 'rc' else pad_inches
    bbox_inches = from_rc('bbox_inches') if bbox_inches == 'rc' else bbox_inches
    facecolor = from_rc('facecolor') if facecolor == 'rc' else facecolor

    facecolor = 'none' if transparent else facecolor

    if retina:
        inl_format = 'retina'
        dpi = dpi * 2
    else:
        inl_format = 'png'

    mpl_inline.set_matplotlib_formats(
        inl_format,
        facecolor=facecolor,
        bbox_inches=bbox_inches,
        dpi=dpi,
        pad_inches=pad_inches,
        **kwargs,
    )
