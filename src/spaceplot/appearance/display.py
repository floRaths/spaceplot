from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from .. import utils
from . import inline, styles, tools


def display(
    theme: str | Theme = 'default',
    **kwargs,
):
    theme = Theme(source_theme=theme, **kwargs) if isinstance(theme, str) else theme
    theme.apply()


@dataclass
class Theme:
    source_theme: str = None
    explicit_rcParams: dict = None
    retina: bool = False
    transparent: bool = False
    figsize: tuple[float, float] = None
    dpi: int = None
    palette: list[str] = None
    cmap: str = None
    text_color: str = None
    line_color: str = None
    ticks: list[str] = None
    minor_visible: bool = None
    spines: bool = None
    margins: float = None
    grid: bool = None
    grid_color: str = None
    grid_alpha: float = None
    grid_linestyle: str = None
    grid_linewidth: float = None
    tick_linewidth: tuple[float, float] = (None, None)
    tick_pad: tuple[float, float] = (None, None)
    spine_linewidth: float = None
    tick_size: tuple[float, float] = (None, None)
    font_family: str = None
    font_size: int = None
    fig_facecolor: str = None
    axes_facecolor: str = None
    labelsize: dict = field(default_factory=lambda: {'axes': None, 'figure': None, 'ticks': None})
    titlesize: dict = field(default_factory=lambda: {'axes': None, 'figure': None})
    titleweight: dict = field(default_factory=lambda: {'axes': None, 'figure': None})
    labelweight: dict = field(default_factory=lambda: {'axes': None, 'figure': None})
    axes_labelpad: float = None
    axes_titlepad: float = None
    inline_config: dict = field(default_factory=dict)

    def parse_source_theme(self):
        if isinstance(self.source_theme, str):
            base_theme = styles.themes.get(self.source_theme, {})
            return Theme(source_theme=base_theme)

        elif isinstance(self.source_theme, dict):
            return Theme(**self.source_theme)

        else:
            return self.source_theme

    def reset_defaults(self):
        plt.rcdefaults()

    @property
    def rcDict(self):
        from cycler import cycler

        tick_unset_val = None if self.ticks is None else False
        tick_t, tick_b, tick_l, tick_r = tools.set_position(positions=self.ticks, unset_value=tick_unset_val)

        spine_unset_val = None if self.spines is None else False
        spine_t, spine_b, spine_l, spine_r = tools.set_position(positions=self.spines, unset_value=spine_unset_val)

        prop_cycle = None if self.palette is None else cycler('color', self.palette)

        rc_dict = {
            key: value
            for key, value in {
                'figure.figsize': self.figsize,
                'figure.dpi': self.dpi,
                'axes.prop_cycle': prop_cycle,
                'image.cmap': self.cmap,
                'xtick.color': self.line_color,
                'ytick.color': self.line_color,
                'axes.grid': self.grid,
                'axes3d.grid': self.grid,
                'polaraxes.grid': self.grid,
                'grid.alpha': self.grid_alpha,
                'grid.color': self.grid_color,
                'grid.linestyle': self.grid_linestyle,
                'grid.linewidth': self.grid_linewidth,
                'text.color': self.text_color,
                'axes.labelcolor': self.text_color,
                'axes.titlecolor': self.text_color,
                'ytick.labelcolor': self.text_color,
                'xtick.labelcolor': self.text_color,
                'axes.edgecolor': self.line_color,
                'axes.facecolor': self.axes_facecolor,
                'figure.facecolor': self.fig_facecolor,
                'font.family': self.font_family,
                'font.size': self.font_size,
                'figure.titlesize': self.titlesize.get('figure', None),
                'axes.titlesize': self.titlesize.get('axes', None),
                'figure.titleweight': self.titleweight.get('figure', None),
                'axes.titleweight': self.titleweight.get('axes', None),
                'figure.labelsize': self.labelsize.get('figure', None),
                'axes.labelsize': self.labelsize.get('axes', None),
                'xtick.labelsize': self.labelsize.get('ticks', None),
                'ytick.labelsize': self.labelsize.get('ticks', None),
                'axes.labelpad': self.axes_labelpad,
                'axes.titlepad': self.axes_titlepad,
                'figure.labelweight': self.labelweight.get('figure', None),
                'axes.labelweight': self.labelweight.get('axes', None),
                'axes.linewidth': self.spine_linewidth,
                'axes.xmargin': self.margins,
                'axes.ymargin': self.margins,
                'axes.zmargin': self.margins,
                'axes.spines.top': spine_t,
                'axes.spines.bottom': spine_b,
                'axes.spines.left': spine_l,
                'axes.spines.right': spine_r,
                'xtick.top': tick_t,
                'xtick.bottom': tick_b,
                'ytick.left': tick_l,
                'ytick.right': tick_r,
                'ytick.major.left': tick_l,
                'ytick.major.right': tick_r,
                'xtick.major.top': tick_t,
                'xtick.major.bottom': tick_b,
                'ytick.minor.left': tick_l,
                'ytick.minor.right': tick_r,
                'xtick.minor.top': tick_t,
                'xtick.minor.bottom': tick_b,
                'ytick.labelleft': tick_l,
                'ytick.labelright': tick_r,
                'xtick.labeltop': tick_t,
                'xtick.labelbottom': tick_b,
                'xtick.major.pad': utils.maj_min_args(self.tick_pad)[0],
                'xtick.minor.pad': utils.maj_min_args(self.tick_pad)[1],
                'ytick.major.pad': utils.maj_min_args(self.tick_pad)[0],
                'ytick.minor.pad': utils.maj_min_args(self.tick_pad)[1],
                'xtick.major.size': utils.maj_min_args(self.tick_size)[0],
                'xtick.minor.size': utils.maj_min_args(self.tick_size)[1],
                'ytick.major.size': utils.maj_min_args(self.tick_size)[0],
                'ytick.minor.size': utils.maj_min_args(self.tick_size)[1],
                'xtick.major.width': utils.maj_min_args(self.tick_linewidth)[0],
                'xtick.minor.width': utils.maj_min_args(self.tick_linewidth)[1],
                'ytick.major.width': utils.maj_min_args(self.tick_linewidth)[0],
                'ytick.minor.width': utils.maj_min_args(self.tick_linewidth)[1],
                'ytick.minor.visible': self.minor_visible,
                'xtick.minor.visible': self.minor_visible,
            }.items()
            if value is not None
        }

        if self.explicit_rcParams is not None:
            rc_dict.update(self.explicit_rcParams)

        if self.source_theme is not None:
            source = self.parse_source_theme()
            rc_dict = {**source.rcDict, **rc_dict}

        return rc_dict

    def apply(self):
        if self.source_theme == 'default':
            print('Theme: resetting to matplotlib defaults')
            plt.rcdefaults()

        plt.rcParams.update(self.rcDict)

        inline.inline_config(retina=self.retina, transparent=self.transparent, **self.inline_config)
