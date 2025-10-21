from . import palettes as plts

STD_PALETTE = plts.set25
STD_LW = 0.65

SP_BASETHEME = {
    'figsize': (10, 6),
    'dpi': 100,
    'palette': STD_PALETTE,
    'cmap': 'cmc.lipari',
    'font_family': 'Lato',
    'spines': True,
    'spine_linewidth': STD_LW,
    'grid': False,
    'grid_alpha': 1,
    'grid_linewidth': STD_LW,
    'ticks': ['bottom', 'left'],
    'minor_visible': False,
    'tick_size': (5, 2.5),
    'tick_linewidth': (STD_LW, STD_LW * 0.8),
    'tick_pad': 3.5,
    'margins': 0.1,
    'titlesize': {'figure': 22, 'axes': 18},
    'labelsize': {'figure': 14, 'axes': 14, 'ticks': 12},
    'labelweight': {'figure': 'bold', 'axes': 'bold'},
    'titleweight': {'figure': 'bold', 'axes': 'bold'},
    'axes_labelpad': 8,
    'axes_titlepad': 12,
    'explicit_rcParams': {
        'figure.constrained_layout.w_pad': 0.05,
        'figure.constrained_layout.h_pad': 0.05,
    },
}

BASE_DARK = {
    'text_color': '#ABB2BF',
    'line_color': '#ABB2BF',
    'axes_facecolor': '#21252B',
    'fig_facecolor': '#282C34',
    'grid_color': '0.25',
}

BASE_LIGHT = {
    'text_color': '0.1',
    'line_color': '0.1',
    'axes_facecolor': '0.95',
    'fig_facecolor': 'white',
    'grid_color': '0.75',
}

VOID = {'spines': False, 'ticks': False}

themes = {
    'dark': {**SP_BASETHEME, **BASE_DARK},
    'light': {**SP_BASETHEME, **BASE_LIGHT},
    'void_dark': {**SP_BASETHEME, **BASE_DARK, **VOID},
    'void_light': {**SP_BASETHEME, **BASE_LIGHT, **VOID},
}
