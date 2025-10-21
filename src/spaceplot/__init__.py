import matplotlib.pyplot as plt

from . import aligner
from . import decorators as decs
from . import utils as ut
from .appearance import palettes as plts
from .appearance.display import Theme, display
from .appearance.layout import layout
from .montage_plot import montage_plot
from .plotting import plt_category, plt_continous, plt_image

__all__ = [
    'aligner',
    'decs',
    'ut',
    'plts',
    'Theme',
    'display',
    'layout',
    'montage_plot',
    'plt_category',
    'plt_continous',
    'plt_image',
    'plt',
]
