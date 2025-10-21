from collections.abc import Iterable
from itertools import cycle

import numpy as np

from .. import decorators as decs
from .. import utils
from . import tools

major_grid_style = 'solid'
minor_grid_style = (0, (1, 2))


def layout(
    axs,
    *,
    axis: str = 'both',
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    abc: str | bool = None,
    make_square: bool = None,
    margins: float = None,
    aspect: str | float | tuple = None,
    ticks: tools._tick_vis = None,
    grid: tools._grid_vis = None,
    minor: bool = None,
    spines: tools._tick_vis = None,
    x_breaks: list[float] = None,
    y_breaks: list[float] = None,
    x_lims: list[float] = None,
    y_lims: list[float] = None,
    x_scale: str = None,
    y_scale: str = None,
    x_tick_labels: list[str] = None,
    y_tick_labels: list[str] = None,
    **kwargs,
):
    # decompose kwargs into title, label, tick, and grid settings
    title_settings = utils.get_hook_dict(kwargs, 'title', remove_hook=True)
    label_settings = utils.get_hook_dict(kwargs, 'label', remove_hook=True)

    tick_settings = utils.get_hook_dict(kwargs, 'tick', remove_hook=True)
    grid_settings = utils.get_hook_dict(kwargs, 'grid', remove_hook=False)
    tick_settings.update(grid_settings)

    # ensure axs is a list
    if not isinstance(axs, Iterable):
        axs = [axs]
    if not isinstance(title, Iterable):
        title = [title]

    handle_abc_labels(axs, abc, **kwargs)

    pairs = list(zip(axs, cycle(title)))

    for ax, title in pairs:
        # handle ticks, grid, and spine visibility
        # NOTE when axis != 'both', ticks=True does weird things... seems to be a matplotlib issue
        handle_tick_settings(ax, axis, ticks, minor, grid, tick_settings)

        # handle other layout elements
        handle_title(ax, title, title_settings)
        handle_labels(ax, axis, x_label, y_label, label_settings)
        handle_tick_labels(ax, x_tick_labels, y_tick_labels)

        handle_spines(ax, spines)
        handle_breaks(ax, x_breaks, y_breaks)
        handle_scales(ax, x_scale, y_scale)
        handle_lims(ax, x_lims, y_lims)

        handle_aspect(ax, aspect)

        # TODO when x_lim/y_lim are set, margins don't work as expected
        handle_margins(ax, margins, make_square)

        if make_square is True:
            tools.axis_ratio(ax, yx_ratio=1, margins=margins, how='lims')


def handle_abc_labels(axs, abc=None, **kwargs):
    if abc:
        ax_labels = np.arange(1, len(axs) + 1)
        if abc == 'ABC':
            ax_labels = [chr(64 + num) for num in ax_labels]
        elif abc == 'abc':
            ax_labels = [chr(96 + num) for num in ax_labels]

        abc_params = utils.get_hook_dict(kwargs, 'abc')
        abc_params['loc'] = abc_params['loc'] if 'loc' in abc_params else 'tl'
        abc_params['size'] = abc_params['size'] if 'size' in abc_params else 18
        for i, ax in enumerate(axs):
            decs.place_abc_label(
                ax,
                label=ax_labels[i],
                pad=0.2,
                **abc_params,
            )


def handle_tick_grid_vis(ax, axis, ticks, minor, grid, tick_settings):
    tools.set_minor_ticks_by_axis(ax, axis=axis)
    minor = False if minor is None else minor

    tools.set_tick_visibility(ax, axis=axis, ticks=ticks, minor=minor)

    # set axis below if no grid zorder is specified to make sure grid lines are below other plot elements
    ax_below = False if 'grid_zorder' in tick_settings else True
    ax.set_axisbelow(ax_below)

    maj_grid, min_grid = tools.set_grid_visibility(ax, axis=axis, grid=grid, minor=minor, apply=False)
    tick_settings['gridOn'] = [maj_grid, min_grid]

    # Set default grid style, since rcParams don't offer minor grid style
    if 'grid_linestyle' not in tick_settings:
        tick_settings['grid_linestyle'] = [major_grid_style, minor_grid_style]


def handle_text_element(getter, setter, text: str = None, params: dict = {}):
    """Generic helper to get current text if needed and set it with params."""
    if text is None and len(params) == 0:
        return

    if text is None and len(params) > 0:
        text = getter()

    setter(text, **params)


def handle_tick_settings(ax, axis, ticks, minor, grid, tick_settings):
    if ticks is None and minor is None and grid is None and len(tick_settings) == 0:
        return

    # first all the visibility settings
    handle_tick_grid_vis(ax, axis, ticks, minor, grid, tick_settings)

    # tick (and grid) settings are applied separately for major and minor ticks
    majmin_settings = {k: utils.maj_min_args(maj_min=v) for k, v in tick_settings.items()}

    for i, which in enumerate(['major', 'minor']):
        tick_settings_select = {k: v[i] for k, v in majmin_settings.items()}
        ax.tick_params(axis=axis, which=which, **tick_settings_select)


def handle_spines(ax, spines):
    if spines is not None:
        tools.set_spine_visibility(ax, spines)


def handle_aspect(ax, aspect):
    if aspect is not None:
        aspect = [aspect] if not isinstance(aspect, (list, tuple)) else aspect
        adjustable = None if len(aspect) < 2 else aspect[1]
        aspect_params = {'aspect': aspect[0], 'adjustable': adjustable}
        ax.set_aspect(**aspect_params)


def handle_breaks(ax, x_breaks, y_breaks):
    if x_breaks is not None:
        ax.set_xticks(x_breaks)

    if y_breaks is not None:
        ax.set_yticks(y_breaks)


def handle_scales(ax, x_scale, y_scale):
    if y_scale is not None:
        scale_params = tools.parse_scale(scale=y_scale)
        ax.set_yscale(**scale_params)

    if x_scale is not None:
        scale_params = tools.parse_scale(scale=x_scale)
        ax.set_xscale(**scale_params)


def handle_lims(ax, x_lims, y_lims):
    if y_lims is not None:
        ax.set_ylim(y_lims)

    if x_lims is not None:
        ax.set_xlim(x_lims)


def handle_title(ax, title, title_params):
    if title is None and len(title_params) == 0:
        return

    if title is None:
        title = ax.get_title()
        title = None if len(title) == 0 else title

    if title is not None or len(title_params) > 0:
        handle_text_element(ax.get_title, ax.set_title, title, title_params)


def handle_labels(ax, axis, x_label, y_label, label_params):
    if x_label is None and y_label is None and len(label_params) == 0:
        return

    loc_lookup = {
        'x': {'start': 'left', 'center': 'center', 'end': 'right'},
        'y': {'start': 'bottom', 'center': 'center', 'end': 'top'},
    }

    def normalize_params(axis_key: str, params: dict | None) -> dict:
        params = params or {}
        loc = params.get('loc')
        if loc is not None:
            try:
                params['loc'] = loc_lookup[axis_key][loc]
            except KeyError:
                raise ValueError(
                    f"Invalid {axis_key} label loc '{loc}'. Valid options are {list(loc_lookup[axis_key])}."
                )
        return params

    x_label_params = normalize_params('x', label_params.copy()) if axis in ['x', 'both'] else {}
    y_label_params = normalize_params('y', label_params.copy()) if axis in ['y', 'both'] else {}

    handle_text_element(ax.get_xlabel, ax.set_xlabel, x_label, x_label_params)
    handle_text_element(ax.get_ylabel, ax.set_ylabel, y_label, y_label_params)


def handle_tick_labels(ax, x_tick_labels, y_tick_labels):
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)

    if y_tick_labels is not None:
        ax.set_yticklabels(y_tick_labels)


def handle_margins(ax, margins, make_square):
    if margins is not None and not make_square:
        xmargin, ymargin = utils.maj_min_args(margins)

        ax.set_xmargin(xmargin)
        ax.set_ymargin(ymargin)
