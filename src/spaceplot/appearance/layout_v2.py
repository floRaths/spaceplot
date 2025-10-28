from collections.abc import Iterable
from itertools import cycle
from typing import Literal

import spaceplot.appearance.layout as ly

from .. import utils

_tick_vis = Literal[True, False, '1', '2', 'both', 'all', None]
_grid_vis = Literal[True, False, 'both', 'major', 'minor', None]

major_grid_style = 'solid'
minor_grid_style = (0, (1, 2))

# TODO tick param hygiene... there are likely some grid, tick, minor params that don't belong
# I think currently, tick visibility will override settings from other setters... need to fix that
# implement allowed_params check like started in layout_v1
# find solution for tick_labels


def layout_v2(
    axs,
    *,
    title: str | list = None,
    abc: str | bool = None,
    margins: float = None,
    aspect: str | float | tuple = None,
    ticks: _tick_vis = None,
    spines: _tick_vis = None,
    grid: _grid_vis = None,
    minor: bool = None,
    label: str = None,
    breaks: list[float] = None,
    lims: list[float] = None,
    scale: str = None,
    tick_labels: list[str] = None,
    make_square: bool = None,
    **kwargs,
):
    local_args = {
        k: v
        for k, v in {
            'label': label,
            'margins': margins,
            'breaks': breaks,
            'lims': lims,
            'scale': scale,
        }.items()
        if v is not None
    }

    merged = {**local_args, **kwargs}

    axis_params, static_kwargs = merge_axis_kwargs(merged)
    x_label_params, x_tick_params, x_params = compile_axis_settings(axis_params, axis='x')
    y_label_params, y_tick_params, y_params = compile_axis_settings(axis_params, axis='y')

    title_settings = utils.get_hook_dict_v2(static_kwargs, 'title_', remove_hook=True)

    # ensure axs is a list
    if not isinstance(axs, Iterable):
        axs = [axs]
    if not isinstance(title, list):
        title = [title]

    ly.handle_abc_labels(axs, abc, **kwargs)

    pairs = list(zip(axs, cycle(title)))

    for ax, title in pairs:
        viz(ax, ticks=ticks, grid=grid, minor=minor, **kwargs)

        handle_tick_settings(ax, 'y', y_tick_params)
        handle_tick_settings(ax, 'x', x_tick_params)

        # handle other layout elements
        ly.handle_title(ax, title, title_settings)
        ly.handle_text_element(ax.get_xlabel, ax.set_xlabel, x_params['label'], x_label_params)
        ly.handle_text_element(ax.get_ylabel, ax.set_ylabel, y_params['label'], y_label_params)

        # ly.handle_tick_labels(ax, x_tick_labels, y_tick_labels)

        ly.handle_spines(ax, spines)
        ly.handle_breaks(ax, x_params['breaks'], y_params['breaks'])
        ly.handle_scales(ax, x_params['scale'], y_params['scale'])
        ly.handle_lims(ax, x_params['lims'], y_params['lims'])

        ly.handle_aspect(ax, aspect)

        # ly.handle_margins(ax, margins, make_square)
        ax.set_xmargin(x_params['margins']) if x_params['margins'] is not None else None
        ax.set_ymargin(y_params['margins']) if y_params['margins'] is not None else None


def handle_tick_settings(ax, axis, tick_settings):
    # Set default grid style, since rcParams don't offer minor grid style
    if 'grid_linestyle' not in tick_settings:
        tick_settings['grid_linestyle'] = [major_grid_style, minor_grid_style]

    # if len(tick_settings) == 0:
    #     return

    majmin_settings = {k: utils.maj_min_args(maj_min=v) for k, v in tick_settings.items()}

    for i, which in enumerate(['major', 'minor']):
        tick_settings_select = {k: v[i] for k, v in majmin_settings.items()}
        ax.tick_params(axis=axis, which=which, **tick_settings_select)


def viz(
    ax,
    *,
    ticks: _tick_vis = None,
    grid: _grid_vis = None,
    minor: _tick_vis = None,
    **kwargs,
):
    def value_check(param, value):
        if value is None:
            return
        if type(value) is bool:
            return
        if value not in [v for v in _tick_vis.__args__ if type(v) is str]:
            raise ValueError(f'Invalid {param} param: {value} ')

    value_check('ticks', ticks)
    value_check('minor', minor)
    for k, v in kwargs.items():
        if k.endswith('_ticks'):
            value_check(k, v)
        if k.endswith('_minor'):
            value_check(k, v)

    viz_args = {
        k: v
        for k, v in {
            'ticks': ticks,
            'grid': grid,
            'minor': minor,
        }.items()
        if v is not None
    }

    ax_below = False if 'grid_zorder' in kwargs else True
    ax.set_axisbelow(ax_below)

    # filter kwargs and merge with args to a single dict
    viz_kwargs = {k: v for k, v in kwargs.items() if any([k.endswith(h) for h in ['_ticks', '_grid', '_minor']])}
    merged = {**viz_args, **viz_kwargs}

    # break down merged into x_ and y_ specific params
    viz_params = parse_axis_params(params=merged)
    xviz = utils.get_hook_dict_v2(viz_params, 'x_', remove_hook=True, check=False)
    yviz = utils.get_hook_dict_v2(viz_params, 'y_', remove_hook=True, check=False)

    # apply settings
    set_tick_grid_visibility(ax, axis='x', **xviz)
    set_tick_grid_visibility(ax, axis='y', **yviz)


def set_tick_grid_visibility(ax, *, axis='x', ticks=None, minor=None, grid=None):
    axis_obj = getattr(ax, f'{axis}axis')

    # mapping for tick visibility in x/y axis
    p1 = 'bottom' if axis == 'x' else 'left'
    p2 = 'top' if axis == 'x' else 'right'

    # this determines the logic for tick visibility
    def apply_logic(value):
        logic_1 = True if value in ('1', 'both', 'all', True) else False
        logic_2 = True if value in ('2', 'both', 'all') else False

        viz = {
            p1: logic_1,
            f'label{p1}': logic_1,
            p2: logic_2,
            f'label{p2}': logic_2,
        }

        return viz

    if ticks is None:
        viz = {}
    else:
        viz = apply_logic(ticks)

    if minor is None and ticks is None:
        viz_min = {}
    elif minor is None and ticks:
        viz_min = {}
    elif minor is True:
        # axis_obj.minorticks_on()
        viz_min = viz.copy()
    else:
        # axis_obj.minorticks_on()
        viz_min = apply_logic(minor)

    axis_obj.minorticks_on()
    grid_maj, grid_min = parse_grid_visibility(grid=grid, minor=minor)

    axis_obj.set_tick_params(which='major', gridOn=grid_maj, **viz)
    axis_obj.set_tick_params(which='minor', gridOn=grid_min, **viz_min)


def parse_grid_visibility(
    grid: _grid_vis | None = None,
    minor: bool | None = None,
):
    if grid is None:
        return None, None

    if isinstance(grid, bool):
        if not grid:
            major_setter = minor_setter = False
        else:
            major_setter = grid
            minor_setter = grid if minor is None else minor
    elif isinstance(grid, str):
        major_setter = True if grid in ('both', 'major') else False
        minor_setter = True if grid in ('both', 'minor') else False

    return major_setter, minor_setter


def parse_axis_params(params: dict):
    """
    separates explicit axis params (i.e: x_tick_size) from global axis params (i.e: tick_size)
    and merges them into a single dict of xy related axis params
    """
    specific_axis_params = {}
    global_axis_params = {}
    for key, value in params.items():
        if any([key.startswith(h) for h in ['x_', 'y_']]):
            specific_axis_params[key] = value
        else:
            global_axis_params[key] = value

    # 3: expand global axis params to x_ and y_ versions
    x_global = {'x_' + key: value for key, value in global_axis_params.items()}
    y_global = {'y_' + key: value for key, value in global_axis_params.items()}
    axis_global = {**x_global, **y_global}

    # 4: check for duplicates and merge
    duplicate_keys = set(specific_axis_params.keys()) & set(axis_global.keys())
    if duplicate_keys:
        param_names = [p.removeprefix('x_') if p.startswith('x_') else p.removeprefix('y_') for p in duplicate_keys]
        raise ValueError(f'Duplicate axis param for: {duplicate_keys} and {param_names}')

    axis_params = {**specific_axis_params, **axis_global}

    return axis_params


def merge_axis_kwargs(kwargs):
    axis_hooks = ['x_', 'y_', 'tick_', 'margins', 'grid', 'label', 'lims', 'breaks', 'scale']
    # 1: separate axis and static params
    axis_params = {}
    static_params = {}
    for key, value in kwargs.items():
        if any([key.startswith(h) for h in axis_hooks]):
            axis_params[key] = value
        else:
            static_params[key] = value

    axis_params = parse_axis_params(axis_params)

    return axis_params, static_params


def compile_axis_settings(axis_params, axis: str) -> dict:
    settings = utils.get_hook_dict_v2(axis_params, f'{axis}_', remove_hook=True, check=False)
    tick_settings = utils.get_hook_dict_v2(settings, 'tick_', remove_hook=True)
    grid_settings = utils.get_hook_dict_v2(settings, 'grid_', remove_hook=False)
    label_settings = utils.get_hook_dict_v2(settings, 'label_', remove_hook=True)
    tick_settings.update(grid_settings)
    other_params = {k: v for k, v in settings.items() if not (k.startswith('tick_') or k.startswith('grid_'))}

    complete_other = {}
    for i in ['grid', 'margins', 'label', 'breaks', 'lims', 'scale']:
        complete_other[i] = other_params.get(i, None)

    return label_settings, tick_settings, complete_other
