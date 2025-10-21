from typing import Literal

from matplotlib.ticker import AutoMinorLocator, NullLocator

_tick_vis = Literal[True, False, 'top', 'bottom', 'left', 'right', None]
_grid_vis = Literal[True, False, 'both', 'major', 'minor', None]


def set_position(
    positions: _tick_vis | list[_tick_vis] | tuple[_tick_vis, ...] | None = None,
    unset_value: bool | None = False,
):
    # Special cases
    if positions is True:
        return (True, True, True, True)
    if positions is False:
        return (False, False, False, False)
    if positions is None:
        # Respect unset_value (often None when caller wants "leave as-is")
        return (unset_value, unset_value, unset_value, unset_value)

    # Normalize to list
    if not isinstance(positions, (list, tuple)):
        positions = [positions]

    # Validate using a concrete set (Literal is not iterable for membership checks)
    valid_positions = {True, False, 'top', 'bottom', 'left', 'right', None}
    invalid_positions = [pos for pos in positions if pos not in valid_positions]
    if invalid_positions:
        raise ValueError(f'Invalid positions: {invalid_positions}. Valid options are: {valid_positions}')

    # Initialize all to unset_value, then enable specified sides
    pos_t = pos_b = pos_l = pos_r = unset_value
    if 'top' in positions:
        pos_t = True
    if 'bottom' in positions:
        pos_b = True
    if 'left' in positions:
        pos_l = True
    if 'right' in positions:
        pos_r = True

    return pos_t, pos_b, pos_l, pos_r


def set_grid_visibility(
    ax,
    *,
    axis: str = 'both',
    grid: _grid_vis | None = None,
    minor: bool | None = None,
    apply=False,
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

    if apply:
        ax.tick_params(axis=axis, which='major', gridOn=major_setter)
        ax.tick_params(axis=axis, which='minor', gridOn=minor_setter)

    return major_setter, minor_setter


def set_minor_ticks_by_axis(ax, axis='both'):
    locators = {
        'x': (AutoMinorLocator(), NullLocator()),
        'y': (NullLocator(), AutoMinorLocator()),
        'both': (AutoMinorLocator(), AutoMinorLocator()),
    }
    xloc, yloc = locators.get(axis, locators['both'])
    ax.xaxis.set_minor_locator(xloc)
    ax.yaxis.set_minor_locator(yloc)


def set_tick_visibility(ax, *, axis: str = 'both', ticks=None, minor=None):
    if ticks is None and minor is None:
        return

    locs_major = locs_settings(tick_locs=ticks)
    if minor is False:
        locs_minor = locs_settings(tick_locs=False)
    else:
        locs_minor = locs_major

    settings = zip(['major', 'minor'], [locs_major, locs_minor])

    for which, locs in settings:
        if which == 'minor' and minor is None:
            continue

        if locs is not None:
            ax.tick_params(axis=axis, which=which, **locs)


def locs_settings(tick_locs: list | None = None):
    if tick_locs is None:
        return None

    tick_t, tick_b, tick_l, tick_r = set_position(positions=tick_locs, unset_value=False)
    ticklabel_t, ticklabel_b, ticklabel_l, ticklabel_r = set_position(positions=tick_locs, unset_value=False)

    locs = {
        'left': tick_l,
        'bottom': tick_b,
        'right': tick_r,
        'top': tick_t,
        'labelleft': ticklabel_l,
        'labelbottom': ticklabel_b,
        'labelright': ticklabel_r,
        'labeltop': ticklabel_t,
    }

    return locs


def set_spine_visibility(ax, spines):
    if spines is None:
        return

    pos_t, pos_b, pos_l, pos_r = set_position(spines, unset_value=False)

    ax.spines['top'].set_visible(pos_t)
    ax.spines['left'].set_visible(pos_l)
    ax.spines['bottom'].set_visible(pos_b)
    ax.spines['right'].set_visible(pos_r)


def parse_scale(scale: str | None = None) -> dict:
    if scale is None or scale == 'linear':
        return {'value': 'linear'}

    if scale == 'log' or scale.startswith('log_'):
        if scale == 'log':
            base = 10
        else:
            base = float(scale.split('_')[-1])
            scale = 'log'

        return {'value': scale, 'base': base}


def axis_ratio(ax, yx_ratio=None, margins=None, how='margins'):
    if how == 'margins':
        force_aspect_via_margins(ax, yx_ratio=yx_ratio, margins=margins)
    elif how == 'lims':
        force_aspect_via_lims(ax, yx_ratio=yx_ratio, margins=margins, use_view=False)
    else:
        raise ValueError(f'Invalid how: {how}. Must be "margins" or "lims".')

    ax.set_aspect('equal', adjustable='box')


def force_aspect_via_margins(ax, yx_ratio=None, margins=None):
    if margins is None:
        xmargs, ymargs = ax.get_xmargin(), ax.get_ymargin()
        margins = min(xmargs, ymargs)

    xmin, xmax = ax.dataLim.x0, ax.dataLim.x1
    ymin, ymax = ax.dataLim.y0, ax.dataLim.y1
    x_span, y_span = xmax - xmin, ymax - ymin

    data_yx_ratio = ax.get_data_ratio() if yx_ratio is None else yx_ratio

    if x_span > y_span:
        xmargin = margins
        ymargin = ((1 + 2 * xmargin) / data_yx_ratio - 1) / 2
    else:
        ymargin = margins
        xmargin = ((1 + 2 * ymargin) * data_yx_ratio - 1) / 2

    ax.set_xmargin(xmargin)
    ax.set_ymargin(ymargin)


def force_aspect_via_lims(ax, yx_ratio=None, margins=None, use_view=False):
    """
    Make the *plotting area* square by expanding limits.
    The longer data side gets `margins` per-side padding (fraction of its data span).
    The shorter side is padded so that the final x/y spans match.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    yx_ratio : float or None
        Optional y/x ratio to *decide* which side is longer.
        If provided, y-span will be scaled by (1 / yx_ratio) before comparing to x-span.
    margins : float
        Per-side fraction on the *longer* side (e.g., 0.1 = 10% each side).
    use_view : bool
        If True, base on current view limits; else, base on data limits (default).
    """
    # Base spans from data or current view
    if use_view:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
    else:
        xmin, xmax = ax.dataLim.x0, ax.dataLim.x1
        ymin, ymax = ax.dataLim.y0, ax.dataLim.y1

    sx = xmax - xmin
    sy = ymax - ymin

    # Adjust for desired y/x ratio when comparing spans
    if yx_ratio is not None:
        scaled_sy = sy / yx_ratio  # "normalize" y to match x-scale
    else:
        scaled_sy = sy

    margins = min(ax.get_xmargin(), ax.get_ymargin()) if margins is None else margins

    # Decide which side is longer
    if sx >= scaled_sy:
        # x is "longer"
        target_span = sx * (1 + 2 * margins)
        pad_y = max(0.0, (target_span * (yx_ratio or 1) - sy) / 2.0)
        ax.set_xlim(xmin - sx * margins, xmax + sx * margins)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
    else:
        # y is "longer"
        target_span = sy * (1 + 2 * margins)
        pad_x = max(0.0, (target_span / (yx_ratio or 1) - sx) / 2.0)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - sy * margins, ymax + sy * margins)
