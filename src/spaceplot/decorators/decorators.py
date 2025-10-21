from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .. import utils
from . import tools

_legend_pos = Literal['right', 'bottom', 'inset_br', 'inset_bl', 'inset_tr', 'inset_tl']
_unit = Literal['micron', 'px']
_loc = Literal[
    'upper left',
    'upper center',
    'upper right',
    'center left',
    'center',
    'center right',
    'lower left',
    'lower center',
    'lower right',
]


def add_legend(
    ax,
    categories,
    title: str = None,
    palette=list,
    loc: _legend_pos = 'right',
    ncol: str = None,
    text_size=12,
    labelspacing=0.75,
    handletextpad=0,
    borderpad=0,
    columnspacing=1,
    handlelength=2,
    handleheight=1,
):
    def calculate_ncol(in_size):
        if in_size <= 2:
            out = 8
        elif in_size <= 4:
            out = 6
        elif in_size <= 6:
            out = 5
        elif in_size <= 8:
            out = 4
        elif in_size <= 10:
            out = 3
        else:
            out = 2

        return out

    color_dict = {cat: palette[i] for i, cat in enumerate(categories)}

    if loc in ['right', 'inset_tr', 'inset_tl', 'inset_br', 'inset_bl']:
        if ncol is None:
            ncol = 1 if len(categories) <= 14 else 2 if len(categories) <= 30 else 3
        anchor = {
            'right': (1, 1),
            'inset_tr': (1, 1),
            'inset_tl': (0, 1),
            'inset_br': (1, 0),
            'inset_bl': (0, 0),
        }[loc]
        loc = {
            'right': 'upper left',
            'inset_tr': 'upper right',
            'inset_tl': 'upper left',
            'inset_br': 'lower right',
            'inset_bl': 'lower left',
        }[loc]

    elif loc == 'bottom':
        if ncol is None:
            longest_cat = np.max([len(category) for category in color_dict])
            ncol = calculate_ncol(longest_cat)
        loc = 'upper center'
        anchor = (0.5, 0)

    for label in categories:
        ax.scatter([], [], c=color_dict[label], label=label)

        legend = ax.legend(
            title=title,
            fontsize=text_size,
            frameon=False,
            loc=loc,
            bbox_to_anchor=anchor,
            bbox_transform=ax.transAxes,
            ncol=ncol,
            labelspacing=labelspacing,  # Space between legend entries
            handletextpad=handletextpad,  # Space between handle and text
            borderpad=borderpad,  # Space between legend border and content
            columnspacing=columnspacing,  # Space between columns if ncol > 1
            handlelength=handlelength,  # Length of legend handles
            handleheight=handleheight,  # Height of legend handles
        )

    plt.setp(
        legend.get_title(),
        color=plt.rcParams['axes.labelcolor'],
        fontweight='bold',
        size=14,
    )
    legend._legend_box.align = 'left'


def add_colorbar(ax, cax, loc, title='', ax_dist=0.025, size=1.0):
    if loc == 'right':
        loc = 'upper left'
        width = 0.1
        height = f'{int(size * 100)}%'
        anchor = (1 + ax_dist, 0, 1, 1)
        orientation = 'vertical'

    if loc == 'bottom':
        loc = 'upper center'
        width = f'{int(size * 100)}%'
        height = 0.1
        anchor = (0, 0, 1, -ax_dist)
        orientation = 'horizontal'

    axins = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        bbox_to_anchor=anchor,
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = ax.figure.colorbar(cax, cax=axins, label='Test', orientation=orientation)
    cbar.set_label(title, fontsize=12, fontweight='bold')


def place_abc_label(
    ax,
    label='A',
    size=20,
    loc='tl',
    label_pos='c',
    style='alpha_box',
    pad=0.1,
    box=None,
    clip_on=True,
    **kwargs,
):
    label_params, box_params, box = tools.get_abc_style(style, size, box, ax, clip_on, kwargs)

    width = size / 72 * (1 + pad)
    height = size / 72 * (1 + pad)

    dpi = ax.figure.dpi
    w, h = int(width * dpi), int(height * dpi)  # DrawingArea uses pixels

    # Build the content inside a DrawingArea (in pixels)
    da = DrawingArea(w, h, clip=False)

    # Determine label position
    loc_align = 'c' if isinstance(label_pos, tuple) else None
    label_xy, label_hv = tools.parse_loc_input(loc=label_pos, align=loc_align, align_format='name')

    label_y_nudge = 0.9
    dx, dy = label_xy[0] * w, label_xy[1] * h * label_y_nudge
    txt = Text(dx, dy, label, ha=label_hv[0], va=label_hv[1], **label_params)
    da.add_artist(txt)

    # Place the DrawingArea inside an AnnotationBbox (in axes fraction)
    box_xy, box_hv = tools.parse_loc_input(loc=loc, align_format='frac')

    ab = AnnotationBbox(
        da,
        xy=box_xy,
        xycoords='axes fraction',
        box_alignment=box_hv,
        frameon=box,
        boxcoords=('offset pixels'),  # Important: use pixels for the box — this keeps size fixed on screen
        pad=0,
        bboxprops=box_params,
        # xybox=(0, -0.25),
    )

    ax.add_artist(ab)


def place_text(
    ax,
    label,
    loc='tl',
    weight='bold',
    color=None,
    size=20,
    offset=0,
    offset_type='rel',
    **kwargs,
):
    if color is None:
        color = plt.rcParams['text.color']

    loc_xy, loc_hv = tools.parse_loc_input(loc, align_format='name')
    loc_x, loc_y = loc_xy

    offset_x, offset_y = utils.maj_min_args(offset)

    if offset_type == 'abs':
        # Build offset transform
        fig = ax.figure
        offset = ScaledTranslation(offset_x, offset_y, fig.dpi_scale_trans)

        trans = ax.transAxes + offset
    elif offset_type == 'rel':
        trans = ax.transAxes
        loc_x += offset_x
        loc_y += offset_y

    zorder = kwargs.pop('zorder', 100)
    # Place text
    ax.text(
        loc_x,
        loc_y,
        label,
        horizontalalignment=loc_hv[0],
        verticalalignment=loc_hv[1],
        transform=trans,
        fontsize=size,
        weight=weight,
        color=color,
        zorder=zorder,
        **kwargs,
    )


def place_rect(
    ax,
    width=0.1,
    height=0.1,
    loc='tl',
    facecolor='none',
    edgecolor='0.5',
    lw: float = 0,
    offset: float = 0,
    offset_type='rel',  # 'rel' (axes fraction) or 'abs' (pixels)
    size_type='rel',  # 'rel' (axes fraction) or 'abs' (inches)
    apply: bool = True,
    **kwargs,
):
    # Parse anchor (you already have this)
    loc_x, loc_y, ha, va = tools.parse_loc_input(loc, ha=kwargs.get('ha'), va=kwargs.get('va'))
    offset_x, offset_y = utils.maj_min_args(offset)

    # Compute anchor transform + offsets for the *position* of the box
    # We’ll anchor in axes fraction by default.
    xy = (loc_x, loc_y)

    # Convert offsets depending on offset_type
    if offset_type == 'abs':
        # Offset in pixels (like your ScaledTranslation use).
        # With AnnotationBbox we’ll use xybox + boxcoords instead.
        boxcoords = ax.transAxes + ScaledTranslation(0, 0, ax.figure.dpi_scale_trans)
        xybox = (offset_x, offset_y)
    else:
        # Relative (axes fraction) offsets: just add to (loc_x, loc_y)
        xy = (loc_x + offset_x, loc_y + offset_y)
        xybox = (0, 0)
        boxcoords = None

    if size_type == 'abs':
        dpi = ax.figure.dpi
        w_px = width * dpi
        h_px = height * dpi

        da = DrawingArea(w_px, h_px, 0, 0)
        rect = Rectangle((0, 0), w_px, h_px, facecolor=facecolor, edgecolor=edgecolor, lw=lw)
        da.add_artist(rect)

        if offset_type == 'abs':
            # Use relative offset mode:
            # Prefer "offset pixels" (mpl >= 3.8). For wider compatibility, use points.
            # Option A (newer mpl):
            boxcoords = 'offset pixels'
            xybox = (offset_x, offset_y)
            # Option B (portable): comment A and uncomment B
            # boxcoords = "offset points"
            # xybox = (offset_x * 72.0 / dpi, offset_y * 72.0 / dpi)
        else:
            # relative (axes-fraction) offset -> just add to xy and no xybox
            xy = (loc_x + offset_x, loc_y + offset_y)
            xybox = None
            boxcoords = None
        # --- FIX ENDS HERE ---

        ab = AnnotationBbox(
            da,
            xy=(loc_x, loc_y),  # anchor in axes-fraction
            xycoords='axes fraction',
            xybox=xybox,
            boxcoords=boxcoords,  # now "offset pixels" or "offset points"
            box_alignment=(
                {'left': 0, 'center': 0.5, 'right': 1}.get(ha, 0),
                {'bottom': 0, 'center': 0.5, 'top': 1}.get(va, 1),
            ),
            frameon=False,
            zorder=kwargs.pop('zorder', 99),
            clip_on=kwargs.pop('clip_on', True),
        )

        if apply:
            ax.add_artist(ab)
        return ab, ab.get_transform()

    else:
        # --- your original relative-size path (axes fraction) ---
        if offset_type == 'abs':
            trans = ax.transAxes + ScaledTranslation(offset_x, offset_y, ax.figure.dpi_scale_trans)
        else:
            trans = ax.transAxes
        # Adjust anchor for alignment
        loc_x, loc_y = xy
        if ha == 'center':
            loc_x -= width / 2
        elif ha == 'right':
            loc_x -= width
        if va == 'center':
            loc_y -= height / 2
        elif va == 'top':
            loc_y -= height

        rect = Rectangle(
            (loc_x, loc_y),
            width,
            height,
            transform=trans,
            facecolor=facecolor,
            edgecolor=edgecolor,
            lw=lw,
            zorder=kwargs.pop('zorder', 99),
            **kwargs,
        )
        if apply:
            ax.add_patch(rect)

        return rect, trans


def place_rect_v1(
    ax,
    width=0.1,
    height=0.1,
    loc='tl',
    facecolor='none',
    edgecolor='0.5',
    lw: float = 0,
    offset: float = None,
    offset_type='rel',
    size_type='rel',
    apply: bool = True,
    **kwargs,
):
    # Parse anchor point
    loc_x, loc_y, ha, va = tools.parse_loc_input(loc, ha=kwargs.get('ha', None), va=kwargs.get('va', None))

    # Handle offsets
    offset = 0 if offset is None else offset
    offset_x, offset_y = utils.maj_min_args(offset)

    if offset_type == 'abs':
        fig = ax.figure
        offset = ScaledTranslation(offset_x, offset_y, fig.dpi_scale_trans)
        trans = ax.transAxes + offset
    elif offset_type == 'rel':
        trans = ax.transAxes
        loc_x += offset_x
        loc_y += offset_y

    # --- Handle absolute size ---
    if size_type == 'abs':
        dpi = ax.figure.dpi

        # Convert inches to display (pixels)
        w_disp = width * dpi
        h_disp = height * dpi

        # Transform (0,0) and (w,h) into Axes coords
        x0, y0 = trans.transform((loc_x, loc_y))

        inv = ax.transAxes.inverted()
        x1, y1 = inv.transform((x0 + w_disp, y0 + h_disp))
        loc_x, loc_y = inv.transform((x0, y0))
        width, height = x1 - loc_x, y1 - loc_y

    # Adjust (loc_x, loc_y) depending on alignment
    if ha == 'center':
        loc_x -= width / 2
    elif ha == 'right':
        loc_x -= width
    elif ha == 'left':
        loc_x = loc_x

    if va == 'center':
        loc_y -= height / 2
    elif va == 'top':
        loc_y -= height
    elif va == 'bottom':
        loc_y = loc_y

    zorder = kwargs.pop('zorder', 99)
    # Create rectangle
    rect = Rectangle(
        (loc_x, loc_y),
        width,
        height,
        transform=trans,
        facecolor=facecolor,
        edgecolor=edgecolor,
        lw=lw,
        zorder=zorder,
        **kwargs,
    )

    if apply:
        ax.add_patch(rect)

    return rect, trans


def color_badge(ax, color, height=None, width=0.05, pad=None):
    if pad is None:
        pad = plt.rcParams['axes.titlepad']

    if height is None:
        height = plt.rcParams['axes.titlesize']

    fig = ax.get_figure()
    # Convert points to axes-relative coordinates
    conversion = 72 * fig.get_size_inches()[1]  # Convert points to fraction of figure height

    y_pos = 1 + (pad / conversion)
    height = height / conversion

    rectangle = Rectangle(
        (0, y_pos),
        width,
        height,
        transform=ax.transAxes,
        color=color,
        alpha=1,
        clip_on=False,
    )

    ax.add_patch(rectangle)
