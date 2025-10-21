from collections.abc import Sequence

import numpy as np
from matplotlib import gridspec
from matplotlib.pyplot import figure


def montage_plot(
    n_plots: int = None,
    n_rows: int = None,
    n_cols: int = None,
    ref_panel: int = 0,
    design: list[list[int]] = None,
    title: str = None,
    figsize: list[float] = None,
    panel_size: tuple[float, float] | float = 3.5,
    layout: str = 'constrained',
    w_ratios: list[float] = None,
    h_ratios: list[float] = None,
    wspace: float = None,
    hspace: float = None,
    **kwargs,
):
    design, n_rows, n_cols = generate_design_from_inputs(n_plots=n_plots, n_rows=n_rows, n_cols=n_cols, design=design)

    # handle ratios and figure size
    w_ratios = np.ones(n_cols) if w_ratios is None else np.array(w_ratios)
    h_ratios = np.ones(n_rows) if h_ratios is None else np.array(h_ratios)

    figsize = calculate_figure_size(design, ref_panel, panel_size, w_ratios, h_ratios)

    fig = figure(figsize=figsize, layout=layout)
    fig.suptitle(title)

    grid = gridspec.GridSpec(
        nrows=n_rows,
        ncols=n_cols,
        figure=fig,
        width_ratios=w_ratios,
        height_ratios=h_ratios,
        wspace=wspace,
        hspace=hspace,
        **kwargs,
    )

    axes_list, labels, axes_by_label = get_plot_layout(design, grid)
    axs = Axs(axes_list)

    if len(axs) == 1:
        return axs[0]

    return axs


class Axs(Sequence):
    """
    A wrapper around a list of matplotlib Axes that:
      • behaves like a sequence (len, indexing, iteration)
      • broadcasts attribute access & method calls to all axes
        e.g., axs.scatter(...) calls scatter on each axis and returns a list of results
      • still lets you do axs[i].scatter(...) on an individual axis
    """

    def __init__(self, axes):
        # Store a plain list internally
        object.__setattr__(self, '_axes', list(axes))

    # --- list/sequence behavior ---
    def __len__(self):
        return len(self._axes)

    def __getitem__(self, idx):
        return self._axes[idx]

    def __iter__(self):
        return iter(self._axes)

    # --- attribute routing / broadcasting ---
    def __getattr__(self, name):
        """
        If attribute exists on Axes:
          - if callable: return a function that calls it on every axis and returns list of results
          - if not callable: return a list of that attribute from each axis,
            except if all values are equal -> return the common value (feels 'single-object'-y)
        Otherwise, raise as usual.
        """
        # Raise AttributeError if we have no axes
        if not self._axes:
            raise AttributeError(f"'Axs' has no axes to proxy '{name}'")

        # Probe the first axis
        try:
            first_attr = getattr(self._axes[0], name)
        except AttributeError as e:
            # Not an Axes attribute either
            raise AttributeError(f"'Axs' object has no attribute '{name}'") from e

        if callable(first_attr):

            def broadcast(*args, **kwargs):
                return [getattr(ax, name)(*args, **kwargs) for ax in self._axes]

            # Give a helpful repr
            broadcast.__name__ = name
            return broadcast
        else:
            vals = [getattr(ax, name) for ax in self._axes]
            # If all equal, return the single value, else return the list
            return vals[0] if all(v == vals[0] for v in vals) else vals

    def __setattr__(self, name, value):
        """
        Setting attributes: if it's our own private slot, set normally.
        Otherwise, try to set the attribute on each axis.
        If value is a sequence of same length as axes, assign per-axis.
        """
        if name == '_axes':
            object.__setattr__(self, name, value)
            return

        # If it's a per-axis sequence, assign elementwise
        try:
            is_seq = isinstance(value, (list, tuple))
            if is_seq and len(value) == len(self._axes):
                for ax, v in zip(self._axes, value):
                    setattr(ax, name, v)
                return
        except Exception:
            pass

        # Otherwise, set same value on all axes
        for ax in self._axes:
            setattr(ax, name, value)


def _build_design(n_plots, n_rows, n_cols):
    n_rows, n_cols = int(n_rows), int(n_cols)
    n_plots = n_rows * n_cols if n_plots is None else int(n_plots)
    if n_plots == 1:
        design = [[0]]

    else:
        # Create a default design based on the calculated layout
        design = []
        j = 0
        for i in range(n_rows):
            row = []
            for k in range(n_cols):
                row.append(j if j < n_plots else -1)
                j += 1
            design.append(row)

    return design


def generate_design_from_inputs(n_plots=None, n_rows=None, n_cols=None, design=None):
    if design is None and n_plots is None and n_rows is None and n_cols is None:
        raise ValueError('Please specify either a design or the number of plots via `n_plots` and/or `n_rows|n_cols`.')

    # If design is provided, derive rows and columns
    if design is not None:
        if not all(isinstance(row, list) for row in design):
            raise ValueError('Design should be a list of lists.')
        n_rows = len(design)
        n_cols = len(design[0])

    elif design is None and n_plots is not None:
        # Derive n_rows and n_cols from n_plots with a preference for balanced dimensions
        if n_cols is None and n_rows is None:
            n_rows = 1 if n_plots < 4 else 2 if n_plots <= 10 else int(np.floor(np.sqrt(n_plots)))
            n_cols = np.ceil(n_plots / n_rows)
        elif n_rows is not None and n_cols is None:
            n_cols = np.ceil(n_plots / n_rows)
        elif n_cols is not None and n_rows is None:
            n_rows = np.ceil(n_plots / n_cols)

        design = _build_design(n_plots=n_plots, n_rows=n_rows, n_cols=n_cols)

    elif n_plots is None and n_rows is not None and n_cols is not None:
        n_plots = n_cols * n_rows

        design = _build_design(n_plots=n_plots, n_rows=n_rows, n_cols=n_cols)

    return design, int(n_rows), int(n_cols)


def calculate_figure_size(design, ref_panel_idx, ref_panel_size, w_ratios, h_ratios):
    if isinstance(ref_panel_size, (int, float)):
        ref_panel_size = (ref_panel_size, ref_panel_size)
    if len(ref_panel_size) != 2:
        raise ValueError('ref_panel_size must be a float or a tuple of two floats.')

    design = np.array(design)
    rows, cols = np.where(design == ref_panel_idx)
    if len(rows) == 0:
        raise ValueError('Reference panel index not found in design.')

    ref_w_ratio = sum(w_ratios[cols.min() : cols.max() + 1])
    ref_h_ratio = sum(h_ratios[rows.min() : rows.max() + 1])

    scale_x = ref_panel_size[0] / ref_w_ratio
    scale_y = ref_panel_size[1] / ref_h_ratio

    fig_width = sum(w_ratios) * scale_x
    fig_height = sum(h_ratios) * scale_y
    return fig_width, fig_height


def get_plot_layout(design, grid, fig=None, return_order='first_seen'):
    """
    Build axes from a design matrix using a GridSpec.

    Parameters
    ----------
    design : list[list[int]]
        -1 means empty; repeated positive labels span cells.
    grid : matplotlib.gridspec.GridSpec
        GridSpec whose shape matches design.
    fig : matplotlib.figure.Figure or None
        If None, uses grid.figure.
    return_order : {"first_seen", "sorted"}
        Control the ordering of the returned list.

    Returns
    -------
    axes_list : list[Axes]
        Axes ordered by `return_order`.
    labels : np.ndarray
        Corresponding labels.
    axes_by_label : dict[int, Axes]
        Mapping label -> Axes.
    """
    arr = np.asarray(design)
    nrows, ncols = arr.shape
    if (nrows, ncols) != (grid.nrows, grid.ncols):
        raise ValueError(f'design shape {arr.shape} must match GridSpec shape {(grid.nrows, grid.ncols)}')

    if fig is None:
        fig = grid.figure

    # preserve first-seen order by scanning row-major
    seen = {}
    for r in range(nrows):
        for c in range(ncols):
            lab = arr[r, c]
            if lab == -1:
                continue
            if lab not in seen:
                seen[lab] = []
            seen[lab].append((r, c))

    labels = np.array(list(seen.keys()))
    if return_order == 'sorted':
        labels = np.array(sorted(labels, key=int))

    axes_by_label = {}
    axes_list = []
    for lab in labels:
        coords = np.array(seen[lab])
        rows = coords[:, 0]
        cols = coords[:, 1]

        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()

        # rectangularity check: area must equal count
        expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if expected_area != len(coords):
            raise ValueError(
                f'Label {lab} is not a solid rectangle (got {len(coords)} cells, bbox area is {expected_area}).'
            )

        sp = grid[r0 : r1 + 1, c0 : c1 + 1]
        ax = fig.add_subplot(sp)
        axes_by_label[lab] = ax
        axes_list.append(ax)

    return axes_list, labels, axes_by_label


# def get_plot_layout(design, grid):
#     arr = np.array(design)
#     ax_labels = np.unique(arr)
#     ax_labels = ax_labels[ax_labels != -1]

#     plot_layout = []
#     for plot in ax_labels:
#         if plot != -1:
#             ax_row = np.where(arr == plot)[0]
#             ax_col = np.where(arr == plot)[1]

#             plot_grids = grid[min(ax_row) : max(ax_row) + 1, min(ax_col) : max(ax_col) + 1]
#             plot_layout.append(grid.figure.add_subplot(plot_grids))

#     return plot_layout, ax_labels
