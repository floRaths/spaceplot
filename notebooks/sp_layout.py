# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: mpl-spaceplot
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import spaceplot as sp

sp.display('dark', retina=True, transparent=False)


# %%
n_points = 150
datax, datay = np.random.rand(n_points), np.random.rand(n_points)


# %%
axs = sp.montage_plot(1, panel_size=(6.5, 4.5))
axs.scatter(datax, datay, alpha=0.5)

sp.layout(
    axs,
    title='Scatter Plot',
    abc=True,
    # breaks=np.linspace(0, 1, 6),
    # tick_labelcolor='crimson',
    # x_ticklabels=['Low', 'Medium', 'High'],
    lims=(0.01, 2),
    scale='log',
    # margins=0.9,
    grid='both',
    # abc_box=False,
    aspect=1,
)



# %%
axis = 'x'
axis_obj = getattr(axs, f'{axis}axis')

# mapping for tick visibility in x/y axis
p1 = 'bottom' if axis == 'x' else 'left'
p2 = 'top' if axis == 'x' else 'right'

keys = [p1, f'label{p1}', p2, f'label{p2}']
curr_params = axis_obj.get_tick_params()
curr_params = {k: v for k, v in curr_params.items() if k in keys}

# %%
axs = sp.montage_plot(1, panel_size=(4.5, 3.5))
axs.scatter(datax, datay)

l2.tick_grid_visibility(axs, axis='x', ticks='1', minor=True, grid='major')
l2.tick_grid_visibility(axs, axis='y', ticks='1', minor=True, grid='major')


# %%
ref_panel_idx = 0
ref_panel_size = (3, 3)
w_ratios = (1, 1, 2)
h_ratios = (1, 2, 1)

design = [[0, 1, 2], 
          [0, 3, 2], 
          [-1, 3, 4]]

import numpy as np

def calculate_figure_size(design, ref_panel_idx, ref_panel_size, w_ratios, h_ratios):
    """
    Calculate the figure size needed for a given subplot design.
    
    Parameters
    ----------
    design : list of lists
        2D matrix of subplot indices. -1 means empty space.
    ref_panel_idx : int
        Index of the reference panel (the one whose size we want to match).
    ref_panel_size : tuple (width, height)
        Desired size (in inches) of the reference panel.
    w_ratios : tuple
        Relative width ratios of the columns.
    h_ratios : tuple
        Relative height ratios of the rows.

    Returns
    -------
    (fig_width, fig_height) : tuple
        Total figure size in inches.
    """
    
    # Convert to numpy for convenience
    design = np.array(design)
    
    # Find where the reference panel sits
    rows, cols = np.where(design == ref_panel_idx)
    if len(rows) == 0:
        raise ValueError("Reference panel index not found in design.")
    
    row_span = rows.max() - rows.min() + 1
    col_span = cols.max() - cols.min() + 1
    
    # Effective ratios spanned by the reference panel
    ref_w_ratio = sum(w_ratios[cols.min():cols.max()+1])
    ref_h_ratio = sum(h_ratios[rows.min():rows.max()+1])
    
    # Scale factors to convert ratio units -> inches
    scale_x = ref_panel_size[0] / ref_w_ratio
    scale_y = ref_panel_size[1] / ref_h_ratio
    
    # Total figure size
    fig_width = sum(w_ratios) * scale_x
    fig_height = sum(h_ratios) * scale_y
    
    return fig_width, fig_height

fig_size = calculate_figure_size(design, ref_panel_idx, ref_panel_size, w_ratios, h_ratios)
print(fig_size)


# %%
def test_data(n_points, x_limits=(0, 1), y_limits=(0, 1), seed=None):
    np.random.seed(seed) if seed is not None else None

    y = np.random.uniform(y_limits[0], y_limits[1], n_points)
    x = np.random.uniform(x_limits[0], x_limits[1], n_points)
    return x, y



# %%
axs = sp.montage_plot(4, figsize=(6, 4.5), layout='constrained')

sp.layout(axs, grid=True, make_square=False, abc=True, ticks=False, abc_size=18, abc_style='alpha_box')


# %%
axs = sp.montage_plot(2, panel_size=4, layout='constrained')
x, y = test_data(n_points=200, x_limits=(-10, 10), y_limits=(-10, 10), seed=42)
for ax in axs:
    ax.scatter(x, y, alpha=0.75, zorder=2)
# ax.imshow(np.ones((20, 10)), zorder=1, extent=[-5, 5, -10, 10], origin='lower')

sp.layout(axs, grid=False, make_square=False, x_label='X-axis', abc='ABC', abc_style='alpha_box', abc_loc='br')


# %%
x, y = test_data(100, seed=None)

design = [[1, 1, 2, 2, 3, 3], [-1, 4, 4, 5, 5, -1]]

axs = sp.montage_plot(design=design, panel_size=2, layout='constrained', h_ratios=[2, 2], w_ratios=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2])  # , hspace=0.5, wspace=1


# for ax in axs:
axs.scatter(x, y, alpha=0.75, zorder=2)

sp.layout(
    axs,
    title='Montage Plot',
    title_size=15,
    abc=True,
    y_label='y-axis',
    x_label='x-axis',
    grid='both',
    margins=0.2,
    # ticks=False,
    tick_labelsize=8,
    x_breaks=[0, 0.5, 1],
    y_breaks=[0, 0.5, 1],
    # make_square=True,
)


# %%
# design = [[0, 1, 2], [0, 3, 2], [-1, 3, 4]]

# sp.plt.rcParams['figure.constrained_layout.h_pad'] = 0.05
# sp.plt.rcParams['figure.constrained_layout.w_pad'] = 0.05

# axs = sp.montage_plot(
#     # design=design,
#     n_cols=3,
#     n_rows=2,
#     panel_size=(3, 3),
#     ref_panel=0,
#     w_ratios=(1, 1, 2),
#     h_ratios=(1, 1),
#     layout='constrained',
#     wspace=0,
#     hspace=0,
# )

# axs.scatter(datax, datay)
# sp.layout(axs, abc=True, ticks=False, margins=0.5)
# sp.plt.show()


# %%
# from matplotlib import font_manager

# # List all available font names registered with matplotlib
# fonts = sorted(set(f.name for f in font_manager.fontManager.ttflist))
# for font in fonts:
#     print(font)
