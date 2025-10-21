import matplotlib.pyplot as plt
import numpy as np

from .. import decorators as decs
from ..appearance import palettes as plts
from . import tools


def plt_continous(
    ax=None,
    shuffle=None,
    title=None,
    continuous_data=None,
    pt_size: float = 0.5,
    cmap=None,
    legend_loc=None,
    cbar_title=None,
    vmax='q_99',
):
    ax.set_title(title)
    # ax.set_ylabel('y-blobs')
    # ax.set_xlabel('x-blobs')

    plot_data = continuous_data

    if vmax is None:
        vmax = plot_data[:, 2].max()
    elif isinstance(vmax, str):
        q = float(f'0.{vmax.split("_")[1]}')
        vmax = np.quantile(plot_data[:, 2], q)

    if shuffle:
        np.random.shuffle(plot_data)
    else:
        plot_data = plot_data[plot_data[:, 2].argsort()]

    # Use a color map to map continuous values
    if cmap is None:
        cmap = plt.cm.inferno  # You can choose any colormap like 'plasma', 'inferno', 'coolwarm', etc.

    # norm = plt.Normalize(vmin=plot_data[:, 2].min(), vmax=plot_data[:, 2].max())

    # for i, section in enumerate(sections):
    x_sec = plot_data[:, 0].astype(float)
    y_sec = plot_data[:, 1].astype(float)
    continuous_values = plot_data[:, 2].astype(float)

    # Plot continuous data
    cax = ax.scatter(
        x_sec,
        y_sec,
        c=continuous_values,
        cmap=cmap,
        s=pt_size,
        linewidth=0,
        alpha=1,
        antialiased=False,
        zorder=3,
        vmax=vmax,
    )

    # if cbar_title
    decs.add_colorbar(ax=ax, cax=cax, loc=legend_loc, title=cbar_title)

    return ax, cax


def plt_category(
    ax=None,
    df=None,
    x=None,
    y=None,
    color=None,
    title: str = None,
    palette: list = None,
    order=None,
    size=0.25,
    alpha=1,
    shuffle: bool = True,
    shuffle_amt=10,
    legend_loc='right',
    legend_title: str = None,
    legend_cols=None,
):
    if ax is None:
        ax = plt.gca()

    if title is None:
        title = color

    if palette is None:
        palette = plts.mode20b

    ax.set_title(title)
    ax.set_ylabel(y)
    ax.set_xlabel(x)

    if df is not None:
        category_data = df.select(x, y, color).to_numpy()
    else:
        category_data = np.vstack([x, y, color]).T

    if order is not None:
        unique_cats = order
    else:
        unique_cats = df[color].unique().sort()

    color_dict = {cat: palette[i] for i, cat in enumerate(unique_cats)}

    if isinstance(shuffle, int):
        n_sections = shuffle
        shuffle = False
    elif shuffle is True:
        n_sections = shuffle_amt
    elif shuffle is False:
        n_sections = 1

    sections = tools.section_categories(plot_arr=category_data, n_sections=n_sections)

    for section in sections:
        x_sec = section[:, 0].astype(float)
        y_sec = section[:, 1].astype(float)
        labels_true_sec = section[:, 2]

        # Shuffle categories for each section if enabled
        if shuffle:
            color_dict = tools.shuffle_cats(color_dict)

        for category in color_dict:
            mask = labels_true_sec == category
            ax.scatter(
                x_sec[mask],
                y_sec[mask],
                c=color_dict[category],
                s=size,
                linewidth=0,
                alpha=alpha,
                antialiased=False,
                zorder=3,
            )

    if legend_loc is not None:
        if legend_title is None:
            legend_title = color

        decs.add_legend(
            ax,
            categories=unique_cats,
            palette=palette,
            title=legend_title,
            loc=legend_loc,
            ncol=legend_cols,
        )

    return ax


