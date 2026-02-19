import numpy as np
from matplotlib.colors import ListedColormap, to_rgba

allowed_params = {
    'tick_': [
        'size',
        'width',
        'color',
        'pad',
        'direction',
        'labelsize',
        'labelcolor',
        'labelrotation',
        'labelfontfamily',
        'zorder',
        # 'tickdir',
        # 'length',
    ],
    'label_': [
        'size',
        'color',
        'pad',
        'rotation',
        'fontfamily',
        'zorder',
    ],
    'title_': [
        'size',
        'color',
        'pad',
        'rotation',
        'loc',
        'fontfamily',
        'zorder',
    ],
    'abc_': ['style', 'size', 'loc', 'box', 'pad', 'label_pos'],
    'grid_': [
        'grid_color',
        'grid_alpha',
        'grid_linestyle',
        'grid_linewidth',
        'grid_dashes',
        'grid_dash_capstyle',
        'grid_dash_joinstyle',
        'grid_clip_box',
        'grid_clip_on',
        'grid_clip_path',
        'grid_data',
        'grid_drawstyle',
        'grid_figure',
        'grid_fillstyle',
        'grid_gapcolor',
        'grid_gid',
        'grid_in_layout',
        'grid_label',
        'grid_marker',
        'grid_markeredgecolor',
        'grid_markeredgewidth',
        'grid_markerfacecolor',
        'grid_markerfacecoloralt',
        'grid_markersize',
        'grid_markevery',
        'grid_mouseover',
        'grid_path_effects',
        'grid_picker',
        'grid_pickradius',
        'grid_rasterized',
        'grid_sketch_params',
        'grid_snap',
        'grid_solid_capstyle',
        'grid_solid_joinstyle',
        'grid_transform',
        'grid_url',
        'grid_visible',
        'grid_xdata',
        'grid_ydata',
        'grid_zorder',
        'grid_aa',
        'grid_c',
        'grid_ds',
        'grid_ls',
        'grid_lw',
        'grid_mec',
        'grid_mew',
        'grid_mfc',
        'grid_mfcalt',
        'grid_ms',
        'grid_agg_filter',
        'grid_animated',
        'grid_antialiased',
    ],
}


def get_hook_dict_v2(params, hook, remove_hook: bool = True, check: bool = True) -> dict:
    hook_params = {}
    if params == {}:
        return hook_params

    for key, value in params.items():
        if not key.startswith(hook):
            continue
        param = key.removeprefix(hook) if remove_hook else key

        # print(f"recognized '{hook}' parameter: {key}")
        if check:
            if param not in allowed_params.get(hook, []):
                raise ValueError(
                    f"Invalid {hook} parameter: '{param}'.\nSupported parameters are: {allowed_params.get(hook, [])}"
                )

        d = {param: value}
        hook_params.update(d)

    return hook_params


def get_hook_dict(params, hook, remove_hook=True) -> dict:
    hook_dict = {}

    if params == {}:
        return hook_dict

    for key, value in params.items():
        param = key.split('_')
        if param[0] == hook:
            # print(f"recognized '{hook}' parameter: {key}")
            # if param[1] not in allowed_params.get(hook, []):
            #     raise ValueError(
            #         f"Invalid {hook} parameter: '{param[1]}'.\nSupported parameters are: {allowed_params.get(hook, [])}"
            #     )
            d = {param[1]: value} if remove_hook else {key: value}
            hook_dict.update(d)

    return hook_dict


def maj_min_args(maj_min=None):
    if maj_min is None:
        return (None, None)
    if isinstance(maj_min, (list, tuple)) and len(maj_min) == 2:
        return tuple(maj_min)
    return (maj_min, maj_min)


def get_axis_ratio(ax):
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    y_span, x_span = ymax - ymin, xmax - xmin
    print(round((x_span / y_span), 3))


def confetti_cmap(n_labels, bg_color: str = None, bg_alpha: float = None, seed: int = None) -> ListedColormap:
    if seed is None:
        seed = 42

    rng = np.random.default_rng(seed)  # fixed seed for reproducibility
    colors = rng.random((n_labels, 3))  # RGB
    colors = np.hstack([colors, np.ones((n_labels, 1))])

    bg_color = colors[0] if bg_color is None else bg_color
    rgb = to_rgba(bg_color, alpha=bg_alpha)
    colors[0] = np.array(rgb)

    # Make a discrete colormap
    return ListedColormap(colors)
