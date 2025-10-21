import numpy as np
from matplotlib.colors import ListedColormap, to_rgba


def get_hook_dict(params, hook, remove_hook=True) -> dict:
    hook_dict = {}

    if params == {}:
        return hook_dict

    for key, value in params.items():
        param = key.split('_')
        if param[0] == hook:
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
