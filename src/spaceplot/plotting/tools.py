import numpy as np


# region _internal functions
def shuffle_cats(color_dict):
    # Get a list of the dictionary's keys and shuffle them
    keys = list(color_dict.keys())
    np.random.shuffle(keys)

    # Recreate the dictionary with shuffled keys
    shuffled_dict = {key: color_dict[key] for key in keys}

    return shuffled_dict


def section_categories(plot_arr, n_sections: int = 10):
    """Split the data array into sections after shuffling."""
    np.random.shuffle(plot_arr)
    # shuffled_data = _shuffle_array(plot_arr)  # Shuffle before splitting
    section_len = int(len(plot_arr) / n_sections)

    sections = [plot_arr[section_len * i : section_len * (i + 1)] for i in range(n_sections)]

    return sections


def downsample_mask(mask, mask_length=None, ax=None, display_size=None, downsample='auto'):
    def get_ax_size(ax):
        fig = ax.get_figure()

        bbox = ax.get_position()
        fig_width, fig_height = fig.get_size_inches()
        width_in = bbox.width * fig_width
        height_in = bbox.height * fig_height

        ax_dim = (width_in, height_in)
        return ax_dim

    if not downsample or downsample is None:
        return mask

    elif isinstance(downsample, int):
        downsample_fct = int(downsample)

    elif downsample.startswith('auto'):
        if downsample == 'auto':
            target_ppi = 200
        elif downsample.startswith('auto_'):
            target_ppi = int(downsample.split('_')[-1])

        if ax is None and display_size is None:
            raise ValueError("Either ax or display_size must be provided when downsample is 'auto'")
        elif ax is None and display_size:
            panel_size = display_size
        elif ax and display_size is None:
            ax_dim = get_ax_size(ax)
            panel_size = np.array(ax_dim).max()

        if mask_length is None:
            mask_length = np.array(mask.shape).max()
        elif mask_length is not None:
            mask_length = mask_length

        downsample_fct = int((mask_length / panel_size) / target_ppi)
        # print(f'Downsample factor: {downsample_fct}')

    else:
        raise ValueError("downsample must be None, int or 'auto'")

    if downsample_fct > 0:
        sampled_mask = mask[::downsample_fct, ::downsample_fct].copy()
    else:
        sampled_mask = mask.copy()

    return sampled_mask
