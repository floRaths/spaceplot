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

