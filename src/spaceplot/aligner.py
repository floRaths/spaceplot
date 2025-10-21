from typing import Literal

import numpy as np

aligns = Literal[
    'top_left',
    'top_right',
    'bottom_left',
    'bottom_right',
    'center_left',
    'center_right',
    'center_top',
    'center_bottom',
]

align_map = {
    'c': 0.5,
    'l': 0,
    'r': 1,
    'b': 0,
    't': 1,
}

align_full_names = {
    'c': 'center',
    'l': 'left',
    'r': 'right',
    'b': 'bottom',
    't': 'top',
}


def translate_align(how: aligns, format: str = 'frac', xfact: float = None, yfact: float = None) -> tuple[float, float]:
    if how is None and (xfact is None or yfact is None):
        raise ValueError("Either 'how' must be provided or both 'xfact' and 'yfact' must be specified.")

    if how is not None:
        x_a, y_a = parse_alignment(how)
    else:
        x_a, y_a = 'c', 'c'

    x = xfact if xfact else align_map[x_a]
    y = yfact if yfact else align_map[y_a]

    if format == 'name':
        x = align_full_names[x_a] if xfact is None else xfact
        y = align_full_names[y_a] if yfact is None else yfact

    return x, y


def parse_alignment(inpt: aligns) -> tuple[aligns, aligns]:
    """reutrns a tuple of (x, y) alignment based on the input string."""

    def get_inpt_type(inpt):
        if inpt in ['l', 'r']:
            inpt_type = 'horizontal'
        elif inpt in ['t', 'b']:
            inpt_type = 'vertical'
        else:
            inpt_type = 'centered'
        return inpt_type

    if len(inpt) > 2:
        if '_' in inpt:
            parts = inpt.split('_')

            if len(parts) == 2:
                inpt = parts[0][0] + parts[1][0]
            else:
                raise ValueError('Input must be a two-part string separated by an underscore.')
        else:
            inpt = inpt[0]

    x = y = None
    if len(inpt) == 1:
        in_type = get_inpt_type(inpt)
        if in_type == 'horizontal':
            x, y = inpt, 'c'
        elif in_type == 'vertical':
            x, y = 'c', inpt
        else:
            x = y = 'c'

    elif len(inpt) == 2:
        in_type_a = get_inpt_type(inpt[0])
        in_type_b = get_inpt_type(inpt[1])

        types = np.array([in_type_a, in_type_b])

        if in_type_a == in_type_b:
            raise ValueError('Both inputs cannot be of the same kind.')

        h_idx = np.where(types == 'horizontal')[0]
        v_idx = np.where(types == 'vertical')[0]

        if len(h_idx) == 0:
            x = 'c'
        elif len(h_idx) == 1:
            x = inpt[h_idx[0]]
        else:
            raise ValueError('More than one horizontal input provided.')
        if len(v_idx) == 0:
            y = 'c'
        elif len(v_idx) == 1:
            y = inpt[v_idx[0]]
        else:
            raise ValueError('More than one vertical input provided.')
    else:
        raise ValueError('Input must be a single character or a two-character string.')

    return x, y
