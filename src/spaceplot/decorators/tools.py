import matplotlib.pyplot as plt

from .. import aligner, utils


def parse_loc_input(loc, align=None, align_format='frac'):
    if isinstance(loc, (list, tuple)) and len(loc) == 2:
        loc_x, loc_y = loc
        loc = None
    else:
        loc_x = loc_y = None

    if isinstance(align, str):
        align_loc = align
        align_x = align_y = None
    if isinstance(align, (list, tuple)) and len(align) == 2:
        align_loc = None
        align_x, align_y = align
    elif align is None:
        align_loc = loc
        align_x = align_y = None

    xy = aligner.translate_align(how=loc, format='frac', xfact=loc_x, yfact=loc_y)
    hv = aligner.translate_align(how=align_loc, format=align_format, xfact=align_x, yfact=align_y)
    return xy, hv


def get_abc_style(style, size, box, ax, clip_on, kwargs):
    rcfc = plt.rcParams['axes.facecolor']
    rcec = plt.rcParams['axes.edgecolor']
    rctc = plt.rcParams['text.color']
    rclw = plt.rcParams['axes.linewidth']

    styles = {
        'label': {
            'tc': rctc,
            'ec': 'none',
            'fc': rcec,
            'lw': rclw,
            'alpha': 1,
            'box': False,
        },
        'alpha_box': {
            'tc': rctc,
            'ec': 'none',
            'fc': rcec,
            'lw': rclw,
            'alpha': 0.1,
            'box': True,
        },
        'frame': {
            'tc': rctc,
            'ec': rcec,
            'fc': rcfc,
            'lw': rclw,
            'alpha': 1,
            'box': True,
        },
        'box': {
            'tc': rcfc,
            'ec': 'none',
            'fc': rcec,
            'lw': rclw,
            'alpha': 1,
            'box': True,
        },
    }

    if style not in styles:
        raise ValueError(f"Style '{style}' not recognized. Available styles: {list(styles.keys())}")

    used_style = styles[style]

    label_params = utils.get_hook_dict(params=kwargs, hook='label')
    label_params['fontsize'] = label_params['fontsize'] if 'fontsize' in label_params else size
    label_params['fontweight'] = label_params['fontweight'] if 'fontweight' in label_params else 'bold'
    label_params['color'] = label_params['color'] if 'color' in label_params else used_style['tc']

    box_params = utils.get_hook_dict(params=kwargs, hook='box')

    box = used_style['box'] if box is None else box
    box_params['clip_path'] = ax.patch if clip_on else None
    box_params['facecolor'] = box_params['facecolor'] if 'facecolor' in box_params else used_style['fc']
    box_params['edgecolor'] = box_params['edgecolor'] if 'edgecolor' in box_params else used_style['ec']
    box_params['linewidth'] = box_params['linewidth'] if 'linewidth' in box_params else used_style['lw']
    box_params['alpha'] = box_params['alpha'] if 'alpha' in box_params else used_style['alpha']

    return label_params, box_params, box
