# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

from yolov8_pytorch.utils import (DEFAULT_CFG_DICT, LOGGER, RANK, ROOT, RUNS_DIR,
                                  SETTINGS, SETTINGS_YAML, TESTS_RUNNING, IterableSimpleNamespace, colorstr, deprecation_warn, yaml_load, yaml_print)

# Define valid tasks and modes
MODES = 'train', 'val', 'predict', 'export', 'benchmark'
TASKS = 'detect', 'segment'
TASK2DATA = {'detect': 'coco8.yaml', 'segment': 'coco8-seg.yaml'}
TASK2MODEL = {
    'detect': 'yolov8n.pt',
    'segment': 'yolov8n-seg.pt',

}
TASK2METRIC = {
    'detect': 'metrics/mAP50-95(B)',
    'segment': 'metrics/mAP50-95(M)',
}

# Define keys for arg type checks
CFG_FLOAT_KEYS = 'warmup_epochs', 'box', 'cls', 'dfl', 'degrees', 'shear', 'time'
CFG_FRACTION_KEYS = ('dropout', 'iou', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_momentum', 'warmup_bias_lr',
                     'label_smoothing', 'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'perspective', 'flipud',
                     'fliplr', 'mosaic', 'mixup', 'copy_paste', 'conf', 'iou', 'fraction')  # fraction floats 0.0 - 1.0
CFG_INT_KEYS = ('epochs', 'patience', 'batch', 'workers', 'seed', 'close_mosaic', 'mask_ratio', 'max_det', 'vid_stride',
                'line_width', 'workspace', 'nbs', 'save_period')
CFG_BOOL_KEYS = ('save', 'exist_ok', 'verbose', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'overlap_mask', 'val',
                 'save_json', 'save_hybrid', 'half', 'dnn', 'plots', 'show', 'save_txt', 'save_conf', 'save_crop',
                 'save_frames', 'show_labels', 'show_conf', 'visualize', 'augment', 'agnostic_nms', 'retina_masks',
                 'show_boxes', 'keras', 'optimize', 'int8', 'dynamic', 'simplify', 'nms', 'profile')


def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted to a dictionary.

    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data.
        overrides (str | Dict | optional): Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        if 'save_dir' not in cfg:
            overrides.pop('save_dir', None)  # special override keys to ignore
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in 'project', 'name':
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':  # assign model to 'name' arg
        cfg['name'] = cfg.get('model', '').split('.')[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                    f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. "
                                     f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be an int (i.e. '{k}=8')")
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")

    # Return instance
    return IterableSimpleNamespace(**cfg)


def get_save_dir(args, name=None):
    """Return save_dir as created from train/val/predict arguments."""

    if getattr(args, 'save_dir', None):
        save_dir = args.save_dir
    else:
        from yolov8_pytorch.utils.files import increment_path

        project = args.project or (ROOT.parent / 'tests/tmp/runs' if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f'{args.mode}'
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in (-1, 0) else True)

    return Path(save_dir)


def _handle_deprecation(custom):
    """Hardcoded function to handle deprecated config keys."""

    for key in custom.copy().keys():
        if key == 'boxes':
            deprecation_warn(key, 'show_boxes')
            custom['show_boxes'] = custom.pop('boxes')
        if key == 'hide_labels':
            deprecation_warn(key, 'show_labels')
            custom['show_labels'] = custom.pop('hide_labels') == 'False'
        if key == 'hide_conf':
            deprecation_warn(key, 'show_conf')
            custom['show_conf'] = custom.pop('hide_conf') == 'False'
        if key == 'line_thickness':
            deprecation_warn(key, 'line_width')
            custom['line_width'] = custom.pop('line_thickness')

    return custom


def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list. If
    any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Args:
        custom (dict): a dictionary of custom configuration options
        base (dict): a dictionary of base configuration options
        e (Error, optional): An optional error that is passed by the calling function.
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ''
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f'{k}={base[k]}' if base.get(k) is not None else k for k in matches]
            match_str = f'Similar arguments are i.e. {matches}.' if matches else ''
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string) from e


def handle_yolo_settings(args: List[str]) -> None:
    """
    Handle YOLO settings command-line interface (CLI) commands.

    This function processes YOLO settings CLI commands such as reset.
    It should be called when executing a script with arguments related to YOLO settings management.

    Args:
        args (List[str]): A list of command line arguments for YOLO settings management.

    Example:
        ```bash
        python my_script.py yolo settings reset
        ```
    """
    url = 'https://docs.ultralytics.com/quickstart/#ultralytics-settings'  # help URL
    try:
        if any(args):
            if args[0] == 'reset':
                SETTINGS_YAML.unlink()  # delete the settings file
                SETTINGS.reset()  # create new settings
                LOGGER.info('Settings reset successfully')  # inform the user that settings have been reset
            else:  # save a new setting
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)

        LOGGER.info(f'💡 Learn about settings at {url}')
        yaml_print(SETTINGS_YAML)  # print the current settings
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ settings error: '{e}'. Please see {url} for help.")


def parse_key_value_pair(pair):
    """Parse one 'key=value' pair and return key and value."""
    k, v = pair.split('=', 1)  # split on first '=' sign
    k, v = k.strip(), v.strip()  # remove spaces
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v):
    """Convert a string to an underlying type such as int, float, bool, etc."""
    v_lower = v.lower()
    if v_lower == 'none':
        return None
    elif v_lower == 'true':
        return True
    elif v_lower == 'false':
        return False
    else:
        with contextlib.suppress(Exception):
            return eval(v)
        return v
