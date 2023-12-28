# Copyright 2023 Lornatang Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging

from .common import (convert_letterbox_to_bbox, calculate_model_parameters, calculate_model_flops, make_anchors)
from .files import (increment_path, get_latest_run, file_age, file_date, file_size, spaces_in_path, WorkingDirectory)
from .misc import (fuse_conv_and_bn, inference_mode, model_summary, time_sync)
from .ops import (scale_image)
from .profile import (Profile)
from .visualizer import (generate_feature_maps, save_feature_maps)

logger = logging.getLogger(__name__)


def colorstr(*input):
    """
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr('blue', 'bold', 'hello world')
        >>> '\033[34m\033[1mhello world\033[0m'
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


class BaseClass:
    r"""A base class providing helpful string representation, error reporting, and attribute
    access methods for easier debugging and usage.
    """

    def __str__(self):
        r"""Return a human-readable string representation of the object.

        This method iterates over all attributes of the object that are not callable (i.e., methods) and do not start with an underscore (i.e., private or protected).
        For each attribute, it checks if it is an instance of BaseClass. If it is, it only displays the module and class name.
        Otherwise, it displays the attribute name and its value.
        Finally, it returns a string containing the module name, class name, and all attributes of the object.
        """
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith('_'):
                if isinstance(v, BaseClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        r"""Return a machine-readable string representation of the object.

        In this class, the __repr__ method simply calls the __str__ method, so they return the same result.
        """
        return self.__str__()

    def __getattr__(self, attr):
        r"""Custom attribute access error message with helpful information.

        This method is called when trying to access an attribute that does not exist.
        It raises an AttributeError with a custom message that includes the class name and the class's docstring, providing information about valid attributes.
        """
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")
