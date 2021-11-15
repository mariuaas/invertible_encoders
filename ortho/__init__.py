""" OrthoLayers for PyTorch

TODO: Docstring
TODO: Add if missing imports are too annoying ->
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""

import torch
import torchvision
import os
import pickle
import gzip
import json
import math

import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import torchvision.transforms as T

from scipy.signal.windows import gaussian
from PIL import Image, ImageFilter
from IPython.display import clear_output
from collections import OrderedDict
from typing import List, Tuple, Union, Optional

# NOTE: Disabled due to stupid server version control
#from numpy.typing import ArrayLike
