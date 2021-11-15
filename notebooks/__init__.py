import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import torch.fft as fft
import torchvision
import torchvision.transforms as transforms
import os
import math
import matplotlib.pyplot as plt
import sys
import json
import pickle
sys.path.append('../')

from scipy import stats
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from tqdm.notebook import tqdm
from collections import OrderedDict

from ortho import modules, utils, models, datasets

plt.style.use('bmh')
plt.rcParams["image.cmap"] = 'nipy_spectral'
plt.rcParams["figure.figsize"] = (6.5,3)
#plt.rcParams['axes.facecolor'] = (0.0, 0.0, 0.0, 0.075)
#plt.rcParams['figure.facecolor'] = (1.0, 1.0, 1.0, 0.0)
#plt.rcParams['savefig.facecolor'] = (1.0, 1.0, 1.0, 0.0)
plt.rcParams['figure.autolayout'] = True

def imshow(img, cmap='gray', root='../figures/misc/', filename=None, file_format='pdf'):
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.grid(False)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(root + filename + '.' + file_format)
        
classes = [
    'cat',
    'lynx',
    'wolf',
    'coyote',
    'cheetah',
    'jaguar',
    'chimpanzee',
    'orangutan',
    'hamster',
    'guineapig', 
]