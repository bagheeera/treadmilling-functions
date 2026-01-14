import math
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 120
import pandas as pd
import random
import numpy as np
import ipywidgets as widgets
import os
import pyarrow.feather as feather
from tqdm.notebook import tqdm
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)

ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
import pickle
from scipy.optimize import curve_fit

from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes

plt.rcParams.update(
    cycler.cycler(
        color=palettes.pn #paultol_high_contrast #[:3], #marker=markers.x_like_bold[:3]
    )
)
import functions as fct