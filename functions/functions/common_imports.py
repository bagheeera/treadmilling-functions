import math
import matplotlib.pyplot as plt
# plt.rcParams["figure.dpi"] = 120
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
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
import gzip
from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes
from jinja2 import Environment, FileSystemLoader

plt.rcParams.update(
    cycler.cycler(
        color=palettes.pn #paultol_high_contrast #[:3], #marker=markers.x_like_bold[:3]
    )
)
import functions as fct

__all__ = [
    # core numerics / data
    "np",
    "pd",
    "math",
    "random",

    # plotting
    "plt",
    "ccycle",

    # progress / UI
    "tqdm",
    "widgets",

    # IO / utilities
    "os",
    "pickle",
    "feather",

    # scientific helpers
    "curve_fit",

    # tueplots styling
    "cycler",
    "markers",
    "palettes",

    # your own helpers
    "fct",
]
