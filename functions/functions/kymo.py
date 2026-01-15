import numpy as np
def autocorr_1d(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]  # keep non-negative lags
    return result / result[0]  # normalize

def autocorr_along_axis(arr, axis=0):
    return np.apply_along_axis(autocorr_1d, axis=axis, arr=arr)

from pathlib import Path
import numpy as np
from tifffile import imread

def load_tif_as_array(path: str | Path, *,
                      dtype: np.dtype = np.float32,
                      normalize: str | bool = "bitdepth") -> np.ndarray:
    """
    Load a greyscale .tif image and return a NumPy array.

    Parameters
    ----------
    path : str | Path
        Path to the .tif file.
    dtype : np.dtype, default np.float32
        Target dtype for output array.
    normalize : {'bitdepth', 'minmax', False}, default 'bitdepth'
        - 'bitdepth': normalize based on bit depth (e.g. divide by 255 or 65535)
        - 'minmax'  : normalize based on actual min/max of the image
        - False     : no normalization (just convert dtype)

    Returns
    -------
    np.ndarray
        2‑D image as a NumPy array.
    """
    img = imread(path)

    if img.ndim != 2:
        raise ValueError(f"{path} does not appear to be single-channel.")

    if np.issubdtype(dtype, np.integer):
        return img.astype(dtype, copy=False)

    img = img.astype(dtype, copy=False)

    if normalize == "bitdepth":
        bit_depth = img.itemsize * 8
        max_val = (1 << bit_depth) - 1
        img /= max_val

    elif normalize == "minmax":
        img -= img.min()
        ptp = np.ptp(img)
        if ptp > 0:
            img /= ptp
        else:
            img[:] = 0.0  # flat image → normalized to 0

    elif normalize is False:
        pass  # no scaling

    else:
        raise ValueError(f"Invalid normalize option: {normalize!r}")

    return img


def imshow_kymograph(ax, imfile,
                     showhalf=True,
                     aspect=100,
                     cmap="inferno"):
    # extract diameter from filename
    df_kymo = pd.read_csv("/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/2__synthase_setup/9__midcell_condensation/8__check_lifetimes/C__wsynth_kymoparams/exp_data/ftsz_dynamics_div_state_categories.csv")
    def extract_diamter_in_nm(img, df= df_kymo):
        diam = df[df["Image_ROI_Name"].str.contains(
            img.split("/")[-1].split(".tif")[0],
            na=False
        )]["DiameterNm"].values[0]
        return diam
    img = load_tif_as_array(imfile) # [2]
    if showhalf:
        img = img[:,:img.shape[1]//2]

    diameter = extract_diamter_in_nm(imfile)
    ax.imshow(img,
        origin="lower",
                aspect=aspect,
                cmap=cmap,
                extent=[0, diameter*np.pi, 0, img.shape[0]])
