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
