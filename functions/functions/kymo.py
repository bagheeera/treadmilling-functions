import numpy as np
def autocorr_1d(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]  # keep non-negative lags
    return result / result[0]  # normalize

def autocorr_along_axis(arr, axis=0):
    return np.apply_along_axis(autocorr_1d, axis=axis, arr=arr)