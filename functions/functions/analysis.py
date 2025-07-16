## from http://127.0.0.1:7777/notebooks/0__treadmilling/6__balance_out_epsilon/process_synthases.ipynb

def correct_PBC_jumps(x, jumpcut=20, sidelength=200):
    x = x.copy()  # Avoid modifying the input array directly
    for i in range(1, len(x)):
        if x[i] > x[i - 1]:
            if x[i] - x[i - 1] > jumpcut:
                print(f"Jump detected at index {i}: {x[i]} -> adjusting by -{sidelength}")
                x[i:] -= sidelength
        elif x[i - 1] - x[i] > jumpcut:
            print(f"Jump detected at index {i}: {x[i]} -> adjusting by +{sidelength}")
            x[i:] += sidelength
    return x


def correct_PBC_jumps_dataframe(df, jumpcut=20, sidelength=200):
    df = df.copy()  # Work with a copy to avoid modifying the original DataFrame
    for col in ["x", "y"]:  # Apply to specific columns
        if col in df:
            df[col] = correct_PBC_jumps(df[col].values, jumpcut, sidelength)
        else:
            print(f"Column {col} not found in DataFrame")
    return df


import gzip
def compress_pickle(obj, filename):
    with gzip.open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def decompress_pickle(filepath):
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)


import os
import pyarrow.feather as feather
def load_output(rundir, verbose=True):
    """Loads _df.pkl.gz or output.feather
    """
    if os.path.exists(rundir + "/_df.pkl.gz"):
        if verbose:
            print("Using _df.pkl.gz")
        df = decompress_pickle(rundir + "/_df.pkl.gz")
        return df
    elif os.path.exists(rundir + "/output.feather"):
        if verbose:
            print("Using output.feather")
        df = feather.read_feather(rundir + "/output.feather")
        return df
    else:
        print("cannot find output files")