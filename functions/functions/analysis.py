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


import os
import pandas as pd
import json
from tqdm.notebook import tqdm
import subprocess

def read_xyz(tdir="./"):
    if os.path.exists(os.path.join(tdir, "parameters.json")):
        with open(os.path.join(tdir, "parameters.json"), "r") as f:
            config_data = json.load(f)
    
        tscale = config_data.get("tscale", 1)
        tstep = config_data.get("tstep", 1)
    else:
        print("no parameter file found, defaulting to standard timescale parameters")
        tstep = tscale = 1

    filepath = os.path.join(tdir, 'output.xyz')
    data = []
    timestep = None
    reading_atoms = False
    column_names = None  # to be filled dynamically

    # Get total number of lines using wc -l
    result = subprocess.run(['wc', '-l', filepath], stdout=subprocess.PIPE, text=True)
    total_lines = int(result.stdout.strip().split()[0])
    
    # Now read file with tqdm progress
    with open(filepath, 'r') as f:
        for line in tqdm(f, desc="Reading XYZ", total=total_lines):
            line = line.strip()
            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(next(f).strip())
                reading_atoms = False
            elif line.startswith("ITEM: ATOMS"):
                column_names = ["time"] + line.split()[2:]  # skip "ITEM: ATOMS"
                reading_atoms = True
            elif reading_atoms and column_names:
                values = line.split()
                if len(values) == len(column_names) - 1:  # skip malformed rows
                    # prepend current timestep as "time"
                    data.append([timestep] + list(map(float, values)))
    
    df = pd.DataFrame(data, columns=column_names)
    
    if "time" in df.columns:
        df["time"] = df["time"] * tstep / tscale

    # Try to infer types smartly
    for col in df.columns:
        if col in {"id", "mol", "type"}:
            df[col] = df[col].astype("int32")
        elif col not in {"time"}:
            df[col] = df[col].astype("float")

    return df
