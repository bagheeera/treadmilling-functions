from pathlib import Path

def find_pkl_files(root="."):
    """
    Recursively find all .pkl and .pkl.gz files under root,
    returning only the base filenames (without paths).
    
    Parameters
    ----------
    root : str or Path, optional
        Root directory to start searching (default is current directory).
    
    Returns
    -------
    set of str
        Set of base filenames.
    """
    root = Path(root)
    pkl_files = {f.name for f in root.rglob("*.pkl")} | {f.name for f in root.rglob("*.pkl.gz")}
    return pkl_files

import os
import pickle
import gzip
from tqdm.notebook import tqdm

def load_pickles_into_D(D, pkl_files):
    """
    Load specified .pkl/.pkl.gz files into D[key] dicts.

    Parameters
    ----------
    D : dict
        Dictionary where each value has a "rundir" entry (path).
    pkl_files : iterable of str
        List or set of base filenames to search for (e.g. from find_pkl_files).

    Returns
    -------
    int
        Number of successfully loaded files.
    """
    counter = 0
    for key in tqdm(D, desc="Loading pickle files"):
        rundir = Path(D[key]["rundir"])
        for fname in pkl_files:
            target = rundir / fname
            subkey = Path(fname).stem  # removes .pkl or .gz
            
            if subkey not in D[key]:
                if target.exists() and target.stat().st_size > 0:
                    counter += 1
                    if fname.endswith(".gz"):
                        with gzip.open(target, "rb") as f:
                            D[key][subkey] = pickle.load(f)
                    else:
                        with open(target, "rb") as f:
                            D[key][subkey] = pickle.load(f)
    return counter