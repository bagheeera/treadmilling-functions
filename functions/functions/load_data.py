from pathlib import Path

def find_pkl_files(root=".", ending="pkl"):
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
    pkl_files = {f.name for f in root.rglob(f"*.{ending}")} | {f.name for f in root.rglob(f"*.{ending}.gz")}
    return pkl_files

import os
import pickle
import gzip
from tqdm.notebook import tqdm

def load_pickles_into_D(D, pkl_files, usedill=False):
    """
    Load specified .pkl/.pkl.gz or .dill/.dill.gz files into D[key] dicts.

    Parameters
    ----------
    D : dict
        Dictionary where each value has a "rundir" entry (path).
    pkl_files : iterable of str
        Filenames to load relative to each rundir
        (e.g. produced by find_pkl_files).
    usedill : bool, optional
        If True, use dill.loads for all files (requires dill installed).

    Returns
    -------
    int
        Number of successfully loaded files.
    """

    # Select loader
    if usedill:
        import dill
        if dill is None:
            raise ImportError("usedill=True but 'dill' is not installed.")
        loader = dill.load
    else:
        loader = pickle.load

    counter = 0

    for key in tqdm(D, desc="Loading serialized files"):
        rundir = Path(D[key]["rundir"])

        for fname in pkl_files:
            target = rundir / fname
            subkey = Path(fname).stem  # remove extension(s)

            if subkey in D[key]:
                continue  # already loaded

            if not (target.exists() and target.stat().st_size > 0):
                continue  # missing or empty

            try:
                # gzip vs regular open
                open_func = gzip.open if fname.endswith(".gz") else open

                with open_func(target, "rb") as f:
                    D[key][subkey] = loader(f)

                counter += 1

            except Exception as e:
                print(f"Failed to load {target}: {e}")

    return counter
