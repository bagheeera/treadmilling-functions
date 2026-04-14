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
import numpy as np


def load_pickles_into_D(D, pkl_files, prm_select=None):
    """
    Load specified .pkl/.pkl.gz or .dill/.dill.gz files into D[key] dicts.
    
    Parameters
    ----------
    D : dict
        Dictionary where each value has a "rundir" entry (path).
        Keys are tuples of (param_name, param_value) pairs.
    pkl_files : iterable of str
        Filenames to load relative to each rundir
        (e.g. produced by find_pkl_files).
    prm_select : dict, optional
        Dictionary mapping parameter names to lists of allowed values.
        E.g., {'fCurv': [0.5, 2], 'saturate': [6000, 9000, 12000]}
        Only entries matching ALL specified parameters are loaded.
        If None, all entries are processed.
    
    Returns
    -------
    int
        Number of successfully loaded files.
    """
    import gzip
    import pickle
    from pathlib import Path
    from tqdm import tqdm
    
    def sel_loader(fname):
        """Select appropriate loader based on file extension."""
        if ".pkl" in fname:
            return pickle.load
        elif ".dl" in fname:
            import dill
            return dill.load
        elif ".feather" in fname:
            import pyarrow.feather as feather
            return feather.read_feather
        elif ".npy" in fname:
            import numpy as np
            return np.load
        else:
            raise ImportError(f"filetype not recognized: {fname}")
    
    def matches_prm_select(key, prm_select):
        """
        Check if a key (tuple of param pairs) matches the selection criteria.
        
        Parameters
        ----------
        key : tuple
            Key like (('fCurv', 2), ('saturate', 9000), ('seed', 1), ...)
        prm_select : dict or None
            Selection criteria like {'fCurv': [0.5, 2], 'saturate': [6000, 9000, 12000]}
        
        Returns
        -------
        bool
            True if key matches all criteria (or if prm_select is None)
        """
        if prm_select is None:
            return True
        
        # Convert key to dict for easier lookup
        key_dict = dict(key)
        
        # Check if all specified parameters match
        for param_name, allowed_values in prm_select.items():
            if param_name not in key_dict:
                # Parameter not in this key, skip this entry
                return False
            if key_dict[param_name] not in allowed_values:
                # Parameter value doesn't match, skip this entry
                return False
        
        return True
    
    counter = 0
    # Filter keys based on prm_select
    keys_to_process = [k for k in D if matches_prm_select(k, prm_select)]
    
    for key in tqdm(keys_to_process, desc="Loading serialized files"):
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
                    D[key][subkey] = sel_loader(fname)(f)
                counter += 1
            except Exception as e:
                print(f"Failed to load {target}: {e}")
    
    return counter
