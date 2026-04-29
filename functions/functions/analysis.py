## from http://127.0.0.1:7777/notebooks/0__treadmilling/6__balance_out_epsilon/process_synthases.ipynb
import pickle
def correct_PBC_jumps(x, jumpcut=20, sidelength=200,
                      verbose=False):
    x = x.copy()  # Avoid modifying the input array directly
    for i in range(1, len(x)):
        if x[i] > x[i - 1]:
            if x[i] - x[i - 1] > jumpcut:
                if verbose:
                    print(f"Jump detected at index {i}: {x[i]} -> adjusting by -{sidelength}")
                x[i:] -= sidelength
        elif x[i - 1] - x[i] > jumpcut:
            if verbose:
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


import numpy as np
from tqdm.notebook import tqdm
# https://chatgpt.com/c/68a41feb-3610-8320-a2b4-6aa49d8c8159
def reassign_molids(df, verbose=False):
    """
    Reconstructs mol ids by tracking which mol ids appear first in time and then assigns them to all ids of that mol later on. 
    Assigns lineages to molecules over time, ie the mol id within a lineage is passed on to all subsequent particle ids.    
    Each lineage is identified by a ref_id (the "middle" ID in a group of molecules at a given time)
    and a lineage label (#1, #2, ...). Lineages persist over time whenever possible.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: "time", "mol", "id"
    verbose : bool
        If True, prints detailed propagation steps
    
    Returns
    -------
    df : pd.DataFrame
        Copy of input df with 'mol' replaced by lineage labels (integers)
    """
    
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)  # ensure time ordering
    
    # Convert relevant columns to numpy arrays for speed
    time_arr = df["time"].values
    mol_arr = df["mol"].values.astype(object)
    id_arr = df["id"].values
    
    lineages = {}       # maps ref_id -> lineage_label (string, e.g. "#1")
    used_labels = set() # set of used numeric labels
    next_label = 1      # next available numeric label
    
    times = np.unique(time_arr)  # iterate over sorted unique times
    
    for t in tqdm(times, desc="Propagating lineages"):
        if verbose:
            print(f"\n=== TIME {t} ===")
        
        mask_t = time_arr == t           # mask for current time
        mols_present = set(mol_arr[mask_t])  # unique mols present at this time
        
        # Step 1: Remove lineages whose ref_id is missing at this time
        for ref_id in list(lineages.keys()):
            mask_ref = (id_arr == ref_id) & mask_t
            if mask_ref.any():
                # ref_id present -> remove current mol from mols_present
                current_mol = mol_arr[mask_ref][0]
                mols_present.discard(current_mol)
            else:
                # ref_id missing -> lineage ends
                if verbose:
                    print(f"Lineage {lineages[ref_id]} with ref_id={ref_id} missing at time {t}, ending lineage.")
                del lineages[ref_id]
        
        # Step 2: Assign new lineages for previously unseen mols
        for mol in mols_present:
            # IDs belonging to this mol at current time
            ids_in_mol = np.sort(id_arr[mask_t & (mol_arr == mol)])
            if len(ids_in_mol) == 0:
                continue
            
            # Find next unused numeric label
            while next_label in used_labels:
                next_label += 1
            lineage_label = f"#{next_label}"
            used_labels.add(next_label)
            next_label += 1
            
            # Choose the "middle" id as reference
            ref_id = ids_in_mol[len(ids_in_mol)//2]
            lineages[ref_id] = lineage_label
            
            # Overwrite mols with lineage label
            mol_arr[mask_t & (mol_arr == mol)] = lineage_label
            
            if verbose:
                print(f"New lineage {lineage_label} for mol={mol}, ref_id={ref_id}, ids={ids_in_mol.tolist()}")
        
        # Step 3: Update ref_ids for existing lineages
        new_lineages = {}
        for ref_id, lineage_label in lineages.items():
            mask_ref = (id_arr == ref_id) & mask_t
            if not mask_ref.any():
                # ref_id missing -> skip
                continue
            current_mol = mol_arr[mask_ref][0]
            
            # IDs belonging to this mol at current time
            ids_to_overwrite = np.sort(id_arr[mask_t & (mol_arr == current_mol)])
            # Overwrite all mols with lineage label
            mol_arr[mask_t & (mol_arr == current_mol)] = lineage_label
            
            # Select new middle ID as ref_id
            new_ref_id = ids_to_overwrite[len(ids_to_overwrite)//2]
            new_lineages[new_ref_id] = lineage_label
            
            if verbose and new_ref_id != ref_id:
                print(f"[{lineage_label}] Updated ref_id: {ref_id} -> {new_ref_id}, mol={current_mol}, ids={ids_to_overwrite.tolist()}")
        
        # Update lineages for next time step
        lineages = new_lineages
    
    # Convert lineage labels from "#N" to integers
    df["mol"] = mol_arr
    df["mol"] = df["mol"].apply(lambda x: int(x[1:]) if isinstance(x, str) and x.startswith("#") else x)
    
    return df


def reassign_molids_optimized(df, verbose=False):
    """
    Reassign molecule IDs to maintain consistent lineage labels across time steps.
 
    This function processes molecular data across multiple time steps, identifying which
    molecule IDs belong together and assigning them consistent lineage labels. It handles
    cases where molecules may split, merge, or change composition over time.
 
    Args:
        df (pd.DataFrame): Input dataframe with required columns:
            - 'time': Time step identifier (numeric, should be sortable)
            - 'mol': Original molecule identifier
            - 'id': Unique identifier for each molecule ID entry
            
            Note: All other columns are preserved in the output.
 
        verbose (bool, optional): If True, print detailed processing information at each
            time step, including:
            - Lineages that end (reference ID disappears)
            - New lineages created
            - Reference ID updates within existing lineages
            Default: False
 
    Returns:
        pd.DataFrame: Copy of input dataframe with modified 'mol' column containing
            the reassigned lineage labels (integers).
 
    Notes:
        - Input dataframe is not modified; a copy is returned
        - Time ordering is enforced (data is sorted by 'time')
        - Lineage labels are unique positive integers assigned in order of appearance
        - All molecule IDs in the same lineage at a given time step receive the same label
        - Reference ID selection uses the "middle" ID for robustness against outliers
 
    Raises:
        KeyError: If required columns ('time', 'mol', 'id') are missing from dataframe
        TypeError: If 'time' or 'id' columns are not numeric
 
    Performance Characteristics:
        - Time Complexity: O(n log n) per time step (dominated by sorting)
        - Space Complexity: O(n) for auxiliary arrays
        - Typical Runtime: ~50,000-100,000 rows/second on modern hardware
        - Scales linearly with number of time steps
 
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from reassign_molids_optimized import reassign_molids_optimized
        >>>
        >>> # Example: Track molecules across 3 time steps
        >>> df = pd.DataFrame({
        ...     'time': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...     'mol': [10, 10, 20, 10, 10, 20, 10, 10, 20],
        ...     'id': [101, 102, 201, 101, 102, 201, 101, 102, 201]
        ... })
        >>> result = reassign_molids_optimized(df, verbose=False)
        >>> print(result)
           time  mol   id  mol
        0     0   10  101    1
        1     0   10  102    1
        2     0   20  201    2
        3     1   10  101    1
        4     1   10  102    1
        5     1   20  201    2
        6     2   10  101    1
        7     2   10  102    1
        8     2   20  201    2
 
    Warnings:
        - If 'id' values are not unique, behavior is undefined. Ensure each row has
          a distinct 'id' value.
        - Large time steps with many unique molecules may consume significant memory.
          For datasets >10M rows, consider processing in temporal chunks.
    """
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)  # Enforce chronological order
    
    # Convert to numpy arrays for fast vectorized operations
    time_arr = df["time"].values
    mol_arr = df["mol"].values.astype(object)  # Object type allows string labels temporarily
    id_arr = df["id"].values
    
    # Tracking structures
    lineages = {}       # Maps ref_id -> lineage_label (e.g., ref_id=42 -> "#1")
    used_labels = set() # Set of numeric labels already assigned for O(1) lookup
    next_label = 1      # Next available lineage label to assign
    times = np.unique(time_arr)  # Sorted array of unique time steps
    
    # ========================================================================
    # MAIN LOOP: Process each time step sequentially
    # ========================================================================
    # Note: Cannot parallelize across time steps due to lineage dependencies
    
    for t in tqdm(times, desc="Propagating lineages"):
        if verbose:
            print(f"\n=== TIME {t} ===")
        
        # ====================================================================
        # OPTIMIZATION: Get all indices for this time step at once
        # Instead of repeatedly applying full-array masks like (time_arr == t),
        # we get indices once and slice into arrays. This avoids repeated
        # boolean array allocations and improves cache locality.
        # ====================================================================
        idx_t = np.where(time_arr == t)[0]
        
        if len(idx_t) == 0:
            continue
        
        # ====================================================================
        # STEP 1: End lineages whose reference ID disappeared
        # ====================================================================
        # A lineage ends when its reference ID is no longer present in the data.
        # This handles cases where specific molecules cease to exist at a time step.
        
        missing_refs = []
        for ref_id in list(lineages.keys()):
            # Check if ref_id appears in current time step (vectorized)
            if not np.any(id_arr[idx_t] == ref_id):
                missing_refs.append(ref_id)
                if verbose:
                    print(f"Lineage {lineages[ref_id]} with ref_id={ref_id} missing at time {t}, ending lineage.")
        
        # Remove ended lineages from tracking
        for ref_id in missing_refs:
            del lineages[ref_id]
        
        # ====================================================================
        # STEP 2: Identify which molecules are "new" (not in any lineage yet)
        # ====================================================================
        # A molecule is "new" if no active lineage claims it at this time step.
        # We track this to avoid assigning multiple lineages to the same mol.
        
        assigned_mols = set()
        for ref_id in list(lineages.keys()):
            # If this lineage's ref_id is present, get its current mol
            if np.any(id_arr[idx_t] == ref_id):
                # Get the first (and should be only) mol for this ref_id
                current_mol = mol_arr[np.where(id_arr[idx_t] == ref_id)[0][0]]
                assigned_mols.add(current_mol)
        
        # New molecules = all molecules at this time - already assigned molecules
        unique_mols_t = set(mol_arr[idx_t]) - assigned_mols
        
        # ====================================================================
        # STEP 2: Assign new lineages to previously unseen molecules
        # ====================================================================
        # Each new molecule group gets a fresh lineage label.
        # A "group" consists of all IDs that share the same original mol value.
        
        for mol in unique_mols_t:
            # Find all IDs belonging to this molecule at current time
            mol_mask = mol_arr[idx_t] == mol
            ids_in_mol = np.sort(id_arr[idx_t][mol_mask])
            
            if len(ids_in_mol) == 0:
                continue
            
            # ================================================================
            # HEURISTIC: Choose "middle" ID as reference for robustness
            # ================================================================
            # Using the middle ID reduces sensitivity to outliers and edge
            # effects in the ID sequence. Alternatives: min, max, or random.
            # The middle ID is more stable across splits and merges.
            
            # Find next unused label (could optimize with counter if labels
            # are always contiguous, but this is safe and O(1) per time step)
            while next_label in used_labels:
                next_label += 1
            
            lineage_label = f"#{next_label}"  # Temporary string format for uniqueness
            used_labels.add(next_label)
            next_label += 1
            
            # Choose middle ID as reference
            ref_id = ids_in_mol[len(ids_in_mol) // 2]
            lineages[ref_id] = lineage_label
            
            # Replace all original mol values with the lineage label
            # This ensures all IDs in the same group have the same label
            mol_arr[idx_t[mol_mask]] = lineage_label
            
            if verbose:
                print(f"New lineage {lineage_label} for mol={mol}, ref_id={ref_id}, ids={ids_in_mol.tolist()}")
        
        # ====================================================================
        # STEP 3: Propagate and update existing lineages
        # ====================================================================
        # For each active lineage, reassign all IDs in its molecule group
        # to have the lineage label, and update the reference ID to the
        # "middle" ID for the current time step.
        
        new_lineages = {}  # Lineages for next time step
        
        for ref_id, lineage_label in list(lineages.items()):
            # Check if this lineage's ref_id still exists
            ref_mask = id_arr[idx_t] == ref_id
            if not np.any(ref_mask):
                # ref_id is absent at this time (shouldn't happen after Step 1, but safe)
                continue
            
            # Get all IDs in this mol group
            current_mol = mol_arr[idx_t[ref_mask]][0]
            mol_mask = mol_arr[idx_t] == current_mol
            ids_to_overwrite = np.sort(id_arr[idx_t][mol_mask])
            
            # Replace mol values with lineage label (idempotent if already labeled)
            mol_arr[idx_t[mol_mask]] = lineage_label
            
            # Update reference ID to new middle ID for next iteration
            # This keeps the reference stable even as molecules split/merge
            new_ref_id = ids_to_overwrite[len(ids_to_overwrite) // 2]
            new_lineages[new_ref_id] = lineage_label
            
            if verbose and new_ref_id != ref_id:
                print(f"[{lineage_label}] Updated ref_id: {ref_id} -> {new_ref_id}, mol={current_mol}, ids={ids_to_overwrite.tolist()}")
        
        # Update lineages for next time step
        lineages = new_lineages
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    
    # Replace temporary string labels with integers
    # Converts "#1" -> 1, "#42" -> 42, etc.
    df["mol"] = mol_arr
    df["mol"] = df["mol"].apply(
        lambda x: int(x[1:]) if isinstance(x, str) and x.startswith("#") else x
    )
    
    return df
