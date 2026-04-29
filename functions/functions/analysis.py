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
    Vectorized version with these optimizations:
    1. Use np.where() to get indices upfront (fewer full array masks)
    2. Pre-compute assigned mols to avoid set operations on full array
    3. Batch operations per mol
    4. Reduce string formatting overhead
    """
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)
    
    time_arr = df["time"].values
    mol_arr = df["mol"].values.astype(object)
    id_arr = df["id"].values
    lineages = {}
    used_labels = set()
    next_label = 1
    times = np.unique(time_arr)
    
    for t in tqdm(times, desc="Propagating lineages (OPTIMIZED)"):
        if verbose:
            print(f"\n=== TIME {t} ===")
        
        # Get indices for this time step once
        idx_t = np.where(time_arr == t)[0]
        
        if len(idx_t) == 0:
            continue
        
        # Step 1: Identify missing lineages (vectorized)
        missing_refs = []
        for ref_id in list(lineages.keys()):
            # Check if ref_id exists in current time step
            if not np.any(id_arr[idx_t] == ref_id):
                missing_refs.append(ref_id)
                if verbose:
                    print(f"Lineage {lineages[ref_id]} with ref_id={ref_id} missing at time {t}, ending lineage.")
        
        for ref_id in missing_refs:
            del lineages[ref_id]
        
        # Step 2: Find which mols are already assigned in this lineage
        assigned_mols = set()
        for ref_id in list(lineages.keys()):
            # Find which mol this ref_id points to
            ref_mask = id_arr[idx_t] == ref_id
            if np.any(ref_mask):
                current_mol = mol_arr[idx_t[ref_mask]][0]
                assigned_mols.add(current_mol)
        
        # Get unique mols at this time, excluding already-assigned ones
        unique_mols_t = set(mol_arr[idx_t]) - assigned_mols
        
        # Step 2: Assign new lineages
        for mol in unique_mols_t:
            mol_mask = mol_arr[idx_t] == mol
            ids_in_mol = np.sort(id_arr[idx_t][mol_mask])
            
            if len(ids_in_mol) == 0:
                continue
            
            # Find next unused label
            while next_label in used_labels:
                next_label += 1
            
            lineage_label = f"#{next_label}"
            used_labels.add(next_label)
            next_label += 1
            
            ref_id = ids_in_mol[len(ids_in_mol) // 2]
            lineages[ref_id] = lineage_label
            
            # Overwrite mols (use indices, not boolean mask)
            mol_arr[idx_t[mol_mask]] = lineage_label
            
            if verbose:
                print(f"New lineage {lineage_label} for mol={mol}, ref_id={ref_id}, ids={ids_in_mol.tolist()}")
        
        # Step 3: Update ref_ids for existing lineages
        new_lineages = {}
        for ref_id, lineage_label in list(lineages.items()):
            ref_mask = id_arr[idx_t] == ref_id
            if not np.any(ref_mask):
                continue
            
            current_mol = mol_arr[idx_t[ref_mask]][0]
            mol_mask = mol_arr[idx_t] == current_mol
            ids_to_overwrite = np.sort(id_arr[idx_t][mol_mask])
            
            # Overwrite mols
            mol_arr[idx_t[mol_mask]] = lineage_label
            
            # Select new middle ID as ref_id
            new_ref_id = ids_to_overwrite[len(ids_to_overwrite) // 2]
            new_lineages[new_ref_id] = lineage_label
            
            if verbose and new_ref_id != ref_id:
                print(f"[{lineage_label}] Updated ref_id: {ref_id} -> {new_ref_id}, mol={current_mol}, ids={ids_to_overwrite.tolist()}")
        
        lineages = new_lineages
    
    # Convert lineage labels from "#N" to integers
    df["mol"] = mol_arr
    df["mol"] = df["mol"].apply(lambda x: int(x[1:]) if isinstance(x, str) and x.startswith("#") else x)
    
    return df
