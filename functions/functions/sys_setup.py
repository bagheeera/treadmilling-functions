import os
import itertools
import json
from jinja2 import Template
from tqdm.notebook import tqdm  # make sure tqdm is imported
import numpy as np
import math
import random
import subprocess
import os

def format_value(v, decimals=3):
        if isinstance(v, float):
            format_str = f"{{:.{decimals}f}}"
            return format_str.format(v).rstrip('0').rstrip('.')  # Avoid .0 and overprecision
        return str(v)

def write_templates(parameter_values, base_values, template, dontwrite=False, adjust_dependent_params_fn=None,
                   omit_params=None,
                   decimals=3,
                   overwrite=False):
    """
    Write templates for all combinations, with support for dependent parameters.
    
    `parameter_values`: Dictionary of parameters and their values
    `base_values`: Dictionary of base values to include in each config
    `template`: Jinja2 template
    `dontwrite`: Flag to skip actual file writing
    `adjust_dependent_params_fn`: Function for adjusting dependent parameters (optional)
    `omit_params`: Set or list of parameters to exclude from folder name
    `overwrite`: Whether to overwrite existing files
    `decimals`: Number of decimal places for float formatting in directory names

    adjust_dependent_params_fn example usage:
    import random
    def adjust_dependent_params_fn(params):
        params["seed"] = random.randint(1, 6)
        return params
    """

    

    param_keys = list(parameter_values.keys())
    combinations = list(itertools.product(*parameter_values.values()))
    print(len(combinations))

    if not dontwrite:
        for values in tqdm(combinations):
            param_combination = dict(zip(param_keys, values))

            if adjust_dependent_params_fn:
                param_combination = adjust_dependent_params_fn(param_combination)

            params = {**base_values, **param_combination}

            # Construct dir name with proper formatting
            dir_name = "run_" + "_".join(
                f"{k}{format_value(v, decimals=decimals)}"
                for k, v in param_combination.items()
                if not omit_params or k not in omit_params
            )

            os.makedirs(dir_name, exist_ok=True)
            os.makedirs(os.path.join(dir_name, "runfiles"), exist_ok=True)

            config_output = template.render(params)
            config_path = os.path.join(dir_name, "runfiles", "config.sh")
            params_path = os.path.join(dir_name, "runfiles", "parameters.json")

            if overwrite or not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    f.write(config_output)

            if overwrite or not os.path.exists(params_path):
                with open(params_path, "w") as json_file:
                    json.dump(params, json_file, indent=4)

    return combinations, param_keys

import shutil
def copy_reaction_directories(source_dir, base_target_dir, param_keys, combinations, decimals=3):
    """Copies reaction directories based on parameter combinations."""
    for values in tqdm(combinations):
        params = dict(zip(param_keys, values))  # Create dictionary of parameters

        # Generate directory name dynamically
        dir_name = os.path.join(base_target_dir, "run_" + "_".join(f"{k}{format_value(v, decimals=decimals)}" for k, v in params.items()), "runfiles/Reactions_rdis/")
        
        # Use shutil for efficient copying
        shutil.copytree(source_dir, dir_name, dirs_exist_ok=True)
    
    #print(f"All directories copied successfully to {base_target_dir}")



def generate_triangular_grid(min_coord, max_coord, sidelength):
    """Generate a triangular grid of points."""
    import math
    coordinates = []
    y = min_coord
    row = 0
    while y <= max_coord:
        x_start = min_coord if row % 2 == 0 else min_coord + (sidelength / 2)
        x = x_start
        while x <= max_coord:
            coordinates.append((x, y))
            x += sidelength
        y += (math.sqrt(3) / 2) * sidelength
        row += 1
    return coordinates

def check_min_distance(new_x, new_y, atom_table, min_dist):
    """Ensure new coordinates are at least min_dist away from existing atoms."""
    for entry in atom_table:
        existing_x, existing_y = entry[3], entry[4]
        distance = math.sqrt((new_x - existing_x) ** 2 + (new_y - existing_y) ** 2)
        if distance < min_dist:
            return False
    return True

def generate_configuration(Lx, n_synthases, run_dir, add_grid=True, initial_synth_ptype=6, zpos=0.5, 
                           mZ=1, m_process=1, min_dist=1.3, sidelength=8, m_diffu=1, m_t6=1, 
                          yboxsize=None, n_atomtypes_=None, zboxsize=None,
                          n_activating=0,
                           activating_initial_yrange=30,
                           activating_particle_type=8,
                           grid_particle_type=7,
                           zlim=4.25,
                           _3Ddiviplacement=False,
                           Lz_ini_low=0,
                           Lz_ini_high=0,
                           check_distance=True,
                           n_bondtypes=1,
                          ):
    """Generate the configuration file and save it to the given directory."""
    import math
    MIN_COORD = -Lx
    MAX_COORD = Lx
    
    # Generate triangular grid
    triangular_grid = generate_triangular_grid(MIN_COORD, MAX_COORD, sidelength)
    
    
    # Initial atom entries
    atom_table = [
     #   [1, 1, 2, -0.5, 0.0, 0.0],
     #   [2, 1, 3, 0.5, 0.0, 0.0],
        [1, 9, 4, 0.0, -0.5, -2.0],
        [2, 9, 4, 0.0, 0.5, -2.0],
    #    [5, 2, initial_synth_ptype, 0.0, 0.0, DIVI_ZCOORD]
    ]
    
    # Generate additional synthase atoms
    for i in range(len(atom_table) + 1, n_synthases + len(atom_table) + 1):
        while True:
            new_x = round(random.uniform(-Lx, Lx), 2)
            new_y = round(random.uniform(-Lx, Lx), 2)
            if _3Ddiviplacement:
                new_z = round(random.uniform(Lz_ini_low+1, Lz_ini_high-1), 2)
            else:
                new_z = zpos
            if check_distance:
                if check_min_distance(new_x, new_y, atom_table, min_dist):
                    atom_table.append([i, i, initial_synth_ptype, new_x, new_y, new_z])
                    break
            else:
                atom_table.append([i, i, initial_synth_ptype, new_x, new_y, new_z])
                break
    if n_activating:
        for i in range(len(atom_table)+1, n_activating + len(atom_table)+1):
            entry = [i, i, activating_particle_type,
                         round(random.uniform(-Lx, Lx),2), 
                         round(random.uniform(-activating_initial_yrange, activating_initial_yrange),2), 
                         zpos]
            atom_table.append(entry)
    
    # Add grid points to atom table
    if add_grid:
        print("grid entries:", len(triangular_grid))
        for i, coord in enumerate(triangular_grid):
            atom_table.append([atom_table[-1][0] + 1, atom_table[-1][1] + 1, grid_particle_type, coord[0], coord[1], zpos])
        
    n_atoms = len(atom_table)
    #if n_atomtypes_:
    #    n_atomtypes = n_atomtypes_
    #else:
    #    n_atomtypes = int(max(np.array(atom_table)[:, 2]))
    n_atomtypes = 10
    Lhalved = Lx
    if yboxsize:
        yLhalved = yboxsize
    else:
        yLhalved = Lhalved
    
    # Create configuration text
    gridstring = "\n".join(" ".join(map(str, entry)) for entry in atom_table)
    
    config_content = f"""
´Divisome´ setup with grid of reaction ghosts
{n_atoms} atoms
1 bonds
0 angles
{n_atomtypes} atom types
{n_bondtypes} bond types
2 angle types
-{Lhalved} {Lhalved} xlo xhi
-{yLhalved} {yLhalved} ylo yhi
-{zlim} {zlim} zlo zhi

Masses

1 {mZ}
2 {mZ}
3 {mZ}
4 1
5 {m_process}
6 {m_t6}
7 1
8 1
9 {m_process}
10 {m_diffu}

Atoms

{gridstring}

Bonds

1 1 1 2
""".strip()
    
    # Ensure output directory exists
    os.makedirs(run_dir, exist_ok=True)
    
    # Write to file
    with open(os.path.join(run_dir, "configuration.txt"), "w") as f:
        f.write(config_content)
    
    #print(f"Configuration file saved to {os.path.join(run_dir, 'configuration.txt')}")
