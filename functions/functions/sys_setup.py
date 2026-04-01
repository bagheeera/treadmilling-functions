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

    for key in parameter_values:
        if "_" in key:
            raise ValueError('parameter contains _, will mess up detection of keys from directorynames later on!')

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



import math

def generate_triangular_grid(min_x, max_x, min_y, max_y, sidelength):
    """Generate a triangular grid within a rectangular boundary."""
    coordinates = []
    y = min_y
    row = 0
    
    # Calculate vertical spacing for equilateral triangles
    v_spacing = (math.sqrt(3) / 2) * sidelength
    
    while y <= max_y + 1e-9:
        # Offset every other row by half the sidelength
        x_start = min_x if row % 2 == 0 else min_x + (sidelength / 2)
        x = x_start
        
        while x <= max_x + 1e-9:
            coordinates.append((x, y))
            x += sidelength
            
        y += v_spacing
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

def generate_atom_table(Lx, n_synthases, add_grid=True, initial_synth_ptype=6, zpos=0.5,
                        m_process=1, min_dist=1.3, sidelength=8, m_t6=1,
                        n_activating=0, activating_initial_yrange=30,
                        activating_particle_type=8, grid_particle_type=7,
                        _3Ddiviplacement=False, Lz_ini_low=0, Lz_ini_high=0,
                        check_distance=True, mZ=1):
    """Generate and return the atom table (list of atom entries)."""
    # Define rectangular boundaries
    MIN_X, MAX_X = -Lx, Lx
    MIN_Y, MAX_Y = -yboxsize, yboxsize

    # 1. Generate the grid with rectangular bounds
    triangular_grid = generate_triangular_grid(MIN_X, MAX_X, MIN_Y, MAX_Y, sidelength)

    atom_table = [
        [1, 9, 4, 0.0, -0.5, -2.0],
        [2, 9, 4, 0.0,  0.5, -2.0],
    ]

    for i in range(len(atom_table) + 1, n_synthases + len(atom_table) + 1):
        while True:
            new_x = round(random.uniform(-Lx, Lx), 2)
            new_y = round(random.uniform(-Lx, Lx), 2)
            new_z = round(random.uniform(Lz_ini_low + 1, Lz_ini_high - 1), 2) if _3Ddiviplacement else zpos
            if not check_distance or check_min_distance(new_x, new_y, atom_table, min_dist):
                atom_table.append([i, i, initial_synth_ptype, new_x, new_y, new_z])
                break

    if n_activating:
        for i in range(len(atom_table) + 1, n_activating + len(atom_table) + 1):
            atom_table.append([i, i, activating_particle_type,
                                round(random.uniform(-Lx, Lx), 2),
                                round(random.uniform(-activating_initial_yrange, activating_initial_yrange), 2),
                                zpos])

    if add_grid:
        print("grid entries:", len(triangular_grid))
        for coord in triangular_grid:
            atom_table.append([atom_table[-1][0] + 1, atom_table[-1][1] + 1,
                                grid_particle_type, coord[0], coord[1], zpos])

    return atom_table


def generate_configuration(Lx, n_synthases, run_dir, add_grid=True, initial_synth_ptype=6, zpos=0.5,
                           mZ=1, m_process=1, min_dist=1.3, sidelength=8, m_diffu=1, m_t6=1,
                           yboxsize=None, n_atomtypes_=None, zboxsize=None,
                           n_activating=0, activating_initial_yrange=30,
                           activating_particle_type=8, grid_particle_type=7,
                           zlim=4.25, _3Ddiviplacement=False,
                           Lz_ini_low=0, Lz_ini_high=0,
                           check_distance=True, n_bondtypes=1,
                           massdict=None, overwrite=False):
    """Generate the configuration file and save it to the given directory."""
    if os.path.exists(os.path.join(run_dir, "configuration.txt")):
        if not overwrite:
            return
        else:
            print(f"Configuration file already exists in {run_dir}. Overwriting as requested.")

    DEFAULT_MASSDICT = {i: 1 for i in range(1, 15)}

    if massdict is None:
        massdict = {
            1: mZ, 2: mZ, 3: mZ, 4: 1, 5: m_process,
            6: m_t6, 7: 1, 8: 1, 9: m_process, 10: m_diffu
        }
    else:
        massdict = {**DEFAULT_MASSDICT, **massdict}

    atom_table = generate_atom_table(
        Lx=Lx, n_synthases=n_synthases, add_grid=add_grid,
        initial_synth_ptype=initial_synth_ptype, zpos=zpos,
        m_process=m_process, min_dist=min_dist, sidelength=sidelength,
        m_t6=m_t6, n_activating=n_activating,
        activating_initial_yrange=activating_initial_yrange,
        activating_particle_type=activating_particle_type,
        grid_particle_type=grid_particle_type,
        _3Ddiviplacement=_3Ddiviplacement,
        Lz_ini_low=Lz_ini_low, Lz_ini_high=Lz_ini_high,
        check_distance=check_distance, mZ=mZ,
    )

    n_atoms = len(atom_table)
    n_atomtypes = len(DEFAULT_MASSDICT)
    Lhalved = Lx
    yLhalved = yboxsize if yboxsize else Lhalved

    gridstring = "\n".join(" ".join(map(str, entry)) for entry in atom_table)
    masses_block = "\n".join(f"{k} {v}" for k, v in massdict.items())

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

{masses_block}

Atoms

{gridstring}

Bonds

1 1 1 2
""".strip()

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "configuration.txt"), "w") as f:
        f.write(config_content)