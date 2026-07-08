from tqdm.notebook import tqdm
import numpy as np
import math
import random
import subprocess
import os

import os
import sys

import os

def check_dirs_with_file_without_file(required_files=None, missing_files=None, dir_name="runfiles", write_to=None):
    """
    Return a list of directories (searched recursively from cwd) whose name is `dir_name`,
    that contain ALL files in `required_files`, and contain NONE of the files in `missing_files`.

    `required_files` and `missing_files` may be strings or lists of strings.
    """

    # Help
    if dir_name in ("-h", "--help"):
        print("Usage: check_dirs_with_file_without_file(<directory_name>, <required_files>, <missing_files>)")
        print("")
        print("required_files and missing_files may be a string or a list of strings.")
        print("")
        print("Returns: list of matching directory paths.")
        print("")
        print("Example:")
        print("  check_dirs_with_file_without_file('runfiles', ['output.xyz'], ['df.pkl.gz'])")
        return []

    # Normalize args
    if isinstance(required_files, str):
        required_files = [required_files]
    if isinstance(missing_files, str):
        missing_files = [missing_files]

    if not dir_name or not required_files or not missing_files:
        raise ValueError("Missing arguments: dir_name, required_files, missing_files must all be provided.")

    cwd = os.getcwd()
    matches = []

    for root, dirs, files in os.walk(cwd):
        if os.path.basename(root) != dir_name:
            continue

        has_all_required = all(os.path.isfile(os.path.join(root, f)) for f in required_files)
        has_any_missing = any(os.path.isfile(os.path.join(root, f)) for f in missing_files)

        if has_all_required and not has_any_missing:
            matches.append(root)

    if write_to:
        with open(write_to, "w") as f:
            for match in matches:
                f.write(match + "\n")

    return matches



def flatten(arr):
    return [val for subl in arr for val in subl]

import os, fnmatch
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def runtag(inp, prm_sets):
    tag = ""
    for i, p in enumerate(list(prm_sets)):
        tag += p + str(inp[i]) + "_"
    return tag[:-1] ## leave out the final "_"

def filestring_to_dict(lst):
    result = {}
    
    for item in lst:
        # Extract alphabetic characters for the key
        key = ''.join(filter(str.isalpha, item))
        
        # Extract numeric characters (including the decimal point) for the value
        value_str = ''.join(filter(lambda c: c.isdigit() or c == '.', item))
        
        # Convert the numeric string to float or int
        if '.' in value_str:
            value = float(value_str)
        else:
            value = int(value_str)
        
        # Assign the key-value pair to the result dictionary
        result[key] = value
    return result

def parameters_and_paramsets(pkl):
    def split_path_extract_paramlist(p):
        split = p.split("/")
        paramlist = [part for part in split if part.startswith("run_")][0]
        paramlist = paramlistsplit("_")[1:]

    
    params = [filestring_to_dict(p.split("/")[1].split("_")[1:])
        for p in pkl
    ]
    prm_sets = {key: sorted(list(set([prm[key] for prm in params])))
                   for key in list(params[0])
               }
    return params, prm_sets

def tuple_to_tag(params):
    """
    Convert a tuple of parameter-value pairs into a tag with underscores.
    
    Args:
        params (tuple): Tuple of (parameter, value) pairs.
        
    Returns:
        str: A string tag with parameter names and values joined by underscores.
    """
    return "_".join(f"{param}{value}" for param, value in params)

def find_runfiles_dirs(rootdir):
    """
    
    Recursively find all directories named 'runfiles' starting from rootdir.

    Args:
        rootdir (str): Path to start searching from.

    Returns:
        list of str: Full paths to all 'runfiles' directories.
    """
    runfiles_dirs = []
    for dirpath, dirnames, _ in os.walk(rootdir):
        if "runfiles" in dirnames:
            runfiles_dirs.append(os.path.join(dirpath, "runfiles"))
    return runfiles_dirs


def build_rundir_dict(rundirs, params, prm_sets, runtag):
    """
    Match each param set to a rundir and return a dict with keys as sorted param tuples.

    Args:
        rundirs (list of str): List of run directory paths.
        params (list of dict): Parameter dictionaries to create keys from.
        prm_sets: The full set of parameters used in runtag.
        runtag (func): Function to generate tag from param values and prm_sets.

    Returns:
        dict: Mapping from tuple(sorted(param.items())) to {"rundir": path}.

    Run:
    pkl = fct.utils.find("*starting_lammps.txt", "./")
    params, prm_sets = fct.utils.parameters_and_paramsets(pkl)
    rundirs = fct.utils.find_runfiles_dirs("./")
    D = build_rundir_dict(rundirs, params, prm_sets, fct.utils.runtag)
    """
    D = {}
    for prm in params:
        key = tuple(sorted(prm.items()))
        tag = runtag(list(prm.values()), prm_sets)
        for path in rundirs:
            parent = os.path.basename(os.path.dirname(path))  # gives 'run_rdis10_pact0.01_pdeact0.1_Kbend500'
            if tag == parent.replace("run_", ""):
                abs_path = os.path.abspath(path)
                D[key] = {"rundir": abs_path}
                break
    return D

def restrict_prm_sets(prm_sets, select):
    if not select:
        return prm_sets

    prm_sets = prm_sets.copy()

    for key, val in select.items():
        if key not in prm_sets:
            raise KeyError(f"Unknown parameter in select: {key}")

        allowed = prm_sets[key]

        if isinstance(val, (list, tuple, set)):
            prm_sets[key] = [v for v in allowed if v in val]
        else:
            prm_sets[key] = [val] if val in allowed else []

        if not prm_sets[key]:
            raise ValueError(
                f"No valid values left for parameter '{key}' after selection"
            )

    return prm_sets

def filter_params(params, select):
    if not select:
        return params

    def match(prm):
        for k, v in select.items():
            if k not in prm:
                return False
            if isinstance(v, (list, tuple, set)):
                if prm[k] not in v:
                    return False
            else:
                if prm[k] != v:
                    return False
        return True

    return [prm for prm in params if match(prm)]


def load_runs(fname="*starting_lammps.txt", 
                wdir=".", exclude=[],
                select=None
    ):
    pkl = [
        f for f in find(fname, wdir)
        if not any(exc in f for exc in exclude)
    ]
    params, prm_sets = parameters_and_paramsets(pkl)

    prm_sets = restrict_prm_sets(prm_sets, select)
    params = filter_params(params, select)

    rundirs = [
        d for d in find_runfiles_dirs(wdir)
        if not any(exc in d for exc in exclude)
    ]
    D = build_rundir_dict(rundirs, params, prm_sets, runtag)
    print(f"Loaded dict with {len(D)} keys")
    return D, params, prm_sets

import subprocess
import time

def slurm_job_finished(job_id):
    """Check if the given SLURM job has finished by using squeue."""
    try:
        result = subprocess.run(["squeue", "-j", str(job_id)], capture_output=True, text=True)
        return str(job_id) not in result.stdout
    except Exception as e:
        print(f"Error checking SLURM job status: {e}")
        return False

def wait_and_run_commands(job_id, commands, sleep_time=10000):
    """
    Waits for a specified SLURM job to finish, then executes a list of functions.

    Parameters
    ----------
    job_id : int or str
        The SLURM job ID to monitor.
    commands : list of callables
        A list of zero-argument functions (e.g., lambdas) to execute once the job finishes.
    sleep_time : int, optional
        Time in seconds to wait between job status checks. Default is 10000 seconds.

    Example
    -------
    >>> commands = [
    ...     lambda: submit_papermill("job1", "path/to/notebook1.ipynb", "job1", "rdir", ram_gb=40, time_hours=30),
    ...     lambda: submit_papermill("job2", "path/to/notebook2.ipynb", "job2", "rdir", ram_gb=40, time_hours=30),
    ... ]
    >>> wait_and_run_commands(12345678, commands)

    This will wait until SLURM job 12345678 finishes, then submit the two papermill jobs defined in the commands list.
    """
    print(f"Monitoring SLURM job {job_id}...")
    while not slurm_job_finished(job_id):
        print(f"Waiting for job {job_id} to finish...")
        time.sleep(sleep_time)
    
    print(f"Job {job_id} finished! Running commands...")
    for i, cmd in enumerate(commands):
        print(f"Running command {i + 1}...")
        try:
            cmd()
        except Exception as e:
            print(f"Error running command {i + 1}: {e}")


#https://chatgpt.com/c/68396b26-96c4-8011-8cc6-a6c74a421910
import os
import subprocess
import glob

def submit_restart_runs(rdir, job_name, iteration, analysisonly=False, cores=2, time="30:00:00", mem="30G", 
                        lmp_path="~/0__treadmilling/0__treadmilling_git/MD/lammpsSep21/src/lmp_serial",
                        analysis_script="/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/2__synthase_setup/2__vary_potential_size/filament_analysis.ipynb",
                        env_setup="/nfs/scistore26/saricgrp/fhorvath/miniforge3/etc/profile.d",
                        dontsubmit=False, additional_analysis=None, analyzefilaments=True,
                        writeonly=False,
                        load_mamba_env=True,
                        env_name="filaments"):
    """
    Submits a SLURM job to continue LAMMPS simulations by reading the latest restart file.
    Also updates config.sh to config_restart.sh with appropriate modifications.
    If writeonly is True, only write config_restart.sh in each directory and exit.
    """

    run_dirs = [line.strip() for line in open(rdir)]

    for d in run_dirs:
        config_path = os.path.join(d, "config.sh")
        restart_path = os.path.join(d, "config_restart.sh")

        import json
        with open(d + "/parameters.json", "r") as f:
            params = json.load(f)
        tstep = params.get("tstep", 0)  
        runtime = params.get("runtime", 0)  
        nsteps = int(runtime / tstep)
        #print(f"Runtime: {runtime}, Timestep: {tstep}, Nsteps: {nsteps}")
        prev_run_length = nsteps  # Default to total nsteps 

        if not os.path.exists(config_path):
            print(f"Warning: {config_path} not found. Skipping.")
            continue

        # Find the latest restart file
        restart_files = sorted(glob.glob(os.path.join(d, "restart.*")), key=lambda x: int(x.split(".")[-1]) if x.split(".")[-1].isdigit() else -1)
        if not restart_files:
            print(f"No restart.* files found in {d}. Skipping.")
            continue

        latest_restart = os.path.basename(restart_files[-1])

        # Process config.sh into config_restart.sh
        with open(config_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            ## read in restart files
            if line.startswith("read_data"):
                new_lines.append("#" + line)
                new_lines.append(f"read_restart {latest_restart}\n")
            
            ## modify output file names to include iteration number
            elif line.startswith("dump"):
                for ext in [".dump", ".out", ".xyz"]:
                    if ext in line:
                        newline = line.replace(ext, f"_{iteration}{ext}")
                        # print("original line:", line.strip())
                        # print("modified line:", newline.strip())
                        new_lines.append(newline)
                        break
            
            
            elif line.startswith("run"):
                ## extract previous run length
                # prev_run_length = int(line.split()[1]) ## done via json parameters now
                prev_runtime = int(latest_restart.split(".")[-1])
                new_run_length = prev_run_length - prev_runtime # int(latest_restart)
                print(f"Previous run length: {prev_run_length}, latest restart: {prev_runtime}, new run length: {new_run_length}")
                new_lines.append(f"run {new_run_length}\n")


            #elif line.strip().startswith("dump_modify"):
            #    new_lines.append(line.strip() + " append yes\n")
            else:
                new_lines.append(line)

        with open(restart_path, "w") as f:
            f.writelines(new_lines)

  

        #print(f"Wrote {restart_path}")

    if writeonly:
        print("writeonly=True: Not submitting any jobs.")
        return

    # Write SLURM job script
    os.makedirs("logs", exist_ok=True)

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/array_job_%A_task_%a.log
#SBATCH -c {cores}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

dir=$(sed "${{SLURM_ARRAY_TASK_ID}}q;d" {rdir})
#SBATCH --workdir $dir

echo "Running in directory: $dir"
cd $dir
"""

    if not analysisonly:
        script_content += f"""
touch starting_lammps.txt
srun {lmp_path} -i config_restart.sh
"""

    script_content += f"""
# Check for ERROR in log.txt before running analysis
if grep -q "ERROR" log.txt; then
    echo "Skipping analysis due to ERROR in log.txt"
    exit 0
fi
"""

    if analyzefilaments:
        if load_mamba_env:
            script_content += f"""source {env_setup}/conda.sh
source {env_setup}/mamba.sh
eval "$(mamba shell hook --shell bash)"
mamba activate {env_name}"""
        script_content += f"""
srun papermill {analysis_script} analyze_slrm.ipynb -p runfold $dir -p num_cores {cores}
"""

    if additional_analysis:
        if load_mamba_env:
            script_content += f"""source {env_setup}/conda.sh
source {env_setup}/mamba.sh
eval "$(mamba shell hook --shell bash)"
mamba activate {env_name}"""
        for analysis in additional_analysis:
            analysis_filename = os.path.basename(analysis)
            script_content += f"""
srun papermill {analysis} {analysis_filename} -p runfold $dir -p rundir $dir -p num_cores {cores}
"""

    with open("slurm_array.submit", "w") as f:
        f.write(script_content)

    if not dontsubmit:
        line_count = len(run_dirs)
        subprocess.run(["sbatch", f"--array=1-{line_count}", "slurm_array.submit"])


import json

def make_export_lines(extra_args):
    if not extra_args:
        return ""
    return "\n".join(
        f'export {k}={json.dumps(v)}'
        for k, v in extra_args.items()
    )

def make_papermill_args(params):
    if not params:
        return ""
    return " ".join(
        f"-p {k} {json.dumps(v)}"
        for k, v in params.items()
    )


import os
import subprocess

def submit_runs(
    rdir,
    job_name,
    analysisonly=False,
    cores=1,
    time="30:00:00",
    mem="30G",
    lmp_path="/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/lammps_molid/lammps/build/lmp",
    analysis_script="/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/2__synthase_setup/2__vary_potential_size/filament_analysis.ipynb",
    env_setup="/nfs/scistore26/saricgrp/fhorvath/miniforge3/etc/profile.d",
    dontsubmit=False,
    additional_analysis=None,
    analyzefilaments=True,
):
    """
    Create and submit a SLURM array job for running LAMMPS simulations and/or
    post-processing analyses across multiple run directories.

    Each SLURM array task operates in a single run directory read from `rdir`.
    The job can optionally run a simulation step, followed by one or more
    analysis steps executed via Papermill (for notebooks) or Python.

    Parameters
    ----------
    rdir : str
        Path to a text file containing one run directory per line. Each line
        corresponds to one SLURM array task.

    job_name : str
        Name of the SLURM job.

    analysisonly : bool, default False
        If True, skip the LAMMPS simulation step and only run analysis.

    cores : int, default 2
        Number of CPU cores per SLURM task.

    time : str, default "30:00:00"
        Walltime limit for each SLURM task.

    mem : str, default "30G"
        Memory allocation per task.

    lmp_path : str
        Path to the LAMMPS executable.

    analysis_script : str
        Path to the default filament analysis Jupyter notebook.

    env_setup : str
        Path to the conda/mamba environment initialization scripts.

    dontsubmit : bool, default False
        If True, write the SLURM submission script but do not submit it.

    additional_analysis : list, optional
        Additional analysis steps to run after the default analysis.
        Each entry may be either:

        - A string (backward compatible), interpreted as a command fragment,
          e.g. "analysis.ipynb -p foo 1" or "script.py --flag".

        - A dict with the following optional keys:
            * path (str): Path to a .ipynb or .py file (required).
            * env (str): Conda environment to activate (default: "filaments").
            * papermill_params (dict): Parameters passed via papermill `-p`
              (notebooks only).
            * extra_args (dict): Environment variables exported before execution.
            * cli_args (str): Raw command-line arguments appended verbatim.

    analyzefilaments : bool, default True
        If True, run the default filament analysis notebook before any
        additional analysis steps.

    Notes
    -----
    - Analysis type is inferred from file extension:
        * `.ipynb` → executed with `papermill`
        * `.py` → executed with `python`
    - The parameters `runfold` and `rundir` are always passed to notebooks.
    - Environment variables in `extra_args` are JSON-encoded and must be
      decoded inside Python using `json.loads(os.getenv(...))`.
    - Each analysis step is skipped if an ERROR is detected in `log.txt`.

    Returns
    -------
    None
        Writes a SLURM submission script and optionally submits it.
    """

    os.makedirs("logs", exist_ok=True)

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/array_job_%A_task_%a.log
#SBATCH -c {cores}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

dir=$(sed "${{SLURM_ARRAY_TASK_ID}}q;d" {rdir})
#SBATCH --workdir $dir

echo "Running in directory: $dir"
cd $dir
"""

    if not analysisonly:
        script_content += f"""
touch starting_lammps.txt
srun {lmp_path} -i config.sh
"""

    script_content += """
# Check for ERROR in log.txt before running analysis
if grep -q "ERROR" log.txt; then
    echo "Skipping analysis due to ERROR in log.txt"
    exit 0
fi
"""

    # ---- default filament analysis ----
    if analyzefilaments:
        script_content += f"""
# Filament analysis
source {env_setup}/conda.sh
source {env_setup}/mamba.sh
eval "$(mamba shell hook --shell bash)"
mamba activate filaments
srun papermill {analysis_script} analyze_slrm.ipynb \
    -p runfold $dir \
    -p rundir $dir \
    -p num_cores {cores}
"""

    # ---- generalized additional analysis ----
    if additional_analysis:
        for analysis in additional_analysis:

            if isinstance(analysis, str):
                # Backward compatibility:
                # allow "script.ipynb -p foo bar"
                parts = analysis.split()
                path = parts[0]
                extra_cli = " ".join(parts[1:])
                env = "filaments"
                extra_args = {}
                papermill_params = {}
            else:
                path = analysis["path"]
                extra_cli = analysis.get("cli_args", "")
                env = analysis.get("env", "filaments")
                extra_args = analysis.get("extra_args", {})
                papermill_params = analysis.get("papermill_params", {})


            fname = os.path.basename(path)
            export_lines = make_export_lines(extra_args)

            pm_args = make_papermill_args(papermill_params)

            if path.endswith(".ipynb"):
                cmd = (
                    f"srun papermill {path} {fname} "
                    f"-p runfold $dir "
                    f"-p rundir $dir "
                    f"-p num_cores {cores} "
                    f"{pm_args} "
                    f"{extra_cli}"
                )
            elif path.endswith(".py"):
                cmd = f"srun python {path} {extra_cli}"
            else:
                raise ValueError(f"Unsupported analysis type: {path}")

            script_content += f"""
# Additional analysis: {fname}
source {env_setup}/conda.sh
source {env_setup}/mamba.sh
eval "$(mamba shell hook --shell bash)"
mamba activate {env}
{export_lines}
{cmd}
"""

    with open("slurm_array.submit", "w") as file:
        file.write(script_content)

    if not dontsubmit:
        line_count = sum(1 for _ in open(rdir))
        subprocess.run(["sbatch", f"--array=1-{line_count}", "slurm_array.submit"])





def submit_papermill(job_name, ipynb_file, storeoutput, rundirs_file, ram_gb=5, ncores=1, time_hours=30, envname="filaments", extra_args=""):
    """
    Generates a SLURM submission script and submits a batch job to run a Jupyter notebook via papermill.

    Parameters:
    - job_name (str): Name of the SLURM job.
    - ipynb_file (str): Path to the Jupyter Notebook (.ipynb) file to execute.
    - storeoutput (str): Directory where the executed notebooks will be saved.
    - rundirs_file (str): ("runfold") Path to a file containing a list of working directories (one per line).
    - ram_gb (int, optional): Amount of RAM requested in GB (default: 30GB).
    - time_hours (int, optional): Maximum runtime for the job in hours (default: 30 hours).
    - extra_args (str, optional): Additional arguments to pass to papermill (default: "").

    The function:
    1. Reads the `rundirs_file` to determine the number of SLURM array tasks.
    2. Creates a submission script with correct SLURM directives.
    3. Ensures necessary directories exist (`logs/` for logs and `storeoutput/` for output files).
    4. Submits the job via `sbatch`.

    Example usage:
    ```
    fct.utils.submit_papermill(
        job_name="read",
        ipynb_file="~/0__treadmilling/utils/read_xyz.ipynb",
        storeoutput="read",
        rundirs_file="rdir",
        ram_gb=10,
        ncores=1,
        time_hours=24,
        extra_args="-p delete True"
    )
    ```
    """

    submit_file = f"{job_name}.submit"
    ipynb_basename = os.path.splitext(os.path.basename(ipynb_file))[0]  # Extract filename without extension

    # Get the number of lines in rundirs_file
    try:
        with open(rundirs_file, "r") as f:
            max_index = sum(1 for _ in f)
        if max_index == 0:
            raise ValueError(f"Error: {rundirs_file} is empty!")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {rundirs_file} not found!")

    # Ensure necessary directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs(storeoutput, exist_ok=True)

    script_content = f"""#!/bin/bash
#SBATCH --array=1-{max_index}
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%A_task_%a.log
#SBATCH -c {ncores}
#SBATCH --time={time_hours}:00:00
#SBATCH --mem={ram_gb}G
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /nfs/scistore26/saricgrp/fhorvath/miniforge3/etc/profile.d/conda.sh
source /nfs/scistore26/saricgrp/fhorvath/miniforge3/etc/profile.d/mamba.sh
eval "$(mamba shell hook --shell bash)"  # Add this line
mamba activate {envname}

rundir=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {rundirs_file})

echo "Running analysis for directory: $rundir"

srun papermill {ipynb_file} {storeoutput}/{ipynb_basename}_${{SLURM_ARRAY_TASK_ID}}.ipynb --start-timeout 300 \
            -p runfold "$rundir" -p rundir "$rundir" {extra_args}
"""

    # Write the submission script
    with open(submit_file, "w") as f:
        f.write(script_content)
    
    print(f"Submission script '{submit_file}' created.")

    # Submit the job
    command = ["sbatch", submit_file]
    print("Executing command:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"Job '{job_name}' submitted.")

# Example usage:
# submit_papermill("calc_lengths", "calc_filament_lengths.ipynb", "STOREOUTPUT", "rundirs", ram_gb=30, time_hours=30, extra_args="-p some_param 42")



import os
import subprocess

def submit_python(job_name, py_file, rundirs_file, ram_gb=5, ncores=1, time_hours=30, envname="filaments", extra_args=None):
    """
    Generates a SLURM submission script and submits a batch job to run a Python script via srun.

    Parameters:
    - job_name (str): Name of the SLURM job.
    - py_file (str): Path to the Python (.py) file to execute.
    - rundirs_file (str): Path to a file containing a list of working directories (one per line).
    - ram_gb (int, optional): Amount of RAM requested in GB (default: 30GB).
    - ncores (int, optional): Number of cores (default: 1).
    - time_hours (int, optional): Maximum runtime in hours (default: 30).
    - envname (str, optional): Conda environment name to activate (default: "filaments").
    - extra_args (dict, optional): Dictionary of extra environment variables to pass to the script.

    Example usage:
        submit_python(
            job_name="run_analysis",
            py_file="analyze.py",
            rundirs_file="rdirs.txt",
            ram_gb=10,
            ncores=2,
            time_hours=24,
            extra_args={"param1": 42, "param2": 0.5}
        )
    """

    extra_args = extra_args or {}
    submit_file = f"{job_name}.submit"
    py_basename = os.path.splitext(os.path.basename(py_file))[0]

    # Count lines in rundirs_file to define array
    try:
        with open(rundirs_file, "r") as f:
            max_index = sum(1 for _ in f)
        if max_index == 0:
            raise ValueError(f"Error: {rundirs_file} is empty!")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {rundirs_file} not found!")

    os.makedirs("logs", exist_ok=True)

    # Prepare extra environment variables
    export_lines = "\n".join([f'export {k}="{v}"' for k, v in extra_args.items()])

    script_content = f"""#!/bin/bash
#SBATCH --array=1-{max_index}
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%A_task_%a.log
#SBATCH -c {ncores}
#SBATCH --time={time_hours}:00:00
#SBATCH --mem={ram_gb}G
#SBATCH --no-requeue
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /nfs/scistore26/saricgrp/fhorvath/miniforge3/etc/profile.d/conda.sh
source /nfs/scistore26/saricgrp/fhorvath/miniforge3/etc/profile.d/mamba.sh
eval "$(mamba shell hook --shell bash)"
mamba activate {envname}

rundir=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {rundirs_file})

echo "Running analysis for directory: $rundir"

{export_lines}
export rundir="$rundir"

srun python -u {py_file}
"""

    with open(submit_file, "w") as f:
        f.write(script_content)
    
    print(f"Submission script '{submit_file}' created.")
    subprocess.run(["sbatch", submit_file], check=True)
    print(f"Job '{job_name}' submitted.")


import gzip
import pickle
def compress_pickle(obj, filename):
    with gzip.open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def decompress_pickle(filepath):
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)

from pathlib import Path
import time
def modified_within(file_path, minutes=5):
    file = Path(file_path)
    if not file.exists():
        return False
    
    mtime = file.stat().st_mtime  # modification timestamp
    age_seconds = time.time() - mtime
    return age_seconds < minutes * 60

def safe_write_df(df, rundir):
    final = Path(rundir) / "df.pkl.gz"
    tmp = Path(rundir) / "df.pkl.tmp"

    # Write to temp file
    compress_pickle(df, tmp)

    # Force write to disk (optional but safer)
    os.sync()

    # Atomically replace
    tmp.replace(final)  # atomic on POSIX systems

from .analysis import read_xyz
def load(rundir, dontwritedf=False):
    import pyarrow.feather as feather
    import os
    
    if os.path.exists(rundir + "/df.pkl.gz"):
        return decompress_pickle(rundir + "/df.pkl.gz")   
    elif os.path.exists(rundir + "/output.feather"):
        return feather.read_feather(rundir + "/output.feather")
        
    elif os.path.exists(rundir + "/_df.pkl.gz"):
        return decompress_pickle(rundir + "/_df.pkl.gz")
    
    elif os.path.exists(rundir + "/output.feather.gz"):
        with gzip.open(rundir + "/output.feather.gz", "rb") as f:
            return feather.read_feather(f)
    elif os.path.exists(rundir + "/output.xyz"):
        df = read_xyz(rundir)
        if not modified_within(f"{rundir}/output.xyz", minutes=5):
            if not dontwritedf: ## write out compressed output file unless suppressed
                safe_write_df(df, rundir)
        return df

    else:
        print("cannot find output files")

def find_files_recentlyunchanged(root_dir=".", minutes=10, fname="output.xyz", exclude_fname=None):
        import os
        import time
        
        threshold_sec = minutes * 60
        now = time.time()
        result = []

        for dirpath, _, filenames in os.walk(root_dir):
            if fname in filenames:
                if exclude_fname and exclude_fname in filenames:
                    continue  # skip this directory
                
                filepath = os.path.join(dirpath, fname)
                mtime = os.path.getmtime(filepath)
                if now - mtime > threshold_sec:
                    result.append(os.path.abspath(dirpath))

        return result

def date_tag():
    from datetime import datetime
    date_tag = datetime.now().strftime("%Y%m%d")
    return date_tag

from collections import OrderedDict

def update_key(key, **updates):
    d = OrderedDict(key)
    d.update(updates)
    return tuple(d.items())

def is_file_a_more_recent_than_file_b(file_a_path, file_b_path):
    import os
    import datetime
    try:
        # Get the last modification time of file A
        file_a_mod_time = os.path.getmtime(file_a_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_a_path} does not exist.")
    
    try:
        # Get the last modification time of file B
        file_b_mod_time = os.path.getmtime(file_b_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_b_path} does not exist.")
    
    # Compare modification times
    return file_a_mod_time > file_b_mod_time

import os
import re
from pathlib import Path

def clean_restart_files(root_folder, subfolder_name=None, recursive=False, dry_run=False):
    """
    Removes all but the latest 'restart.<number>' files in a folder (or matching subfolders).

    Args:
        root_folder (str or Path): directory to start from
        subfolder_name (str, optional): if given, look for subfolders with this name
        recursive (bool): whether to search arbitrarily deep
        dry_run (bool): if True, don't delete anything—just print what would be done
    """
    restart_pattern = re.compile(r"^restart\.(\d+(?:\.\d+)?)$")
    root_folder = Path(root_folder)
    if not root_folder.is_dir():
        raise ValueError(f"{root_folder} is not a directory")

    # Choose folders to clean
    if subfolder_name:
        if recursive:
            folders = [p for p in root_folder.rglob(subfolder_name) if p.is_dir()]
        else:
            folders = [p for p in root_folder.iterdir() if p.is_dir() and p.name == subfolder_name]
    else:
        folders = [root_folder]

    if not folders:
        print("No matching folders found.")
        return

    for folder in folders:
        restart_files = []
        for f in folder.iterdir():
            match = restart_pattern.match(f.name)
            if match:
                try:
                    number = float(match.group(1))
                    restart_files.append((number, f))
                except ValueError:
                    continue

        if not restart_files:
            continue  # no restarts here

        restart_files.sort(key=lambda x: x[0], reverse=True)
        latest = restart_files[0]
        to_delete = restart_files[1:]

        print(f"\nIn folder: {folder}")
        print(f"  Keeping: {latest[1].name}")
        if not to_delete:
            continue

        for _, fpath in to_delete:
            if dry_run:
                print(f"  [DRY-RUN] Would delete: {fpath.name}")
            else:
                print(f"  Deleting: {fpath.name}")
                try:
                    fpath.unlink()
                except Exception as e:
                    print(f"    Failed to delete {fpath.name}: {e}")

    print("\nCleanup complete." if not dry_run else "\nDry-run complete.")


# Example usage:
# clean_restart_files("/path/to/data", dry_run=True)
# clean_restart_files("/path/to/data", subfolder_name="RESTART", recursive=True)

import numpy as np
import pandas as pd


def get_scalar_metric(D, key, metric_key_or_func):
    """
    Safely extracts a single float from D[key].
    Handles dict keys, lambdas, and array results.
    """
    if key not in D:
        return np.nan
    
    # 1. Get the raw value
    if callable(metric_key_or_func):
        val = metric_key_or_func(D, key)
    else:
        val = D[key].get(metric_key_or_func, np.nan)
    
    # 2. Flatten if it's an array/list (The "Flexibility" part)
    if isinstance(val, (list, np.ndarray, pd.Series)):
        return np.nanmean(val)
    
    # 3. Ensure it's a float or NaN
    try:
        return float(val) if val is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _normalize_value(val):
    """
    Convert a value to a numpy array for stacking purposes.
    Handles scalars, lists, arrays, Series, DataFrames.
    """
    if val is None:
        return None
    
    if isinstance(val, pd.DataFrame):
        return val.values
    elif isinstance(val, pd.Series):
        return val.values
    elif isinstance(val, (list, tuple)):
        return np.array(val)
    elif isinstance(val, np.ndarray):
        return val
    else:
        # Scalar
        try:
            return np.array([float(val)])
        except (TypeError, ValueError):
            return None


import numpy as np
from itertools import product

def key_pooling(D, base_key, metric_fct, seeds=None, pool_params=None, verbose=False):
    """
    Pools metric data across parameter combinations and computes element-wise statistics.

    Iterates through the Cartesian product of parameters in `pool_params` (or `seeds`),
    updates the `base_key` for each combination, extracts data with `metric_fct`, and
    returns the element-wise mean and standard deviation (axis=0).

    Parameters
    ----------
    D : dict
        Source dictionary or database containing the data.
    base_key : object
        Template key updated with parameter values via `update_key`.
    metric_fct : callable
        Function called as `metric_fct(D, key)` that returns an array-like or None.
    seeds : list, optional
        Default seeds `[1, 2, 3, 4, 5]` if `pool_params` is not provided.
    pool_params : dict, optional
        Dict of parameters to pool over (e.g., {'seed': [1,2], 'lr':[0.01,0.1]}).
    verbose : bool, default False
        If True, prints debug information.

    Returns
    -------
    mean_res : np.ndarray or None
        Element-wise mean across pooled data.
    std_res : np.ndarray or None
        Element-wise standard deviation.
    n_found : int
        Number of successfully collected arrays.
    """
    
    # Decide what to pool over
    if pool_params is None:
        if seeds is None:
            seeds = [1, 2, 3, 4, 5]
        pool_params = {'seed': seeds}

    collected_data = []

    # Generate combinations of parameters
    items = sorted(pool_params.items())
    param_names = [it[0] for it in items]
    param_values = [it[1] for it in items]

    if verbose:
        print(f"[DEBUG] Pooling parameters: {pool_params}")
        print(f"[DEBUG] Combinations to iterate over: {list(product(*param_values))}")

    n_attempts = 0
    n_success = 0

    for vals in product(*param_values):
        update_dict = dict(zip(param_names, vals))
        n_attempts += 1
        try:
            s_key = update_key(base_key, **update_dict)
        except Exception as e:
            # Fallback for dict-type keys
            if isinstance(base_key, dict):
                s_key = base_key.copy()
                for pname, pval in update_dict.items():
                    s_key = update_key(s_key, **{pname: pval})
            else:
                if verbose:
                    print(f"[DEBUG] Failed to update key for {update_dict}: {e}")
                continue

        if s_key in D:
            try:
                val = metric_fct(D, s_key)
                if val is not None:
                    collected_data.append(val)
                    n_success += 1
                    if verbose:
                        print(f"[DEBUG] Collected for {update_dict}: shape={np.shape(val)}")
                elif verbose:
                    print(f"[DEBUG] Got None for {s_key}")
            except Exception as e:
                if verbose:
                    print(f"[DEBUG] Metric computation failed for {s_key}: {e}")
        else:
            if verbose:
                print(f"[DEBUG] Key not found in D: {s_key}")

    if verbose:
        print(f"[DEBUG] Finished looping. Attempted={n_attempts}, Collected={n_success}")

    if not collected_data:
        if verbose:
            print("[DEBUG] No data collected.")
        return None, None, 0

    # Normalize to 1‑D numeric arrays
    normalized = [np.atleast_1d(np.asarray(v, float)) for v in collected_data if v is not None]
    if not normalized:
        if verbose:
            print("[DEBUG] No normalized data.")
        return None, None, 0

    # Handle unequal lengths by NaN-padding to the max length
    max_len = max(len(a) for a in normalized)
    padded_arrays = []
    for a in normalized:
        if len(a) < max_len:
            a = np.pad(a, (0, max_len - len(a)), constant_values=np.nan)
        padded_arrays.append(a)

    data_stack = np.vstack(padded_arrays)  # shape = (n_arrays, max_len)

    mean_res = np.nanmean(data_stack, axis=0)
    std_res = np.nanstd(data_stack, axis=0)
    n_found = len(normalized)

    if verbose:
        print(f"[DEBUG] Stacked shape: {data_stack.shape}")
        print(f"[DEBUG] Mean (first few): {mean_res[:5]}")
        print(f"[DEBUG] Std  (first few): {std_res[:5]}")

    return mean_res, std_res, n_found

def collect_trend_from_base(
    D, 
    base_key, 
    param_name, 
    param_values, 
    analysis_func, 
    pool_params=None,
    cache_name="circ_v1"
):
    """
    Varies a parameter from a base key, pools multiple runs, and returns trend data.
    
    Parameters:
    -----------
    D : dict
        The global simulation results dictionary.
    base_key : tuple or dict
        The 'anchor' key used as a template for the parameter sweep.
    param_name : str
        The parameter to vary on the X-axis (e.g., 'tauhyd').
    param_values : list
        The range of values for param_name to evaluate.
    analysis_func : callable
        Function(D, key) -> result (scalar, array, or matrix).
    pool_params : dict, optional
        Parameters to pool over (e.g., {'seed': [1,2,3,4,5]}).
        If None, no pooling is performed.
    cache_name : str
        Unique identifier to store/retrieve results to avoid re-calculation.
    
    Returns:
    --------
    x_vals : np.ndarray
        Sorted parameter values.
    y_means : np.ndarray
        Mean results (shape depends on analysis_func output).
    y_stds : np.ndarray
        Standard deviation (same shape as y_means).
    
    Examples:
    ---------
    # Simple sweep with pooling across seeds
    x, y_mean, y_std = collect_trend_from_base(
        D, base_key, 'tauhyd', [0.1, 0.2, 0.3],
        analysis_func, pool_params={'seed': [1,2,3,4,5]}
    )
    
    # Sweep with pooling over multiple parameters
    x, y_mean, y_std = collect_trend_from_base(
        D, base_key, 'tauhyd', [0.1, 0.2, 0.3],
        analysis_func, pool_params={'seed': [1,2,3], 'trial': [0,1,2]}
    )
    """
    results = []
    storage_key = f"cache_{cache_name}"
    
    for val in param_values:
        # Create key for this X-axis point
        current_key = update_key(base_key, **{param_name: val})
        
        # Define metric function that uses caching
        def cached_analysis_func(D, key):
            res = D[key].get(storage_key)
            if res is None:
                try:
                    res = analysis_func(D, key)
                    D[key][storage_key] = res
                except Exception:
                    res = np.nan
            return res
        
        # Pool across secondary parameters if specified
        if pool_params:
            mean_res, std_res, n_found = key_pooling(
                D, current_key, cached_analysis_func, pool_params=pool_params
            )
        else:
            # No pooling, just get the single result
            try:
                mean_res = cached_analysis_func(D, current_key)
                std_res = None
                n_found = 1
            except Exception:
                mean_res = np.nan
                std_res = np.nan
                n_found = 0
        
        results.append({
            'param_val': val,
            'mean': mean_res,
            'std': std_res,
            'n_found': n_found
        })
    
    # Sort by parameter value
    results.sort(key=lambda x: x['param_val'])
    
    x_vals = np.array([r['param_val'] for r in results])
    y_means = np.array([r['mean'] if r['mean'] is not None else np.nan for r in results])
    y_stds = np.array([r['std'] if r['std'] is not None else np.nan for r in results])
    
    return x_vals, y_means, y_stds


