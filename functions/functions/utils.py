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

import os
import re
import json
import shlex
import subprocess
import textwrap


def _make_export_lines(extra_args):
    """
    Convert a dict into bash export lines.

    Example:
        {"foo": 3, "bar": [1, 2]}

    becomes:
        export foo='3'
        export bar='[1, 2]'

    Values are JSON-encoded so Python scripts can later decode them with
    json.loads(os.getenv("foo")) if needed.
    """
    if not extra_args:
        return ""

    lines = []

    for key, value in extra_args.items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise ValueError(f"Invalid environment variable name: {key}")

        encoded = json.dumps(value)
        lines.append(f"export {key}={shlex.quote(encoded)}")

    return "\n".join(lines)


def _make_papermill_args(params):
    """
    Convert a dict into papermill -p arguments.

    Example:
        {"foo": 1, "bar": "abc"}

    becomes:
        -p foo 1 -p bar abc
    """
    if not params:
        return ""

    args = []

    for key, value in params.items():
        args.extend(["-p", str(key), str(value)])

    return shlex.join(args)


def _make_cli_args(cli_args):
    """
    Convert additional CLI args into a safely quoted string.

    Accepts either:
        "--flag value"
    or:
        ["--flag", "value"]
    """
    if not cli_args:
        return ""

    if isinstance(cli_args, str):
        parts = shlex.split(cli_args)
    else:
        parts = list(map(str, cli_args))

    return shlex.join(parts)


def submit_runs(
    rdir,
    job_name,
    analysisonly=False,
    cores=1,
    time="30:00:00",
    mem="30G",
    lmp_path="/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/lammps_molid/lammps/build/lmp",
    analysis_script="/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/2__synthase_setup/2__vary_potential_size/filament_analysis.ipynb",
    env_setup=None,
    dontsubmit=False,
    additional_analysis=None,
    analyzefilaments=True,
    load_mamba_env=True,
    env_name="filaments",
    env_runner="/nfs/scistore26/saricgrp/fhorvath/miniforge3/bin/mamba",
    env_prefix=None,
    setup_commands="",
):
    """
    Create and optionally submit a SLURM array job for LAMMPS runs and analyses.

    Important environment behavior:
    - This version does NOT source hard-coded conda.sh or mamba.sh files.
    - Instead it runs analysis commands via:

          micromamba run -n ENV ...
          mamba run -n ENV ...
          conda run -n ENV ...

      or, if env_prefix is given:

          micromamba run -p /path/to/env ...

    Parameters
    ----------
    rdir : str
        Text file with one run directory per non-empty line.

    job_name : str
        SLURM job name.

    analysisonly : bool
        If True, skip the LAMMPS simulation and only run analysis.

    cores : int
        CPU cores per SLURM task.

    time : str
        SLURM walltime, e.g. "30:00:00".

    mem : str
        SLURM memory, e.g. "30G".

    lmp_path : str
        Path to LAMMPS executable.

    analysis_script : str
        Default filament analysis notebook.

    env_setup : str or None
        Deprecated. Kept only for backwards compatibility.
        Prefer `env_runner`, `env_prefix`, and `setup_commands`.

    dontsubmit : bool
        If True, write the submit script but do not call sbatch.

    additional_analysis : list or None
        Additional analysis steps.

        String form:
            "analysis.ipynb -p foo 1"
            "script.py --flag value"

        Dict form:
            {
                "path": "analysis.ipynb",
                "env": "filaments",
                "env_prefix": "/optional/full/env/path",
                "papermill_params": {"foo": 1},
                "extra_args": {"MY_VAR": [1, 2, 3]},
                "cli_args": "--flag value",
            }

    analyzefilaments : bool
        If True, run the default filament analysis notebook.

    load_mamba_env : bool
        If True, run analysis commands inside the selected env.
        If False, just run papermill/python directly.

    env_name : str
        Default environment name.

    env_runner : str
        "auto", "micromamba", "mamba", "conda", or a full path.

    env_prefix : str or None
        Optional full path to the default environment.

    setup_commands : str
        Optional shell setup before finding micromamba/mamba/conda.

        Examples:
            setup_commands='module load micromamba'
            setup_commands='export PATH="/path/to/miniforge3/bin:$PATH"'
    """

    os.makedirs("logs", exist_ok=True)

    submit_file = "slurm_array.submit"

    # Make important paths absolute so the SLURM job does not depend on
    # the directory from which it starts.
    rdir = os.path.abspath(os.path.expanduser(rdir))
    lmp_path = os.path.abspath(os.path.expanduser(lmp_path))
    analysis_script = os.path.abspath(os.path.expanduser(analysis_script))

    if env_prefix is not None:
        env_prefix = os.path.abspath(os.path.expanduser(env_prefix))
    else:
        env_prefix = ""

    # Count non-empty run-directory lines.
    # Each non-empty line becomes one SLURM array task.
    try:
        with open(rdir, "r") as f:
            line_count = sum(1 for line in f if line.strip())

        if line_count == 0:
            raise ValueError(f"Error: {rdir} is empty!")

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {rdir} not found!")

    # This block is inserted into the SLURM script.
    # It finds the environment runner inside the batch job.
    #
    # With --export=NONE, your interactive PATH is not automatically inherited,
    # so use setup_commands or a full env_runner path if needed.
    if load_mamba_env:
        if env_runner == "auto":
            runner_block = r"""
if command -v micromamba >/dev/null 2>&1; then
    ENV_RUNNER="$(command -v micromamba)"
elif command -v mamba >/dev/null 2>&1; then
    ENV_RUNNER="$(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
    ENV_RUNNER="$(command -v conda)"
else
    echo "ERROR: could not find micromamba, mamba, or conda in PATH." >&2
    echo "Use setup_commands='module load ...' or env_runner='/full/path/to/mamba'." >&2
    exit 1
fi
"""
        else:
            runner_block = f"""
ENV_RUNNER={shlex.quote(env_runner)}

if [[ "$ENV_RUNNER" == */* ]]; then
    if [[ ! -x "$ENV_RUNNER" ]]; then
        echo "ERROR: env runner is not executable: $ENV_RUNNER" >&2
        exit 1
    fi
else
    if ! command -v "$ENV_RUNNER" >/dev/null 2>&1; then
        echo "ERROR: env runner not found: $ENV_RUNNER" >&2
        exit 1
    fi
    ENV_RUNNER="$(command -v "$ENV_RUNNER")"
fi
"""
    else:
        runner_block = """
ENV_RUNNER=""
"""

    script_content = f"""#!/bin/bash -l
#SBATCH --array=1-{line_count}
#SBATCH --job-name={job_name}
#SBATCH --output=logs/array_job_%A_task_%a.log
#SBATCH -c {cores}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail

unset SLURM_EXPORT_ENV

# Limit numerical libraries to the number of cores requested from SLURM.
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

# Optional cluster-specific setup.
# Examples:
#   module load micromamba
#   export PATH="/path/to/miniforge3/bin:$PATH"
{setup_commands}

{runner_block}

LOAD_MAMBA_ENV={1 if load_mamba_env else 0}

RDIR_FILE={shlex.quote(rdir)}
LMP_PATH={shlex.quote(lmp_path)}
DEFAULT_ANALYSIS_SCRIPT={shlex.quote(analysis_script)}
DEFAULT_ENV_NAME={shlex.quote(env_name)}
DEFAULT_ENV_PREFIX={shlex.quote(env_prefix)}

# Run a command either directly or inside a mamba/micromamba/conda environment.
#
# Usage:
#   srun_in_env ENV_NAME ENV_PREFIX command arg1 arg2 ...
#
# If ENV_PREFIX is non-empty, it uses:
#   micromamba run -p ENV_PREFIX command ...
#
# Otherwise it uses:
#   micromamba run -n ENV_NAME command ...
srun_in_env() {{
    local env_name="$1"
    shift

    local env_prefix="$1"
    shift

    if [[ "$LOAD_MAMBA_ENV" == "1" ]]; then
        if [[ -n "$env_prefix" ]]; then
            srun "$ENV_RUNNER" run -p "$env_prefix" "$@"
        else
            srun "$ENV_RUNNER" run -n "$env_name" "$@"
        fi
    else
        srun "$@"
    fi
}}

# Select the Nth non-empty line from the run-directory file.
dir="$(awk -v task="${{SLURM_ARRAY_TASK_ID}}" 'NF {{ n++; if (n == task) {{ print; exit }} }}' "$RDIR_FILE")"

if [[ -z "$dir" ]]; then
    echo "ERROR: empty run directory for task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "SLURM job ID: $SLURM_JOB_ID"
echo "SLURM array task: $SLURM_ARRAY_TASK_ID"
echo "Running in directory: $dir"

cd "$dir"
"""

    if not analysisonly:
        script_content += f"""
touch starting_lammps.txt

echo "Starting LAMMPS"
srun "$LMP_PATH" -i config.sh
"""

    script_content += """
# If LAMMPS produced an ERROR in log.txt, skip analysis.
if [[ -f log.txt ]] && grep -q "ERROR" log.txt; then
    echo "Skipping analysis because ERROR was found in log.txt"
    exit 0
fi
"""

    # Default filament analysis notebook.
    if analyzefilaments:
        script_content += f"""
echo "Running default filament analysis"

srun_in_env "$DEFAULT_ENV_NAME" "$DEFAULT_ENV_PREFIX" \\
    papermill "$DEFAULT_ANALYSIS_SCRIPT" analyze_slrm.ipynb \\
    -p runfold "$dir" \\
    -p rundir "$dir" \\
    -p num_cores {cores}
"""

    # Additional analysis steps.
    if additional_analysis:
        for analysis in additional_analysis:
            if isinstance(analysis, str):
                # Backward-compatible form:
                #   "notebook.ipynb -p foo 1"
                #   "script.py --flag value"
                parts = shlex.split(analysis)

                if not parts:
                    continue

                path = parts[0]
                cli_args = parts[1:]
                env = env_name
                analysis_env_prefix = env_prefix
                extra_args = {}
                papermill_params = {}

            else:
                path = analysis["path"]
                cli_args = analysis.get("cli_args", "")
                env = analysis.get("env", env_name)

                # Allows per-analysis env prefix.
                # If not given, reuse the default env_prefix.
                analysis_env_prefix = analysis.get("env_prefix", env_prefix)

                if analysis_env_prefix is None:
                    analysis_env_prefix = ""

                extra_args = analysis.get("extra_args", {})
                papermill_params = analysis.get("papermill_params", {})

            path = os.path.expanduser(path)
            fname = os.path.basename(path)

            export_lines = _make_export_lines(extra_args)
            pm_args = _make_papermill_args(papermill_params)
            cli_arg_string = _make_cli_args(cli_args)

            quoted_path = shlex.quote(path)
            quoted_fname = shlex.quote(fname)
            quoted_env = shlex.quote(env)
            quoted_env_prefix = shlex.quote(analysis_env_prefix or "")

            if path.endswith(".ipynb"):
                cmd = f"""srun_in_env {quoted_env} {quoted_env_prefix} \\
    papermill {quoted_path} {quoted_fname} \\
    -p runfold "$dir" \\
    -p rundir "$dir" \\
    -p num_cores {cores}"""

                if pm_args:
                    cmd += f" \\\n    {pm_args}"

                if cli_arg_string:
                    cmd += f" \\\n    {cli_arg_string}"

            elif path.endswith(".py"):
                cmd = f"""srun_in_env {quoted_env} {quoted_env_prefix} \\
    python {quoted_path}"""

                if cli_arg_string:
                    cmd += f" \\\n    {cli_arg_string}"

            else:
                raise ValueError(f"Unsupported analysis type: {path}")

            script_content += f"""

echo "Running additional analysis: {fname}"

{export_lines}

{cmd}
"""

    script_content = textwrap.dedent(script_content)

    with open(submit_file, "w") as file:
        file.write(script_content)

    print(f"Submission script written to: {submit_file}")

    if not dontsubmit:
        command = ["sbatch", submit_file]
        print("Executing command:", " ".join(command))
        subprocess.run(command, check=True)
        print(f"Job '{job_name}' submitted.")



import os
import shlex
import subprocess
import textwrap


def submit_papermill(
    job_name,
    ipynb_file,
    storeoutput,
    rundirs_file,
    ram_gb=5,
    ncores=1,
    time_hours=30,
    envname="filaments",
    extra_args="",
    env_runner="/nfs/scistore26/saricgrp/fhorvath/miniforge3/bin/mamba",
    env_prefix=None,
    setup_commands="",
):
    """
    Generate and submit a SLURM array job that runs a notebook with papermill.

    Parameters
    ----------
    job_name : str
        Name of this SLURM job.
    ipynb_file : str
        Path to input notebook.
    storeoutput : str
        Directory where executed notebooks are written.
    rundirs_file : str
        File containing one run directory per line.
    ram_gb : int
        Memory in GB.
    ncores : int
        Number of CPU cores.
    time_hours : int
        Runtime limit in hours.
    envname : str
        Conda/mamba/micromamba environment name.
    extra_args : str
        Extra arguments passed to papermill, e.g. "-p delete True".
    env_runner : str
        One of "auto", "micromamba", "mamba", "conda", or an absolute path.
        "auto" tries micromamba, then mamba, then conda.
    env_prefix : str or None
        Optional full path to environment prefix. If given, uses `run -p`.
        Otherwise uses `run -n envname`.
    setup_commands : str
        Optional shell commands to run before detecting the env runner.
        Useful for clusters, e.g. "module load micromamba".
    """

    submit_file = f"{job_name}.submit"

    # Convert paths like "~" to full absolute paths.
    # This makes the SLURM script less dependent on the starting directory.
    ipynb_file = os.path.abspath(os.path.expanduser(ipynb_file))
    storeoutput = os.path.abspath(os.path.expanduser(storeoutput))
    rundirs_file = os.path.abspath(os.path.expanduser(rundirs_file))

    # Used later to make output notebook names.
    ipynb_basename = os.path.splitext(os.path.basename(ipynb_file))[0]

    # Count non-empty lines in rundirs_file.
    # Each non-empty line corresponds to one SLURM array task.
    try:
        with open(rundirs_file, "r") as f:
            max_index = sum(1 for line in f if line.strip())

        if max_index == 0:
            raise ValueError(f"Error: {rundirs_file} is empty!")

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {rundirs_file} not found!")

    # Create output/log directories before submitting.
    os.makedirs("logs", exist_ok=True)
    os.makedirs(storeoutput, exist_ok=True)

    # extra_args can be either a string:
    #     "-p delete True"
    # or a list:
    #     ["-p", "delete", "True"]
    #
    # The list form is safer if arguments contain spaces.
    if isinstance(extra_args, (list, tuple)):
        extra_args = shlex.join(map(str, extra_args))

    # Decide whether to use an environment name or an environment path.
    #
    # Name:
    #     micromamba run -n filaments ...
    #
    # Prefix/path:
    #     micromamba run -p /path/to/env ...
    if env_prefix is not None:
        env_prefix = os.path.abspath(os.path.expanduser(env_prefix))
        env_args_line = f"ENV_ARGS=(-p {shlex.quote(env_prefix)})"
    else:
        env_args_line = f"ENV_ARGS=(-n {shlex.quote(envname)})"

    # Bash code that will be inserted into the submit script.
    # It finds micromamba/mamba/conda on PATH.
    if env_runner == "auto":
        runner_block = r"""
if command -v micromamba >/dev/null 2>&1; then
    ENV_RUNNER="$(command -v micromamba)"
elif command -v mamba >/dev/null 2>&1; then
    ENV_RUNNER="$(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
    ENV_RUNNER="$(command -v conda)"
else
    echo "ERROR: could not find micromamba, mamba, or conda in PATH." >&2
    echo "Use setup_commands='module load ...' or env_runner='/full/path/to/micromamba'." >&2
    exit 1
fi
"""
    else:
        runner_block = f"""
ENV_RUNNER={shlex.quote(env_runner)}

if ! command -v "$ENV_RUNNER" >/dev/null 2>&1; then
    echo "ERROR: env runner not found: $ENV_RUNNER" >&2
    exit 1
fi

ENV_RUNNER="$(command -v "$ENV_RUNNER")"
"""

    # This is the actual SLURM/bash script that will be written to disk.
    #
    # Important:
    #   Lines beginning with # inside this string are BASH comments,
    #   not Python comments.
    #
    #   Avoid putting # comments in the middle of commands continued with "\".
    script_content = f"""#!/bin/bash
#SBATCH --array=1-{max_index}
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%A_task_%a.log
#SBATCH -c {ncores}
#SBATCH --time={time_hours}:00:00
#SBATCH --mem={ram_gb}G
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

{setup_commands}

{runner_block}

{env_args_line}

PM_EXTRA_ARGS=({extra_args})

IPYNB_FILE={shlex.quote(ipynb_file)}
STOREOUTPUT={shlex.quote(storeoutput)}
RUNDIRS_FILE={shlex.quote(rundirs_file)}
IPYNB_BASENAME={shlex.quote(ipynb_basename)}

mkdir -p "$STOREOUTPUT"

RUNDIR="$(awk -v task="${{SLURM_ARRAY_TASK_ID}}" 'NF {{ n++; if (n == task) {{ print; exit }} }}' "$RUNDIRS_FILE")"

if [ -z "$RUNDIR" ]; then
    echo "ERROR: empty rundir for task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

OUT_IPYNB="${{STOREOUTPUT}}/${{IPYNB_BASENAME}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.ipynb"

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Run directory: $RUNDIR"
echo "Input notebook: $IPYNB_FILE"
echo "Output notebook: $OUT_IPYNB"
echo "Environment runner: $ENV_RUNNER"
echo "Environment args: ${{ENV_ARGS[*]}}"

"$ENV_RUNNER" run "${{ENV_ARGS[@]}}" \\
    papermill "$IPYNB_FILE" "$OUT_IPYNB" \\
    --start-timeout 300 \\
    -p runfold "$RUNDIR" \\
    -p rundir "$RUNDIR" \\
    "${{PM_EXTRA_ARGS[@]}}"
"""

    # Remove accidental leading indentation from the generated bash script.
    script_content = textwrap.dedent(script_content)

    # Write the SLURM submit script.
    with open(submit_file, "w") as f:
        f.write(script_content)

    print(f"Submission script '{submit_file}' created.")

    # Submit the script with sbatch.
    command = ["sbatch", submit_file]
    print("Executing command:", " ".join(command))
    subprocess.run(command, check=True)

    print(f"Job '{job_name}' submitted.")
    
    


# Example usage:
# submit_papermill("calc_lengths", "calc_filament_lengths.ipynb", "STOREOUTPUT", "rundirs", ram_gb=30, time_hours=30, extra_args="-p some_param 42")




import os
import json
import shlex
import subprocess
import textwrap
import re


def submit_python(
    job_name,
    py_file,
    rundirs_file,
    ram_gb=5,
    ncores=1,
    time_hours=30,
    envname="filaments",
    extra_args=None,
    # env_runner="/nfs/scistore26/saricgrp/fhorvath/miniforge3/bin/mamba",
    env_runner="auto",
    # env_prefix=None,
    env_prefix="/nfs/scistore26/saricgrp/fhorvath/miniforge3/envs/filaments",
    setup_commands="",
    dontsubmit=False,
):
    """
    Generate a SLURM array submission script and optionally submit it with sbatch.

    Each SLURM array task reads one non-empty line from `rundirs_file`, changes
    into that directory, exports the selected run directory as environment
    variables, and runs the requested Python script with `srun`.

    The Python script is executed inside a mamba/micromamba/conda environment
    using `run`, for example:

        mamba run -n filaments python -u analyze.py

    or, if `env_prefix` is given:

        mamba run -p /path/to/env python -u analyze.py

    This avoids hard-coded `source conda.sh`, `source mamba.sh`, and
    `mamba activate ...` lines in the generated SLURM script.

    Parameters
    ----------
    job_name : str
        Name of the SLURM job. Also used to name the generated submission file
        as `{job_name}.submit` and the log files as
        `logs/{job_name}_%A_task_%a.log`.

    py_file : str
        Path to the Python script to execute. The path is expanded with
        `os.path.expanduser` and converted to an absolute path before being
        written into the SLURM script.

    rundirs_file : str
        Path to a text file containing one run directory per line. Each
        non-empty line corresponds to one SLURM array task. Empty lines are
        ignored when counting and selecting tasks.

    ram_gb : int or float, default 5
        Amount of memory requested per SLURM array task, in GB. Written to the
        submission script as `#SBATCH --mem={ram_gb}G`.

    ncores : int, default 1
        Number of CPU cores requested per SLURM task. Written as
        `#SBATCH -c {ncores}`. Also used to set common numerical-library thread
        variables such as `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
        `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`.

    time_hours : int or float, default 30
        Maximum walltime requested for the job, in hours. Written to the
        submission script as `#SBATCH --time={time_hours}:00:00`.

    envname : str, default "filaments"
        Name of the mamba/micromamba/conda environment in which the Python
        script should run. Used as:

            ENV_RUNNER run -n envname python -u py_file

        Ignored if `env_prefix` is provided.

    extra_args : dict, optional
        Dictionary of additional environment variables to export before running
        the Python script.

        Example:

            extra_args={"param1": 42, "param2": 0.5}

        will create exports in the SLURM script. Values are JSON-encoded, so
        the Python script can recover them with for example:

            import os, json
            param1 = json.loads(os.environ["param1"])

        Variable names must be valid shell environment variable names.

    env_runner : str, default "/nfs/scistore26/saricgrp/fhorvath/miniforge3/bin/mamba"
        Command or full path used to run the environment. Valid examples:

            "auto"
            "micromamba"
            "mamba"
            "conda"
            "/full/path/to/micromamba"
            "/full/path/to/mamba"

        If set to `"auto"`, the generated SLURM script searches, in order, for
        `micromamba`, `mamba`, and `conda` on `PATH`.

        Note that the script uses `#SBATCH --export=NONE`, so your interactive
        shell `PATH` may not be available inside the job. If auto-detection
        fails, either provide a full path here or use `setup_commands` to modify
        `PATH` or load a module.

    env_prefix : str or None, default None
        Optional full path to the environment prefix. If provided, the generated
        script uses:

            ENV_RUNNER run -p env_prefix python -u py_file

        instead of:

            ENV_RUNNER run -n envname python -u py_file

        This is often more robust for micromamba environments.

    setup_commands : str, default ""
        Optional shell commands inserted into the SLURM script before searching
        for `micromamba`, `mamba`, or `conda`.

        Useful on clusters where environments are made available through
        modules or manual PATH modification.

        Examples:

            setup_commands="module load micromamba"

            setup_commands='export PATH="/path/to/miniforge3/bin:$PATH"'

    dontsubmit : bool, default False
        If True, write the SLURM submission script but do not call `sbatch`.
        Useful for debugging the generated script.

    Behavior
    --------
    The function:
    1. Counts non-empty lines in `rundirs_file` to determine the SLURM array
       size.
    2. Creates a `logs/` directory if needed.
    3. Writes a SLURM submission script named `{job_name}.submit`.
    4. In each array task, selects the corresponding non-empty run directory.
    5. Changes into that run directory.
    6. Exports `rundir` and `runfold` environment variables.
    7. Exports any variables provided through `extra_args`.
    8. Runs the Python script inside the selected environment using
       `mamba run`, `micromamba run`, or `conda run`.
    9. Submits the script with `sbatch`, unless `dontsubmit=True`.

    Example usage
    -------------
    Basic usage with the default mamba path:

        submit_python(
            job_name="run_analysis",
            py_file="analyze.py",
            rundirs_file="rdirs.txt",
            ram_gb=10,
            ncores=2,
            time_hours=24,
            envname="filaments",
            extra_args={"param1": 42, "param2": 0.5}
        )

    Using automatic runner detection:

        submit_python(
            job_name="run_analysis",
            py_file="analyze.py",
            rundirs_file="rdirs.txt",
            env_runner="auto",
            envname="filaments",
        )

    Using micromamba with a full environment prefix:

        submit_python(
            job_name="run_analysis",
            py_file="analyze.py",
            rundirs_file="rdirs.txt",
            env_runner="/path/to/micromamba",
            env_prefix="/path/to/envs/filaments",
        )

    Debug without submitting:

        submit_python(
            job_name="run_analysis",
            py_file="analyze.py",
            rundirs_file="rdirs.txt",
            dontsubmit=True,
        )

    Returns
    -------
    None
        Writes a SLURM submission script and optionally submits it with sbatch.
    """

    extra_args = extra_args or {}

    submit_file = f"{job_name}.submit"

    # Make paths absolute so the SLURM script does not depend on the directory
    # from which the job starts.
    py_file = os.path.abspath(os.path.expanduser(py_file))
    rundirs_file = os.path.abspath(os.path.expanduser(rundirs_file))

    if env_prefix is not None:
        env_prefix = os.path.abspath(os.path.expanduser(env_prefix))
    else:
        env_prefix = ""

    # Count non-empty lines. Each non-empty line becomes one array task.
    try:
        with open(rundirs_file, "r") as f:
            max_index = sum(1 for line in f if line.strip())

        if max_index == 0:
            raise ValueError(f"Error: {rundirs_file} is empty!")

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {rundirs_file} not found!")

    os.makedirs("logs", exist_ok=True)

    # Convert extra_args into bash export lines.
    # Values are JSON-encoded so scripts can decode them if needed.
    export_lines = []
    for key, value in extra_args.items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise ValueError(f"Invalid environment variable name: {key}")

        encoded_value = json.dumps(value)
        export_lines.append(f"export {key}={shlex.quote(encoded_value)}")

    export_lines = "\n".join(export_lines)

    # Bash block for finding micromamba/mamba/conda inside the SLURM job.
    #
    # If env_runner="auto", try micromamba, then mamba, then conda.
    # If env_runner is a full path, check that it is executable.
    # If env_runner is just "mamba", check that it exists on PATH.
    if env_runner == "auto":
        runner_block = r"""
if command -v micromamba >/dev/null 2>&1; then
    ENV_RUNNER="$(command -v micromamba)"
elif command -v mamba >/dev/null 2>&1; then
    ENV_RUNNER="$(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
    ENV_RUNNER="$(command -v conda)"
else
    echo "ERROR: could not find micromamba, mamba, or conda in PATH." >&2
    echo "Use setup_commands='module load ...' or env_runner='/full/path/to/mamba'." >&2
    exit 1
fi
"""
    else:
        runner_block = f"""
ENV_RUNNER={shlex.quote(env_runner)}

if [[ "$ENV_RUNNER" == */* ]]; then
    if [[ ! -x "$ENV_RUNNER" ]]; then
        echo "ERROR: env runner is not executable: $ENV_RUNNER" >&2
        exit 1
    fi
else
    if ! command -v "$ENV_RUNNER" >/dev/null 2>&1; then
        echo "ERROR: env runner not found: $ENV_RUNNER" >&2
        exit 1
    fi

    ENV_RUNNER="$(command -v "$ENV_RUNNER")"
fi
"""

    script_content = f"""#!/bin/bash -l
#SBATCH --array=1-{max_index}
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%A_task_%a.log
#SBATCH -c {ncores}
#SBATCH --time={time_hours}:00:00
#SBATCH --mem={ram_gb}G
#SBATCH --no-requeue
#SBATCH --export=NONE

set -euo pipefail

unset SLURM_EXPORT_ENV

# Limit numerical libraries to the number of cores requested from SLURM.
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

# Optional cluster-specific setup.
# Examples:
#   module load micromamba
#   export PATH="/path/to/miniforge3/bin:$PATH"
{setup_commands}

# Only validate/find ENV_RUNNER if ENV_PREFIX is empty
if [[ -z {shlex.quote(env_prefix)} ]]; then
{runner_block}
fi

PY_FILE={shlex.quote(py_file)}
RUNDIRS_FILE={shlex.quote(rundirs_file)}
ENV_NAME={shlex.quote(envname)}
ENV_PREFIX={shlex.quote(env_prefix)}

# Select the Nth non-empty line from the rundirs file.
rundir="$(awk -v task="${{SLURM_ARRAY_TASK_ID}}" 'NF {{ n++; if (n == task) {{ print; exit }} }}' "$RUNDIRS_FILE")"

if [[ -z "$rundir" ]]; then
    echo "ERROR: empty rundir for task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "SLURM job ID: $SLURM_JOB_ID"
echo "SLURM array task: $SLURM_ARRAY_TASK_ID"
echo "Running in directory: $rundir"
echo "Python script: $PY_FILE"
echo "Environment runner: ${{ENV_RUNNER:-N/A}}"
echo "Environment name: $ENV_NAME"
echo "Environment prefix: $ENV_PREFIX"

cd "$rundir"

# Export the rundir so the Python script can read it with os.environ["rundir"].
export rundir="$rundir"
export runfold="$rundir"

# Extra user-provided environment variables.
{export_lines}

# Run Python inside the requested mamba/micromamba/conda environment.
if [[ -n "$ENV_PREFIX" ]]; then
    echo "running with $ENV_PREFIX/bin/python"
    srun "$ENV_PREFIX/bin/python" -u "$PY_FILE"
else
    echo "running via $ENV_RUNNER"
    srun "$ENV_RUNNER" run -n "$ENV_NAME" python -u "$PY_FILE"
fi
"""

    script_content = textwrap.dedent(script_content)

    with open(submit_file, "w") as f:
        f.write(script_content)

    print(f"Submission script '{submit_file}' created.")

    if not dontsubmit:
        command = ["sbatch", submit_file]
        print("Executing command:", " ".join(command))
        subprocess.run(command, check=True)
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


def load(rundir, dontwritedf=False):
    from .analysis import read_xyz
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


