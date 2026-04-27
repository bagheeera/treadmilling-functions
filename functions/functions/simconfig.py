from copy import deepcopy
import os

def build_config(base: dict, *overrides: dict) -> dict:
    """Return a merged copy of base + any number of overrides (later wins)."""
    result = deepcopy(base)
    for override in overrides:
        result.update(override)
    return result



base_values = {
    # ---- Basic simulation configuration ----
    "read_data": "configuration.txt",
    "outputBonds": False,  # Flag to enable bond output

    # ---- Runtime and timing parameters ----
    "tstep": 0.001,     # 0.001,    # Time step size
    "run_time": 3000.0,    # Total simulation time
    "stab_steps": 1,       # Stabilization steps before main run
    "dT": 10000,          # Steps between septum coverage evaluations
    #"threshold": 0.75,     # Fraction of filled bins to trigger constriction
    "frame_rate": 1,     # Frequency of output frames
    "switchtime": 150,      # Switching timescale for events
    "modtime": 0.0,        # Modifier time offset
    "seed": 987987,        # Random seed for reproducibility

    # ---- System size and geometry ----
    "Lx": 200,  # Box size in x-direction
    "Ly": 90,   # Box size in y-direction

    # ---- Particle and reaction counts ----
    "n_synthases": 700,
    "nsynth": 700,          # Redundant with n_synthases
    "n_activating": 600,    # Number of activator particles
    "passive_type_initial": 10,

    # ---- Interaction and physical parameters ----
    "ron": 8.0,    # Outer cutoff for potential
    "rdis": 1.0,   # Characteristic distance for interactions
    "rnuc": 5.0,   # Nucleation radius
    "Kbond": 1000.0,      # Bond spring constant
    "KbondPrimed": 5.0,   # Bond spring constant for primed complexes
    "Kbend": 1000.0,      # Bending stiffness
    "Kobst": 100.0,       # Obstacle stiffness
    "tauhyd": 1.0,        # Hydrodynamic parameter
    "fAlignCmplx": 1000,  # Alignment force when forming complex
    "fAlignPrimed": 100,    # Alignment force for primed enzymes
    "fCurv": 0.5,           # Curvature modulation factor
    "fSwim": 0.0,         # Active swim force
    "addforce": False,    # Enable extra force flag
    "fsynth": 50,         # Synthesis force
    "saturate": 15000,    # Saturation threshold
    "rc": 0.8,            # Reaction cutoff
    "activationrange": 1.5,
    "range": 0.8,
    "complex_distance_cutoff": 1.2,  # Max distance for complex formation
    "epscore": 4,    # Effective energy parameter
    "eps": 11,       # LJ interaction strength
    "profW": 10,     # Results in better transport per notebook:
                     # https://jupyterhub.ista.ac.at/user/fhorvath/lab/workspaces/auto-7/tree/0__treadmilling/B__valency/D__Zring/4__newmodel/Zrng_newmodel.ipynb

    # ---- Molecular types ----
    "act_ptype": 8,
    "complex_ptypeA": 11,  # Synthase
    "complex_ptypeB": 12,  # Activator
    "intermediate_ptypeA": 13,
    "intermediate_ptypeB": 14,
    "primed_ptypeA": 15,
    "primed_ptypeB": 16,
    # "primed_ptypeAunbound": 17,
    # "primed_ptypeBunbound": 18,
    "topdir": os.getcwd(),

    # ---- Reaction probabilities and timings ----
    "treact": 0.1,     # Reaction timescale
    "tnuc": 0.1,       # Nucleation timescale
    "ratesratio": 1.0, # Scaling ratio for rate constants
    "pdeact": 0.5,     # Deactivation probability
    "pact": 0.01,      # Activation probability
                       # tuned to match Kevin's lifetimes, see:
                       # https://jupyterhub.ista.ac.at/user/fhorvath/lab/workspaces/auto-K/tree/0__treadmilling/5__processive_states/6__FtsN_newmodel/N_newmodel.ipynb
    "StabStepsPrime": 10,  # Stabilization steps for priming
    "prefresh": 0.05,  # Refreshing reaction probability
    "pcomplexformation": 0.01,  # Complex formation probability
    "pprime": 0.003,  # Priming probability


    # ---- Flags and modes ----
    "arrt": False,
    "calcIE": False,
    "attracted_synth_type": 6,

    # ---- Mobility and dynamic parameters ----
    "mZ": 1,
    "mdiffu": 0.01,   # Diffusion coefficient for diffusive states
    "mprocess": 10,   # Mobility for processive state
    "mprimed": 10,    # Mobility for primed state
    # "mprimed_unbound": 10, # Mobility for unbound primed state

    # ---- Reactions (paths to external files) ----
    "synthreactions": "/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/5__processive_states/8__FtsN_newmodel_eitherdir/A/reactions/",
    "reactions": "/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/Reactions_explicit/",
    "complex_reactions": "/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/0__ring/I__activcomplexes/A/reactions/",
    "complex_reactions_rendered": f"{os.getcwd()}/complex_reactions/",

    # ---- Geometry bins for constriction ----
    "Nangularbins": 200,  # Number of circumferential bins
    "strandheight": 3.0
}
# print(os.getcwd())
import subprocess
git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
base["git_hash"] = git_hash
    