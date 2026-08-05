import numpy as np
import functions.sPG_tracker as pgt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def d_alpha(t, tau_c, alpha,):
    diam = (1-(t/tau_c)**alpha)**(1/alpha)
    return diam
t_model = np.linspace(0,50, 1000)
tau_c = 51
alpha = 1.3

def diam_plot(D, key, ax, modelonly=False, overlay="Overlay",
              axislabels=False, width="45%", height="45%",
              legendtitle=None, coltharp_label=None, ncol=1,
              coltharp_color="k",
              inset_ylim=None,
              color=None,
              inset_yaxis_right=False,
              borderpad=3,
              show_inset=True,):
    # 1. Main Plot Logic
    t, r = np.array(D[key]["t_r"]).T
    diam_md = r * 2 * 5
    ax.plot(t / 1000 / 60, diam_md / diam_md[0], lw=3, label=overlay,
            color=color,
    )
    if axislabels:
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(r"$D(t) / D(t_0)$")
    
    if not modelonly and 't_model' in globals():
        ax.plot(t_model, d_alpha(t_model, tau_c, alpha), color=coltharp_color, label=coltharp_label,
                ls="--")

    if "H_total" in D[key] and show_inset:
        # 2. Handle the Inset (Check if it already exists to avoid duplicaates)
        # We store the inset reference in the main axes object to retrieve it later
        if not hasattr(ax, "my_inset"):
            ax.my_inset = inset_axes(ax, width=width, height=height, loc='lower left', borderpad=borderpad)
            #ax.my_inset.set_title("H_total Mean", fontsize=9)
        
        ax_ins = ax.my_inset

        # 3. Data Processing for Inset
        z_range_tuple = (-3 * 70, 3 * 70)
        strand_width_su = pgt.strand_thickness_width / 5.0
        z_min, z_max = z_range_tuple
        
        z_edges = np.arange(z_min, z_max + strand_width_su, strand_width_su)
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        z_nm = z_centers * 5.0

        # 4. Plot into the existing inset
        ax_ins.plot(z_nm, D[key]["H_total"].mean(axis=0), label=overlay,
                    color=color,
        )
        
        # Refresh legends for both
        if overlay:
            ax.legend(loc='upper right', fontsize=8, title=legendtitle, ncol=ncol)
        ax_ins.set_xlim(-300,300)
        if inset_ylim is not None:
            ax_ins.set_ylim(inset_ylim)
        if axislabels:
            ax_ins.set_xlabel("Long cell axis (nm)", fontsize=7)
            ax_ins.set_ylabel("Septum height (nm)", fontsize=7)
        # ax_ins.legend(fontsize=7)
        if inset_yaxis_right:
            ax_ins.yaxis.set_label_position("right")
            ax_ins.yaxis.tick_right()


import numpy as np
import pandas as pd
import pyarrow.feather as feather
import os
from itertools import product

def plot_center_fraction_trend(D, keys, ax, param_name, stype=8, prm=None, label=None, color=None,
                               discardbefore=0):
    """
    Plots the fraction of particles in the center (0-50) vs a specific parameter,
    pooling all other provided parameters into mean/std at each X-point.

    Parameters:
    -----------
    D : dict
        The data dictionary containing 'rundir'.
    keys : list
        List of base keys (representing the X-axis points).
    param_name : str
        The parameter name to use for the X-axis (e.g., 'epscore').
    prm : dict, optional
        Parameters to pool over (e.g., {'seed': [1,2,3], 'mdiffu': [1, 2]}).
    """
    import functions as fct  # Ensure this is imported for key updates
    def _get_frac(k):
        # Path handling as per your environment
        fname = D[k]["rundir"] + "/df_synth.feather" # .replace("data", "bak")
        if not os.path.exists(fname): 
            return None
        
        df = feather.read_feather(fname, columns=["type", "time", "y"])
        df = df[df["time"]>discardbefore]
        data = df[df["type"] == stype]["y"].abs()
        total = len(data)
        if total == 0: 
            return None
        
        # Consistent bins for Center vs Periphery
        counts, _ = np.histogram(data, bins=[0, 50, 300])
        return counts[0] / total

    # Dictionary to group all found fractions by their X-axis value
    # This prevents "mdiffu" values from appearing as separate steps on the X-axis
    x_data_groups = {}

    for base_key in keys:
        # Extract the X-axis value (e.g., the current epscore)
        p_val = dict(base_key).get(param_name)
        
        if p_val not in x_data_groups:
            x_data_groups[p_val] = []
        
        if prm is None:
            # Single run mode
            f = _get_frac(base_key)
            if f is not None:
                x_data_groups[p_val].append(f)
        else:
            # Generalized Pooling Mode: Iterate over combinations of prm
            items = sorted(prm.items())
            p_keys = [item[0] for item in items]
            p_values = [item[1] for item in items]
            
            for vals in product(*p_values):
                update_dict = dict(zip(p_keys, vals))
                # Update the key with pooling params (seed, mdiffu, etc.)
                k = fct.utils.update_key(base_key, **update_dict)
                
                if k in D:
                    f = _get_frac(k)
                    if f is not None:
                        x_data_groups[p_val].append(f)

    # Process grouped data into means and standard deviations
    final_x = []
    final_y = []
    final_err = []

    for p_val, fracs in x_data_groups.items():
        if fracs:
            final_x.append(p_val)
            final_y.append(np.mean(fracs))
            final_err.append(np.std(fracs))

    if not final_x:
        print(f"No data found for stype {stype} with param {param_name}")
        return

    # Sort numerically by the parameter value
    idx = np.argsort(final_x)
    x_sorted = np.array(final_x)[idx]
    y_sorted = np.array(final_y)[idx]
    err_sorted = np.array(final_err)[idx]

    # Plot using an ordinal X-axis (0, 1, 2...) for clean spacing
    x_positions = range(len(x_sorted))
    ax.errorbar(x_positions, y_sorted, yerr=err_sorted, fmt='o-', 
                capsize=4, label=label, color=color, lw=1.5)
    
    # Label the X-axis with the actual parameter values
    ax.set_xticks(x_positions)
    # ax.set_xticklabels([f"{round(v)}" for v in x_sorted])
    ax.set_xticklabels([f"{v}" for v in x_sorted])
    
    ax.set_xlabel(param_name)
    ax.set_ylabel(f'Fraction in Center (stype {stype})')
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.3)


def process_synth_to_fractions(rundir):
    fname = os.path.join(rundir, "df_synth.feather")
    out_name = os.path.join(rundir, "center_frac_timeseries.feather")
    if not os.path.exists(fname): return

    df = feather.read_feather(fname, columns=["time", "type", "y"])
    df["is_center"] = df["y"].abs() <= 50  # Boolean: True if in center
    
    # Calculate fraction of 'True' values per time and type
    summary = df.groupby(["time", "type"])["is_center"].mean().unstack(level="type")
    
    # Save with LZ4 compression
    feather.write_feather(summary.reset_index(), out_name, compression='lz4')
    # print("Saved center fraction timeseries to:", out_name, "for key:", rundir  )
import functions as fct
def plot_center_fraction_vs_time(D, key, ax, stype=8, prm=None, label=None, color=None):
    
    """
    Plots the fraction of particles in center vs time, pooling across parameters.
    """
    def _get_timeseries(k):
        # Look for the slim summary file
        path = D[k]["rundir"]+ "/center_frac_timeseries.feather" # .replace("data", "bak") 
        if not os.path.exists(path): 
            print("not found:", path)
            return None
        
        df = feather.read_feather(path).set_index("time")
        df.columns = df.columns.astype(int)
        
        if stype not in df.columns: return None
        return df[stype]

    all_series = []

    if prm is None:
        s = _get_timeseries(key)
        if s is not None: all_series.append(s)
    else:
        # Generate all combinations for pooling (seed, mdiffu, etc.)
        items = sorted(prm.items())
        p_keys, p_vals = [it[0] for it in items], [it[1] for it in items]
        
        for vals in product(*p_vals):
            update_dict = dict(zip(p_keys, vals))
            k = fct.utils.update_key(key, **update_dict)
            if k in D:
                s = _get_timeseries(k)
                if s is not None: all_series.append(s)

    if not all_series:
        print(f"No time-series data found for {label}")
        return

    # Combine all series into one DataFrame (aligned by time index)
    combined = pd.concat(all_series, axis=1)
    
    mean_ts = combined.mean(axis=1)
    std_ts = combined.std(axis=1)
    time = mean_ts.index

    # Plot mean line
    line, = ax.plot(time, mean_ts, label=label, color=color, lw=2)
    
    # Plot shaded error region (Std Dev)
    ax.fill_between(time, mean_ts - std_ts, mean_ts + std_ts, 
                    color=line.get_color(), alpha=0.2, lw=0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Fraction recruited to the ring")
    # ax.set_ylim(0, 1.05)
    # ax.grid(True, alpha=0.3)


import numpy as np
import pandas as pd
import pyarrow.feather as feather
import os

def plot_nr_processive_pooled(D, key, ax, stype=11, prm=None, overlay=None, color=None):
    """
    Plots counts over time, storing the result in D[key] for instant replotting.
    """
    
    # Check if we have already calculated and stored this specific summary
    # We use a sub-key that includes the stype to avoid conflicts
    storage_key = f"summary_type_{stype}"
    
    if storage_key in D[key]:
        # DATA RECOVERY: Load from memory
        mean_counts = D[key][storage_key]['mean']
        std_counts = D[key][storage_key]['std']
        time = mean_counts.index
    else:
        # DATA PROCESSING: Load from disk
        def _get_counts(k):
            path = D[k]["rundir"] + "/df_synth.feather"
            if not os.path.exists(path): return None
            df = feather.read_feather(path, columns=["time", "type"])
            return df[df["type"] == stype].groupby("time").size()

        all_series = []
        seeds = prm.get('seed', [1]) if prm else [1]
        
        for s_val in seeds:
            k = fct.utils.update_key(key, seed=s_val)
            if k in D:
                counts = _get_counts(k)
                if counts is not None:
                    counts.name = s_val
                    all_series.append(counts)

        if not all_series: return

        # Align, Fill, and Sort (to prevent the straight line issue)
        df_pooled = pd.concat(all_series, axis=1).fillna(0).sort_index()
        
        mean_counts = df_pooled.mean(axis=1)
        std_counts = df_pooled.std(axis=1)
        time = mean_counts.index

        # STORAGE: Save into the dictionary for next time
        D[key][storage_key] = {
            'mean': mean_counts,
            'std': std_counts
        }

    # Plotting
    line, = ax.plot(time/60, mean_counts, label=overlay, color=color, lw=2)
    ax.fill_between(time/60, 
                    mean_counts - std_counts, 
                    mean_counts + std_counts, 
                    color=line.get_color(), alpha=0.2, lw=0)

    ax.set_ylabel(f"Nr of processive particles")
    if overlay:
        ax.legend()