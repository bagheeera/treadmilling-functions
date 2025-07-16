from collections import OrderedDict
import itertools

import matplotlib.pyplot as plt
import numpy as np

def plot_data_(prm_sets, D, grid_params, overlay_params, figure_params, plot_function, check_key, 
              fixed_params=None, axis_edits=None, figsize_=None, 
              sharex=True, sharey=True):
    """
    Generates a grid of subplots for each combination of parameters in `figure_params`,
    using `grid_params` to define the subplot layout and optionally overlaying data
    along a third parameter. Each subplot is populated by calling a user-defined
    `plot_function`.

    Parameters
    ----------
    prm_sets : dict
        Dictionary mapping parameter names to lists of possible values.
    D : dict
        Dictionary containing data, indexed by parameter tuples. Each key is a tuple
        of (param_name, value) pairs, e.g., (('rc', 3), ('deact', 0.1), ...).
    grid_params : list of str
        Two parameters that define the subplot grid layout (rows × columns).
    overlay_params : list of str or None
        Parameter(s) to use for overlaid plots in each subplot. Pass None or an empty list
        for no overlays.
    figure_params : list of str
        Parameters to sweep over for generating separate figures.
    plot_function : callable
        Function called for each valid subplot. Signature:
        `plot_function(key_tuple, ax, overlay_value)`
    check_key : str
        A key to look for inside `D[key_tuple]` to decide if a subplot should be drawn.
    fixed_params : dict, optional
        Dictionary of parameter-value pairs to fix for all plots (default: {}).
    axis_edits : dict, optional
        Dictionary of axis methods to call on each subplot, e.g.:
        {
            "set_ylim": [0, 1],
            "vlines": [(5, 0, 1, {"color": "red", "linestyle": "--"})],
            "set_xticks": [[0, 5, 10]],
            "axvline": [
                (0.5, {"color": "red", "linestyle": "--", "linewidth": 2}),
                (1.0, {"color": "blue", "linestyle": ":", "linewidth": 1}),
            ]
        }
    figsize_ : tuple, optional
        Size of the entire figure. If None, it will be computed automatically.
    sharex : bool, default=True
        Whether to share the x-axis across subplots.
    sharey : bool, default=True
        Whether to share the y-axis across subplots.

    Behavior
    --------
    - Creates a figure for each combination of `figure_params`.
    - Within each figure, arranges subplots in a grid defined by `grid_params`.
    - Each subplot overlays plots along `overlay_params` values.
    - Only plots if the `check_key` exists in the corresponding `D[key_tuple]`.
    - Applies optional `axis_edits` to each subplot.
    - Uses `OrderedDict` to preserve parameter order when constructing keys.

    Notes
    -----
    - The user-defined `plot_function` is responsible for formatting the plots.
    - This function assumes that `plt` (from matplotlib.pyplot) is already imported.
    """
    import matplotlib.pyplot as plt
    fixed_params = fixed_params or {}

    figure_combinations = list(itertools.product(*[prm_sets[p] for p in figure_params]))
    
    if figsize_:
        figsize = figsize_
    else:
        figsize = (4 * len(prm_sets[grid_params[1]]), 3 * len(prm_sets[grid_params[0]]))
        
    for fig_vals in figure_combinations:
        fig_param_dict = dict(zip(figure_params, fig_vals))

        # Skip combinations that contradict fixed_params
        skip = any(fig_param_dict.get(k) != v for k, v in fixed_params.items() if k in fig_param_dict)
        if skip:
            continue

        fig_title = " ".join(f"{p}={v}" for p, v in fig_param_dict.items())

        fig, ax = plt.subplots(len(prm_sets[grid_params[0]]), 
                               len(prm_sets[grid_params[1]]), 
                               sharex=sharex, 
                               sharey=sharey, 
                               figsize=figsize)
        
        fig.subplots_adjust(wspace=.4)
        fig.suptitle(fig_title)

        for i, grid1 in enumerate(prm_sets[grid_params[0]]):
            for j, grid2 in enumerate(prm_sets[grid_params[1]]):
                subplot_ax = ax[i, j] if len(prm_sets[grid_params[0]]) > 1 else ax[j]

                overlay_values = prm_sets[overlay_params[0]] if overlay_params else [None]

                for overlay in overlay_values:
                    key_dict = OrderedDict()
                    key_dict[grid_params[0]] = grid1
                    key_dict[grid_params[1]] = grid2
                    if overlay_params:
                        key_dict[overlay_params[0]] = overlay
                    for p, v in fig_param_dict.items():
                        key_dict[p] = v
                    for p, v in fixed_params.items():
                        key_dict[p] = v

                    key_tuple = tuple(sorted(key_dict.items()))

                    if key_tuple in D and check_key in D[key_tuple]:
                        plot_function(key_tuple, subplot_ax, overlay)

                if j == 0:
                    subplot_ax.annotate(f"{grid_params[0]}={grid1}", xy=(-0.3, 0.5), 
                                        xycoords="axes fraction",
                                        ha="right", va="center", fontsize=10, fontweight="bold")
                if i == 0:
                    subplot_ax.annotate(f"{grid_params[1]}={grid2}", xy=(0.5, 1.1), 
                                        xycoords="axes fraction",
                                        ha="center", va="bottom", fontsize=10, fontweight="bold")

                if axis_edits:
                    for prop, value in axis_edits.items():
                        method = getattr(subplot_ax, prop, None)

                        if callable(method):
                            if isinstance(value, list) and all(isinstance(v, tuple) for v in value):
                                for args in value:
                                    if isinstance(args[-1], dict):
                                        *positional_args, kwargs = args
                                        method(*positional_args, **kwargs)
                                    else:
                                        method(*args)
                            else:
                                method(*value if isinstance(value, list) else [value])
                        elif prop == "vlines":
                            for x, ymin, ymax, kwargs in value:
                                subplot_ax.vlines(x, ymin=ymin, ymax=ymax, **kwargs)
                        elif prop == "hlines":
                            for y, xmin, xmax, kwargs in value:
                                subplot_ax.hlines(y, xmin=xmin, xmax=xmax, **kwargs)
                        elif prop in ['set_ylim', 'set_xlim', 'set_xticks', 'set_yticks']:
                            getattr(subplot_ax, prop)(*value)

        plt.legend()
        fig.tight_layout()
        plt.show()



def histos_w_mean(key, ax, subkey, bins, D, overlaylabel, overlay=None):
# Plot the histogram
    n, bins, patches = ax.hist(D[key][subkey], histtype="step", lw=3,
            density=True,
            bins=bins,
            label=f"{overlaylabel}={overlay}")

    # Calculate and plot the mean as a vertical line
    # Extract the color from the first patch (line)
    color = patches[0].get_edgecolor()
    mean_value = np.mean(D[key][subkey])
    ax.axvline(mean_value, color=color, linestyle='--', lw=2, 
               label=f"Mean={mean_value:.2f}")


import os
import glob
from IPython.display import Video, display

def show_mp4(D, key, index=0):
    """
    List and optionally display .mp4 files in D[key]['rundir'].

    Parameters
    ----------
    D : dict
        Must contain D[key]["rundir"] → str (path).
    key : hashable
        Key into D.
    index : int, optional
        Which video to display (0 = newest).  
        Set to None if you only want the listing.
    """
    rundir = D[key].get("rundir")
    if not rundir or not os.path.isdir(rundir):
        print(f"[!] Invalid rundir: {rundir}")
        return

    mp4s = sorted(
        glob.glob(os.path.join(rundir, "*.mp4")),
        key=os.path.getmtime,
        reverse=True,         # newest first
    )

    if not mp4s:
        print("No .mp4 files found.")
        return

    # ── Listing ────────────────────────────────────────────────
    print(f"Found {len(mp4s)} .mp4 file(s) in '{rundir}':")
    for i, f in enumerate(mp4s):
        ts = os.path.getmtime(f)
        print(f"[{i}] {os.path.basename(f)} ") # (mod: {ts:%Y‑%m‑%d %H:%M:%S})")

    # ── Display ────────────────────────────────────────────────
    if index is None:
        return  # user only wanted the list

    if not (0 <= index < len(mp4s)):
        print(f"[!] Invalid index {index}. Choose 0‑{len(mp4s)-1}.")
        return

    print(f"\nShowing [{index}] → {os.path.basename(mp4s[index])}")
    display(Video(mp4s[index], embed=True))
