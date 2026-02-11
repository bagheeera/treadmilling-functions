from collections import OrderedDict
import itertools

import matplotlib.pyplot as plt
import numpy as np

def plot_data_(prm_sets, D, grid_params, overlay_params, plot_function, check_key, 
              fixed_params=None, axis_edits=None, figsize_=None, 
              sharex=True, sharey=True, renamedict=None):
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
    #figure_params : list of str
    #    Parameters to sweep over for generating separate figures.
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

    Usage:
    fct.plot.plot_data_(prm_sets, D,
        grid_params=grid_params,
        overlay_params=["rdis"],
        plot_function=lambda key, ax, overlay: 
            ax.hist2d(*D[key]["synthcoords"], label=f'{overlay}'),
        check_key="synthcoords",
        axis_edits={"set_ylim": (-25,25),
                    },
    )
    """
    import matplotlib.pyplot as plt
    fixed_params = fixed_params or {}
    figure_params = list(set(prm_sets) - set(grid_params + overlay_params))
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
        fig.suptitle(fig_title, y=0.999)

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
                if renamedict:
                    column_label = renamedict.get(grid_params[0], grid_params[0])
                    row_label = renamedict.get(grid_params[1], grid_params[1])
                else:
                    column_label = grid_params[0]
                    row_label = grid_params[1]
                if j == 0:
                    subplot_ax.annotate(f"{column_label}={grid1}", xy=(-0.3, 0.5), 
                                        xycoords="axes fraction",
                                        ha="right", va="center", fontsize=10, fontweight="bold")
                if i == 0:
                    subplot_ax.annotate(f"{row_label}={grid2}", xy=(0.5, 1.1), 
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


import numpy as np
from matplotlib.colors import to_rgb, to_hex

def interpolate_colors(color1, color2, factor=0.5):
    """Helper function to linearly interpolate between two colors."""
    return (1 - factor) * np.array(color1) + factor * np.array(color2)

def generate_numbers_and_colors(N, colors):
    """
    Interpolate a list of colors to obtain a continuous palette
    from https://chatgpt.com/c/66e2e2f4-b470-8011-8323-b89d94472e23
    # Example usage with 20 numbers and the 9-color palette
    colors = ['#FFBE0B', '#FB5607', '#FF006E', '#8338EC', '#3A86FF', '#FD8A09', '#FD2B3B', '#C11CAD', '#5E5FF6']
    N = 20
    numbers, interpolated_colors = generate_numbers_and_colors(N, colors)
    """
    # Convert hex colors to RGB
    rgb_colors = np.array([to_rgb(c) for c in colors])
    color_count = len(rgb_colors)
    
    # Initialize arrays for numbers and interpolated colors
    numbers = np.zeros(N)
    interpolated_colors = [''] * N
    
    # Calculate how many numbers correspond to each color interval
    segment_length = N / (color_count - 1)
    
    # Loop over each color pair and interpolate between them
    for i in range(color_count - 1):
        start = int(i * segment_length)
        end = int((i + 1) * segment_length)
        
        # Interpolate numbers
        numbers[start:end] = np.linspace(start, end-1, end - start)
        
        # Interpolate colors
        for j in range(start, end):
            factor = (j - start) / (end - start)  # Factor for interpolation
            interpolated_color = interpolate_colors(rgb_colors[i], rgb_colors[i + 1], factor)
            interpolated_colors[j] = to_hex(interpolated_color)
    
    # Ensure the last color and number are properly set
    numbers[-1] = N - 1
    interpolated_colors[-1] = to_hex(rgb_colors[-1])
    
    return numbers, interpolated_colors



def fancy_scatter(ax, x, y, radius=0.5, facecolor='lightblue',
                  edgecolor='black', shadow_color='black', shadow_alpha=0.4,
                  shadow_offset=(-0.15, -0.15), shadow_scale=0.8,
                  zorder_base=1, zorder_shadow=2):
    """scatter plot with little crescents to mimic 3d behavior. example usage:
    dft = df[(df["time"]==t_frame) & (df["x"].between(xlim[0], xlim[1])) & (df["y"].between(ylim[0], ylim[1]))]
    fancy_scatter(ax, *dft[["x", "y"]].values.T, radius=.47, #facecolor='#f72585', 
          facecolor="#24cedbff",
          shadow_alpha=.1,
          shadow_offset=(-0.35, -0.35)
                 )
                        """
    dx, dy = shadow_offset
    for xi, yi in zip(x, y):
        base = Circle((xi, yi), radius,
                      transform=ax.transData,
                      facecolor=facecolor,
                      edgecolor=edgecolor,
                      lw=0.8, zorder=zorder_base)
        ax.add_patch(base)

        shadow = Circle((xi + radius * dx, yi + radius * dy),
                        radius * shadow_scale,
                        transform=ax.transData,
                        facecolor=shadow_color,
                        alpha=shadow_alpha,
                        edgecolor='none',
                        zorder=zorder_shadow)
        shadow.set_clip_path(base)  # comment out this line if shadows vanish too much near edges
        ax.add_patch(shadow)

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from cycler import cycler
import numpy as np

import numpy as np
import matplotlib as mpl
from matplotlib import cm
from cycler import cycler

def set_cmap_colorcycle(cmap_name="viridis", N=10, portion=(0.0, 1.0)):
    """
    Set the matplotlib color cycle using a colormap.

    Parameters
    ----------
    cmap_name : str
        Name of the matplotlib colormap.
    N : int
        Number of colors to generate.
    portion : tuple(float, float)
        Fraction of the colormap to use (start, end), each in [0, 1].
        E.g., (1/3, 1) uses the final 2/3 of the colormap.
    """
    start, end = portion
    cmap = cm.get_cmap(cmap_name, N)
    colors = cmap(np.linspace(start, end, N))
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
    return cycler(color=colors)



def use_tue():
    """Set matplotlib to use tueplots style with paultol high contrast color palette."""
     #
     # Set color cycle to paultol high contrast
     #
    from tueplots import cycler
    from tueplots.constants import markers
    from tueplots.constants.color import palettes
    plt.rcParams.update(
        cycler.cycler(
            color=palettes.pn #paultol_high_contrast #[:3], #marker=markers.x_like_bold[:3]
        )
    )

def reset_matplotlib():
    """Reset matplotlib to default settings."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def load_pretty_figure_setup():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import os
    import sys

    # Prepend your TeX Live to PATH
    os.environ["PATH"] = os.path.expanduser("~/texlive/bin/x86_64-linux") + ":" + os.environ["PATH"]

    _preamble_shared = R"""
        \usepackage{graphicx}
        \DeclareMathOperator{\arcsinh}{arcsinh}
        \DeclareMathOperator{\km}{k_\mathrm{m}}
        \DeclareMathOperator{\fbi}{f_\mathrm{bi}}
        \DeclareMathOperator{\eps}{\epsilon_{\mathrm{mc}}}
        \DeclareMathOperator{\epscrit}{\epsilon_{\mathrm{mc}}^*}
        \DeclareMathOperator{\uf}{u_{\mathrm{f}}}
        \DeclareMathOperator{\kBT}{k_\mathrm{B}T}
        """[
        1:
    ]



    def mpl_rcParams_avenir():
        rcParams = {}
        rcParams["font.family"] = "sans-serif"
        rcParams["font.cursive"] = ["Optima"]
        rcParams["text.usetex"] = True
        # rcParams['text.latex.unicode']= True
        rcParams["pgf.texsystem"] = "lualatex"
        rcParams["pgf.rcfonts"] = False
        rcParams["pgf.preamble"] = (
            R"""
        \usepackage[utf8x]{inputenc}
        \usepackage[T1]{fontenc}
        \usepackage{fontspec}
        \usepackage{amsmath}
        \setmainfont{Avenir}[Scale=.9]
        \renewcommand{\setmainfont}{}
        \renewcommand{\sffamily}{}
        """[
                1:
            ]
            + "\n"
            + _preamble_shared
        )
        return rcParams

    def rc_params_setup():
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["figure.constrained_layout.use"] = True
        mpl.rcParams.update(mpl_rcParams_avenir())
        # mpl.rcParams["pgf.texsystem"] = "lualatex"
        # mpl.rcParams["text.latex.preamble"] = mpl.rcParams['pgf.preamble'] #R"\usepackage{amsmath}\usepackage{lmodern}"
        mpl.rcParams["text.latex.preamble"] = (
            R"""
        \usepackage{lmodern}
        \usepackage{amsmath}
        """
            + "\n"
            + _preamble_shared
        )

    rc_params_setup()
    print("Pretty figure set-up loaded.")  


def create_footer(custompath=None):
    from pathlib import Path
    from datetime import datetime

    # Full path and notebook name for footer
    full_path = str(Path.cwd().resolve())
    timestamp = datetime.now().isoformat()
    if custompath is not None:
        full_path = custompath
    footer_text = f"{full_path} | {timestamp}"
    return footer_text

def save_fig(fig, name, sources, notes="", footersize=4, notebook_name=""):
    """
    Save a matplotlib figure with organized metadata and a descriptive footer.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    name : str
        Name for this figure folder and file.
    sources : list of str
        Paths to the simulation runs or data used to make this figure.
    notes : str
        Additional notes (optional).
    """
    from pathlib import Path
    from datetime import datetime
    import yaml
    import matplotlib.pyplot as plt

    # Full path and notebook name for footer
    full_path = str(Path.cwd().resolve())
    timestamp = datetime.now().isoformat()
    footer_text = create_footer() #f"{full_path} | {timestamp}"

    # Add footer to figure
    fig.text(0.01, 0.01, footer_text, 
             fontsize=footersize, alpha=0.6)

    # Create folder for figure
    out = Path("../results") / name
    out.mkdir(parents=True, exist_ok=True)

    # Save figure with descriptive filenames
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg")

    def yaml_safe(obj):
        if isinstance(obj, dict):
            return {k: yaml_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [yaml_safe(v) for v in obj]
        else:
            return obj


    # Save metadata
    meta = {
        "sources": yaml_safe(sources),
        "path": full_path,
        "created": timestamp,
        "notes": notes,
        "notebook_used_for_figure_creation": notebook_name
    }

    with open(out / "meta.yaml", "w") as f:
        yaml.dump(meta, f)

    print(f"Figure saved: {out}/{name}.png and .svg with metadata")
