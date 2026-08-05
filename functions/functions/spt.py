import pandas as pd
import numpy as np
from functions.common_imports import *
def read_xml(c):
    from lxml import etree
    import pandas as pd

    tree = etree.parse(c)

    rows = []
    for pid, particle in enumerate(tree.xpath(".//particle")):
        for det in particle.xpath(".//detection"):
            rows.append({
                "id": pid,
                "time": int(det.get("t")),
                "x": float(det.get("x")),
                "y": float(det.get("y")),
                "z": float(det.get("z")),
            })
    df = pd.DataFrame(rows)
    df.loc[:, "t_zeroed"] = df["time"] - df.groupby("id")["time"].transform("min")
    return df


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def df_scatter(df, ax, s=0.5, vmax=None, vmin=None,
               colorby="t_zeroed", cmap="jet", zorder=99,
               useplot=False, separate_cmap=False,
               plotalpha=0.5, plotlw=0.5):
    
    df = df.copy()
    # Ensure t_zeroed exists for coloring if requested
    df.loc[:, "t_zeroed"] = df["time"] - df.groupby("id")["time"].transform("min")
    
    # Get the colormap object
    cm = plt.get_cmap(cmap)
    
    if useplot:
        # Loop through each unique track ID
        for pid, data in df.groupby("id"):
            points = data[["x", "y"]].values
            colors = data[colorby].values
            
            if separate_cmap:
                # Normalize colors locally for this specific ID (0 to 1)
                norm = mcolors.Normalize(vmin=colors.min(), vmax=colors.max())
                path_colors = cm(norm(colors))
            else:
                # Use global normalization (or provided vmax/vmin)
                norm = mcolors.Normalize(
                    vmin=vmin if vmin is not None else df[colorby].min(), 
                    vmax=vmax if vmax is not None else df[colorby].max()
                )
                path_colors = cm(norm(colors))
            
            # 1. Plot the continuous line (using a single color or basic plot)
            # Note: standard ax.plot doesn't support multicolored segments easily.
            # We plot the line in a light grey or the first color to show connectivity.
            ax.plot(points[:, 0], points[:, 1], color='gray', alpha=plotalpha, linewidth=plotlw, zorder=zorder-1)
            
            # 2. Overlay the scatter points for the gradient effect
            sc = ax.scatter(points[:, 0], points[:, 1],
                            c=path_colors,
                            s=s,
                            zorder=zorder)
    else:
        # Loop through each unique track ID to maintain the same logic as the 'if' block
        # but without the ax.plot() calls.
        for pid, data in df.groupby("id"):
            points = data[["x", "y"]].values
            colors = data[colorby].values
            
            if separate_cmap:
                # Normalize colors locally for this specific ID (0 to 1)
                norm = mcolors.Normalize(vmin=colors.min(), vmax=colors.max())
                path_colors = cm(norm(colors))
            else:
                # Use global normalization
                norm = mcolors.Normalize(
                    vmin=vmin if vmin is not None else df[colorby].min(), 
                    vmax=vmax if vmax is not None else df[colorby].max()
                )
                path_colors = cm(norm(colors))
            
            # Overlay the scatter points for the gradient effect (no line plotted)
            sc = ax.scatter(points[:, 0], points[:, 1],
                            c=path_colors,
                            s=s,
                            zorder=zorder)
    
    ax.set_aspect("equal")
    # return sc


import pathlib
def plot_pid(xmlfile, pid, data, MSD_analysis, df_read_fn):
    alpha = MSD_analysis[xmlfile]["alpha"].get(pid, np.nan)
    r2 = MSD_analysis[xmlfile]["r2"].get(pid, np.nan)
    D = MSD_analysis[xmlfile]["D"].get(pid, np.nan)
    df = df_read_fn(xmlfile)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{pathlib.Path(xmlfile).stem} | pid={pid} | α={alpha:.3f} | r2={r2:.3f}")

    # MSD curve + fit
    msd = data[(data["file"] == xmlfile) & (data["pid"] == pid)]
    # 1. Filter the data for the specific file and PID
    msd_df = data[(data["file"] == xmlfile) & (data["pid"] == pid)]

    # 2. Calculate the bounds for the shaded area
    lower_bound = msd_df["MSD"] - msd_df["MSD_std"]
    upper_bound = msd_df["MSD"] + msd_df["MSD_std"]

    # 3. Plot the main line
    axes[0].plot(msd_df["tau"], msd_df["MSD"], label=f"PID: {pid}", linewidth=2)

    # 4. Add the shaded uncertainty region
    axes[0].fill_between(
        msd_df["tau"], 
        lower_bound, 
        upper_bound, 
        alpha=0.2,       # Transparency (0.2 is usually a sweet spot)
        label="± Std Dev"
    )


    # tau_fit = np.linspace(msd["tau"].min(), msd["tau"].max(), 200)
    axes[0].plot(msd["tau"], D * msd["tau"]**alpha, "--", color="tomato", label="fit")
    axes[0].set_xlabel("tau")
    axes[0].set_ylabel("MSD")

    # trajectory
    df_scatter(df[df["id"] == pid], axes[1], s=20, useplot=True)

    plt.tight_layout()
    plt.show()


def plot_top_alpha_pids(files, data, MSD_analysis, df_read_fn, N_head=5):
    alpha_records = []
    for xmlfile in files:
        if xmlfile not in MSD_analysis:
            continue
        for pid, alpha in MSD_analysis[xmlfile]["alpha"].items():
            if not np.isnan(alpha):
                alpha_records.append({"file": xmlfile, "pid": pid, "alpha": alpha})

    top = (pd.DataFrame(alpha_records)
             .sort_values("alpha", ascending=False)
             .head(N_head))
    print(top)

    for _, row in top.iterrows():
        plot_pid(row["file"], row["pid"], data, MSD_analysis, df_read_fn)

def get_msds_for_file(data, xmlfile):
    return data[data["file"] == xmlfile].groupby("pid")["MSD"].apply(list).to_dict()

def merge_msd_analysis(files, MSD_analysis):
    merged = {"alpha_values": []}
    for xmlfile in files:
        if "ZapA" not in xmlfile:
            merged["alpha_values"].extend([val for val in MSD_analysis[xmlfile]["alpha"].values()
                                           if not np.isnan(val)
                                           ])
    return merged

from pathlib import Path
# from synthana import analysis, utils


def plot_msd_alpha(xmlfile, MSD_analysis, ax, color="steelblue", show_filtered=True,
cmap="viridis", add_inset=True, s_scatter=0.1):
# 1. Load Data
    df = read_xml(xmlfile)
    ax.set_xlim(df["x"].min(), df["x"].max())
    ax.set_ylim(df["y"].min(), df["y"].max())
    # 2. Map Alpha values
    alpha_series = df["id"].map(MSD_analysis[xmlfile]["alpha"])
    df["alpha"] = alpha_series
    
    # 3. Optional Background Layer
    if show_filtered:
        # Filter for IDs that are NOT in the MSD_analysis results
        df_nan = df[df["alpha"].isna()]
        ax.scatter(df_nan["x"], df_nan["y"], 
                   s=s_scatter, color="lightgrey", alpha=0.2, 
                   zorder=1, rasterized=True)

    # 4. Foreground: Plot analyzed traces
    df_valid = df[df["alpha"].notna()]
    sc = df_scatter(df_valid, ax, s=0.5, colorby="alpha", cmap=cmap, 
                    vmax=2, vmin=0.8, zorder=2)
    
    # ax.set_title(Path(xmlfile).stem, fontsize=10)
    if add_inset:
        # 4. Create Inset (Lower Right Corner)
        # [x0, y0, width, height] in normalized axis coordinates
        ax_ins = ax.inset_axes([0.0, 0.05, 0.12, 0.3])
        ax_ins.grid(axis="y")

        # 5. Extract alpha values for distribution
        alpha_values = alpha_series.dropna().unique()
    

        # Plot Distribution in Inset
        if len(alpha_values) > 0:
            # Violin Plot
            parts = ax_ins.violinplot([alpha_values], positions=[0], 
                                    showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.3)
            
            # Boxplot
            bp = ax_ins.boxplot([alpha_values], positions=[0], 
                                widths=0.4, patch_artist=True, 
                                showfliers=False,
                                medianprops=dict(color="black", linewidth=1.5))
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.7)
        
        # Clean up inset appearance
        ax_ins.set_xticks([]) # Hide x-axis ticks
        ax_ins.set_ylabel(r"$\alpha$", fontsize=8)
        ax_ins.tick_params(labelsize=7)
        ax_ins.set_ylim(0.5, 2.5) # Consistent scale for alpha
        
    return sc



def plot_pid_loglog(xmlfile, pid, data, MSD_analysis, df_read_fn,
                    skip_ballistic=1, tau_range=None, show_loglog=True):
    
    alpha = MSD_analysis[xmlfile]["alpha"].get(pid, np.nan)
    r2    = MSD_analysis[xmlfile]["r2"].get(pid, np.nan)
    D     = MSD_analysis[xmlfile]["D"].get(pid, np.nan)

    df      = df_read_fn(xmlfile)
    msd_df  = data[(data["file"] == xmlfile) & (data["pid"] == pid)]
    tau     = msd_df["tau"].values
    msd     = msd_df["MSD"].values

    # ── determine fit region bounds ────────────────────────────────────────────
    tau_min_fit = max(skip_ballistic, tau_range[0] if tau_range is not None else 0)
    tau_max_fit = tau_range[-1] if tau_range is not None else tau.max()
    def shade_regions(ax, xmin, xmax, tau_min_fit, tau_max_fit, log=False):
        """Shade excluded (grey) and fitted (green) regions."""
        if log:
            xmin, xmax         = np.log(xmin), np.log(xmax)
            tau_min_fit_plot   = np.log(max(tau_min_fit, xmin))
            tau_max_fit_plot   = np.log(min(tau_max_fit, xmax))
        else:
            tau_min_fit_plot   = tau_min_fit
            tau_max_fit_plot   = tau_max_fit

        # excluded: before fit window
        if tau_min_fit_plot > xmin:
            ax.axvspan(xmin, tau_min_fit_plot,
                       color="grey", alpha=0.12, label="excluded (ballistic)")
        # fit window
        ax.axvspan(tau_min_fit_plot, tau_max_fit_plot,
                   color="green", alpha=0.08, label="fit region")
        # excluded: after fit window
        if tau_max_fit_plot < xmax:
            ax.axvspan(tau_max_fit_plot, xmax,
                       color="grey", alpha=0.12)

    ncols = 3 if show_loglog else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.suptitle(
        f"{pathlib.Path(xmlfile).stem} | pid={pid} | "
        f"α={alpha:.3f} | R²(log)={r2:.3f}"
    )

    # ── panel 0: linear scale ──────────────────────────────────────────────────
    ax = axes[0]
    shade_regions(ax, tau.min(), tau.max(), tau_min_fit, tau_max_fit, log=False)
    ax.plot(tau, msd, linewidth=2, label=f"PID {pid}")
    ax.fill_between(tau,
                    msd_df["MSD"] - msd_df["MSD_std"],
                    msd_df["MSD"] + msd_df["MSD_std"],
                    alpha=0.2, label="± std")
    if not np.isnan(alpha):
        tau_fit_line = np.linspace(tau_min_fit, tau_max_fit, 300)
        ax.plot(tau_fit_line, D * tau_fit_line**alpha,
                "--", color="tomato", label=f"fit: Dτ^α")
    ax.set_xlabel("τ")
    ax.set_ylabel("MSD")
    ax.legend(fontsize=8)

    # ── panel 1: trajectory ────────────────────────────────────────────────────
    df_scatter(df[df["id"] == pid], axes[1], s=20, useplot=True)

    # ── panel 2: log-log ───────────────────────────────────────────────────────
    if show_loglog:
        ax2 = axes[2]
        pos = (tau > 0) & (msd > 0) & np.isfinite(msd)

        log_tau = np.log(tau[pos])
        log_msd = np.log(msd[pos])
        tau_pos = tau[pos]
        msd_pos = msd[pos]

        # ── all boundaries in log-space ────────────────────────────────────────
        log_tau_min_fit = np.log(max(tau_min_fit, tau_pos.min()))
        log_tau_max_fit = np.log(min(tau_max_fit, tau_pos.max()))

        # shading directly in log-space (no helper needed)
        if log_tau_min_fit > log_tau.min():
            ax2.axvspan(log_tau.min(), log_tau_min_fit,
                        color="grey", alpha=0.12, label="excluded")
        ax2.axvspan(log_tau_min_fit, log_tau_max_fit,
                    color="green", alpha=0.08, label="fit region")
        if log_tau_max_fit < log_tau.max():
            ax2.axvspan(log_tau_max_fit, log_tau.max(),
                        color="grey", alpha=0.12)

        # ── masks in log-space to match shading exactly ────────────────────────
        in_fit  = (log_tau > log_tau_min_fit) & (log_tau <= log_tau_max_fit)
        out_fit = ~in_fit

        ax2.scatter(log_tau[in_fit],  log_msd[in_fit],
                    s=18, zorder=3, label="fitted points")
        ax2.scatter(log_tau[out_fit], log_msd[out_fit],
                    s=18, zorder=3, marker="x", color="grey", label="excluded")

        if not np.isnan(alpha):
            ax2.plot(log_tau, np.log(D) + alpha * log_tau,
                     "--", color="tomato", label=f"α={alpha:.3f}")

        ax2.set_xlabel("log τ")
        ax2.set_ylabel("log MSD")
        ax2.set_title(f"log-log | R²={r2:.3f}")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


import ipywidgets as widgets
from pathlib import Path
from IPython.display import display

def browse_datasets(file_list, msd_results, color="steelblue", label="Dataset",
                    widths=0.15):
    """
    Creates an interactive browser for any list of XML files.
    """
    def update_plot(index):
        if not file_list:
            print("File list is empty.")
            return
            
        xmlfile = file_list[index]
        
        fig, ax = plt.subplots(figsize=(6,6))
        
        # Using the function we built previously
        sc = plot_msd_alpha(
            xmlfile, 
            msd_results, 
            ax, 
            color=color, 
            show_filtered=True
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add colorbar for context
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'$\alpha$')
        
        # Add extra info to title
        # ax.set_title(f"{label} [{index}]: {Path(xmlfile).name}", fontsize=10)
        print(xmlfile)
        
        plt.tight_layout()
        plt.show()

    # Create the slider
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max(0, len(file_list) - 1),
        description='Index',
        continuous_update=False,
        layout=widgets.Layout(width='50%')
    )

    # Use interactive to bundle the UI
    ui = widgets.VBox([widgets.Label(f"Exploring: {label}"), slider])
    out = widgets.interactive_output(update_plot, {'index': slider})
    
    display(ui, out)


def violin_boxplot(ax, positions, data_dict, colors, widths=0.15, alpha=0.3):
    """
    Creates a combined Violin and Boxplot visualization on a given Matplotlib axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis object where the plot will be drawn.
    positions : list or array-like
        The x-axis coordinates where each plot should be placed (e.g., [1, 2, 3]).
    data_dict : dict
        A dictionary containing the data to plot. 
        Keys are used as X-axis labels, Values are lists/arrays of numerical data.
    colors : list
        List of colors for each plot (must match the number of keys in data_dict).
    widths : float, optional
        The width of the boxplots. Default is 0.15.
    alpha : float, optional
        Transparency for both the violin bodies and the boxes. Default is 0.3.
    """
    
    # 1. Generate Violin Plot (Background)
    # data_dict.values() extracts the numerical arrays for each category
    parts = ax.violinplot(data_dict.values(), positions=positions,
                          showmedians=False, showextrema=False)
    
    # Apply custom colors to violin bodies
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(alpha=alpha)

    # 2. Generate Boxplot (Foreground)
    # patch_artist=True is required to fill the boxes with color
    # bp = ax.boxplot(data_dict.values(), positions=positions,
    #                 widths=widths, patch_artist=True,
    #                 showfliers=False, # Removes outliers to keep the plot clean
    #                 medianprops=dict(color="black", linewidth=2))

    bp = ax.boxplot(
        data_dict.values(),
        positions=positions,
        widths=widths,
        patch_artist=True,         # needed to access box face color
        showfliers=False,          # hide outlier markers
        boxprops=dict(color="black", linewidth=1, facecolor="none"),  # outlines only
        capprops=dict(color="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        medianprops=dict(color="black", linewidth=2)
    )
    
    # Apply custom colors and alpha to the box patches
    for patch, color in zip(bp["boxes"], colors):
        # patch.set_facecolor(color)
        patch.set_alpha(alpha)

    # 3. Final Formatting
    # Use the dictionary keys to label the categories on the X-axis
    ax.set_xticks(positions)
    ax.set_xticklabels(data_dict.keys())



import re

def extract_timeunit(s):
    match = re.search(r'(\d+)\s*(ms|s)(?=[^a-zA-Z]|$)', s, re.IGNORECASE)
    if match:
        return match.group(1) + " " + match.group(2).lower()
    return "0 ms" #None, None
def timeunit_to_ms(filename):
    label = extract_timeunit(filename)
    if label is None:
        return float("inf")
    number, unit = label.split()
    return int(number) if unit == "ms" else int(number) * 1000


import ipywidgets as widgets
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt
import numpy as np

def plot_traces(file_index=0, file_list=[]):
        # 1. Select and load file
        f = file_list[file_index]
        df = fct.spt.read_xml(f)
        
        # 2. Select IDs for sampling
        pids = df["id"].unique()
        sample_size = min(len(pids), 1500)
        sel = np.random.choice(pids, sample_size, replace=False)
        
        # 3. Filter and Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Updated scatter call per your request
        df_filtered = df[df["id"].isin(sel)]
        fct.spt.df_scatter(df_filtered, ax, colorby="id",
                           cmap="cividis")
        
        # ax.set_xlim(20, 70)
        # ax.set_ylim(20, 70)
        ax.set_title(f"Index {file_index}: {fct.spt.extract_timeunit(f)}")
        
        plt.show()

def start_csv_browser(file_list):
    """
    Pass any file list to this function to create a slider-based explorer.
    """
    # Use 'fixed' for the file_list so interact doesn't try to make a widget for it
    interact(
        plot_traces, 
        file_list=fixed(file_list), 
        file_index=widgets.IntSlider(
            min=0, 
            max=len(file_list) - 1, 
            step=1, 
            value=0,
            description='File Index:',
            continuous_update=False
        )
    )

def plot_pid_windows_eigenvectors(xmlfile, pid, df_read_fn, L,
                                  fontsize=6,
                                  offset=0,
                                  cmap=plt.cm.viridis):
    df     = df_read_fn(xmlfile)
    df_pid = df[df["id"] == pid].copy().reset_index(drop=True)

    n_frames  = len(df_pid)
    n_windows = n_frames // L
    
    fig, ax = plt.subplots(figsize=(5, 5))

    for i in range(n_windows):
        window = df_pid.iloc[i*L : (i+1)*L]
        color  = cmap(i / max(n_windows - 1, 1))

        ax.plot(window["x"], window["y"], color=color, linewidth=2, alpha=0.7)
        ax.plot(window["x"].iloc[0], window["y"].iloc[0],
                "o", color=color, markersize=7, zorder=3,
                markeredgecolor="k", markeredgewidth=0.5)

        # ── gyration tensor ────────────────────────────────────────────────────
        xy     = window[["x", "y"]].values
        center = xy.mean(axis=0)
        cov    = np.cov(xy.T)
        eigvals, eigvecs = np.linalg.eigh(cov)  # eigvals ascending
        lam2, lam1 = np.sqrt(eigvals)            # λ1 >= λ2
        v1,   v2   = eigvecs[:, 1], eigvecs[:, 0]

        asymmetry = -np.log(1 - (lam1**2 - lam2**2)**2 / (2*(lam1**2 + lam2**2)**2))

        # draw principal axes scaled by eigenvalues
        for vec, lam, ls in [(v1, lam1, "-"), (v2, lam2, "-")]:
            ax.annotate("", 
                        xy=center + lam * vec,
                        xytext=center - lam * vec,
                        arrowprops=dict(arrowstyle="<->", color=color,
                                        lw=1.5, linestyle=ls))

        # annotate asymmetry at segment center
        ax.text(center[0]+offset, center[1]+offset, f"{asymmetry:.2f}",
                fontsize=fontsize, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none"))

    # end marker
    ax.plot(df_pid["x"].iloc[n_windows*L - 1],
            df_pid["y"].iloc[n_windows*L - 1],
            "s", color=cmap(1.0), markersize=7, zorder=3,
            markeredgecolor="k", markeredgewidth=0.5)

    cmap_discrete = plt.cm.colors.BoundaryNorm(
        boundaries=np.arange(-0.5, n_windows), ncolors=n_windows)
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.colors.ListedColormap([cmap(i / max(n_windows-1, 1)) 
                                           for i in range(n_windows)]),
        norm=plt.Normalize(vmin=-0.5, vmax=n_windows-0.5))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="segment index",
                        ticks=np.arange(n_windows))
    cbar.set_ticklabels(np.arange(n_windows))
    sm.set_array([])
    # plt.colorbar(sm, ax=ax, label="segment index")
#     ax.set_title(f"{pathlib.Path(xmlfile).stem} | pid={pid} | L={L}")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def parse_param_set(param_set_str):
    import re
    """Parses 'N:30, nf:6' -> (30, 6)"""
    match = re.match(r"N:(\d+),\s*nf:(\d+)", param_set_str)
    return int(match.group(1)), int(match.group(2))
def plot_pid_colored_by_D(pid, all_configs, original_df, 
                          selected_param_set,
                           metric="D", cmap=plt.cm.coolwarm):
    window_pts, n_f = parse_param_set(selected_param_set)

    configs = all_configs[(all_configs["id"] == pid) &
                          (all_configs["param_set"] == selected_param_set)].reset_index(drop=True)

    df_pid_reset = original_df[original_df["id"] == pid].reset_index(drop=True)
    # print(f"pid={pid}, df_pid_reset len={len(df_pid_reset)}, "
    #       f"configs rows={len(configs)}, "
    #       f"start range={configs['window_start_idx'].min()}-{configs['window_start_idx'].max()}, "
    #       f"window_pts={window_pts}")


    # ── color scale ────────────────────────────────────────────────────────────
    values = configs[metric].values
    vmax = np.nanpercentile(np.abs(values), 80)
    vmin = -vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # ── per-point metric array ─────────────────────────────────────────────────
    metric_per_point = np.full(len(df_pid_reset), np.nan)
    for _, row in configs.iterrows():
        start = int(row["window_start_idx"])
        metric_per_point[start : start + window_pts] = row[metric]

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(df_pid_reset["x"], df_pid_reset["y"],
            color="grey", linewidth=1, alpha=0.4, zorder=1)
    ax.scatter(df_pid_reset["x"], df_pid_reset["y"],
               c=metric_per_point, cmap=cmap, norm=norm,
               s=30, zorder=3, edgecolors="k", linewidths=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=metric, shrink=0.3)
    ax.set_title(f"pid={pid} | {selected_param_set} | {metric}")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

def window_max_span(coords):
    """Maximum pairwise distance within a window."""
    from scipy.spatial.distance import pdist
    if len(coords) < 2:
        return np.nan
    return pdist(coords).max()

# def sliding_window_span(df, window_sizes, step_size=1):
#     """
#     Computes mean max-span per id for multiple window sizes.
#     Returns dict: {pid: {L: mean_span}}
#     """
#     results = {}

#     for pid, group in df.groupby("id"):
#         coords = group[["x", "y"]].to_numpy()
#         n_points = len(coords)
#         results[pid] = {}

#         for L in window_sizes:
#             if n_points < L:
#                 continue
#             spans = [
#                 window_max_span(coords[start : start + L])
#                 for start in range(0, n_points - L + 1, step_size)
#             ]
#             results[pid][L] = np.nanmean(spans)

#     return results
# def sliding_window_span(df, window_sizes, step_size=1):
#     records = []
#     for pid, group in df.groupby("id"):
#         coords = group[["x", "y"]].to_numpy()
#         n_points = len(coords)

#         for L in window_sizes:
#             if n_points < L:
#                 continue
#             for start in range(0, n_points - L + 1, step_size):
#                 window = coords[start : start + L]
#                 span = window_max_span(window)
#                 if np.isfinite(span):
#                     records.append({"pid": pid, "L": L, "span": span})

#     return pd.DataFrame(records)

def sliding_window_span(df, window_times, step_size=1):
    """
    window_times: list of window durations in seconds (e.g. [0.5, 1.0, 2.0])
    """
    records = []
    for pid, group in df.groupby("id"):
        coords = group[["x", "y"]].to_numpy()
        n_points = len(coords)
        dt = group["time"].diff().iloc[1]

        for window_time in window_times:
            window_pts = int(np.round(window_time / dt))
            if n_points < window_pts or window_pts < 2:
                continue
            for start in range(0, n_points - window_pts + 1, step_size):
                window = coords[start : start + window_pts]
                span = window_max_span(window)
                if np.isfinite(span):
                    records.append({"pid": pid, "L": window_time, "span": span})

    return pd.DataFrame(records)

def plot_pid_colored_by_window_metric(
        df, pid,
        window_sizes,
        metric_fn,
        metric_name="metric",
        step_size=1,
        cmap=plt.cm.plasma,
        fig=None,
        ax=None,
        s=5, linewidth=1):

    df_pid = df[df["id"] == pid].reset_index(drop=True)
    n_pts  = len(df_pid)

    metric_arrays = {}
    for L in window_sizes:
        arr = np.full(n_pts, np.nan)
        for start in range(0, n_pts - L + 1, L):  # step_size=L always
            val = metric_fn(df_pid.iloc[start : start + L])
            if np.isfinite(val):
                arr[start : start + L] = val
        metric_arrays[L] = arr
    

    all_vals = np.concatenate([v[np.isfinite(v)] for v in metric_arrays.values()])
    vmin, vmax = np.nanpercentile(all_vals, 2), np.nanpercentile(all_vals, 98)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # ── create fig/axes only if not provided ──────────────────────────────────
    created_fig = fig is None
    if created_fig:
        fig, axes = plt.subplots(1, len(window_sizes),
                                 figsize=(3 * len(window_sizes), 3))
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(ax)

    for i, L in enumerate(window_sizes):
        a = axes[i]
        a.plot(df_pid["x"], df_pid["y"],
               color="grey", linewidth=linewidth, alpha=0.4, zorder=1)
        a.scatter(df_pid["x"], df_pid["y"],
                  c=metric_arrays[L], cmap=cmap, norm=norm,
                  s=s, zorder=3, edgecolors="k", linewidths=0.1)
        a.set_title(f"L={L}", fontsize=8)
        a.set_aspect("equal")
        # a.set_xticklabels([])
        # a.set_yticklabels([])

    # always add colorbar if fig is available
    if fig is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.tolist(), label=metric_name, shrink=0.6)

    if created_fig:
        fig.suptitle(f"pid={pid} | {metric_name}", fontsize=9)
        plt.tight_layout()
        plt.show()

    # return axes

import ipywidgets as widgets
from IPython.display import display

def interactive_window_metric(df, pid, window_sizes, metric_fn,
                               metric_name="metric", step_size=1,
                               cmap=plt.cm.plasma, s=20):
    df_pid = df[df["id"] == pid].reset_index(drop=True)
    n_pts  = len(df_pid)

    # ── precompute metric for all window sizes ─────────────────────────────────
    metric_arrays = {}
    for L in window_sizes:
        arr    = np.full(n_pts, np.nan)
        counts = np.zeros(n_pts)
        for start in range(0, n_pts - L + 1, step_size):
            val = metric_fn(df_pid.iloc[start : start + L])
            if np.isfinite(val):
                arr[start : start + L] = np.nansum(
                    [arr[start : start + L], np.full(L, val)], axis=0)
                counts[start : start + L] += 1
        valid = counts > 0
        arr[valid] /= counts[valid]
        metric_arrays[L] = arr

    all_vals = np.concatenate([v[np.isfinite(v)] for v in metric_arrays.values()])
    vmin, vmax = np.nanpercentile(all_vals, 2), np.nanpercentile(all_vals, 98)
    norm  = plt.Normalize(vmin=vmin, vmax=vmax)

    # ── widgets ────────────────────────────────────────────────────────────────
    L_slider     = widgets.SelectionSlider(options=window_sizes,
                                           description="Window size L:",
                                           continuous_update=True,
                                           style={"description_width": "auto"})
    start_slider = widgets.IntSlider(value=0, min=0,
                                     max=n_pts - window_sizes[0],
                                     description="Window start:",
                                     continuous_update=True,
                                     style={"description_width": "auto"})

    def update_start_range(change):
        start_slider.max = n_pts - L_slider.value - 1

    L_slider.observe(update_start_range, names="value")

    out = widgets.Output()

    def plot(L, start):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(5, 5))

            metric_arr = metric_arrays[L]

            # grey for all points outside window
            outside = np.ones(n_pts, dtype=bool)
            outside[start : start + L] = False

            ax.plot(df_pid["x"], df_pid["y"],
                    color="lightgrey", linewidth=1, alpha=0.5, zorder=1)

            # colored scatter for outside points (faint)
            ax.scatter(df_pid["x"][outside], df_pid["y"][outside],
                       c=metric_arr[outside], cmap=cmap, norm=norm,
                       s=s * 0.5, zorder=2, alpha=0.3,
                       edgecolors="none")

            # highlighted window
            win = df_pid.iloc[start : start + L]
            win_metric = metric_arr[start : start + L]
            sc = ax.scatter(win["x"], win["y"],
                            c=win_metric, cmap=cmap, norm=norm,
                            s=s * 2, zorder=4,
                            edgecolors="k", linewidths=0.4)

            # window outline box
            ax.plot(win["x"], win["y"],
                    color="black", linewidth=1.5, alpha=0.6, zorder=3)
            ax.plot(win["x"].iloc[0], win["y"].iloc[0],
                    "o", color="lime", markersize=8, zorder=5,
                    markeredgecolor="k", markeredgewidth=0.5)
            ax.plot(win["x"].iloc[-1], win["y"].iloc[-1],
                    "s", color="red", markersize=8, zorder=5,
                    markeredgecolor="k", markeredgewidth=0.5)

            plt.colorbar(sc, ax=ax, label=metric_name, shrink=0.6)
            ax.set_title(f"pid={pid} | L={L} | start={start} | "
                         f"{metric_name}={np.nanmean(win_metric):.3f}")
            ax.set_aspect("equal")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.tight_layout()
            plt.show()

    interactive_plot = widgets.interactive(plot,
                                           L=L_slider,
                                           start=start_slider)
    display(widgets.VBox([L_slider, start_slider, out]))
    plot(L_slider.value, start_slider.value)

    # link sliders to plot manually for responsiveness
    def on_change(_):
        plot(L_slider.value, start_slider.value)

    L_slider.observe(on_change, names="value")
    start_slider.observe(on_change, names="value")



import numpy as np
import orientationpy


def calc_orientation(
    image,
    boxSizePixels=10,
    sigma=1,
    intensity_threshold=0.05,
    mode="fiber",
    time_axis=0,
    return_pixel_maps=False,
):
    """
    Calculate local orientation vectors for a 2D image or a time stack of 2D images.

    Parameters
    ----------
    image : ndarray
        Either a 2D image with shape (y, x), or a 3D stack with shape
        (t, y, x) by default.

    boxSizePixels : int or tuple of int, optional
        Size of the local boxes used to average orientation vectors.
        If int, the same size is used in y and x.

    sigma : float, optional
        Smoothing scale used for the full-resolution structure tensor.

    intensity_threshold : float, optional
        Normalized intensity threshold below which vectors are hidden by
        setting them to zero.

    mode : str, optional
        Orientation mode passed to orientationpy.computeOrientation.
        For fiber-like structures, use "fiber".

    time_axis : int, optional
        Axis corresponding to time if image is 3D.

    return_pixel_maps : bool, optional
        If True, also return full-resolution intensity, directionality,
        and orientation maps.

    Returns
    -------
    result : dict or list of dict
        If input is 2D, returns a single dictionary.
        If input is 3D, returns a list of dictionaries, one per time frame.

        Each dictionary contains:
            "box_centres_x"
            "box_centres_y"
            "vectors_yx"
            "intensity_boxes"
            "intensity_boxes_normalized"
            "orientations_boxes"

        vectors_yx has shape:
            (2, n_boxes_y, n_boxes_x)

        vectors_yx[0] is the y-component.
        vectors_yx[1] is the x-component.
    """

    image = np.asarray(image)

    if image.ndim not in (2, 3):
        raise ValueError(
            "image must be either a 2D image with shape (y, x) "
            "or a 3D stack with shape (t, y, x)."
        )

    if np.isscalar(boxSizePixels):
        box_y = box_x = int(boxSizePixels)
    else:
        box_y, box_x = map(int, boxSizePixels)

    if box_y <= 0 or box_x <= 0:
        raise ValueError("boxSizePixels must be positive.")

    def _analyse_single_frame(frame):
        frame = np.asarray(frame, dtype=float)

        if frame.ndim != 2:
            raise ValueError("Each frame must be 2D.")

        # Compute image gradients
        Gy, Gx = orientationpy.computeGradient(frame)

        # Compute local structure tensor in boxes
        structure_tensor_boxes = orientationpy.computeStructureTensorBoxes(
            [Gy, Gx],
            [box_y, box_x],
        )

        # Box-wise intensity
        intensity_boxes = orientationpy.computeIntensity(structure_tensor_boxes)

        # Avoid divide-by-zero if the image is empty or uniform
        max_intensity = np.nanmax(intensity_boxes)

        if np.isfinite(max_intensity) and max_intensity > 0:
            intensity_boxes_normalized = intensity_boxes / max_intensity
        else:
            intensity_boxes_normalized = np.zeros_like(intensity_boxes, dtype=float)

        # Box-wise orientations
        orientations_boxes = orientationpy.computeOrientation(
            structure_tensor_boxes,
            mode=mode,
        )

        # Convert orientations to y/x vector components
        vectors_yx = orientationpy.anglesToVectors(orientations_boxes)
        vectors_yx = np.asarray(vectors_yx, dtype=float)

        # Hide vectors in low-signal boxes
        low_signal = intensity_boxes_normalized < intensity_threshold
        vectors_yx[:, low_signal] = 0.0

        # Compute box centre coordinates
        n_boxes_y, n_boxes_x = intensity_boxes.shape

        box_centres_y = np.arange(n_boxes_y) * box_y + box_y / 2
        box_centres_x = np.arange(n_boxes_x) * box_x + box_x / 2

        result = {
            "box_centres_x": box_centres_x,
            "box_centres_y": box_centres_y,
            "vectors_yx": vectors_yx,
            "intensity_boxes": intensity_boxes,
            "intensity_boxes_normalized": intensity_boxes_normalized,
            "orientations_boxes": orientations_boxes,
        }

        if return_pixel_maps:
            structure_tensor = orientationpy.computeStructureTensor(
                [Gy, Gx],
                sigma=sigma,
            )

            intensity = orientationpy.computeIntensity(structure_tensor)
            directionality = orientationpy.computeStructureDirectionality(
                structure_tensor
            )
            orientations = orientationpy.computeOrientation(
                structure_tensor,
                mode=mode,
            )

            result.update(
                {
                    "intensity": intensity,
                    "directionality": directionality,
                    "orientations": orientations,
                    "structure_tensor": structure_tensor,
                }
            )

        return result

    # Single image
    if image.ndim == 2:
        return _analyse_single_frame(image)

    # Time stack
    stack = np.moveaxis(image, time_axis, 0)

    results = []
    for t in range(stack.shape[0]):
        results.append(_analyse_single_frame(stack[t]))

    return results

def plot_orientation_frame(
    image_stack,
    results,
    t=0,
    cmap="Greys_r",
    vector_color="r",
    scale=0.1,
    vmin=0,
):
    """
    Plot local orientation vectors for one frame of an image stack.

    Parameters
    ----------
    image_stack : ndarray
        Image stack with shape (time, y, x).

    results : list of dict
        Output from calc_orientation(image_stack).

    t : int
        Time frame to plot.

    cmap : str
        Matplotlib colormap for the image.

    vector_color : str
        Color of orientation vectors.

    scale : float
        Quiver scaling parameter. Smaller values make vectors longer.

    vmin : float or None
        Minimum display intensity for imshow.
    """

    frame = image_stack[t]
    res = results[t]

    boxCentresX = res["box_centres_x"]
    boxCentresY = res["box_centres_y"]
    boxVectorsYX = res["vectors_yx"]

    plt.figure(figsize=(4,4))
    plt.title(f"Local orientation vectors in boxes, frame {t}")

    plt.imshow(frame, cmap=cmap, vmin=vmin)

    plt.quiver(
        boxCentresX,
        boxCentresY,
        boxVectorsYX[1],
        boxVectorsYX[0],
        angles="xy",
        scale_units="xy",
        scale=scale,
        color=vector_color,
        headwidth=0,
        headlength=0,
        headaxislength=1,
    )

    plt.axis("image")
    plt.show()


import numpy as np
import pandas as pd


def calc_orientation_displacement_alignment(
    stack,
    df,
    calc_orientation_func,
    boxSizePixels=10,
    sigma=1,
    coordinate_scale=3.2,
    x_range=None,
    y_range=None,
    x_col="x",
    y_col="y",
    time_col="time",
    id_col="id",
    time_values=None,
    min_n=1,
    require_consecutive=True,
    max_dt=1,
    normalize_by_dt=False,
    weight_by=None,
):
    """
    Calculate orientation/displacement alignment for an image stack and trajectories.

    Parameters
    ----------
    stack : ndarray
        Image stack, usually shape (time, y, x).

    df : pandas.DataFrame
        Trajectory dataframe containing x, y, time, and particle id columns.

    calc_orientation_func : callable
        Function used to calculate local orientations.
        For your case, pass fct.spt.calc_orientation.

    boxSizePixels : int
        Orientation box size in pixels.

    sigma : float
        Sigma passed to calc_orientation_func.

    coordinate_scale : float
        Scale factor applied to df[x_col], df[y_col].
        In your example this is 3.2.

    x_range, y_range : tuple or None
        Binning ranges in pixel coordinates.
        If None, uses image dimensions: x_range=(0, width), y_range=(0, height).

    min_n : int
        Minimum number of displacement vectors per bin required for analysis.

    require_consecutive : bool
        If True, only use displacements between consecutive time points.

    max_dt : int or float
        Required frame step if require_consecutive=True.

    normalize_by_dt : bool
        If True, computes velocity-like dx/dt, dy/dt instead of raw displacement.

    weight_by : {"n", "disp_mag", "orient_mag", None}
        How to average the per-bin alignment over space for each time frame.

    Returns
    -------
    out : dict
        Contains orientation fields, displacement fields, dot products,
        cosine alignment, nematic alignment, and dataframe summaries.
    """

    stack = np.asarray(stack)

    if stack.ndim != 3:
        raise ValueError("Expected stack with shape (time, y, x).")

    n_stack_time, image_height, image_width = stack.shape

    if x_range is None:
        x_range = (0, image_width)

    if y_range is None:
        y_range = (0, image_height)

    # ---------------------------------------------------------------------
    # 1. Calculate local orientation field
    # ---------------------------------------------------------------------
    results = calc_orientation_func(
        stack,
        boxSizePixels=boxSizePixels,
        sigma=sigma,
    )

    # vectors_yx[0] = y-component, vectors_yx[1] = x-component
    U = np.stack([r["vectors_yx"][1] for r in results])  # x-component
    V = np.stack([r["vectors_yx"][0] for r in results])  # y-component

    n_time, n_bins_y, n_bins_x = U.shape

    # ---------------------------------------------------------------------
    # 2. Construct bin edges matching the orientation field
    # ---------------------------------------------------------------------
    x_edges = np.linspace(x_range[0], x_range[1], n_bins_x + 1)
    y_edges = np.linspace(y_range[0], y_range[1], n_bins_y + 1)

    # ---------------------------------------------------------------------
    # 3. Calculate per-particle displacements
    # ---------------------------------------------------------------------
    d = df.copy()

    d[x_col] = coordinate_scale * d[x_col]
    d[y_col] = coordinate_scale * d[y_col]

    if id_col is not None and id_col in d.columns:
        d = d.sort_values([id_col, time_col]).reset_index(drop=True)

        d["x_next"] = d.groupby(id_col)[x_col].shift(-1)
        d["y_next"] = d.groupby(id_col)[y_col].shift(-1)
        d["time_next"] = d.groupby(id_col)[time_col].shift(-1)
    else:
        d = d.sort_values(time_col).reset_index(drop=True)

        d["x_next"] = d[x_col].shift(-1)
        d["y_next"] = d[y_col].shift(-1)
        d["time_next"] = d[time_col].shift(-1)

    d["dt"] = d["time_next"] - d[time_col]
    d["dx"] = d["x_next"] - d[x_col]
    d["dy"] = d["y_next"] - d[y_col]

    d = d.dropna(subset=["dx", "dy", "dt"]).copy()

    if require_consecutive:
        d = d[np.isclose(d["dt"], max_dt)].copy()

    if normalize_by_dt:
        d["dx"] = d["dx"] / d["dt"]
        d["dy"] = d["dy"] / d["dt"]

    # ---------------------------------------------------------------------
    # 4. Assign starting positions to bins
    # ---------------------------------------------------------------------
    d["x_bin"] = np.searchsorted(x_edges, d[x_col], side="right") - 1
    d["y_bin"] = np.searchsorted(y_edges, d[y_col], side="right") - 1

    # Include points exactly on the rightmost edge
    d.loc[d[x_col] == x_edges[-1], "x_bin"] = n_bins_x - 1
    d.loc[d[y_col] == y_edges[-1], "y_bin"] = n_bins_y - 1

    # Discard out-of-range points
    d = d[
        (d["x_bin"] >= 0) & (d["x_bin"] < n_bins_x) &
        (d["y_bin"] >= 0) & (d["y_bin"] < n_bins_y)
    ].copy()

    d["x_bin"] = d["x_bin"].astype(int)
    d["y_bin"] = d["y_bin"].astype(int)

    # ---------------------------------------------------------------------
    # 5. Average displacement vectors per time/bin
    # ---------------------------------------------------------------------
    agg = (
        d.groupby([time_col, "y_bin", "x_bin"])
        .agg(
            dx=("dx", "mean"),
            dy=("dy", "mean"),
            n=("dx", "size"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------------------
    # 6. Convert displacement dataframe into dense arrays
    # ---------------------------------------------------------------------
    if time_values is None:
        # Assumes dataframe time values correspond directly to stack frame indices
        time_values = np.arange(n_time)
    else:
        time_values = np.asarray(time_values)

    if len(time_values) != n_time:
        raise ValueError(
            "time_values must have the same length as the number of frames "
            "in the orientation result."
        )

    time_to_index = {t: i for i, t in enumerate(time_values)}

    agg["t_index"] = agg[time_col].map(time_to_index)
    agg = agg.dropna(subset=["t_index"]).copy()
    agg["t_index"] = agg["t_index"].astype(int)

    DX = np.full((n_time, n_bins_y, n_bins_x), np.nan)
    DY = np.full((n_time, n_bins_y, n_bins_x), np.nan)
    N = np.zeros((n_time, n_bins_y, n_bins_x), dtype=int)

    t_idx = agg["t_index"].to_numpy(int)
    y_idx = agg["y_bin"].to_numpy(int)
    x_idx = agg["x_bin"].to_numpy(int)

    DX[t_idx, y_idx, x_idx] = agg["dx"].to_numpy()
    DY[t_idx, y_idx, x_idx] = agg["dy"].to_numpy()
    N[t_idx, y_idx, x_idx] = agg["n"].to_numpy()

    # ---------------------------------------------------------------------
    # 7. Dot product, cross product, and normalized alignment
    # ---------------------------------------------------------------------
    orient_mag = np.hypot(U, V)
    disp_mag = np.hypot(DX, DY)

    dot = U * DX + V * DY
    cross = U * DY - V * DX

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = dot / (orient_mag * disp_mag)

    invalid = (
        (orient_mag == 0) |
        (disp_mag == 0) |
        (N < min_n) |
        ~np.isfinite(cos_angle)
    )

    cos_angle[invalid] = np.nan
    dot[invalid] = np.nan
    cross[invalid] = np.nan

    # For fiber/nematic orientation, direction sign does not matter.
    # S = +1 means parallel or antiparallel.
    # S =  0 means random/45 degrees.
    # S = -1 means perpendicular.
    nematic_S = 2 * cos_angle**2 - 1
    abs_cos_angle = np.abs(cos_angle)

    # ---------------------------------------------------------------------
    # 8. Spatial average per time frame
    # ---------------------------------------------------------------------
    def nanmean_per_time(A):
        return np.nanmean(A, axis=(1, 2))

    def weighted_nanmean_per_time(A, W):
        valid = np.isfinite(A) & np.isfinite(W) & (W > 0)
        numerator = np.sum(np.where(valid, A * W, 0), axis=(1, 2))
        denominator = np.sum(np.where(valid, W, 0), axis=(1, 2))

        out = np.full(A.shape[0], np.nan)
        good = denominator > 0
        out[good] = numerator[good] / denominator[good]
        return out

    if weight_by is None:
        alignment_nematic = nanmean_per_time(nematic_S)
        alignment_abs_cos = nanmean_per_time(abs_cos_angle)
        alignment_signed_cos = nanmean_per_time(cos_angle)
    elif weight_by == "n":
        W = N.astype(float)
        alignment_nematic = weighted_nanmean_per_time(nematic_S, W)
        alignment_abs_cos = weighted_nanmean_per_time(abs_cos_angle, W)
        alignment_signed_cos = weighted_nanmean_per_time(cos_angle, W)
    elif weight_by == "disp_mag":
        W = disp_mag
        alignment_nematic = weighted_nanmean_per_time(nematic_S, W)
        alignment_abs_cos = weighted_nanmean_per_time(abs_cos_angle, W)
        alignment_signed_cos = weighted_nanmean_per_time(cos_angle, W)
    elif weight_by == "orient_mag":
        W = orient_mag
        alignment_nematic = weighted_nanmean_per_time(nematic_S, W)
        alignment_abs_cos = weighted_nanmean_per_time(abs_cos_angle, W)
        alignment_signed_cos = weighted_nanmean_per_time(cos_angle, W)
    else:
        raise ValueError("weight_by must be one of {'n', 'disp_mag', None}.")

    out = {
        "results": results,
        "U": U,
        "V": V,
        "DX": DX,
        "DY": DY,
        "N": N,
        "dot": dot,
        "cross": cross,
        "cos_angle": cos_angle,
        "abs_cos_angle": abs_cos_angle,
        "nematic_S": nematic_S,
        "alignment_nematic": alignment_nematic,
        "alignment_abs_cos": alignment_abs_cos,
        "alignment_signed_cos": alignment_signed_cos,
        "agg": agg,
        "displacements": d,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "time_values": time_values,
    }

    return out



from pathlib import Path
import tifffile as tiff

def load_triplet(folder: str, exposure_dict):
    """
    Load the XML dataframe and both wavelength STK stacks
    for one exposure group.

    Returns
    -------
    df : pandas.DataFrame
        Data read from the XML file.
    stk_488 : ndarray
        Stack containing 'Tirf488' in the filename.
    stk_640 : ndarray
        Stack containing 'Tirf640' in the filename.
    """
    xml_path = Path(exposure_dict["xml"][0])
    stk_files = [Path(s) for s in exposure_dict["stk"]]

    stk_488 = next((p for p in stk_files
                    if "488" in p.name and "tirf" in p.name.lower()), None)
    stk_640 = next((p for p in stk_files
                    if "640" in p.name and "tirf" in p.name.lower()), None)

    if not stk_488 or not stk_640:
        raise ValueError(f"Could not find both 488‑ and 640‑channel STKs in {folder}")

    df = fct.spt.read_xml(xml_path)
    stack_488 = tiff.imread(stk_488)
    stack_640 = tiff.imread(stk_640)

    return df, stack_488, stack_640
    

def parse_all_files(path, timeunit_to_ms):

    """Group stk/xml files by (folder, exposure-label) via timeunit parsing.

    timeunit_to_ms: callable, e.g. fct.spt.timeunit_to_ms
    """
    from collections import defaultdict

    experiments = defaultdict(lambda: defaultdict(lambda: {"stk": [], "xml": [], "tifxml": []}))
    with open(path) as f:
        for line in f:
            file = Path(line.strip())
            name = file.stem
            exp_key = str(file.parent)
            sub_key = f"{timeunit_to_ms(name):g}ms"
            suffix = file.suffix.lower()
            if suffix == ".stk":
                experiments[exp_key][sub_key]["stk"].append(str(file))
            elif suffix == ".xml":
                kind = "tifxml" if name.endswith(".tif") else "xml"
                experiments[exp_key][sub_key][kind].append(str(file))
    return experiments


def apply_manual_overrides(experiments, manual_groups):
    """Replace auto-grouped entries for folders with hand-specified groupings."""
    for folder, groups in manual_groups.items():
        if folder not in experiments:
            continue
        experiments[folder].clear()
        for label, filespec in groups.items():
            for kind in ("stk", "xml"):
                for name in filespec.get(kind, []):
                    experiments[folder][label][kind].append(str(Path(folder) / name))
    return experiments


def print_summary(experiments):
    for folder, subdict in experiments.items():
        print(f"\n{folder}")
        for exposure, data in sorted(subdict.items()):
            print(f"  {exposure}:  {len(data['stk'])} stk,  {len(data['xml'])} xml,  {len(data['tifxml'])} tifxml")


def check_experiments(experiments, expect=(2, 1, 0)):
    """Split exposures into (ok, flagged) by (n_stk, n_xml, n_tifxml) counts.

    Folders that contain only tif.xml entries (no stk/xml at all) are skipped.
    """
    ok, flagged = [], []
    for folder, subdict in experiments.items():
        if all(len(v["stk"]) == 0 and len(v["xml"]) == 0 and len(v["tifxml"]) > 0
               for v in subdict.values()):
            continue
        for exposure, data in subdict.items():
            counts = (len(data["stk"]), len(data["xml"]), len(data["tifxml"]))
            (ok if counts == expect else flagged).append((folder, exposure, *counts))
    # ok entries only need (folder, exposure); trim the counts back off
    ok = [(folder, exposure) for folder, exposure, *_ in ok]
    return ok, flagged


def classify_folder(folder):
    """Assign a folder to a strain group based on its name, or None if unmatched."""
    name = folder.lower()
    if "sumo" in name:
        return "SUMO"
    if "star" in name:
        return "star"
    if "wt" in name:
        return "wt"
    return None


def build_results(experiments, ok_experiments):
    results = {"star": {}, "wt": {}, "SUMO": {}}
    for folder, exposure in ok_experiments:
        group = classify_folder(folder)
        if group is None:
            continue  # e.g. ZipA folders currently fall through unclassified
        label = f"{Path(folder).name}_{exposure}"
        results[group][label] = {
            "folder": folder,
            "exposure": exposure,
            **experiments[folder][exposure],
        }
    return results


def load_triplet(folder: str, exposure_dict):
    """
    Load the XML dataframe and both wavelength STK stacks
    for one exposure group.

    Returns
    -------
    df : pandas.DataFrame
        Data read from the XML file.
    stk_488 : ndarray
        Stack containing 'Tirf488' in the filename.
    stk_640 : ndarray
        Stack containing 'Tirf640' in the filename.
    """
    xml_path = Path(exposure_dict["xml"][0])
    stk_files = [Path(s) for s in exposure_dict["stk"]]

    stk_488 = next((p for p in stk_files if "488" in p.name and "tirf" in p.name.lower()), None)
    stk_640 = next((p for p in stk_files if "640" in p.name and "tirf" in p.name.lower()), None)

    if not stk_488 or not stk_640:
        raise ValueError(f"Could not find both 488- and 640-channel STKs in {folder}")

    df = fct.spt.read_xml(xml_path)
    stack_488 = tiff.imread(stk_488)
    stack_640 = tiff.imread(stk_640)

    return df, stack_488, stack_640


def flatten_results(results_all):
    """Flatten results_all into a list of experiment records, one per exposure.
 
    Each record carries group/label plus everything load_triplet needs, so a
    job just needs an integer index into this list -- no dict traversal.
    """
    flat = []
    for group, exps in results_all.items():
        for label, data in exps.items():
            flat.append({"group": group, "label": label, **data})
    return flat
 
 
def iter_loaded_experiments(results_all):
    """Lazily yield (group, label, df, stack_488, stack_640) for each experiment.
 
    Loads one experiment at a time so large STK stacks aren't all held in
    memory at once.
    """
    for record in flatten_results(results_all):
        df, stack_488, stack_640 = load_triplet(record["folder"], record)
        yield record["group"], record["label"], df, stack_488, stack_640
 
 
def save_index(flat, path="experiment_index.json"):
    import json
    with open(path, "w") as f:
        json.dump(flat, f, indent=2)
 
 
def load_index(path="experiment_index.json"):
    import json
    with open(path) as f:
        return json.load(f)

def collect_analysis(results_all, filename, analysis_dir="./", key="analysis"):
    """Attach each job's pickled output back onto results_all, matched by index.
 
    Looks in <record["folder"]>/<analysis_dir>/<filename> for each experiment.
 
    filename : str.format template, given idx/group/label, e.g.
        "{idx}.pkl", "{group}_{label}.pkl", "{idx}_tracking.pkl"
    key : name under which the loaded object is stored in results_all
    """
    import pickle
 
    flat = flatten_results(results_all)
    for idx, record in enumerate(flat):
        pkl_path = Path(record["folder"]) / analysis_dir / filename.format(idx=idx, **record)
        if not pkl_path.exists():
            print(f"missing: {pkl_path} ({record['group']}/{record['label']})")
            continue
        with open(pkl_path, "rb") as f:
            analysis = pickle.load(f)
        results_all[record["group"]][record["label"]][key] = analysis
    return results_all


