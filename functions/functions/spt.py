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
        # Standard global scatter logic
        sc = ax.scatter(df["x"], df["y"],
                        c=df[colorby],
                        cmap=cmap,
                        vmax=vmax,
                        vmin=vmin,
                        s=s,
                        zorder=zorder)
    
    ax.set_aspect("equal")
    return sc


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


def plot_msd_alpha(xmlfile, MSD_analysis, ax, color="steelblue", show_filtered=True):
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
                   s=0.1, color="lightgrey", alpha=0.2, 
                   zorder=1, rasterized=True)

    # 4. Foreground: Plot analyzed traces
    df_valid = df[df["alpha"].notna()]
    sc = df_scatter(df_valid, ax, s=0.5, colorby="alpha", cmap="viridis", 
                    vmax=2, vmin=0.8, zorder=2)
    
    # ax.set_title(Path(xmlfile).stem, fontsize=10)
    
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
    bp = ax.boxplot(data_dict.values(), positions=positions,
                    widths=widths, patch_artist=True,
                    showfliers=False, # Removes outliers to keep the plot clean
                    medianprops=dict(color="black", linewidth=2))
    
    # Apply custom colors and alpha to the box patches
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
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
def sliding_window_span(df, window_sizes, step_size=1):
    records = []
    for pid, group in df.groupby("id"):
        coords = group[["x", "y"]].to_numpy()
        n_points = len(coords)

        for L in window_sizes:
            if n_points < L:
                continue
            for start in range(0, n_points - L + 1, step_size):
                window = coords[start : start + L]
                span = window_max_span(window)
                if np.isfinite(span):
                    records.append({"pid": pid, "L": L, "span": span})

    return pd.DataFrame(records)