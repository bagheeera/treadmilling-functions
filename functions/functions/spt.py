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
from synthana import analysis, utils


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
    # violin in background
    parts = ax.violinplot(data_dict.values(), positions=positions,
                    showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(alpha=alpha)

    # boxplot in foreground
    bp = ax.boxplot(data_dict.values(), positions=positions,
                widths=widths, patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)

    ax.set_xticks(positions)
    ax.set_xticklabels(data_dict.keys())



import re

def extract_timeunit(s):
    match = re.search(r'(\d+)\s*(ms|s)(?=[^a-zA-Z]|$)', s, re.IGNORECASE)
    if match:
        return match.group(1) + " " + match.group(2).lower()
    return "0 ms" #None, None