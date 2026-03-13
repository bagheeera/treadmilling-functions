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


def df_scatter(df, ax, s=0.5, vmax=None, vmin=None,
    colorby="t_zeroed", cmap="jet", zorder=99,
    useplot=False,
):
    df = df.copy()
    df.loc[:, "t_zeroed"] = df["time"] - df.groupby("id")["time"].transform("min")
    if useplot:
        sc = ax.plot(*df[["x", "y"]].values.T,)
        sc=ax.scatter(*df[["x", "y"]].values.T,
                            c=df[colorby],
                            cmap=cmap,
                            vmax=vmax,
                            vmin=vmin,
                            s=s,
                            zorder=zorder)
    else:
        sc=ax.scatter(*df[["x", "y"]].values.T,
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
                    show_loglog=True):
    """
    Like plot_pid but:
      - fit overlay uses parameters from log-log fitting
      - optionally adds a third panel with the log-log fit for visual QC
    """
    alpha = MSD_analysis[xmlfile]["alpha"].get(pid, np.nan)
    r2    = MSD_analysis[xmlfile]["r2"].get(pid, np.nan)
    D     = MSD_analysis[xmlfile]["D"].get(pid, np.nan)

    df      = df_read_fn(xmlfile)
    msd_df  = data[(data["file"] == xmlfile) & (data["pid"] == pid)]

    ncols = 3 if show_loglog else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.suptitle(
        f"{pathlib.Path(xmlfile).stem} | pid={pid} | "
        f"α={alpha:.3f} | R²(log)={r2:.3f}"
    )

    # ── panel 0: MSD linear scale ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(msd_df["tau"], msd_df["MSD"], linewidth=2, label=f"PID {pid}")
    ax.fill_between(
        msd_df["tau"],
        msd_df["MSD"] - msd_df["MSD_std"],
        msd_df["MSD"] + msd_df["MSD_std"],
        alpha=0.2, label="± std"
    )
    if not np.isnan(alpha):
        tau_fit = np.linspace(msd_df["tau"].min(), msd_df["tau"].max(), 300)
        ax.plot(tau_fit, D * tau_fit**alpha, "--", color="tomato",
                label=f"fit: Dτ^α")
    ax.set_xlabel("τ")
    ax.set_ylabel("MSD")
    ax.legend(fontsize=8)

    # ── panel 1: trajectory ────────────────────────────────────────────────────
    df_scatter(df[df["id"] == pid], axes[1], s=20, useplot=True)

    # ── panel 2: log-log fit (optional, great for QC) ─────────────────────────
    if show_loglog:
        ax2 = axes[2]
        pos = (msd_df["tau"] > 0) & (msd_df["MSD"] > 0)
        log_tau = np.log(msd_df["tau"][pos])
        log_msd = np.log(msd_df["MSD"][pos])
        ax2.scatter(log_tau, log_msd, s=15, label="log MSD", zorder=3)
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

def browse_datasets(file_list, msd_results, color="steelblue", label="Dataset"):
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


