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
colorby="t_zeroed", cmap="jet", zorder=99):
    df = df.copy()
    df.loc[:, "t_zeroed"] = df["time"] - df.groupby("id")["time"].transform("min")
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
    D = MSD_analysis[xmlfile]["D"].get(pid, np.nan)
    df = df_read_fn(xmlfile)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{pathlib.Path(xmlfile).stem} | pid={pid} | α={alpha:.3f}")

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
    df_scatter(df[df["id"] == pid], axes[1], s=5)

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
    
    ax.set_title(Path(xmlfile).stem, fontsize=10)
    
    # 4. Create Inset (Lower Right Corner)
    # [x0, y0, width, height] in normalized axis coordinates
    ax_ins = ax.inset_axes([0.15, 0.05, 0.2, 0.35])
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