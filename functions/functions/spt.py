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
colorby="t_zeroed", cmap="jet"):
    df = df.copy()
    df.loc[:, "t_zeroed"] = df["time"] - df.groupby("id")["time"].transform("min")
    sc=ax.scatter(*df[["x", "y"]].values.T,
                          c=df[colorby],
                          cmap=cmap,
                          vmax=vmax,
                          vmin=vmin,
                          s=s)
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

