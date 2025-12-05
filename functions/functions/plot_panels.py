import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_func(
    df,
    t,
    savefig=False,
    savepath=None,
    figsize=(6, 4),
    nearest_frame=True,
    scatter_kwargs=None,
    **kwargs
):
    """
    Wrapper to produce a single-frame plot.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ["time", "x", "y", "type"].
    t : float or int
        Requested time.
    savefig : bool
        Whether to save figure.
    savepath : str
        Optional override for output filename.
    nearest_frame : bool
        If True → use closest frame.
        If False → require exact df["time"] == t.
    scatter_kwargs : dict
        Extra arguments passed into scatter_fct().
    kwargs : dict
        Passed into scatter_fct().
    """

    scatter_kwargs = scatter_kwargs or {}

    # --- Get frame ---
    times = df["time"].values
    if nearest_frame:
        t_frame = times[np.abs(times - t).argmin()]
    else:
        # exact match
        if t not in times:
            raise ValueError(f"Exact time {t} not found in df['time']")
        t_frame = t

    fig, ax = plt.subplots(figsize=figsize)

    scatter_fct(df, ax, t_frame, **scatter_kwargs, **kwargs)

    if savefig:
        # Determine base filename (without extension)
        if savepath:
            # remove .png/.svg/etc if user supplied full filename
            import os
            root, _ = os.path.splitext(savepath)
        else:
            root = f"plot_{t}"

        # Save both PNG + SVG
        fig.savefig(f"{root}.png", dpi=500, bbox_inches="tight")
        fig.savefig(f"{root}.svg", bbox_inches="tight")

    return fig, ax



def scatter_fct(
    df,
    ax,
    t_frame,
    # Feature toggles
    plot_processive=True,
    plot_type6=True,
    plot_filaments=True,
    plot_activator=False,
    display_time=True,
    display_legend=True,
    add_arrows=False,
    tracewindow=0,

    # Display tuning
    sZ=10,
    s_synth=20,
    s_processive=20,
    filament_color="#4cc9f0", #"#24cedbff",
    type6_color="#f72585",
    processive_color="#f72585",
    activator_color="#023047",

    marker_filament="o",
    marker_type6="s",
    marker_processive="s",
    marker_activator="v",

    # coordinate scaling
    scale_xy=5,

    # axis limits
    xlim=None,
    ylim=(-150, 150),

    # quantile lines for type6
    show_quantiles=True,
    quantile_range=(-30, 30),
    quantiles=(0.2, 0.8),

    # legend options
    legend_loc="lower center",
    legend_ncol=3,

    # axis label / ticks
    hideticklabels=True,
    xlabel="Cell circumference (nm)",
    ylabel="Long cell axis (nm)",
):
    """
    Scatter plot for one time frame.

    All inputs are optional & customizable.
    """

    # --- Extract frame ---
    D = df.loc[df["time"] == t_frame]

    D_fil = D.loc[D["type"].isin([1, 2, 3])]
    D_6   = D.loc[D["type"].isin([5, 6])]
    D_proc = D.loc[D["type"].isin([5, 9])]
    D_act = D.loc[D["type"].isin([8])]

    # --- Filaments ---
    if plot_filaments and not D_fil.empty:
        ax.scatter(*(D_fil[["x","y"]].values.T * scale_xy),
                   c=filament_color, s=sZ, marker=marker_filament)

    # --- Type 6 ---
    if plot_type6 and not D_6.empty:
        ax.scatter(*(D_6[["x","y"]].values.T * scale_xy),
                   c=type6_color, s=s_synth, marker=marker_type6)

    # --- Processive synthases ---
    if plot_processive:

        # traces
        if tracewindow > 0:
            prev_times = [t_frame - i for i in range(1, tracewindow + 1)]
            for j, tp in enumerate(prev_times):
                D_prev = df.loc[(df["time"] == tp) & (df["type"].isin([5, 9]))]
                if D_prev.empty:
                    continue
                alpha = 1 - (j + 1) / (tracewindow + 1)
                ax.scatter(*(D_prev[["x","y"]].values.T * scale_xy),
                           c=processive_color, s=s_processive,
                           marker=marker_processive, alpha=alpha)

        # current
        if not D_proc.empty:
            ax.scatter(*(D_proc[["x","y"]].values.T * scale_xy),
                       c=processive_color, s=s_processive,
                       marker=marker_processive)

        # arrows
        if add_arrows and not D_proc.empty:
            D5 = D_proc.loc[D_proc["type"] == 5]
            D9 = D_proc.loc[D_proc["type"] == 9]

            width = 0.002

            if not D5.empty:
                ax.quiver(D5["x"]*scale_xy, D5["y"]*scale_xy,
                          [100]*len(D5), [0]*len(D5),
                          angles="xy", scale_units="xy", scale=5,
                          width=width, color=processive_color)

            if not D9.empty:
                ax.quiver(D9["x"]*scale_xy, D9["y"]*scale_xy,
                          [-100]*len(D9), [0]*len(D9),
                          angles="xy", scale_units="xy", scale=5,
                          width=width, color=processive_color)

    # --- Activator ---
    if plot_activator and not D_act.empty:
        ax.scatter(*(D_act[["x","y"]].values.T * scale_xy),
                   c=activator_color, s=20, marker=marker_activator)

    # --- Time text ---
    if display_time:
        txt = ax.text(0.8, 0.8, f"t = {round(t_frame)} s",
                      transform=ax.transAxes, ha="center", va="center")
        txt.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))

    # --- Quantile lines ---
    if show_quantiles and not D_6.empty:
        yvals = D_6.loc[(np.abs(D_6["y"]) < abs(quantile_range[1]))]["y"] * scale_xy
        for q in quantiles:
            qv = np.quantile(yvals, q)
            ax.hlines(qv, xmin=-150*scale_xy, xmax=150*scale_xy,
                      ls="--", lw=1, color=type6_color, alpha=0.7)

        # --- Axes ---
    ax.set_aspect("equal")
    if xlim is None:
        xlim = (df["x"].min() * scale_xy, df["x"].max() * scale_xy)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if hideticklabels:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # --- Legend ---
    if display_legend:
        handles = []

        if plot_filaments:
            handles.append(Line2D([0], [0], label="FtsZ",
                                  c=filament_color, markersize=6,
                                  marker=marker_filament, linestyle=""))

        if plot_type6:
            handles.append(Line2D([0], [0], label="Synthase",
                                  c=type6_color, markersize=8,
                                  marker=marker_type6, linestyle=""))

        if plot_processive:
            handles.append(Line2D([0], [0], label="Processive",
                                  c=processive_color, markersize=8,
                                  marker=marker_processive, linestyle=""))

        if plot_activator:
            handles.append(Line2D([0], [0], label="Activator",
                                  c=activator_color, markersize=6,
                                  marker=marker_activator, linestyle=""))

        ax.legend(handles=handles,
                  bbox_to_anchor=(0.5, 1.02),
                  loc=legend_loc, ncol=legend_ncol,
                  frameon=False)

    return ax
