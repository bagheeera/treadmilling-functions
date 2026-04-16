import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tqdm as tqdm

def plot_func(
    df,
    t,
    savefig=False,
    savepath=None,
    figsize=(7, 4),
    # xlim=(-150*5, 150*5),
    # sZ=1,
    # s_synth=5,
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

    fig, ax = plt.subplots(figsize=figsize,
                            # constrained_layout=True
                            )

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


import matplotlib.pyplot as plt

def plot_panels_with_hist(
    df,
    t_frame,
    scale=1.0,
    Nbins=25,
    colors=('#f72585', '#4cc9f0'),
    y_window=30,
    y_scale=5,
    xlim=(-700, 700),
    figsize_base=(12, 2.5),
    width_ratios=(4, 1.2),
    wspace=0.05,
    scatter_kwargs=None,
    show=True,
    savepath=None,
    histxlim=None
):
    """
    Combined transport scatter plot with vertical density histograms.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    t_frame : int
        Time frame to plot.
    fct : module or object
        Must provide fct.midcell_transport.scatter_fct.
    scale : float
        Global figure scaling.
    Nbins : int
        Number of histogram bins.
    colors : tuple
        Colors for (Synthase, FtsZ).
    y_window : float
        Y-range (±) used for histogram filtering.
    y_scale : float
        Scaling applied to y-values in histogram.
    xlim : tuple
        X-axis limits for scatter plot.
    figsize_base : tuple
        Base figure size before scaling.
    width_ratios : tuple
        Width ratios for (scatter, histogram).
    wspace : float
        Horizontal spacing between axes.
    scatter_kwargs : dict
        Extra keyword arguments passed to scatter_fct.
    show : bool
        Whether to call plt.show().
    savepath : str or None
        If provided, saves figure to this path.

    Returns
    -------
    fig, (ax0, ax1)
    """

    if scatter_kwargs is None:
        scatter_kwargs = {}

    fig, (ax0, ax1) = plt.subplots(
        1, 2,
        figsize=(scale * figsize_base[0], scale * figsize_base[1]),
        sharey=True,
        gridspec_kw={
            "width_ratios": width_ratios,
            "wspace": wspace,
        },
    )

    # === LEFT AX: scatter ===
    scatter_fct(
        df,
        ax0,
        t_frame,
        xlim=list(xlim),
        **scatter_kwargs,
    )

    # === RIGHT AX: histograms ===
    df_Z = df[(df["type"].isin([1, 2, 3])) & (df["time"] == t_frame)]
    df_synth = df[(df["type"].isin([5, 6])) & (df["time"] == t_frame)]

    for dfi, color, label in zip(
        [df_synth, df_Z],
        colors,
        ["Synthase", "FtsZ"],
    ):
        dfi = dfi[dfi["y"].between(-y_window, y_window)]
        ax1.hist(
            dfi["y"] * y_scale,
            bins=Nbins,
            histtype="step",
            lw=2,
            density=True,
            alpha=0.7,
            orientation="horizontal",
            color=color,
            label=label,
        )

    ax1.set_xlabel("Density")
    ax1.legend()
    if histxlim is not None:
        ax1.set_xlim(*histxlim)
    ax1.set_ylim(-y_window * y_scale, y_window * y_scale)

    # === axis consistency ===
    ax0.set_aspect("equal", adjustable="datalim")
    ax1.set_ylim(ax0.get_ylim())

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, (ax0, ax1)


## monomer displacement functions
from scipy.stats import binned_statistic_2d

def filter_by_time(xy, d_xy, time, T_range):
    # https://chatgpt.com/c/6790ce99-4950-8011-a937-642f9f7f5c2a
    """Filter the results based on a given time range."""
    # Flatten the lists and filter based on the time range
    all_xy = np.concatenate(xy)
    all_d_xy = np.concatenate(d_xy)
    all_time = np.concatenate(time)
    
    time_mask = (all_time >= T_range[0]) & (all_time <= T_range[1])
    
    # Apply the mask to each array
    filtered_xy = all_xy[time_mask]
    filtered_d_xy = all_d_xy[time_mask]
    filtered_time = all_time[time_mask]
    
    return filtered_xy, filtered_d_xy, filtered_time

def bin_dxy(xy, d_xy, N_bins=25, yrange=35):

    
    x_minmax = -yrange, yrange # np.min(xy[:,0]), np.max(xy[:,0])
    y_minmax = -yrange, yrange #np.min(xy[:,1]), np.max(xy[:,1]),
    x_minmax, y_minmax
    
    
    bins = [np.linspace(x_minmax[0], x_minmax[1], N_bins),
           np.linspace(y_minmax[0], y_minmax[1], N_bins)]
    
    d_x_mean, x_edges, y_edges, binnumber = binned_statistic_2d(
        xy[:, 0],
        xy[:, 1],
        d_xy[:, 0],  # x-component of vectors
        statistic='mean',
        bins=bins
    )
    
    d_y_mean, _, _, _ = binned_statistic_2d(
        xy[:, 0],
        xy[:, 1],
        d_xy[:, 1],  # y-component of vectors
        statistic='mean',
        bins=bins
    )

    return x_minmax, y_minmax, d_x_mean, d_y_mean, x_edges, y_edges, binnumber

def disp_plot(ax,
    x_minmax, y_minmax, d_y_mean, x_edges, y_edges, 
    binnumber, cut=1,
    arrowscale=1,
    vmax=.01):
    import matplotlib.pyplot as plt
    import numpy as np
    
    im = ax.imshow(
        d_y_mean.T,
        origin='lower',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        vmin=-vmax, vmax=vmax,
        cmap="bwr"
    )
    #ax.axhline(y=0, c="k", alpha=.3)
    
    # Define the arrow grid density
    y_density = 4  # e.g., use every 4th bin along the y-axis
    
    # Calculate the center points for the arrows
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Slice the y-centers and d_y_mean.T to reduce the number of arrows along y
    y_quiver_centers = y_centers[::y_density]
    d_y_mean_quiver = d_y_mean.T[::y_density, :]
    
    # Create a meshgrid for the arrow origins
    X, Y = np.meshgrid(x_centers, y_quiver_centers)
    
    # Define the arrow components
    U = np.zeros_like(d_y_mean_quiver)
    V = d_y_mean_quiver
    V = d_y_mean_quiver * 1000
    
    #cut = 1
    
    V[V>cut]=cut
    
    V[V<-cut]=-cut 
    
    # Overlay the quiver plot with a more direct approach
    # Use the V values for coloring
    ax.quiver(X, Y, U, V, #V, cmap='bwr',
              pivot='mid',
              scale_units='xy',
              scale=arrowscale, 
              #scale=1000.0 / (2 * vmax))  # Dynamic scaling based on vmax
             )
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #cbar = fig.colorbar(im, shrink=.4)
    #cbar.ax.invert_yaxis()
    return im
    #plt.show()

def orientation_plot(ax, x_minmax, y_minmax, d_x_mean, d_y_mean, x_edges, y_edges,
    y_density=4, cut=300,
    quiverscale=5):
    ## plot orientation of displacement vectors
    ## color by cosine of angle with inward normal

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    y_quiver_centers = y_centers[::y_density]

    d_y_mean_quiver = d_y_mean.T[::y_density, :]
    d_x_mean_quiver = d_x_mean.T[::y_density, :]

    X, Y = np.meshgrid(x_centers, y_quiver_centers)

    U = d_x_mean_quiver #* 1000
    V = d_y_mean_quiver #* 1000

    # U = np.clip(U, -cut, cut)
    # V = np.clip(V, -cut, cut)


    ## mark center
    x0 = np.mean(x_centers)
    y0 = np.mean(y_quiver_centers)
    # inward vectors
    RX = x0 - X
    RY = y0 - Y

    # calc cosine of the angle between the arrow direction and the inward normal
    Rnorm = np.sqrt(RX**2 + RY**2)
    RXn = RX / Rnorm
    RYn = RY / Rnorm

    Vnorm = np.sqrt(U**2 + V**2)

    # avoid division by zero
    mask = Vnorm > 0
    Un = np.zeros_like(U)
    Vn = np.zeros_like(V)

    Un[mask] = U[mask] / Vnorm[mask]
    Vn[mask] = V[mask] / Vnorm[mask]

    C = Un * RXn + Vn * RYn



    q = ax.quiver(
        X, Y, U, V,
        C,
        cmap='bwr',
        #vmin=-1, vmax=1,
        pivot='mid',
        scale_units='xy',
        scale=quiverscale #.01
    )
    #ax.colorbar(q, ax=ax, label="cosθ")
    return ax



def plot_filament_age_hist(ax, processed_data, label=None, 
                           color="orangered", ylabel="Long cell axis (nm)"):
    """
    Plots the pre-processed mean profile and shaded error bars.
    ###
    Usage:
    profile_data = get_averaged_age_profile(
        D[key]["filament_age_profiles"], 
        t_min=10, 
        t_max=50, 
        as_density=True
    )

    plot_filament_age_hist(ax, profile_data, label="Filament A", color="blue")
    """
    if processed_data is None:
        return
    
    y_vals = processed_data["y_bins"]
    mu = processed_data["mean"]
    sigma = processed_data["std"]

    # Plot the mean line
    line, = ax.plot(mu, y_vals, lw=2, label=label, color=color)
    
    # Shaded area for ±1 STD
    ax.fill_betweenx(
        y_vals,
        mu - sigma,
        mu + sigma,
        color=line.get_color(),
        alpha=0.3,
    )
    
    if ylabel:
        ax.set_ylabel(ylabel)

def get_averaged_age_profile(age_profiles, t_min, t_max, as_density=False):
    """
    Filters age profiles by time and computes the mean and std across the window.
    """
    # Select times in window
    times = [t for t in sorted(age_profiles.keys()) if t_min <= t <= t_max]

    if not times:
        return None
    
    mean_list = []
    y_bins = None
    
    for t in times:
        dft = age_profiles[t]
        values = dft["mean"].values.copy()
        
        if as_density:
            # Calculate bin width for density normalization
            bin_width = np.diff(dft["y_bin_center"].values).mean()
            total = np.nansum(values)
            if total > 0:
                values = values / (total * bin_width)
        
        mean_list.append(values)
        
        if y_bins is None:
            y_bins = dft["y_bin_center"].values
    
    # Compute stats
    mean_array = np.array(mean_list)
    return {
        "mean": np.nanmean(mean_array, axis=0),
        "std": np.nanstd(mean_array, axis=0),
        "y_bins": 5 * np.array(y_bins)  # Applying the 5x scaling here
    }


import pandas as pd
def compute_max_age_profiles(df, times, y_range=(-35, 35), nbins_y=30):
    """
    Compute mean and SEM of max mol ages binned by y position for given time points.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['id', 'time', 'mol', 'y']
        times (list): List of time points to compute profiles at
        y_bin_width (float): Width of each y bin
        y_range (tuple): Range multiplier for y bins (used with y_bin_width)
        nbins_y (int): Number of y bins (overrides y_bin_width and y_range if given)

    Returns:
        dict: {time: result_df with columns ['y_bin_center', 'mean', 'sem']}
    """
    df = df.copy()

    # Compute first_time and age
    df['first_time'] = df.groupby('id')['time'].transform('first')
    df['age'] = df['time'] - df['first_time']

    # Compute max age per mol
    mol_max_age = df.groupby('mol')['age'].max()

    # Define y bins
    y_min = y_range[0] #* y_bin_width
    y_max = y_range[1] #* y_bin_width
    y_bins = np.linspace(y_min, y_max, nbins_y)

    profiles = {}

    for t in tqdm(times):
        df_t = df[df['time'] == t].copy()

        df_t['mol_age'] = df_t['mol'].map(mol_max_age)
        df_t['y_bin'] = pd.cut(df_t['y'], bins=y_bins)

        grouped = df_t.groupby('y_bin')['mol_age']
        result_df = grouped.mean().reset_index(name='mean')
        result_df['sem'] = grouped.sem().values
        result_df['y_bin_center'] = result_df['y_bin'].apply(lambda x: (x.left + x.right) / 2)

        profiles[t] = result_df
    return profiles


import numpy as np
import matplotlib.pyplot as plt

def plot_division(df, ax,
                  plot_processive=True,
                  plot_type6=True,
                  display_time=False,
                  display_legend=True,
                  tracewindow=0,   # number of previous frames to show traces
                  add_arrows=False, # add arrows to type 5 (→) and type 9 (←)
                  synths=6,
                  hideticklabels=False,
                  ylim=30,
                  show_quantiles=True,
                  sZ=1):
    """
    Plot the current frame of a filament/division dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'time', 'x', 'y', 'type'.
    ax : matplotlib.axes.Axes
        Axis on which to draw.
    Other parameters:
        See function signature.
    """
    # Get latest frame
    D = df[df["time"] == df["time"].max()]
    
    # Separate types
    D_filament = D[D["type"].isin([1,2,3])]
    D_divi = D[D["type"].isin([5,6])]
    
    # Plot filaments
    ax.scatter(D_filament["x"]*5, D_filament["y"]*5, c='#4cc9f0', s=sZ)
    
    # Plot type 6 (division) if requested
    if plot_type6 and not D_divi.empty:
        ax.scatter(D_divi["x"]*5, D_divi["y"]*5, c="#f72585", s=synths, marker="s")
        
        # Optional quantile lines for type 5/6 division
        for q in [0.2, 0.8]:
            mask = abs(D_divi["y"]) < 30
            if mask.any():
                quant = np.quantile(D_divi.loc[mask, "y"]*5, q)
                if show_quantiles:
                    ax.hlines(y=quant, xmin=df["x"].min()*5, xmax=df["x"].max()*5, ls="--", lw=1,
                          color="#f72585", alpha=0.7)
    
    # Axes settings
    ax.set_xlim(df["x"].min()*5, df["x"].max()*5)
    ax.set_ylim(-ylim*5, ylim*5)
    ax.set_aspect("equal")
    
    if hideticklabels:
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
    
    if display_time:
        t_frame = df["time"].max()
        ax.set_title(f"Time: {t_frame:.1f}")


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import functions as fct

def plot_demograph_for_key(
    key,
    D,
    t_select=1500,
    time_step=10,
    y_lim=60,
    hist_y_cut=40,
    synth_y_cut=30,
    heatmap_bins=(-40, 40, 70),
    cmap="RdPu",
    scale=0.8,
    fct=fct,
    savepath=None,
    histo_bins=40,
    quant=0.2,
):
    """
    Generate heatmap + final-time histogram for a single key.

    Parameters
    ----------
    key : hashable
        Key identifying the run in D
    D : dict
        Data dictionary containing rundir and storage for results
    t_select : int
        Time point used for final distributions
    time_step : int
        Time bin width
    y_lim : float
        Y-axis limit for plots (nm)
    hist_y_cut : float
        Absolute y-cut for histogram preprocessing
    synth_y_cut : float
        Absolute y-cut for synth histogram
    heatmap_bins : tuple
        (ymin, ymax, nbins) for heatmap histograms
    cmap : str
        Colormap for heatmap
    scale : float
        Overall figure scaling
    """

    print(key)
    print(os.path.join(os.getcwd(), D[key]["rundir"]))
    # --- Load and preprocess ---
    df = fct.utils.load(D[key]["rundir"])
    df = df[df["time"] < t_select]

    timebins = np.arange(0, int(df["time"].values[-1]), time_step)

    # --- Z quantiles (types 1,2,3) ---
    filtered = df[
        df["type"].isin([1, 2, 3]) & df["time"].isin(timebins)
    ].copy()

    quantiles = [
        filtered.groupby("time")["y"].quantile(q)
        for q in (quant, 1-quant)
    ]

    D[key]["Zquantiles_.2"] = quantiles

    # --- Synth histograms over time (type 5) ---
    df_synth_all = df[df["type"] == 5]

    ymin, ymax, nbins = heatmap_bins
    ybins_heat = np.linspace(ymin, ymax, nbins)

    histos = [
        np.histogram(
            df_synth_all[df_synth_all["time"] == t]["y"],
            bins=ybins_heat,
        )[0]
        for t in timebins
    ]

    D[key]["synth_histos"] = histos

    # --- Final-time histograms ---
    df = fct.utils.load(D[key]["rundir"])
    df = df[np.abs(df["y"]) < hist_y_cut]

    df_Z = df[
        (df["time"] == t_select) &
        (df["type"].isin([1, 2, 3]))
    ]

    df_synth = df[
        (df["time"] == t_select) &
        (df["type"].isin([5]))
    ]
    df_synth = df_synth[np.abs(df_synth["y"]) < synth_y_cut]

    # ybins = np.linspace(df["y"].min(), df["y"].max(), histo_bins)
    ybins = np.linspace(ymin, ymax, histo_bins)
    y_centers = (ybins[:-1] + ybins[1:]) / 2

    y_counts_Z, _ = np.histogram(df_Z["y"], bins=ybins, density=True)
    y_counts_synth, _ = np.histogram(df_synth["y"], bins=ybins, density=True)

    # --- Plot ---
    fig = plt.figure(figsize=(scale * 9, scale * 3.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharey=ax)

    # Heatmap
    im = ax.imshow(
        np.array(histos).T,
        origin="lower",
        extent = (
            0,
            time_step * len(timebins),
            ymin,
            ymax,
        ),
        aspect="auto",
        cmap=cmap,
    )
    fig.colorbar(im, ax=ax, label="Count")

    for q in quantiles:
        ax.plot(
            time_step * np.arange(len(q)),
            5 * q,
            color="#4cc9f0",
            ls="--",
            lw=2,
        )

    ax.plot([], ls="--", lw=2, color="#4cc9f0", label="Z-ring outline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Long cell axis (nm)")
    ax.set_ylim(-y_lim, y_lim)
    ax.set_title("Synthase distribution")
    ax.legend(loc="center left")

    # Histogram
    ax_hist.plot(
        y_counts_Z, y_centers,
        drawstyle="steps-mid",
        color="#4cc9f0",
        lw=2,
        label="FtsZ",
    )
    ax_hist.plot(
        y_counts_synth, y_centers,
        drawstyle="steps-mid",
        color="#fea6c0",
        lw=2,
        label="Synthase",
    )

    ax_hist.set_xlabel("Density")
    ax_hist.set_title("Final distributions")
    ax_hist.legend()
    plt.setp(ax_hist.get_yticklabels(), visible=False)

    # plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)

    return fig
    #plt.show()



import matplotlib.image as mpimg
def plot_sept(D, key, ax, pngscale=3):
    #print(D[key]["rundir"])
    f = D[key]["rundir"] + f"/septumcross_{pngscale}.png"
    if os.path.exists(f):
        img = mpimg.imread(f)
        #print(img.shape)
        #img = img[800:-300,1400:-1400,:]
        img = img[300:-100,500:-500,:]
        ax.imshow(img)
        #plt.axis('off')  # hide axes
    else:
        print(key,  "not found")

import pyarrow.feather as feather
import matplotlib.image as mpimg
def display_png(fname, ax,
crop=(300, -100, 500, -500)):
    #print(D[key]["rundir"])
    if os.path.exists(fname):
        img = mpimg.imread(fname)
        img = img[crop[0]:crop[1],crop[2]:crop[3],:]
        ax.imshow(img)
        ax.axis('off')  # hide axes
    else:
        print("missing", fname)

def plot_nr_active(D, key, ax, overlay):
    if "nr_active" not in D[key]:
        df = feather.read_feather(D[key]["rundir"] + "/df_synth.feather")
        D[key]["nr_active"] = df.groupby("time").size()
        nr_active = D[key]["nr_active"]
    else:
        nr_active = D[key]["nr_active"]
    return ax.plot(nr_active, label=overlay)

import os
import pandas as pd
import pyarrow.feather as feather
def process_synth_to_slim(rundir, output_name="absy_summary.feather"):
    """
    Reads df_synth.feather from a single rundir and saves a tiny summary.
    The summary has 'time' as the index and 'type' as columns.
    """
    input_path = os.path.join(rundir, "df_synth.feather")
    output_path = os.path.join(rundir, output_name)
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # 1. Load only the essential columns
    df = feather.read_feather(input_path, columns=["time", "type", "y"])
    
    # 2. Calculate absolute Y
    df["absy"] = df["y"].abs()
    
    # 3. Pivot the data: Rows=Time, Columns=Type, Values=Mean Absolute Y
    # This creates a "Wide" table: [time, type_12, type_13, etc.]
    summary = df.groupby(["time", "type"])["absy"].mean().unstack(level="type")
    
    # 4. Save with LZ4 compression (Standard for Feather/Arrow speed)
    # We reset index so 'time' is a column for feather compatibility
    feather.write_feather(summary.reset_index(), output_path, compression='lz4')
    
    print(f"Processed {rundir}: Slim file saved ({os.path.getsize(output_path)/1024:.1f} KB)")


def absy_plot(D, key, ax, synthtype=[12], overlay=None, prm=None):
    def _get_mean_absy(k):
        fname = os.path.join(D[k]["rundir"], "absy_summary.feather")
        if not os.path.exists(fname):
            return None
        
        df = feather.read_feather(fname).set_index("time")
        
        # Convert column names (types) to integers to ensure match
        df.columns = df.columns.astype(int)
        
        # Ensure input synthtypes are also integers
        target_types = [int(t) for t in synthtype]
        available_types = [t for t in target_types if t in df.columns]
        
        if not available_types:
            return None
            
        return 5*df[available_types].mean(axis=1).sort_index()

    if prm is None:
        s = _get_mean_absy(key)
        if s is not None:
            ax.plot(s.index, s.values, label=overlay)
    else:
        all_series = []
        for seed in prm["seed"]:
            k = fct.utils.update_key(key, seed=seed)
            if k in D:
                s = _get_mean_absy(k)
                if s is not None:
                    s.name = seed
                    all_series.append(s)

        if not all_series:
            return

        pooled = pd.concat(all_series, axis=1)
        mean = pooled.mean(axis=1).sort_index()
        std  = pooled.std(axis=1).sort_index()

        ax.plot(mean.index, mean.values, label=overlay)
        ax.fill_between(mean.index, 
                        (mean - std).values, 
                        (mean + std).values, 
                        alpha=0.25)
        # ax.set_title(f"n={len(all_series)} seeds")

    ax.legend()

import numpy as np
import pandas as pd

def compute_synth_absy(df, synthtype, yconsider, keep_types_separate=True):
    """
    Compute absolute y-values and optionally their means by time and type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least ["time", "type", "y"] columns.
    synthtype : list
        List of types to include in the computation.
    yconsider : float
        The y-value range for filtering (±yconsider).
    keep_types_separate : bool, default True
        If True, computes means grouped by time and type.
        If False, collects individual abs(y) values for each time step.
    
    Returns
    -------
    synth_absy_full : pd.DataFrame
        DataFrame containing absolute values of y by time (and type if applicable).
    synth_means : pd.DataFrame or None
        DataFrame containing mean absolute y-values by time and type, or None if not computed.
    """
    
    if keep_types_separate:
        # 1. Filter once
        mask = (
            (df["type"].isin(synthtype)) &
            (df["y"].between(-yconsider, yconsider))
        )
        subset = df[mask].copy()

        # 2. Compute absolute values
        subset["absy"] = subset["y"].abs()

        # 3. Group by both 'time' and 'type' to get the mean
        synth_means = subset.groupby(["time", "type"])["absy"].mean().reset_index()

        # Return both the full absolute values and the means
        synth_absy_full = subset[["time", "type", "absy"]]

    else:
        # Generate time range
        trange = np.arange(0, int(df["time"].values[-1]) + 20, 20)

        records = []
        for t in trange:
            tmp = df.loc[
                (df["type"].isin(synthtype)) &
                (df["y"].between(-yconsider, yconsider)) &
                (df["time"] == t)
            ]
            for val in tmp["y"].abs().values:
                records.append({"time": t, "absy": val})

        synth_absy_full = pd.DataFrame(records)
        synth_means = None

    return synth_absy_full, synth_means