import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

    
    x_minmax = np.min(xy[:,0]), np.max(xy[:,0])
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
    y_density=4, cut=300):
    ## plot orientation of displacement vectors
    ## color by cosine of angle with inward normal

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    y_quiver_centers = y_centers[::y_density]

    d_y_mean_quiver = d_y_mean.T[::y_density, :]
    d_x_mean_quiver = d_x_mean.T[::y_density, :]

    X, Y = np.meshgrid(x_centers, y_quiver_centers)

    U = d_x_mean_quiver * 1000
    V = d_y_mean_quiver * 1000

    U = np.clip(U, -cut, cut)
    V = np.clip(V, -cut, cut)


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
        scale=5 #.01
    )
    #ax.colorbar(q, ax=ax, label="cosθ")
    return ax


def windowed_filament_age_hist(ax, key, D, t_min, t_max, label=None,
                               color="orangered", as_density=False):
    # Select times in window
    times = [t for t in sorted(D[key]["filament_age_profiles"].keys())
             if t_min <= t <= t_max]

    if len(times) == 0:
        return  # No data in this window
    
    # Collect all (possibly normalized) mean values in the time window
    mean_list = []
    y_bins = None
    for t in times:
        dft = D[key]["filament_age_profiles"][t]
        values = dft["mean"].values.copy()
        
        if as_density:
            bin_width = np.diff(dft["y_bin_center"].values).mean()  # approximate
            values = values / (np.nansum(values) * bin_width)
        
        mean_list.append(values)
        
        if y_bins is None:
            y_bins = dft["y_bin_center"].values
    
    # Convert to array for nanmean/nanstd
    mean_array = np.array(mean_list)
    
    # Compute mean and std over time window
    mean_over_window = np.nanmean(mean_array, axis=0)
    std_over_window = np.nanstd(mean_array, axis=0)
    
    # Scale y_bins if needed
    y_bins_scaled = 5 * np.array(y_bins)
    
    # Plot horizontal line
    p = ax.plot(mean_over_window, y_bins_scaled, lw=2, label=label,
    color=color)
    
    # Add shaded area for ±1 STD
    ax.fill_betweenx(
        y_bins_scaled,
        mean_over_window - std_over_window,
        mean_over_window + std_over_window,
        color=p[0].get_color(),
        alpha=0.3,
    )
    
    # Labels
    ax.set_ylabel("Long cell axis (nm)")
    
    # if as_density:
    #     ax.set_xlabel(f"Filament age probability density ({t_min}-{t_max}s)")
    # else:
    #     ax.set_xlabel(f"Mean filament age\n({t_min}-{t_max}s)")


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

    for t in times:
        df_t = df[df['time'] == t].copy()

        df_t['mol_age'] = df_t['mol'].map(mol_max_age)
        df_t['y_bin'] = pd.cut(df_t['y'], bins=y_bins)

        grouped = df_t.groupby('y_bin')['mol_age']
        result_df = grouped.mean().reset_index(name='mean')
        result_df['sem'] = grouped.sem().values
        result_df['y_bin_center'] = result_df['y_bin'].apply(lambda x: (x.left + x.right) / 2)

        profiles[t] = result_df
    return profiles
