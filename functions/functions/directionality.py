import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_finalframe_ghosts(
    ax,
    D,
    key,
    zoom_region=[-60, -30,-200, -170],
    scatter_size_main=0.03,
    scatter_size_inset=0.4,
    legend=True,
    markerscale=50,
    hide_ticks=True,
    rectlinewidth=1
):
    """
    Plot occupied / available ghost particles with optional zoom inset.

    Parameters
    ----------
    ax_row : array-like of matplotlib Axes
        A row of axes, e.g. ax[i], expected length >= 2
    D : dict
        Data dictionary
    key : hashable
        Key into D
    zoom_region : tuple
        (x_min, x_max, y_min, y_max)
    """

    if "finalframe_ghosts" not in D[key]:
        return

    ax_main = ax

    df = D[key]["finalframe_ghosts"]
    df7 = df[df["type"] == 7]
    df8 = df[df["type"] == 8]

    # --- main scatter ---
    for j, dfi in enumerate([df7, df8]):
        ax_main.scatter(
            *dfi[["x", "y"]].values.T,
            s=scatter_size_main,
            label=["Available", "Occupied"][j],
        )

    ax_main.set_aspect("equal")

    if legend:
        ax_main.legend(markerscale=markerscale, loc="upper right")

    # clean axes
    #for ax in ax_row[:2]:
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    # shared limits
    xmin, xmax = df["x"].min(), df["x"].max()
    ymin, ymax = df["y"].min(), df["y"].max()
    # for ax in ax_row[:2]:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # --- zoom rectangle ---
    x_min, x_max, y_min, y_max = zoom_region
    rect = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=rectlinewidth,
        edgecolor="black",
        facecolor="none",
    )
    ax_main.add_patch(rect)

    # --- inset ---
    ax_inset = inset_axes(
        ax_main,
        width="30%",
        height="30%",
        loc="lower right",
    )

    for dfi in [df7, df8]:
        ax_inset.scatter(
            *dfi[["x", "y"]].values.T,
            s=scatter_size_inset,
        )

    ax_inset.set_xlim(x_min, x_max)
    ax_inset.set_ylim(y_min, y_max)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])


import pandas as pd
import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def bound_traces(key, ax, D, nshow=60, cmap="jet", ccolor="id",
                 zoom_inset=False, zoom_region=None, s=.1,
                 show_ids=False, id_fontsize=6, id_color='black',
                 rng_seed=None):
    """
    Plots scatter data with an optional inset zooming into a specific region.

    Parameters:
    - key: str, key to access data from D.
    - ax: matplotlib axis, the main axis to plot on.
    - D: dict, contains the dataframes.
    - nshow: int, number of random traces to show.
    - cmap: str, colormap for the scatter plot.
    - ccolor: str, column name for coloring the scatter points.
    - zoom_inset: bool, whether to add an inset zoom.
    - zoom_region: tuple (x_min, x_max, y_min, y_max) defining the zoom region.
    - s: float, scatter size.
    - show_ids: bool, whether to display an ID label at each trace center.
    - id_fontsize: int, font size of the ID labels.
    - id_color: str, color of the ID label text.
    - rng_seed: int or None, if provided, fixes the random seed for reproducible sampling.
    """

    # --- Optional reproducible random selection ---
    if rng_seed is not None:
        random.seed(rng_seed)

    # Load dataframe
    if "df_bound_only" in D[key]:
        dfs = D[key]["df_bound_only.pkl"]
    else:
        dfs = pd.read_pickle(D[key]["rundir"] + "/df_bound_only.pkl.gz")

    # Randomly sample traces to display
    ids = list(dfs["id"].unique())
    selected_ids = random.sample(ids, min(nshow, len(ids)))
    dfs = dfs[dfs["id"].isin(selected_ids)]

    # Main scatter plot
    scatter = ax.scatter(
        *dfs[["x", "y"]].values.T,
        c=dfs[ccolor],
        cmap=cmap,
        s=s
    )
    ax.set_aspect("equal")

    # === Add ID labels at centers ===
    if show_ids:
        centers = dfs.groupby("id")[["x", "y"]].mean().reset_index()
        for _, row in centers.iterrows():
            ax.text(row["x"], row["y"], str(int(row["id"])),
                    color=id_color, fontsize=id_fontsize,
                    ha='center', va='center')

    # === Add inset if requested ===
    if zoom_inset and zoom_region:
        x_min, x_max, y_min, y_max = zoom_region
        ax_inset = inset_axes(ax, width="30%", height="30%", loc="lower right")
        ax_inset.scatter(*dfs[["x", "y"]].values.T,
                         c=dfs[ccolor],
                         cmap=cmap,
                         s=.1)
        ax_inset.set_xlim(x_min, x_max)
        ax_inset.set_ylim(y_min, y_max)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

        if show_ids:
            centers_inset = centers.query("x >= @x_min and x <= @x_max and y >= @y_min and y <= @y_max")
            for _, row in centers_inset.iterrows():
                ax_inset.text(row["x"], row["y"], str(int(row["id"])),
                              color=id_color, fontsize=id_fontsize-1,
                              ha='center', va='center')

    return scatter

def alpha_hist(key, ax, overlay, D, specifylabel=False, legendtitle=False,bins=None,
               showlegend=True,
               usemean=False,
              subkey="MSD_fit", legendloc="best",
              meanls="--"):
    # Plot histogram and capture the returned artists
    if bins is not None:
        usebins=True
    else:
        usebins=False
    n, bins, patches = ax.hist(
        D[key][subkey]["alpha"].values(),
        histtype="step",
        lw=3,
        bins=bins if usebins else 30,
        label=specifylabel if specifylabel else f"{overlay}",
        density=True,
    )
    if showlegend:
        if legendtitle:
            ax.legend(title=legendtitle, loc=legendloc)
        else:
            ax.legend(loc=legendloc)
    # Extract the line color used by the histogram
    color = patches[0].get_edgecolor()

    # Compute mean
    if usemean:
        mean_alpha = np.nanmean(list(D[key][subkey]["alpha"].values()))
    else:
        mean_alpha = np.nanmedian(list(D[key][subkey]["alpha"].values()))

    # Draw axvline with the same color
    ax.axvline(
        mean_alpha,
        color=color,
        linestyle=meanls,  # optional: improves visibility
        linewidth=1.2
    )


def sliding_window_asymmetry(df, window_size=20, step_size=5, return_dict=False):
    """
    Sliding window segmentation for asymmetry analysis.
    df: DataFrame with columns ['id', 'x', 'y']
    window_size: Number of frames per segment
    step_size: How many frames to skip between windows
    """
    segment_asymmetries = []
    id_asymmetries = {}  # {pid: [a_val, ...]}

    for pid, group in df.groupby("id"):
        coords = group[["x", "y"]].to_numpy()
        n_points = len(coords)
        if n_points < window_size:
            continue

        pid_vals = []
        for start in range(0, n_points - window_size + 1, step_size):
            window = coords[start : start + window_size]
            a_val = compute_huet_asymmetry_from_array(window)
            if np.isfinite(a_val):
                segment_asymmetries.append(a_val)
                pid_vals.append(a_val)

        if pid_vals:
            id_asymmetries[pid] = np.mean(pid_vals)

    if return_dict:
        return segment_asymmetries, id_asymmetries
    return segment_asymmetries

def compute_huet_asymmetry_from_array(coords):
    """Helper to process raw numpy arrays."""
    centered = coords - np.mean(coords, axis=0)
    tensor = np.dot(centered.T, centered) / len(coords)
    eigvals = np.linalg.eigvalsh(tensor)
    l1, l2 = np.sort(eigvals)[::-1]
    
    if (l1**2 + l2**2) == 0: return 0.0
    
    num = (l1**2 - l2**2)**2
    den = 2 * (l1**2 + l2**2)**2
    val = 1 - (num / den)
    return -np.log(np.clip(val, 1e-10, 1.0))

from tqdm.notebook import tqdm
def calculate_density_overlap_grouped(
    df, 
    type_groups,
    x_bins=20, 
    y_bins=20, 
    metric='intersection'
):
    """
    Calculate overlap between grouped spatial densities across all time slices.
    
    Parameters:
    -----------
    df : DataFrame with columns [type, time, x, y]
    type_groups : list of lists
        E.g. [[1,2,3], [6], [7,8]] groups types and compares all pairs
    x_bins, y_bins : int, number of bins per axis
    metric : str, 'bhattacharyya'|'hellinger'|'intersection'|'kl_divergence'
    
    Returns:
    --------
    pandas.DataFrame with columns:
      - time
      - group_pair (e.g., "Group_0 vs Group_1")
      - overlap
      - n_samples_g1, n_samples_g2
    """
    
    results = []
    
    # Process each time slice
    for time in tqdm(sorted(df['time'].unique())):
        subset = df[df['time'] == time]
        
        group_densities = {}
        group_samples = {}
        bin_edges = {'x': None, 'y': None}
        
        # Bin each group
        for group_idx, type_list in enumerate(type_groups):
            group_data = subset[subset['type'].isin(type_list)]
            group_samples[group_idx] = len(group_data)
            
            if len(group_data) == 0:
                group_densities[group_idx] = None
                continue
            
            density_2d, xedges, yedges = np.histogram2d(
                group_data['x'], 
                group_data['y'],
                bins=[x_bins, y_bins]
            )
            
            # Normalize to PDF
            density_2d = density_2d / density_2d.sum()
            group_densities[group_idx] = density_2d
            bin_edges['x'] = xedges
            bin_edges['y'] = yedges
        
        # Pairwise comparisons
        group_indices = list(group_densities.keys())
        
        for i, g1 in enumerate(group_indices):
            for j, g2 in enumerate(group_indices):
                if i < j:
                    d1 = group_densities[g1]
                    d2 = group_densities[g2]
                    
                    if d1 is None or d2 is None:
                        overlap_val = np.nan
                    else:
                        d1_flat = d1.flatten()
                        d2_flat = d2.flatten()
                        
                        if metric == 'bhattacharyya':
                            overlap_val = np.sum(np.sqrt(d1_flat * d2_flat))
                        
                        elif metric == 'hellinger':
                            bc = np.sum(np.sqrt(d1_flat * d2_flat))
                            overlap_val = np.sqrt(1 - bc)
                        
                        elif metric == 'intersection':
                            overlap_val = np.sum(np.minimum(d1_flat, d2_flat))
                        
                        elif metric == 'kl_divergence':
                            eps = 1e-10
                            kl_pq = np.sum(d1_flat * np.log((d1_flat + eps) / 
                                                            (d2_flat + eps)))
                            kl_qp = np.sum(d2_flat * np.log((d2_flat + eps) / 
                                                            (d1_flat + eps)))
                            overlap_val = (kl_pq + kl_qp) / 2
                    
                    results.append({
                        'time': time,
                        'group_pair': f'Group_{g1}_vs_Group_{g2}',
                        'overlap': overlap_val,
                        'n_samples_g1': group_samples[g1],
                        'n_samples_g2': group_samples[g2]
                    })
    
    return pd.DataFrame(results)

