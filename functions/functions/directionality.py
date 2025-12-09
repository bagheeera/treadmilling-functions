import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib.pyplot as plt

def bound_traces(key, ax, D, nshow=60, cmap="jet", ccolor="id", zoom_inset=False, zoom_region=None):
    """
    Plots scatter data with an optional inset zooming into a specific region.

    Parameters:
    - key: str, key to access data from D.
    - ax: matplotlib axis, the main axis to plot on.
    - D: dict, contains the dataframes.
    - nshow: int, number of random points to show.
    - cmap: str, colormap for the scatter plot.
    - ccolor: str, column name for coloring the scatter points.
    - zoom_inset: bool, whether to add an inset zoom.
    - zoom_region: tuple (x_min, x_max, y_min, y_max) defining the zoom region.
    """
    dfs = D[key]["df_bound_only.pkl"]
    dfs = dfs[dfs["id"].isin(
        random.sample(list(dfs["id"].unique()), nshow)
    )]
    scatter = ax.scatter(*dfs[["x", "y"]].values.T,
                          c=dfs[ccolor],
                          cmap=cmap,
                          s=.1)
    ax.set_aspect("equal")

    # Add inset if zoom_inset is True and zoom_region is provided
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
        #ax_inset.set_title("Zoomed Inset", fontsize=8)

def alpha_hist(key, ax, overlay, D, specifylabel=False, legendtitle=False,bins=None,
               showlegend=True,
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
    mean_alpha = np.nanmedian(list(D[key][subkey]["alpha"].values()))

    # Draw axvline with the same color
    ax.axvline(
        mean_alpha,
        color=color,
        linestyle=meanls,  # optional: improves visibility
        linewidth=1.2
    )
