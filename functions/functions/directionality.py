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


def bound_traces(key, ax, D, nshow=60, cmap="jet", ccolor="id", zoom_inset=False, zoom_region=None,
                 s=.1):
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
                          s=s)
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
