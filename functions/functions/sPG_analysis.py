import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import colormaps
from ipywidgets import interact, IntSlider, IntRangeSlider
import ipywidgets as widgets

def coverage_widget(inwrd, coverage_mask, deform, threshold,
                    window=10, N_Y_BINS=None):
    """
    Interactive widget to visualize cumulative strand coverage and constriction threshold.

    Layout
    ------
    Top panel : 2D heatmap of cumulative strand counts over the selected time window.
                Rows = x-bins (circumference), cols = y-bins (strand slots).
                Columns where deform=True (threshold met) are highlighted.

    Bottom panel : coverage_mask (number of occupied y-slots) per x-bin over the
                   selected window, with threshold line overlaid.

    Parameters
    ----------
    inwrd : np.ndarray, shape (n_iterations, timesteps, N, N_Y_BINS)
        Raw per-frame histograms saved from main loop.
        Note: if inwrd was saved per-iteration as (timesteps, N, N_Y_BINS),
        stack them first: inwrd = np.array([entry[5] for entry in circumferences])
    coverage_mask : np.ndarray, shape (n_iterations, N)
        Occupied y-slot count per x-bin per iteration.
    deform : np.ndarray of bool, shape (n_iterations, N)
        Which x-bins triggered constriction per iteration.
    threshold : float
        Fraction of y-slots required to trigger constriction — used to draw threshold line.
    window : int
        Default time window width (number of iterations to sum over).
    N_Y_BINS : int, optional
        Number of y-bins. Inferred from inwrd if not provided.
    """
    n_iter = inwrd.shape[0]
    N      = inwrd.shape[2]
    if N_Y_BINS is None:
        N_Y_BINS = inwrd.shape[3]

    threshold_line = threshold * N_Y_BINS  # absolute threshold in slot units

    def _plot(t1):
        t0 = max(0, t1 - window)

        # ── Data for this window ──────────────────────────────────────────────
        # Sum inwrd over selected iterations and all internal timesteps
        # (window, timesteps, N, N_Y_BINS) → (N, N_Y_BINS)
        inwrd_window  = inwrd[t0:t1].sum(axis=(0, 1))
        mask_window   = coverage_mask[t0:t1]          # (window, N)
        deform_window = deform[t0:t1].any(axis=0)     # (N,) — triggered at least once

        # ── Figure ────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(13, 7), facecolor="#0e1117")
        gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.15,
                                left=0.07, right=0.97, top=0.91, bottom=0.08)
        ax_heat = fig.add_subplot(gs[0])
        ax_mask = fig.add_subplot(gs[1], sharex=ax_heat)  # shared x-axis → aligned widths

        for ax in [ax_heat, ax_mask]:
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="#aaaaaa", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

        # ── Top: 2D heatmap ───────────────────────────────────────────────────
        # inwrd_window.T: x-bins on horizontal axis, y-bins vertical
        ax_heat.imshow(
            inwrd_window.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            extent=[0, N, 0, N_Y_BINS],  # x from 0..N to match ax_mask x-axis
        )

        # Highlight deforming x-bins
        for xi in np.where(deform_window)[0]:
            ax_heat.axvline(xi, color="#00ffcc", alpha=0.25, lw=0.6)

        ax_heat.set_ylabel("y-bin (strand slot)", color="#aaaaaa", fontsize=9)
        ax_heat.set_title(
            f"Cumulative strand coverage  |  iterations {t0}–{t1}  "
            f"|  {deform_window.sum()} / {N} x-bins triggered",
            color="#dddddd", fontsize=10, pad=8
        )
        plt.setp(ax_heat.get_xticklabels(), visible=False)  # hidden — shared with bottom

        # ── Bottom: coverage mask lines (blue → red over window) ─────────────
        colors = [colormaps["coolwarm"](x) for x in np.linspace(0, 1, len(mask_window))]
        for i, mask in enumerate(mask_window):
            ax_mask.plot(mask, color=colors[i], lw=0.7, alpha=0.6)

        # Threshold line
        ax_mask.axhline(threshold_line, color="#00ffcc", lw=1.2, ls="--",
                        label=f"threshold ({threshold:.0%} × {N_Y_BINS} = {threshold_line:.1f})")

        # Shade deforming x-bins
        for xi in np.where(deform_window)[0]:
            ax_mask.axvline(xi, color="#00ffcc", alpha=0.15, lw=0.6)

        ax_mask.set_xlabel("x-bin (circumference)", color="#aaaaaa", fontsize=9)
        ax_mask.set_ylabel("occupied y-slots", color="#aaaaaa", fontsize=9)
        ax_mask.set_title(f"Coverage mask — last {t1 - t0} iterations (blue → red)",
                          color="#dddddd", fontsize=10, pad=8)
        ax_mask.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333333",
                       labelcolor="#dddddd", loc="upper right")
        ax_mask.set_ylim(0, N_Y_BINS + 1)

        plt.show()

    slider = widgets.IntSlider(
        value=min(window, n_iter),
        min=window, max=n_iter, step=1,
        description="Up to iter:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="70%")
    )

    interact(_plot, t1=slider)



def circle_plot(key, ax, tcut=None, lw=.8, alpha=.9):
    """
    Plot the evolving constriction ring over time, colored by viridis.

    Center position accumulates across iterations (tracks ring drift).
    Coordinates converted from simulation units to nm (*5).

    Parameters
    ----------
    key : str
        Key into D for the dataset to plot.
    ax : matplotlib Axes
    tcut : float, optional
        Only plot iterations with t < tcut.
    lw : float
        Line width for circle outlines.
    alpha : float
        Opacity for circle outlines.
    """
    # circumferences entries are [circumference_updated, t, xc, yc, r]
    circ = np.array(D[key]["circumference"])  # shape (n_iterations, 5)
    if tcut is not None:                      # note: `if tcut` fails when tcut=0
        circ = circ[circ[:, 1] < tcut]

    colors = [colormaps["viridis"](x) for x in np.linspace(0, 1, len(circ))]
    angles = np.linspace(0, 2 * np.pi, 300, endpoint=False)  # precompute once

    # cumulative center tracks drift of ring position over time (in nm)
    xc_cumulative, yc_cumulative = 0.0, 0.0

    for i, (circumference_updated, t, xc, yc, r) in enumerate(circ):
        # convert simulation units → nm
        xc_nm = xc * 5
        yc_nm = yc * 5
        r_nm  = r  * 5

        # accumulate center drift
        xc_cumulative += xc_nm
        yc_cumulative += yc_nm

        if i % 10 == 0:  # plot every 10th iteration
            ax.plot(
                xc_cumulative + r_nm * np.cos(angles),
                yc_cumulative + r_nm * np.sin(angles),
                color=colors[i], lw=lw, alpha=alpha
            )
            ax.scatter(
                xc_cumulative, yc_cumulative,
                marker="x", color=colors[i], s=10
            )

    ax.set_aspect('equal')    