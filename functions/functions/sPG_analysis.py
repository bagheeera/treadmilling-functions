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


import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
import ipywidgets as widgets
from ipywidgets import interact
import functions.sPG_tracker as pgt


# ── Config helpers ────────────────────────────────────────────────────────────

def load_run_config(rundir):
    """Load parameters.json and initialize pgt septal bins. Returns config dict."""
    with open(f"{rundir}/parameters.json") as f:
        cfg = json.load(f)
    pgt.set_septal_bins(
        strand_width_nm=4.5,
        septal_thickness_nm=5 * cfg.get("profW", 40 / 5)
    )
    return cfg


def _dark_ax(ax):
    """Apply dark theme styling to an axes."""
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def _time_window(t, t_center, window):
    """
    Return iteration indices and raw time bounds for a centered time window.

    Parameters
    ----------
    t : np.ndarray
        Simulation time per iteration (post /1000 normalization).
    t_center, window : float
        Center and full width of the time window.

    Returns
    -------
    i0, i1 : int
        Iteration index bounds.
    t0, t1 : float
        Corresponding simulation times (same units as t and fulldf["time"]).
    """
    t0_time = t_center - window / 2
    t1_time = t_center + window / 2
    idx = np.where((t >= t0_time) & (t < t1_time))[0]
    if len(idx) == 0:
        return None
    i0, i1 = idx[0], idx[-1]
    return i0, i1, t[i0], t[i1]


# ── Coverage heatmap + scatter ────────────────────────────────────────────────

def coverage_scatter(inwrd_window, df_window, fulldf_window, N, ax=None,
                     ylim=None):
    """
    Overlay individual particle positions on a coverage heatmap.

    Both the heatmap and scatter are displayed in simulation coordinates:
    - x-axis: simulation units (mapped from bin indices via fulldf x-extent)
    - y-axis: simulation units (mapped from y_edges module global)

    Parameters
    ----------
    inwrd_window : np.ndarray, shape (N, N_Y_BINS)
        Summed histogram for the time window: inwrd[i0:i1].sum(axis=(0,1)).
    df_window : pd.DataFrame
        Strand particles for the time window (already type-filtered).
    fulldf_window : pd.DataFrame
        All particles for the time window — sets x-extent for coordinate mapping.
    N : int
        Number of x-bins.
    ax : matplotlib Axes, optional
        Axes to plot on. Creates a new figure if None.
    ylim : tuple of float, optional
        (y_min, y_max) in simulation units to restrict y-axis view.
        Defaults to full y_edges range.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 4), facecolor="#0e1117")
    _dark_ax(ax)

    # Simulation-unit coordinate ranges
    x_min = fulldf_window["x"].min()
    x_max = fulldf_window["x"].max()
    y_min_su = pgt.y_edges[0]   # simulation units
    y_max_su = pgt.y_edges[-1]

    # ── Heatmap in simulation units ───────────────────────────────────────────
    ax.imshow(
        inwrd_window.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        extent=[x_min, x_max, y_min_su, y_max_su],  # simulation units on both axes
    )

    # ── Scatter in simulation units ───────────────────────────────────────────
    # No coordinate transformation needed — particles already in simulation units
    ax.scatter(df_window["x"], df_window["y"],
               s=8, alpha=0.5, color="#00ffcc", linewidths=0)

    ax.set_xlabel("x (simulation units)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("y (simulation units)", color="#aaaaaa", fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ylim if ylim is not None else (y_min_su, y_max_su))

    return ax


# ── Scatter widget ────────────────────────────────────────────────────────────

def scatter_widget(D, key, fulldf, inwrd_all, window=50, ylim=None):
    """
    Interactive widget: overlay particle positions on coverage heatmap,
    with a simulation-time slider and optional y-axis limits.

    Parameters
    ----------
    D : dict
        Data dict with D[key]["rundir"] and D[key]["t"].
    key : hashable
        Dataset key.
    fulldf : pd.DataFrame
        Full particle dataframe (all types). fulldf["time"] in same units as D[key]["t"].
    inwrd_all : np.ndarray, shape (n_iterations, timesteps, N, N_Y_BINS)
    window : float
        Time window width in simulation time units.
    ylim : tuple of float, optional
        (y_min, y_max) in simulation units. Defaults to full y_edges range.
    """
    cfg            = load_run_config(D[key]["rundir"])
    N_angular_bins = cfg.get("Nangularbins", 200)
    t              = D[key]["t"]
    t_min, t_max   = float(t.min()), float(t.max())
    inwrd_all      = np.array(inwrd_all)

    def plot(t_center):
        result = _time_window(t, t_center, window)
        if result is None:
            print(f"No iterations in window centered at t={t_center}")
            return
        i0, i1, t0, t1 = result

        inwrd_window  = inwrd_all[i0:i1].sum(axis=(0, 1))
        df_window     = fulldf[(fulldf["type"] == 11) &
                               (fulldf["time"] >= t0) & (fulldf["time"] < t1)]
        fulldf_window = fulldf[(fulldf["time"] >= t0) & (fulldf["time"] < t1)]

        fig, ax = plt.subplots(figsize=(13, 4), facecolor="#0e1117")
        coverage_scatter(inwrd_window, df_window, fulldf_window,
                         N=N_angular_bins, ax=ax, ylim=ylim)
        ax.set_title(
            f"t = {t0:.1f} – {t1:.1f}  |  iterations {i0}–{i1}  |  "
            f"{len(df_window)} strand particles",
            color="#dddddd", fontsize=10
        )
        plt.tight_layout()
        plt.show()

    interact(plot, t_center=widgets.FloatSlider(
        value=t_min + window / 2,
        min=t_min + window / 2,
        max=t_max - window / 2,
        step=window / 2,
        description="t center:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="70%"),
        readout_format=".1f",
    ))


# ── Coverage widget ───────────────────────────────────────────────────────────


def coverage_widget(inwrd, coverage_mask, deform, threshold,
                    window=10, N_Y_BINS=None):
    """
    Interactive widget: coverage heatmap + coverage_mask lines with threshold.

    A toggle switches between two heatmap modes:
      - "window"     : sum over last `window` iterations only
      - "cumulative" : sum over all iterations up to slider position (inwrd[:t1])

    A colorbar is placed above the heatmap using an inset axes so it doesn't
    affect the alignment between heatmap and coverage mask panels.

    Parameters
    ----------
    inwrd : np.ndarray, shape (n_iterations, timesteps, N, N_Y_BINS)
    coverage_mask : np.ndarray, shape (n_iterations, N)
    deform : np.ndarray of bool, shape (n_iterations, N)
    threshold : float
    window : int
        Number of iterations to look back in "window" mode.
    N_Y_BINS : int, optional
        Inferred from inwrd if not provided.
    """
    n_iter = inwrd.shape[0]
    N      = inwrd.shape[2]
    if N_Y_BINS is None:
        N_Y_BINS = inwrd.shape[3]
    threshold_line = threshold * N_Y_BINS

    # Precompute cumulative-with-decay snapshots once at widget init
    # Replays tracker logic: accumulate inwrd, decrement at deform, clamp
    # Shape: (n_iter, N, N_Y_BINS)
    _cumulative = np.zeros_like(inwrd[0].sum(axis=0))
    snapshots = []
    for i in range(n_iter):
        _cumulative = _cumulative + inwrd[i].sum(axis=0)
        _cumulative[deform[i]] -= 1
        _cumulative = np.maximum(0, _cumulative)
        snapshots.append(_cumulative.copy())
    snapshots = np.array(snapshots)  # (n_iter, N, N_Y_BINS)

    def _plot(t1, mode):
        t0_heat = max(0, t1 - window) if mode == "window" else 0
        t0_mask = max(0, t1 - window)  # mask always shows last `window` iterations

        # heatmap data depends on mode:
        #   window      — sum of raw counts over last `window` iters
        #   cumulative  — sum of raw counts from iter 0 to t1
        #   consumption — cumulative with decay applied (shows tracker state)
        if mode == "consumption":
            inwrd_window = snapshots[t1 - 1]                        # (N, N_Y_BINS)
        else:
            inwrd_window = inwrd[t0_heat:t1].sum(axis=(0, 1))       # (N, N_Y_BINS)
        # mask/deform: always last `window` iterations
        mask_window   = coverage_mask[t0_mask:t1]            # (window, N)
        deform_window = deform[t0_mask:t1].any(axis=0)       # (N,)

        fig = plt.figure(figsize=(13, 7), facecolor="#0e1117")
        # 3 rows: colorbar strip (thin), heatmap, coverage mask
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.08, 1, 1], hspace=0.25,
                               left=0.07, right=0.97, top=0.93, bottom=0.08)
        ax_cb   = fig.add_subplot(gs[0])   # colorbar row
        ax_heat = fig.add_subplot(gs[1])
        ax_mask = fig.add_subplot(gs[2], sharex=ax_heat)
        _dark_ax(ax_heat)
        _dark_ax(ax_mask)
        ax_cb.set_visible(False)           # hide axes frame; colorbar draws into it

        # ── Heatmap ───────────────────────────────────────────────────────────
        im = ax_heat.imshow(
            inwrd_window.T,
            aspect="auto", origin="lower", cmap="magma", interpolation="nearest",
            extent=[0, N, 0, N_Y_BINS],
        )

        # Colorbar above heatmap — inset into the invisible ax_cb row
        cbar = fig.colorbar(im, ax=ax_cb, orientation="horizontal",
                            fraction=1.0, pad=0.0)
        cbar.ax.tick_params(colors="#aaaaaa", labelsize=7)
        cbar.set_label("strand count", color="#aaaaaa", fontsize=8)
        ax_cb.set_visible(True)
        ax_cb.axis("off")  # hide the axes box, keep colorbar

        for xi in np.where(deform_window)[0]:
            ax_heat.axvline(xi, color="#00ffcc", alpha=0.25, lw=0.6)

        if mode == "consumption":
            mode_label = f"tracker state at iter {t1}"
        elif mode == "cumulative":
            mode_label = f"cumulative (0–{t1})"
        else:
            mode_label = f"last {t1 - t0_heat} iters"
        ax_heat.set_ylabel("y-bin (strand slot)", color="#aaaaaa", fontsize=9)
        ax_heat.set_title(
            f"Strand coverage  |  {mode_label}  |  {deform_window.sum()} / {N} x-bins triggered",
            color="#dddddd", fontsize=10, pad=8
        )
        plt.setp(ax_heat.get_xticklabels(), visible=False)

        # ── Coverage mask ─────────────────────────────────────────────────────
        colors = [colormaps["coolwarm"](x) for x in np.linspace(0, 1, len(mask_window))]
        for i, mask in enumerate(mask_window):
            ax_mask.plot(mask, color=colors[i], lw=0.7, alpha=0.6)
        ax_mask.axhline(threshold_line, color="#00ffcc", lw=1.2, ls="--",
                        label=f"threshold ({threshold:.0%} × {N_Y_BINS} = {threshold_line:.1f})")
        for xi in np.where(deform_window)[0]:
            ax_mask.axvline(xi, color="#00ffcc", alpha=0.15, lw=0.6)
        ax_mask.set_xlabel("x-bin (circumference)", color="#aaaaaa", fontsize=9)
        ax_mask.set_ylabel("occupied y-slots", color="#aaaaaa", fontsize=9)
        ax_mask.set_title(f"Coverage mask — last {t1 - t0_mask} iterations (blue → red)",
                          color="#dddddd", fontsize=10, pad=8)
        ax_mask.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333333",
                       labelcolor="#dddddd", loc="upper right")
        ax_mask.set_ylim(0, N_Y_BINS + 1)
        plt.show()

    slider = widgets.IntSlider(
        value=min(window, n_iter), min=window, max=n_iter, step=1,
        description="Up to iter:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="70%")
    )
    toggle = widgets.ToggleButtons(
        options=["window", "cumulative", "consumption"],
        value="window",
        description="Heatmap:",
        style={"description_width": "80px", "button_width": "120px"},
    )
    interact(_plot, t1=slider, mode=toggle)


# ── Circle plot ───────────────────────────────────────────────────────────────

def circle_plot(D, key, ax, tcut=None, lw=.8, alpha=.9):
    """
    Plot the evolving constriction ring over time, colored by viridis.

    Center position accumulates across iterations (tracks ring drift).
    Coordinates converted from simulation units to nm (*5).

    Parameters
    ----------
    D : dict
    key : hashable
    ax : matplotlib Axes
    tcut : float, optional
        Only plot iterations with t < tcut.
    lw, alpha : float
    """
    circ = np.array(D[key]["circumference"])   # (n_iterations, 5)
    if tcut is not None:
        circ = circ[circ[:, 1] < tcut]

    colors = [colormaps["viridis"](x) for x in np.linspace(0, 1, len(circ))]
    angles = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    xc_cumulative, yc_cumulative = 0.0, 0.0

    for i, (circumference_updated, t, xc, yc, r) in enumerate(circ):
        xc_nm = xc * 5
        yc_nm = yc * 5
        r_nm  = r  * 5
        xc_cumulative += xc_nm
        yc_cumulative += yc_nm
        if i % 10 == 0:
            ax.plot(xc_cumulative + r_nm * np.cos(angles),
                    yc_cumulative + r_nm * np.sin(angles),
                    color=colors[i], lw=lw, alpha=alpha)
            ax.scatter(xc_cumulative, yc_cumulative,
                       marker="x", color=colors[i], s=10)
    ax.set_aspect("equal")