import json
import numpy as np
from scipy.ndimage import gaussian_filter
import functions.sPG_tracker as pgt


# import tqdm as tqdm #.notebook as tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import functions as fct
# from functions.common_imports import *



import json
import numpy as np
from scipy.ndimage import gaussian_filter
import functions.sPG_tracker as pgt
from functions.sPG_tracker import calc_inward_deformations


def histogram_mesh(df, fulldf, rundir,
                   z_range_tuple=(-150, 150),
                   blur_nm=(20.0, 5.0),
                   per_interval=False,
                   verbose=False,
                #    height_per_count_nm=None # if None = strand_width_nm (4.5 nm default)):
                   ):  
    """
    Build a deformed cylindrical mesh from a 2D histogram of strand particle
    positions, with bins sized to strand dimensions.

    This is a physically cleaner alternative to Gaussian splatting:
    - bins are sized to strand_thickness_width (set via pgt.set_septal_bins)
    - each count contributes strandwidth_nm of radial displacement
    - a final Gaussian blur creates a smooth membrane envelope

    Coordinate system
    -----------------
    - theta (axis 0): angular position [0, 2pi], mapped from x per-interval
      using fulldf x-extent to track shrinking box (same as calc_inward_deformations)
    - z (axis 1): long-axis position in simulation units, binned at strand_width spacing

    Parameters
    ----------
    df : pd.DataFrame
        Processive strand particles (types 5/9). Columns: x, y, time.
    fulldf : pd.DataFrame
        All particle types — for x-extent per interval.
    rundir : str
        Path to run directory containing parameters.json.
        Reads: dT, tstep, Lx.
    z_range_tuple : tuple of float
        (z_min, z_max) in simulation units for z-bin range.
        Default (-150, 150) su = ±750 nm.
    blur_nm : tuple of float
        (sigma_theta_nm, sigma_z_nm) — Gaussian blur widths in nm.
        Converted internally to pixels using bin spacings.
        sigma_theta_nm : smoothing along circumference in nm
        sigma_z_nm     : smoothing along z-axis in nm

    Returns
    -------
    t_grid : np.ndarray, shape (n_theta, n_z)
        Angular bin centers in radians.
    z_grid : np.ndarray, shape (n_theta, n_z)
        Z bin centers in simulation units.
    H_total : np.ndarray, shape (n_theta, n_z)
        Raw cumulative histogram counts (before scaling and blur).
    H_blurred : np.ndarray, shape (n_theta, n_z)
        Smoothed deformation map in nm.
    x_coords, y_coords : np.ndarray, shape (n_theta, n_z)
        Cartesian coordinates of deformed surface in nm.
    z_coords : np.ndarray, shape (n_theta, n_z)
        Z coordinates in simulation units.
    """
    # ── Load run parameters ───────────────────────────────────────────────────
    with open(f"{rundir}/parameters.json") as f:
        d = json.load(f)

    dT               = d["dT"] #/ d["tstep"]
    NM_PER_SU        = 5.0
    circumference_su = 2 * d["Lx"]               # full box in sim units
    R_nm             = NM_PER_SU * circumference_su / (2 * np.pi)  # run-start radius in nm

    # ── Bin geometry ─────────────────────────────────────────────────────────
    strand_width_nm     = pgt.strand_thickness_width          # nm — controls bin size
    height_per_count_nm = pgt.strand_height_nm       # nm — height per count (scaling), default = bin size
    strand_width_su = strand_width_nm / NM_PER_SU         # simulation units
    circumference_nm = NM_PER_SU * circumference_su

    # z-bins: spaced at strand_width, custom range for mesh visualization
    z_min, z_max = z_range_tuple
    z_edges_mesh = np.arange(z_min, z_max + strand_width_su, strand_width_su)
    n_z          = len(z_edges_mesh) - 1
    z_centers    = (z_edges_mesh[:-1] + z_edges_mesh[1:]) / 2

    # theta: N_fine=400 fixed grid, same as calc_inward_deformations default
    n_theta      = 400
    t_centers    = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    t_grid, z_grid = np.meshgrid(t_centers, z_centers, indexing='ij')

    # ── Gaussian blur pixel sigmas ────────────────────────────────────────────
    theta_bin_nm   = circumference_nm / n_theta
    z_bin_nm       = strand_width_nm
    sigma_theta_nm, sigma_z_nm = blur_nm
    sigma_theta_px = sigma_theta_nm / theta_bin_nm
    sigma_z_px     = sigma_z_nm     / z_bin_nm

    # ── Accumulate histograms via calc_inward_deformations ────────────────────
    # Split df into dT intervals and call calc_inward_deformations per interval,
    # passing z_edges_mesh as y_edges_override for extended z-range.
    # Each call returns shape (timesteps=1, N_fine, n_z) which we sum into H_total.
    t_min  = df["time"].min()
    t_max  = df["time"].max()
    t_bins = np.arange(t_min, t_max + dT, dT)

    H_total     = np.zeros((n_theta, n_z))
    r_snapshots = []
    t_snapshots = []
    if verbose:
        print("iterating over", len(t_bins), "time bins")
    for t0, t1 in tqdm(zip(t_bins[:-1], t_bins[1:]), total=len(t_bins[:-1])):
        if verbose:
            print(t0)
        df_i     = df[(df["time"] >= t0) & (df["time"] < t1)]
        full_i   = fulldf[(fulldf["time"] >= t0) & (fulldf["time"] < t1)]
        if len(df_i) == 0 or len(full_i) == 0:
            continue

        # print(len(df_i), len(full_i))
        # one dT interval: timesteps=1 gives shape (1, N_fine, n_z)
        H_interval = calc_inward_deformations(
            df_i, full_i, timesteps=10, N=200, N_fine=n_theta,
            y_edges_override=z_edges_mesh,
            verbose=False
        )
        H_total += H_interval.sum(axis=0)   # (N_fine, n_z)

        if per_interval:
            H_s    = gaussian_filter(H_total * height_per_count_nm, #strand_width_nm,
                                     sigma=(sigma_theta_px, sigma_z_px),
                                     mode=('wrap', 'reflect'))
            r_snap = R_nm - H_s
            r_snapshots.append(r_snap.copy())
            t_snapshots.append(t1)

    # ── Scale and blur ────────────────────────────────────────────────────────
    H_scaled  = H_total * height_per_count_nm #strand_width_nm   # nm
    H_blurred = gaussian_filter(H_scaled,
                                sigma=(sigma_theta_px, sigma_z_px),
                                mode=('wrap', 'reflect'))

    # ── Deformed surface geometry ─────────────────────────────────────────────
    r_final  = R_nm - H_blurred                  # nm
    x_coords = r_final * np.cos(t_grid)
    y_coords = r_final * np.sin(t_grid)
    z_coords = z_grid * NM_PER_SU               # simulation units → nm

    if per_interval:
        return t_grid, z_grid, H_total, H_blurred, x_coords, y_coords, z_coords, \
               np.array(t_snapshots), np.array(r_snapshots)
    return t_grid, z_grid, H_total, H_blurred, x_coords, y_coords, z_coords

import numpy as np
from tqdm import tqdm


import numpy as np
from tqdm import tqdm


def render_time_movie(
    t_grid, z_grid, r_snapshots, t_snapshots, filename, cam_dict,
    clip_normal, clip_origin,
    image_scale=3,
    select_view=None,
    render_movie=True,
    clim=None,
):
    """
    Render a movie over time using pre-computed r_snapshots from histogram_mesh
    with per_interval=True. Compatible with both interactive Jupyter and
    headless SLURM/papermill execution.
  
    Parameters
    ----------
    t_grid, z_grid : np.ndarray, shape (n_theta, n_z)
        Grid coordinates from histogram_mesh.
        z_grid in simulation units — converted to nm internally.
    r_snapshots : np.ndarray, shape (n_intervals, n_theta, n_z)
        Radius at each interval in nm.
    t_snapshots : np.ndarray, shape (n_intervals,)
        Simulation time at each interval.
    filename : str
        Output movie filename (.mp4).
    cam_dict : dict
        Keys: position, focal_point, view_up.
        Ignored if select_view is set.
    clip_normal, clip_origin : array-like
        Clipping plane normal and origin for cross-section view.
        Should be in nm to match r_snapshots units.
    image_scale : int
        Supersampling factor for anti-aliasing. Default 3.
    select_view : str or None
        Preset camera view. Options: 'front', 'side', or None to use cam_dict.
    render_movie : bool, optional
        If True (default), render and save a movie to `filename`.
        If False, skip frame writing and save only the final rendered
        frame as a high-resolution `.png` image instead.
    """

    import pyvista as pv
    import numpy as np
    from tqdm import tqdm
    import os

    pv.start_xvfb()
    pv.global_theme.jupyter_backend = "static"

    # preset camera views
    if select_view == "front":
        cam_dict = {
            'position':    (-1439.267348837681, 367.91393768244296, 2.6378280610901945),
            'focal_point': (0.0, 0.0, 0.15),
            'view_up':     (0, -0.9687882884811285, 0)
        }
        print("using preset front view")
    elif select_view == "side":
        cam_dict = {
            'position':    (-1166.082220657981, -38.3627876539593, 919.7357259921413),
            'focal_point': (0.0, 0.0, 0.15),
            'view_up':     (-0.011985074051557796, -0.9983110926201956, -0.05684470381178824)
        }
        print("using preset side view")

    n_theta, n_z = t_grid.shape
    z_nm = z_grid * 5.0

    # precompute faces — fixed topology, only points change each frame
    faces = []
    for i in range(n_theta):
        i_next = (i + 1) % n_theta
        for j in range(n_z - 1):
            p0 = i * n_z + j
            p1 = i_next * n_z + j
            p2 = i_next * n_z + (j + 1)
            p3 = i * n_z + (j + 1)
            faces.append([4, p0, p1, p2, p3])
    faces = np.hstack(faces)

    # always off_screen for movie rendering — works in both environments
    plotter = pv.Plotter(off_screen=True)
    plotter.enable_anti_aliasing('ssaa')
    plotter.image_scale = image_scale

    if render_movie:
        plotter.open_movie(filename)
        iterable = tqdm(
            zip(r_snapshots, t_snapshots),
            total=len(t_snapshots),
            desc="rendering frames"
        )
    else:
        iterable = [(r_snapshots[-1], t_snapshots[-1])]
        print("Rendering only final frame as image...")

    for r_snap, t in iterable:
        x = r_snap * np.cos(t_grid)
        y = r_snap * np.sin(t_grid)

        points = np.column_stack([x.ravel(), y.ravel(), z_nm.ravel()])
        radial = np.sqrt(x**2 + y**2).ravel()

        mesh = pv.PolyData(points, faces)
        mesh['radius'] = radial
        mesh['H'] = (r_snapshots[0] - r_snap).ravel()

        clipped = mesh.clip(normal=clip_normal, origin=clip_origin, invert=False)

        if plotter.actors.get('clipped_mesh_actor'):
            plotter.remove_actor('clipped_mesh_actor')

        plotter.add_mesh(
            clipped,
            name='clipped_mesh_actor',
            scalars='radius',
            cmap='Purples_r',
            smooth_shading=True,
            show_scalar_bar=False,
            clim=clim,
        )

        plotter.camera.position = cam_dict['position']
        plotter.camera.focal_point = cam_dict['focal_point']
        plotter.camera.up = cam_dict['view_up']

        if render_movie:
            plotter.write_frame()

    if not render_movie:
        cwd = os.getcwd()
        outname = cwd.split("/")[-2] + f"_view_{select_view}.png"
        plotter.screenshot(outname, scale=image_scale)
        print(f"Saved final frame screenshot: {outname}")

    plotter.close()

    if render_movie:
        print(f"Movie saved to {filename}")

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def d_alpha(t, tau_c, alpha):
    """Model for diameter (or radius) evolution over time."""
    return (1 - (t / tau_c) ** alpha) ** (1 / alpha)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

whitley_constriction = np.array([[   0.        , 1155.99320666],
                                [   1.25334741, 1154.63087252],
                                [   2.50669482, 1153.26853838],
                                [   3.76004223, 1151.90620424],
                                [   5.03617777, 1136.51802087],
                                [   6.26673705, 1087.03770711],
                                [   7.52008446, 1030.27003159],
                                [   8.77343187,  973.25278245],
                                [   9.96980894,  912.9032565 ],
                                [  11.10921568,  851.81701928],
                                [  12.19165208,  790.42666505],
                                [  13.21711814,  727.6950768 ],
                                [  14.18561386,  663.23333565],
                                [  15.04016892,  601.95668127],
                                [  15.78078329,  541.73531982],
                                [  16.46442734,  481.05296651],
                                [  17.09110104,  417.59571744],
                                [  17.60383407,  359.935957  ],
                                [  18.05959677,  298.35742176],
                                [  18.34444845,  260.29979188]])

def plot_septum_height(
    ax_ins,
    D,
    key,
    pgt,
    z_range_tuple,
    strandheight,
    label=None,
    md_line_color=None,
    axislabels=True,
    xlim=(-120, 120),
):
    """
    Plot septum height profile into an inset axis.
    """

    z_min, z_max = z_range_tuple

    strand_width_su = getattr(pgt, "strand_thickness_width", 4.5) / 5.0
    z_edges = np.arange(z_min, z_max + strand_width_su, strand_width_su)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    z_nm = z_centers * 5.0

    H_total_mean = D[key]["H_total"].mean(axis=0) * strandheight  # nm

    ax_ins.plot(
        z_nm,
        H_total_mean,
        color=md_line_color,
        label=label,
    )

    ax_ins.set_xlim(*xlim)

    if axislabels:
        ax_ins.set_xlabel("Long cell axis (nm)", fontsize=7)
        ax_ins.set_ylabel("Septum height (nm)", fontsize=7)

def diam_plot(
    D,
    key,
    ax,
    modelonly=False,
    label=None,
    mdlabel=None,
    overlay=None,
    coltharp_color="k",
    mdcolor=None,
    normalize_diameter=True,
    display_radius=False,
    D_0=578,
    inset=True,
    axislabels=True,
    width="45%",
    height="45%",
    legendtitle=None,
    coltharp_label=None,
    whitley_label = "Whitley 2021, Fig4a",
    ncol=1,
    pgt=None,
    z_range_tuple=(-370, 370),
    lw=3,
    t_model_max=50,
    plot_std=False,
    vline_pos=None,
    verbose=False,
    plot_coltharp=True,
    plot_whitley=False,
    legendloc="upper right",
    strandheight=3,
):
    """
    Plot diameter (default) or radius evolution over time, with optional inset,
    optional std shading, and optional vertical line(s) matching the measured data color.
    """

    # ---------------------------
    # Extract data
    # ---------------------------
    t, r = np.array(D[key]["t_r"]).T  # time (ms?), sPG tracker says radius is saved in su, so we convert to nm
    if display_radius:
        val_md = r * 5
        val_name = "Radius"
        val_0 = D_0 / 2
    else:
        val_md = r * 2 * 5
        val_name = "Diameter"
        val_0 = D_0

    # ---------------------------
    # Model parameters
    # ---------------------------
    t_model = np.linspace(0, t_model_max, 1000)
    tau_c, alpha = 51, 1.3

    # ---------------------------
    # Normalize / compute model
    # ---------------------------
    if normalize_diameter:
        val_md_plot = val_md / val_md[0]
        val_model = d_alpha(t_model, tau_c, alpha)
        ylabel = f"Normalized {val_name}"
    else:
        val_md_plot = val_md
        val_model = d_alpha(t_model, tau_c, alpha) * val_0
        ylabel = f"{val_name} (nm)"

    # ---------------------------
    # Plot measured data
    # ---------------------------
    md_line_color = mdcolor  # default fallback
    if not modelonly:
        (md_line,) = ax.plot(
            t / 1000 / 60,
            val_md_plot,
            lw=lw,
            label=mdlabel or overlay,
            color=mdcolor
        )
        md_line_color = md_line.get_color()

    # ---------------------------
    # Optional std fill
    # ---------------------------
    if plot_std:
        std_data = None
        if verbose:
            print(f"[DEBUG] Looking for std data for key={key}")
        if f"{key}_std" in D:
            std_data = np.array(D[f"{key}_std"])
            if verbose:
                print(f"  Found D['{key}_std'], shape={std_data.shape}")
        elif "t_r_std" in D[key]:
            std_data = np.array(D[key]["t_r_std"])
            if verbose:
                print(f"  Found D[key]['t_r_std'], shape={std_data.shape}")
        else:
            if verbose:
                print("  No std data found!")

        if std_data is not None:
            if std_data.ndim == 2 and std_data.shape[1] == 2:
                r_std = std_data[:, 1]
                if display_radius:
                    std_data = r_std * 5.0          # radius in nm
                else:
                    std_data = r_std * 2 * 5.0      # diameter in nm
            std_data = np.asarray(std_data).flatten()
            if len(std_data) == len(val_md_plot):
                if normalize_diameter:
                    std_plot = std_data / val_md[0]
                else:
                    std_plot = std_data
                std_plot = np.nan_to_num(std_plot, nan=0.0)
                time_min = t / 1000 / 60
                ax.fill_between(
                    time_min,
                    val_md_plot - std_plot,
                    val_md_plot + std_plot,
                    color=md_line_color,
                    alpha=0.3,
                    linewidth=0,
                )
                if verbose:
                    print("  fill_between executed successfully!")
            elif verbose:
                print("  Length mismatch! Skipping std plot.")

    # ---------------------------
    # Plot exp curve
    # ---------------------------
    if plot_coltharp:
        ax.plot(t_model, val_model, color=coltharp_color, label=label or coltharp_label, ls="--")
    if plot_whitley:
        if normalize_diameter:
             whitley_constriction[:,1] = whitley_constriction[:,1] / whitley_constriction[:,1][0]
        ax.plot(*whitley_constriction.T, color=coltharp_color, label=label or whitley_label, ls="--")
    # ---------------------------
    # Optional vertical line(s)
    # ---------------------------
    if vline_pos is not None:
            if not isinstance(vline_pos, (list, tuple, np.ndarray)):
                vline_pos = [vline_pos]

            # Determine actual color safely
            vline_color = md_line_color if md_line_color is not None else "k"

            for xpos in vline_pos:
                ax.axvline(x=xpos, color=vline_color, ls="--", lw=lw / 2)

            if verbose:
                print(f"[DEBUG] Drew dashed vline(s) at {vline_pos} with color={vline_color}")

    # ---------------------------
    # Axis labels and legend
    # ---------------------------
    if axislabels:
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(ylabel)
        ax.legend(loc=legendloc, fontsize=8, title=legendtitle, ncol=ncol)

    # ---------------------------
    # Optional inset
    # ---------------------------
    if inset and "H_total" in D[key]:

        if not hasattr(ax, "my_inset"):
            ax.my_inset = inset_axes(
                ax,
                width=width,
                height=height,
                loc="lower left",
                borderpad=4,
            )

        ax_ins = ax.my_inset

        plot_septum_height(
            ax_ins=ax_ins,
            D=D,
            key=key,
            pgt=pgt,
            z_range_tuple=z_range_tuple,
            strandheight=strandheight,
            md_line_color=md_line_color,
            axislabels=axislabels,
            xlim=(-120, 120),
        )
        #ax_ins.set_xlim(230-60, 230+60)
        ax_ins.set_xlim(-120,120)
        if axislabels:
            ax_ins.set_xlabel("Long cell axis (nm)", fontsize=7)
            ax_ins.set_ylabel("Septum height (nm)", fontsize=7)

    return ax


def pooled_diam_plot(D, prm, key, ax, overlay, vline_pos=None, mdcolor=None, normalize_diameter=False, display_radius=True,
                     coltharp_label=None, lw=1.9, t_model_max=44, plot_coltharp=True):
    """
    Plot pooled diameter data.

    Parameters
    ----------
    D : dict
        Main data structure.
    prm : dict
        Parameter dictionary.
    key : str
        Dataset key.
    ax : matplotlib.axes.Axes
        Axis on which to plot.
    overlay : bool
        Whether to overlay on existing plot.
    vline_pos : float or None
        Position for optional vertical line.
        """

    # --- Load time–radius data ----------------------------------------
    if "t_r" not in D[key]:
        D[key]["t_r"] = pd.read_pickle(D[key]["rundir"] + "/t_r.pkl")

    mean_r, std_r, nseeds = fct.utils.key_pooling(
        D, key,
        lambda D, key: [v[1] for v in D[key]["t_r"]],
        seeds=prm["seed"],
        verbose=False,
    )

    # Create pooled data
    times = np.array([v[0] for v in D[key]["t_r"]])
    D["pooled"] = {}
    D["pooled"]["t_r"] = np.column_stack((times, mean_r))
    D["pooled"]["t_r_std"] = np.column_stack((times, std_r))

    # --- BACK-FILL / PREPEND SECTION ---
    # how far you want to go back (same spacing as in your data)
    dt = times[1] - times[0] if len(times) > 1 else 0
    t_first = times[0]
    # for example, extend back by 1 or 2 steps
    prepend_times = np.arange(0, t_first, dt)  # adjust as needed
    first_value = mean_r[0]
    first_std = std_r[0]

    prepend_block = np.column_stack((
        prepend_times,
        np.full_like(prepend_times, first_value, dtype=float)
    ))
    prepend_std_block = np.column_stack((
        prepend_times,
        np.full_like(prepend_times, first_std, dtype=float)
    ))

    # combine earlier + existing data
    D["pooled"]["t_r"] = np.vstack((prepend_block, D["pooled"]["t_r"]))
    D["pooled"]["t_r_std"] = np.vstack((prepend_std_block, D["pooled"]["t_r_std"]))

    fct.cylinder.diam_plot(
        D, "pooled", ax,
        mdlabel=overlay,
        mdcolor=mdcolor,
        normalize_diameter=normalize_diameter,
        display_radius=display_radius,
        plot_std=True,
        lw=lw,
        t_model_max=t_model_max,
        plot_coltharp=plot_coltharp,
        coltharp_label=coltharp_label,
        axislabels=None,
    )

def pooled_diam_plot_(key, ax, D, prm, overlay):
    # Load time–radius data
    D[key]["t_r"] = pd.read_pickle(D[key]["rundir"] + "/t_r.pkl")

    # --- Pool seeds --------------------------------------------------------
    mean_r, std_r, nseeds = fct.utils.key_pooling(
        D, key,
        lambda D, key: [v[1] for v in D[key]["t_r"]],
        seeds=prm["seed"],
        verbose=False,
    )

    # --- Assemble pooled data structures -----------------------------------
    D["pooled"] = {}
    D["pooled"]["t_r"] = [[t, r] for t, r in zip(
        [v[0] for v in D[key]["t_r"]], mean_r)]
    D["pooled"]["t_r_std"] = [[t, s] for t, s in zip(
        [v[0] for v in D[key]["t_r"]], std_r)]

    # --- Plot using your existing diameter plotter -------------------------
    # arrt_value = dict(key)["arrt"]

    fct.cylinder.diam_plot(
        D,
        "pooled",
        ax,
        overlay=overlay, axislabels=None,
        plot_std=True, plot_coltharp=False,
        plot_whitley=True, whitley_label=None,
        vline_pos=dict(key)["arrt"]/60
    )


def render_clipping_movie(mesh, filename, cam_dict,
                          image_scale=2, num_frames=10,
                          start_x=-150, end_x=-18,
                          clip_normal=[1, 0, 0],
                          scalars='radius', cmap='Purples_r',
                          demo=False,
                          show_scalar_bar=True,
                          clim=None,):
    import pyvista as pv
    """
    Renders a movie of a clipping plane passing through a mesh.

    Parameters
    ----------
    mesh : pv.PolyData
        Input mesh with point data scalars.
    filename : str
        Output movie filename. Ignored in demo mode.
    cam_dict : dict
        Keys: position, focal_point, view_up.
    image_scale : int
        Supersampling factor. Default 2.
    num_frames : int
        Number of frames in the movie. Default 10.
    start_x, end_x : float
        Range of clipping plane x-origin.
    clip_normal : list
        Clipping plane normal vector.
    scalars : str
        Point data array to color by. Default 'radius'.
    cmap : str
        Colormap. Default 'viridis'.
    demo : bool
        If True, render a single mid-clip frame interactively via trame
        instead of writing a movie. Useful for testing camera/clip settings.
    """
    if demo:
        # single interactive frame at midpoint of clip range for testing
        mid_x      = (start_x + end_x) / 2
        clipped    = mesh.clip(normal=clip_normal, origin=[mid_x, 0, 0], invert=False)
        plotter    = pv.Plotter()
        plotter.add_mesh(clipped, scalars=scalars, cmap=cmap,
                         smooth_shading=True, show_scalar_bar=True)
        plotter.camera_position = [
            cam_dict['position'],
            cam_dict['focal_point'],
            cam_dict['view_up']
        ]
        #plotter.add_text(f'Demo clip at x={mid_x:.2f}', position='upper_left')
        plotter.show(jupyter_backend="trame")
        # print camera state after interaction so you can copy it into cam_dict
        print("camera position:   ", plotter.camera.position)
        print("camera focal_point:", plotter.camera.focal_point)
        print("camera view_up:    ", plotter.camera.up)
        return plotter

    pv.start_xvfb()
    x_origins = np.linspace(start_x, end_x, num_frames)

    plotter = pv.Plotter(off_screen=True)
    plotter.enable_anti_aliasing('ssaa')
    plotter.camera_position = [
        cam_dict['position'],
        cam_dict['focal_point'],
        cam_dict['view_up']
    ]
    plotter.image_scale = image_scale
    plotter.open_movie(filename)

    for origin_x in tqdm(x_origins, total=len(x_origins)):
        clipped = mesh.clip(normal=clip_normal, origin=[origin_x, 0, 0], invert=False)

        if plotter.actors.get('clipped_mesh_actor'):
            plotter.remove_actor('clipped_mesh_actor')

        plotter.add_mesh(clipped,
                         name='clipped_mesh_actor',
                         scalars=scalars,
                         cmap=cmap,
                         smooth_shading=True,
                         show_scalar_bar=show_scalar_bar,
                         clim=clim,)
        #plotter.add_text(f'Clip at x={origin_x:.2f}',
        #                 name='origin_label', position='upper_left')
        plotter.camera.position    = cam_dict['position']
        plotter.camera.focal_point = cam_dict['focal_point']
        plotter.camera.up          = cam_dict['view_up']
        plotter.write_frame()

    plotter.close()
    print(f"Movie saved to {filename}")


def reconstruct_H_blurred(H_total, rundir, blur_nm=(25, 4)):
    """
    Reconstruct H_blurred from saved H_total using the same blur
    parameters as histogram_mesh.

    Parameters
    ----------
    H_total : np.ndarray, shape (n_theta, n_z)
        Raw cumulative histogram counts, loaded from H_total.npy.
    rundir : str
        Path to run directory containing parameters.json.
    blur_nm : tuple of float
        (sigma_theta_nm, sigma_z_nm) — must match what was used in histogram_mesh.

    Returns
    -------
    H_blurred : np.ndarray, shape (n_theta, n_z)
        Smoothed deformation map in nm.
    R_nm : float
        Run-start radius in nm.
    """
    import json
    from scipy.ndimage import gaussian_filter

    with open(f"{rundir}/parameters.json") as f:
        d = json.load(f)

    NM_PER_SU        = 5.0
    circumference_su = 2 * d["Lx"]
    circumference_nm = NM_PER_SU * circumference_su
    R_nm             = circumference_nm / (2 * np.pi)
    strand_width_nm  = pgt.strand_thickness_width

    n_theta          = H_total.shape[0]
    theta_bin_nm     = circumference_nm / n_theta
    z_bin_nm         = strand_width_nm

    sigma_theta_nm, sigma_z_nm = blur_nm
    sigma_theta_px = sigma_theta_nm / theta_bin_nm
    sigma_z_px     = sigma_z_nm     / z_bin_nm

    H_scaled  = H_total * strand_width_nm
    H_blurred = gaussian_filter(H_scaled,
                                sigma=(sigma_theta_px, sigma_z_px),
                                mode=('wrap', 'reflect'))
    return H_blurred, R_nm

def plot_circle_projection(H_blurred, R_nm, ax=None, cmap='magma', lw=1.5,
                           use_cbar=False, alpha=1.0,
                           label=None):
    """
    Project H_blurred onto a circle and plot.
    
    If cmap is None, it uses the next color from the axis color cycler 
    and applies the specified alpha (perfect for seed overlays).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # max deformation along z for each circumferential position
    H_max = H_blurred.max(axis=1)
    r = R_nm - H_max

    n_theta = len(r)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Close the loop for plotting
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    if cmap is not None:
        # Use LineCollection for multicolored segments based on deformation
        from matplotlib.collections import LineCollection
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        H_max_closed = np.append(H_max, H_max[0])
        norm = plt.Normalize(H_max.min(), H_max.max())
        
        lc = LineCollection(segments, cmap=cmap, norm=norm, lw=lw, alpha=alpha)
        lc.set_array(H_max_closed[:-1])
        ax.add_collection(lc)
        
        if use_cbar:
            plt.colorbar(lc, ax=ax, label='max deformation (nm)')
    else:
        # Use standard plot with the next color in the cycler
        # This is much faster for overlays and supports alpha natively
        ax.plot(x, y, lw=lw, alpha=alpha,
            label=label)

    # Reference circle (dashed)
    theta_ref = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R_nm * np.cos(theta_ref), R_nm * np.sin(theta_ref),
            'k--', lw=0.8, alpha=0.2)

    ax.set_aspect('equal')
    # Use cleaner limits if needed: ax.set_xlim(-R_nm*1.2, R_nm*1.2)
    return ax

import numpy as np

def analyze_circularity_centered(H_blurred, R_nm):
    """
    Calculates circularity metrics by first finding the shape's 
    actual centroid to handle off-center coordinates.
    """
    # 1. Generate Cartesian coordinates of the boundary
    H_max = H_blurred.max(axis=1)
    r_raw = R_nm - H_max
    n_theta = len(r_raw)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    
    x = r_raw * np.cos(theta)
    y = r_raw * np.sin(theta)

    # 2. Find the Centroid (Center of Mass)
    # For a closed loop, the simple mean is usually sufficient
    cx, cy = np.mean(x), np.mean(y)

    # 3. Calculate "True" Radius relative to Centroid
    # This removes the 'shift' error
    r_centered = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # 4. Area and Perimeter (Geometry-based, center-independent)
    # Perimeter is the same regardless of origin
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    perimeter = np.sum(np.sqrt(dx**2 + dy**2))
    
    # Area (Shoelace formula is robust for off-center shapes)
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # 5. Metrics
    circularity = (4 * np.pi * area) / (perimeter**2)
    rms_roughness = np.std(r_centered)
    ovality = (r_centered.max() - r_centered.min()) / r_centered.mean()
    center_offset = np.sqrt(cx**2 + cy**2)

    return {
        "Circularity": circularity,
        "Roughness\n(nm)": rms_roughness,
        "Ovality (Ratio)": ovality,
        "Center Offset (nm)": center_offset,
        "Centroid": (cx, cy)
    }

def circularity_analysis(D, key):
    H_blurred, R_nm = fct.cylinder.reconstruct_H_blurred(D[key]["H_total"], D[key]["rundir"], blur_nm=(25, 4))
    circularity = fct.cylinder.analyze_circularity_centered(H_blurred, R_nm)
    return circularity["RMS Roughness (nm)"]

def circularity_analysis_full(D, key):
    import functions as fct
    """Returns the full dictionary of metrics from the circularity analysis."""
    if key not in D or D[key].get("H_total") is None:
        return None
    
    H_blurred, R_nm = fct.cylinder.reconstruct_H_blurred(
        D[key]["H_total"], D[key]["rundir"], blur_nm=(25, 4)
    )
    # This returns a dict like {'RMS Roughness (nm)': 0.5, 'Circularity': 0.8, ...}
    return fct.cylinder.analyze_circularity_centered(H_blurred, R_nm)


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

def d_alpha(t, tau_c, alpha):
    """Coltharp model for normalized diameter/radius evolution."""
    return (1 - (t / tau_c) ** alpha) ** (1 / alpha)

def front_render_display(
    D,
    key,
    ax,
    img_fname="septum_front",
    crop=(300, -100, 500, -500),
    show_inset=False,
    inset_width="40%",
    inset_height="40%",
    inset_loc="center left",
    overlay=None,
    mdcolor="tab:blue",
    coltharp_color="k",
    coltharp_label=None, #"Coltharp model",
    normalize=True,
    display_radius=False,
    modelonly=False,
):
    """
    Display the septum 'front' PNG and optionally an inset plot showing the 
    measured diameter/radius evolution with Coltharp model overlay.

    Parameters
    ----------
    D : dict
        Dictionary containing at least D[key]["rundir"] and D[key]["t_r"].
    key : hashable
        Dataset key.
    ax : matplotlib.axes.Axes
        Axis to render into.
    label : str
        Image label (default 'front').
    crop : tuple
        Crop window passed to display_png.
    show_inset : bool
        If True, include an inset showing time evolution.
    inset_width, inset_height : str or float
        Size of inset in parent axes units.
    inset_loc : str
        Position of inset (e.g., 'lower left').
    overlay : str
        Label for inset line (data/measurement).
    mdcolor : str
        Color for measurement line.
    coltharp_color : str
        Color for model line.
    coltharp_label : str
        Legend label for model line.
    normalize : bool
        If True, normalize D(t)/D(t₀).
    display_radius : bool
        If True, plot radius instead of diameter.
    modelonly : bool
        If True, only plot model in inset (no measured data).
    """

    # ---------------------------------------------------------
    # 1. Display PNG image
    # ---------------------------------------------------------
    rundir = D[key]["rundir"]
    fname = f"{rundir}/{rundir.split('/')[-2]}{img_fname}.png"
    fct.midcell_transport.display_png(fname, ax, crop=crop)

    # ---------------------------------------------------------
    # 2. Optional inset plot
    # ---------------------------------------------------------
    if show_inset and "t_r" in D[key]:
        if not hasattr(ax, "my_inset"):
            ax.my_inset = inset_axes(
                ax,
                width=inset_width,
                height=inset_height,
                loc=inset_loc,
                borderpad=1,
            )
        ax_ins = ax.my_inset

        # Extract and process data
        t, r = np.array(D[key]["t_r"]).T  # expects ms and nm
        val_md = r if display_radius else r * 2  # radius–>diameter
        val_md_plot = val_md / val_md[0] if normalize else val_md

        # Data line (if not model-only)
        if not modelonly:
            ax_ins.plot(
                t / 1000 / 60,
                val_md_plot,
                lw=3,
                label=overlay,
                color=mdcolor,
            )

        # Model line (Coltharp)
        tau_c, alpha = 51, 1.3
        # t_model = np.linspace(0, np.max(t) / 1000 / 60 * 1.1, 500)
        t_model = np.linspace(0, 50, 1000)  # safe range

        val_model = d_alpha(t_model * 1000 * 60, tau_c, alpha)
        val_model = d_alpha(t_model, tau_c, alpha)  # t_model already in minutes

        if not normalize:
            val_model *= val_md[0]
        # print(t_model[:5], val_model[:5])  # sanity check first few values
        ax_ins.plot(
            t_model,
            val_model,
            color=coltharp_color,
            ls="--",
            label=coltharp_label,
        )

        # Inset formatting
        ax_ins.set_xlabel("Time (min)", fontsize=7)
        # ax_ins.set_ylabel(
        #     r"$D/D_0$" if normalize else "Diameter (nm)", fontsize=7
        # )
        ax_ins.tick_params(axis="both", labelsize=7)
        # ax_ins.legend(fontsize=6, loc="best")

    return ax


import numpy as np
import pandas as pd

def map_to_cylinder(df, fulldf=None, radial_offset=None, NM_PER_SU=1.0):
    """
    Map particle x-positions onto a cylinder by converting x → θ (theta), per time point.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'x', 'y', 'type', 'time'.
    fulldf : pd.DataFrame, optional
        Reference dataframe to determine x-range (defaults to df if None).
    radial_offset : dict, optional
        {particle_type: offset_nm} — per-type radial offsets in nm.
    NM_PER_SU : float
        Conversion factor from simulation units to nm.

    Returns
    -------
    pd.DataFrame
        Columns: ['time','x','y','z','theta','y_nm','R_nm','type',...]
    """

    def _map_single_frame(frame):
        """Map a single time slice to cylindrical coordinates."""
        ref = fulldf[fulldf["time"] == frame["time"].iloc[0]] if fulldf is not None else frame

        x_min = ref["x"].min()
        x_max = ref["x"].max()
        box_width_su = x_max - x_min
        R_nm = box_width_su * NM_PER_SU / (2 * np.pi)

        theta = ((frame["x"].values - x_min) / box_width_su) * 2 * np.pi

        # base radius
        r = np.full(len(frame), R_nm)
        if radial_offset is not None:
            for ptype, offset_nm in radial_offset.items():
                mask = frame["type"].values == ptype
                r[mask] += offset_nm

        out = frame.copy()
        out["theta"] = theta
        out["y_nm"] = frame["y"].values * NM_PER_SU
        out["x"] = r * np.cos(theta)
        out["y"] = r * np.sin(theta)
        out["z"] = frame["y"].values * NM_PER_SU
        out["R_nm"] = R_nm
        return out

    # ---- Apply per-time ----
    if "time" not in df.columns:
        raise ValueError("Input dataframe must contain a 'time' column.")
    
    mapped = pd.concat(
        [_map_single_frame(g) for _, g in df.groupby("time")],
        ignore_index=True
    )
    return mapped




def plot_H_blurred_profile(D, key, ax,
                           color="#6c8895ff",
                           label=None,
                           z_slice_range=(180, 290),
                           z_range_tuple=(-3*70, 3*70),
                           blur_nm=(25, 4),
                           df_filter_type=[11],
                           alpha_fill=0.25,
                           lw=2):
    """
    Plot the mean ± std of H_blurred profile along the z-axis.

    Parameters
    ----------
    D : dict
        Simulation dictionary containing rundir and potentially H_blurred entries.
    key : hashable
        Key of the simulation configuration to plot.
    ax : matplotlib.axes.Axes
        Axis on which to draw.
    color : str
        Line color (hex or named color).
    label : str
        Legend label for the plot.
    z_slice_range : tuple
        Slice range (start, stop) used to index H_blurred.T.
    z_range_tuple : tuple
        Tuple defining z range (in simulation units) used in histogram mesh.
    blur_nm : tuple
        Blur in nanometers applied during histogram mesh.
    df_filter_type : list
        Particle types to select when building the histogram mesh.
    alpha_fill : float
        Alpha for the filled area representing the std.
    lw : float
        Linewidth.

    Returns
    -------
    None
    """
    import functions as fct
    import numpy as np
    import matplotlib.pyplot as plt
    import functions.sPG_tracker as pgt
    # Load H_blurred if not yet computed
    if "H_blurred" not in D[key]:
        df = fct.utils.load(D[key]["rundir"])
        _, _, _, H_blurred, *_ = fct.cylinder.histogram_mesh(
            df[df["type"].isin(df_filter_type)],
            df,
            D[key]["rundir"],
            z_range_tuple=z_range_tuple,
            blur_nm=blur_nm,
        )
        D[key]["H_blurred"] = H_blurred

    # Extract and slice
    H = D[key]["H_blurred"].T[slice(*z_slice_range), :]  # shape: (n_z_slice, n_theta)

    # Compute z-centers in nanometers
    strand_width_nm = pgt.strand_thickness_width
    NM_PER_SU       = 5.0
    strand_width_su = strand_width_nm / NM_PER_SU
    z_min, z_max    = z_range_tuple
    z_edges         = np.arange(z_min, z_max + strand_width_su, strand_width_su)
    z_centers_nm    = (z_edges[:-1] + z_edges[1:]) / 2 * NM_PER_SU
    z_slice_nm      = z_centers_nm[slice(*z_slice_range)]

    # Mean and standard deviation along the ring direction
    mean = H.mean(axis=1)
    std  = H.std(axis=1)

    # Plot
    ax.plot(z_slice_nm, mean, color=color, lw=lw, label=label)
    ax.fill_between(z_slice_nm, mean - std, mean + std,
                    color=color, alpha=alpha_fill)

    # Axis styling (optional)
    ax.set_aspect("equal")
    ax.set_xlabel("Long cell axis (nm)")
    ax.set_ylabel("Septum height (nm)")

def cross_section_plot(D, key, ax, overlay=None, z_range_tuple=None, color=None):    
    import functions.sPG_tracker as pgt
    zrt             = z_range_tuple or D[key].get("z_range_tuple", (-3*70, 3*70))
    strand_width_su = pgt.strand_thickness_width / 5.0
    z_min, z_max    = zrt
    z_edges         = np.arange(z_min, z_max + strand_width_su, strand_width_su)
    z_centers       = (z_edges[:-1] + z_edges[1:]) / 2
    z_nm            = z_centers * 5.0

    if len(z_nm) != D[key]["H_total"].shape[1]:
        print(f"Warning: z_nm {len(z_nm)} != H_total z-dim {D[key]['H_total'].shape[1]}")
        return

    ax.plot(z_nm, D[key]["H_total"].mean(axis=0) * pgt.strand_height_nm,
            label=overlay, color=color)
    ax.set_xlabel("Long cell axis (nm)")
    ax.set_ylabel("Septum height (nm)")
    ax.legend()