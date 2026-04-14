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
                   per_interval=False):
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

    dT               = d["dT"] * d["tstep"]
    NM_PER_SU        = 5.0
    circumference_su = 2 * d["Lx"]               # full box in sim units
    R_nm             = NM_PER_SU * circumference_su / (2 * np.pi)  # run-start radius in nm

    # ── Bin geometry ─────────────────────────────────────────────────────────
    strand_width_nm = pgt.strand_thickness_width          # nm
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

    for t0, t1 in zip(t_bins[:-1], t_bins[1:]):
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
            H_s    = gaussian_filter(H_total * strand_width_nm,
                                     sigma=(sigma_theta_px, sigma_z_px),
                                     mode=('wrap', 'reflect'))
            r_snap = R_nm - H_s
            r_snapshots.append(r_snap.copy())
            t_snapshots.append(t1)

    # ── Scale and blur ────────────────────────────────────────────────────────
    H_scaled  = H_total * strand_width_nm   # nm
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
    render_movie=True
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
            show_scalar_bar=False
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

def diam_plot(D, key, ax, modelonly=False, label=None, mdlabel=None, 
              coltharp_color="k", mdcolor=None, normalize_diameter=True, D_0=578):
    """
    Plot diameter evolution over time.
    
    Parameters
    ----------
    D : dict
        Main dictionary with data
    key : hashable
        Key to access the data in D
    ax : matplotlib.axes.Axes
        Axes object to plot on
    modelonly : bool, optional
        If True, only plot the model, not the data (default: False)
    label : str, optional
        Label for the model line
    mdlabel : str, optional
        Label for the data line
    coltharp_color : str, optional
        Color for the model line (default: "k")
    mdcolor : str, optional
        Color for the data line
    normalize_diameter : bool, optional
        If True, plot diameter/diameter_0. If False, plot absolute diameter in nm (default: True)
    D_0 : float, optional
        Initial diameter in nm, used when normalize_diameter=False (default: 578 based on Coltharp 2026)
    """
    import numpy as np
    
    def d_alpha(t, tau_c, alpha):
        diam = (1-(t/tau_c)**alpha)**(1/alpha)
        return diam
    
    t_model = np.linspace(0, 50, 1000)
    tau_c = 51
    alpha = 1.3
    t, r = np.array(D[key]["t_r"]).T
    diam_md = r * 2 * 5
    
    if normalize_diameter:
        diam_md_plot = diam_md / diam_md[0]
        diam_model = d_alpha(t_model, tau_c, alpha)
    else:
        diam_md_plot = diam_md
        diam_model = d_alpha(t_model, tau_c, alpha) * D_0
    
    ax.plot(t/1000/60, diam_md_plot, lw=3, label=mdlabel, color=mdcolor)
    
    if not modelonly:
        ax.plot(t_model, diam_model,
                color=coltharp_color,
                label=label)







def render_clipping_movie(mesh, filename, cam_dict,
                          image_scale=2, num_frames=10,
                          start_x=-150, end_x=-18,
                          clip_normal=[1, 0, 0],
                          scalars='radius', cmap='Purples_r',
                          demo=False,
                          show_scalar_bar=True):
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
                         show_scalar_bar=show_scalar_bar)
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