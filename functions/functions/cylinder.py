import json
import numpy as np
from scipy.ndimage import gaussian_filter
import functions.sPG_tracker as pgt

import pyvista as pv
# import tqdm as tqdm #.notebook as tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# For remote/headless use
pv.start_xvfb()           # virtual framebuffer (avoids X server errors)
pv.global_theme.jupyter_backend = "trame"  # use web frontend in Jupyter
import functions as fct
# from functions.common_imports import *



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

    # ── Bin geometry from pgt globals ────────────────────────────────────────
    # strand_thickness_width: bin size in nm (set by pgt.set_septal_bins)
    # z-bins: fixed edges spaced at strand_width from z_range_tuple
    strand_width_nm = pgt.strand_thickness_width          # nm
    strand_width_su = strand_width_nm / NM_PER_SU         # simulation units

    z_min, z_max = z_range_tuple
    z_edges = np.arange(z_min, z_max + strand_width_su, strand_width_su)  # sim units
    n_z     = len(z_edges) - 1
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2         # bin centers in sim units

    # theta: fixed 200 bins to match peakweighted tracker scheme
    circumference_nm = NM_PER_SU * circumference_su
    n_theta   = 200
    t_edges   = np.linspace(0, 2 * np.pi, n_theta + 1)
    t_centers = (t_edges[:-1] + t_edges[1:]) / 2

    t_grid, z_grid = np.meshgrid(t_centers, z_centers, indexing='ij')
    H_total        = np.zeros((n_theta, n_z))

    # ── Gaussian blur pixel sigmas — computed once before the loop ────────────
    # theta bin spacing in nm: arc_length / n_theta
    theta_bin_nm   = circumference_nm / n_theta
    z_bin_nm       = strand_width_nm               # z bins spaced at strand_width
    sigma_theta_nm, sigma_z_nm = blur_nm
    sigma_theta_px = sigma_theta_nm / theta_bin_nm
    sigma_z_px     = sigma_z_nm     / z_bin_nm

    # ── Accumulate histogram over dT intervals ────────────────────────────────
    t_min  = df["time"].min()
    t_max  = df["time"].max()
    t_bins = np.arange(t_min, t_max + dT, dT)

    r_snapshots = []
    t_snapshots = []

    for t0, t1 in zip(t_bins[:-1], t_bins[1:]):
        full_i = fulldf[(fulldf["time"] >= t0) & (fulldf["time"] < t1)]
        if len(full_i) == 0:
            continue
        x_min = full_i["x"].min()
        x_max = full_i["x"].max()

        df_i = df[(df["time"] >= t0) & (df["time"] < t1)]
        if len(df_i) == 0:
            continue

        # map x → theta using this interval's box extent
        data_theta = ((df_i["x"].values - x_min) / (x_max - x_min)) * 2 * np.pi
        data_z     = df_i["y"].values   # simulation units

        H, _, _ = np.histogram2d(data_theta, data_z, bins=[t_edges, z_edges])
        H_total += H

        # snapshot r_final after each interval if requested
        if per_interval:
            H_s = gaussian_filter(H_total * strand_width_nm,
                                  sigma=(sigma_theta_px, sigma_z_px),
                                  mode=('wrap', 'reflect'))
            r_snap = R_nm - H_s
            r_snapshots.append(r_snap.copy())
            t_snapshots.append(t1)

    # ── Scale: each count = one strandwidth of radial displacement ────────────
    H_scaled  = H_total * strand_width_nm   # nm

    H_blurred = gaussian_filter(H_scaled,
                                sigma=(sigma_theta_px, sigma_z_px),
                                mode=('wrap', 'reflect'))

    # ── Deformed surface geometry ─────────────────────────────────────────────
    r_final  = R_nm - H_blurred                  # nm
    x_coords = r_final * np.cos(t_grid)
    y_coords = r_final * np.sin(t_grid)
    z_coords = z_grid                            # simulation units

    if per_interval:
        return t_grid, z_grid, H_total, H_blurred, x_coords, y_coords, z_coords, \
               np.array(t_snapshots), np.array(r_snapshots)
    return t_grid, z_grid, H_total, H_blurred, x_coords, y_coords, z_coords


def render_time_movie(t_grid, z_grid, r_snapshots, t_snapshots, filename, cam_dict,
                      clip_normal, clip_origin,
                      image_scale=3,
                      select_view=None):
    """
    Render a movie over time using pre-computed r_snapshots from histogram_mesh
    with per_interval=True.

    Parameters
    ----------
    t_grid, z_grid : np.ndarray, shape (n_theta, n_z)
        Grid coordinates from histogram_mesh.
    r_snapshots : np.ndarray, shape (n_intervals, n_theta, n_z)
        Radius at each interval in nm.
    t_snapshots : np.ndarray, shape (n_intervals,)
        Simulation time at each interval.
    filename : str
        Output movie filename.
    cam_dict : dict
        Keys: position, focal_point, view_up.
    clip_normal, clip_origin : array-like
        Clipping plane normal and origin for cross-section view.
    image_scale : int
        Supersampling factor for anti-aliasing.
    """
    n_theta, n_z = t_grid.shape

    if select_view == "front":
        cam_dict = {
            'position': (-1439.267348837681, 367.91393768244296, 2.6378280610901945),
            'focal_point': (0.0, 0.0, 0.15),
            'view_up': (-0.2476655423778592, -0.9687882884811285, -0.010537135307405729)
        }
        print("using preset front view")
    elif select_view == "side":
        cam_dict = {'position': (-1166.082220657981, -38.3627876539593, 919.7357259921413),
        'focal_point': (0.0, 0.0, 0.15),
        'view_up': (-0.011985074051557796, -0.9983110926201956, -0.05684470381178824)
        }
        print("using preset side view")

    # print(pv)  # should show <module 'pyvista' ...>
    # print(pv.Plotter)  # should show the class
    plotter = pv.Plotter(off_screen=True)
    plotter.enable_anti_aliasing('ssaa')
    plotter.camera_position = [
        cam_dict['position'],
        cam_dict['focal_point'],
        cam_dict['view_up']
    ]
    plotter.image_scale = image_scale
    plotter.open_movie(filename)


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
    # print("starting loop")
    for snap_idx, (r_snap, t) in enumerate(tqdm(zip(r_snapshots, t_snapshots),
                                                total=len(t_snapshots),
                                                desc="rendering frames")):
        # print("rad def")
        # r_snap: (n_theta, n_z) in nm
        x = r_snap * np.cos(t_grid)
        y = r_snap * np.sin(t_grid)
        z = z_grid  # simulation units

        points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        radial = np.sqrt(x**2 + y**2).ravel()
        # print("mesh def")
        mesh = pv.PolyData(points, faces)
        
        mesh['radius'] = radial
        mesh['H']      = (r_snapshots[0] - r_snap).ravel()  # deformation relative to first frame

        clipped = mesh.clip(normal=clip_normal, origin=clip_origin, invert=False)

        if plotter.actors.get('clipped_mesh_actor'):
            plotter.remove_actor('clipped_mesh_actor')

        # print(f"snap {snap_idx}: r_snap range {r_snap.min():.1f}–{r_snap.max():.1f}  "
        #         f"points={len(points)}  clipped points={clipped.n_points}")
        plotter.add_mesh(clipped,
                         name='clipped_mesh_actor',
                         scalars='radius',
                         cmap='Purples_r',
                         smooth_shading=True,
                         show_scalar_bar=False)
        plotter.camera.position    = cam_dict['position']
        plotter.camera.focal_point = cam_dict['focal_point']
        plotter.camera.up          = cam_dict['view_up']
        # plotter.render()  # force render before writing frame
        plotter.write_frame()

    plotter.close()
    print(f"Movie saved to {filename}")