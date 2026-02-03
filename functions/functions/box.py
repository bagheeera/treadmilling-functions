import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import pyarrow.feather as feather

def synth_scatter(ax, key, D, boxcolor="r", cmap="viridis",
    use_finalframe=True,
    use_df_rot=False):
    if use_finalframe:
        df = D[key]["finalframe"]
        df = df[df["type"]==5]
    elif use_df_rot:
        if "df_rot" not in D[key]:
            raise ValueError("df_rot not found in D[key]")
        df = D[key]["df_rot"].copy()
        df = df[df["type"]==5]
        df = df[df["time"]==df["time"].max()]
        df["x"] = df["x_rot"]
        df["y"] = df["y_rot"]
    else:
        df = D[key]["df_synth"]
        df = df[df["time"]==df["time"].max()]
    #ax.scatter(*df[["x", "y"]].values.T,
    #s=1)
    # times 5
    df.loc[:, ["x", "y"]] = df.loc[:, ["x", "y"]] * 5
    ax.hist2d(*df[["x", "y"]].values.T, bins=50,
    cmap=cmap)
    # synth_scatter(ax, key)
    ax.set_aspect("equal")
    # Square corners
    xs = [-250, 250, 250, -250, -250]
    ys = [250, 250, -250, -250, 250]
    #xs = [-50, 50, 50, -50, -50]
    #ys = [50, 50, -50, -50, 50]
    
    ax.plot(xs, ys, color=boxcolor)        # Draw borders only

def rotate_points(pts, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return pts @ R.T

def plot_before_after_rotation(
    D,
    key,
    mean,
    mean_window=5000,
    feather_name="df_synth.feather",
    s=2,
):
    """
    Showcase particle positions before and after rotation, and visualize
    the mean signal used to compute the rotation angle.

    Parameters
    ----------
    D : dict
        Data dictionary
    key : hashable
        Key into D
    mean : array-like
        Mean signal used to compute rotation angle
    mean_window : int
        Number of last points in `mean` used for averaging
    feather_name : str
        Name of feather file containing df_synth
    s : float
        Scatter marker size
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pyarrow.feather as feather

    # --- load df_synth if needed ---
    if "df_synth" not in D[key]:
        D[key]["df_synth"] = feather.read_feather(
            os.path.join(D[key]["rundir"], feather_name)
        )

    df = D[key]["df_synth"]

    # final time frame
    df_t = df[df["time"] == df["time"].max()]
    pos_before = df_t[["x", "y"]].values

    # --- rotation angle ---
    mean = np.asarray(mean)
    mean_slice = mean[-mean_window:]
    theta = -np.radians(mean_slice.mean())

    pos_after = rotate_points(pos_before, theta)

    # --- common limits ---
    xmin = min(pos_before[:, 0].min(), pos_after[:, 0].min())
    xmax = max(pos_before[:, 0].max(), pos_after[:, 0].max())
    ymin = min(pos_before[:, 1].min(), pos_after[:, 1].min())
    ymax = max(pos_before[:, 1].max(), pos_after[:, 1].max())

    # --- figure layout ---
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # --- BEFORE ---
    ax0.scatter(pos_before[:, 0], pos_before[:, 1], s=s)
    ax0.axhline(pos_before[:, 1].mean(), color="r")
    ax0.set_title("Before rotation")
    ax0.set_aspect("equal")
    ax0.grid()

    # --- AFTER ---
    ax1.scatter(pos_after[:, 0], pos_after[:, 1], s=s)
    ax1.axhline(pos_after[:, 1].mean(), color="r")
    ax1.set_title("After rotation")
    ax1.set_aspect("equal")
    ax1.grid()

    for ax in (ax0, ax1):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # --- MEAN SIGNAL ---
    ax2.plot(mean, lw=1, label="mean")
    ax2.axvspan(
        len(mean) - mean_window,
        len(mean),
        color="orange",
        alpha=0.3,
        label=f"averaging window (last {mean_window})",
    )
    ax2.axhline(mean_slice.mean(), color="r", ls="--", label="window mean")
    ax2.set_title("Mean used for rotation")
    ax2.set_xlabel("index")
    ax2.set_ylabel("value")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()


def store_rotated_df(
    D,
    key,
    mean,
    mean_window,
    feather_name="df_synth.feather",
    store_key="df_rot",
):
    """
    Compute rotation angle from mean signal and store rotated coordinates
    in D[key] without plotting.

    Stores:
      - D[key][store_key] : rotated dataframe (copy)
      - D[key]["rotation_angle_deg"]
      - D[key]["rotation_mean_window"]
    """

    import os
    import numpy as np

    # --- load df_synth if needed ---
    if "df_synth" not in D[key]:
        D[key]["df_synth"] = feather.read_feather(
            os.path.join(D[key]["rundir"], feather_name)
        )

    df = D[key]["df_synth"]

    # --- compute rotation angle ---
    mean = np.asarray(mean)
    mean_slice = mean[-mean_window:]
    angle_deg = mean_slice.mean()
    theta = -np.radians(angle_deg)

    # --- rotate all points ---
    xy_rot = rotate_points(df[["x", "y"]].values, theta)

    df_rot = df.copy()
    df_rot["x_rot"] = xy_rot[:, 0]
    df_rot["y_rot"] = xy_rot[:, 1]

    # --- store ---
    D[key][store_key] = df_rot
    D[key]["rotation_angle_deg"] = angle_deg
    D[key]["rotation_mean_window"] = mean_window


def plot_rotated_transport(
    df_rot,
    cols=("y_rot", "x_rot"),
    figsize=(5, 4),
    synth_type=5
):
    """
    Plot mean ± std of |coordinate| vs time for rotated data.
    """

    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)

    df_rot = df_rot[df_rot["type"]==synth_type]

    for col in cols:
        g = (
            df_rot.assign(abs_val=df_rot[col].abs())
            .groupby("time")["abs_val"]
            .agg(["mean", "std"])
            .sort_index()
        )

        plt.fill_between(
            g.index,
            g["mean"] - g["std"],
            g["mean"] + g["std"],
            alpha=0.3,
        )
        plt.plot(g.index, g["mean"], label=col)

    plt.grid()
    plt.legend()
    plt.xlabel("time")
    plt.show()

def plot_rotation_angle(D, key, ax, overlay=None,
                        tozero=True,
                        fillbeween=False,
                        color=None):
        angles = D[key]["filament_orientation"]
        angles = [np.mod(a, 180) for a in angles]
        mean = np.array([np.mean(a) for a in angles]
                        )
        
        mean = mean - np.mean(mean[-500:]) if tozero else mean

        std = np.array([np.std(a) for a in angles])
        if fillbeween:
            ax.fill_between(np.arange(len(mean)),
                            mean - std,
                            mean + std,
                            alpha=0.3)
        ax.plot(mean, label=overlay,
                color=color if color is not None else None)


def orientation_plot(ax, x_minmax, y_minmax, d_x_mean, d_y_mean, x_edges, y_edges,
y_density=4, cut=300):

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

def plot_fract_in_box(
    ax, key, D, Nsynth, seeds, overlay,
    pad_with_nan=False,
    factor=1,
    overlayonly=False,
):
    from functions.utils import update_key
    all_fract = []

    for seed in seeds:
        key_seed = update_key(key, **{"seed": seed})
        if "nr_within_40" in D[key_seed]:
            fract_inside = D[key_seed]["nr_within_40"] / Nsynth
            all_fract.append(np.asarray(fract_inside))

    counter = len(all_fract)
    if counter == 0:
        return

    if pad_with_nan:
        # --- NaN padding mode ---
        max_len = max(len(f) for f in all_fract)
        data = np.full((counter, max_len), np.nan)

        for i, f in enumerate(all_fract):
            data[i, :len(f)] = f

        mean_fract = np.nanmean(data, axis=0)
        std_fract  = np.nanstd(data, axis=0)
        x = np.arange(max_len)

    else:
        # --- truncate to common length ---
        min_len = min(len(f) for f in all_fract)
        data = np.stack([f[:min_len] for f in all_fract])

        mean_fract = data.mean(axis=0)
        std_fract  = data.std(axis=0)
        x = np.arange(min_len)

    ax.plot(x, factor*mean_fract, label=overlay if overlayonly else f"{overlay} ({counter} seeds)")
    # ax.fill_between(x, mean_fract - std_fract, mean_fract + std_fract, alpha=0.3)
    ax.legend()

