import numpy as np
def autocorr_1d(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]  # keep non-negative lags
    return result / result[0]  # normalize

def autocorr_along_axis(arr, axis=0):
    return np.apply_along_axis(autocorr_1d, axis=axis, arr=arr)

from pathlib import Path
import numpy as np
from tifffile import imread

def load_tif_as_array(path: str | Path, *,
                      dtype: np.dtype = np.float32,
                      normalize: str | bool = "bitdepth") -> np.ndarray:
    """
    Load a greyscale .tif image and return a NumPy array.

    Parameters
    ----------
    path : str | Path
        Path to the .tif file.
    dtype : np.dtype, default np.float32
        Target dtype for output array.
    normalize : {'bitdepth', 'minmax', False}, default 'bitdepth'
        - 'bitdepth': normalize based on bit depth (e.g. divide by 255 or 65535)
        - 'minmax'  : normalize based on actual min/max of the image
        - False     : no normalization (just convert dtype)

    Returns
    -------
    np.ndarray
        2‑D image as a NumPy array.
    """
    img = imread(path)

    if img.ndim != 2:
        raise ValueError(f"{path} does not appear to be single-channel.")

    if np.issubdtype(dtype, np.integer):
        return img.astype(dtype, copy=False)

    img = img.astype(dtype, copy=False)

    if normalize == "bitdepth":
        bit_depth = img.itemsize * 8
        max_val = (1 << bit_depth) - 1
        img /= max_val

    elif normalize == "minmax":
        img -= img.min()
        ptp = np.ptp(img)
        if ptp > 0:
            img /= ptp
        else:
            img[:] = 0.0  # flat image → normalized to 0

    elif normalize is False:
        pass  # no scaling

    else:
        raise ValueError(f"Invalid normalize option: {normalize!r}")

    return img

import pandas as pd
import numpy as np
def imshow_kymograph(ax, imfile,
                     showhalf=True,
                     aspect=100,
                     cmap="inferno", 
                     return_extent=False,
                     cropy=None):
    # extract diameter from filename
    df_kymo = pd.read_csv("/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/2__synthase_setup/9__midcell_condensation/8__check_lifetimes/C__wsynth_kymoparams/exp_data/ftsz_dynamics_div_state_categories.csv")
    def extract_diamter_in_nm(img, df= df_kymo):
        diam = df[df["Image_ROI_Name"].str.contains(
            img.split("/")[-1].split(".tif")[0],
            na=False
        )]["DiameterNm"].values[0]
        return diam
    img = load_tif_as_array(imfile) # [2]
    if cropy is not None:
        # print(img.shape)
        img = img[:-cropy, :]
        # print(img.shape)
    if showhalf:
        img = img[:,:img.shape[1]//2]

    diameter = extract_diamter_in_nm(imfile)
    extent = [0, diameter*np.pi, 0, img.shape[0]]
    ax.imshow(img,
        origin="lower",
                aspect=aspect,
                cmap=cmap,
                extent=extent)
    if return_extent:
        return extent

def normalized_kymo_roughness(key, D, startfrom=0):
    """as in load_tif_as_array from https://jupyterhub.ista.ac.at/user/fhorvath/lab/workspaces/auto-g/tree/0__treadmilling/2__synthase_setup/9__midcell_condensation/8__check_lifetimes/C__wsynth_kymoparams/notebooks/kymograph_analysis.ipynb"""
    if "center_xcounts" in D[key]:
        img = D[key]["center_xcounts"]
    else: 
        img = D[key]["center_xcounts_wider"]
    img = img[startfrom:,:]
    img -= img.min()
    ptp = np.ptp(img)
    if ptp > 0:
        img = img / ptp
    else:
        img[:] = 0.0  # flat image → normalized to 0

    return img

def load_kymo_stds():
    import pickle
    with open("/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/2__synthase_setup/9__midcell_condensation/8__check_lifetimes/C__wsynth_kymoparams/exp_data/kymo_STDs.pkl", "rb") as f:
        kymo_stds = pickle.load(f)
    return kymo_stds

import matplotlib.pyplot as plt
def plot_sim_exp_comparison(
    key,
    D,
    lifetimes_plot,
    fct,
    exp_images,
    t0=400,
    marker_size=10,
    skip=4,
    scale=1.3,
    cmap="plasma",
    aspect_sim=40,
    aspect_exp=60,
    kymo_window=120,
    ylim_std=39.5,
    figsize_base=(7.5, 2.5),
    show=True,
    savepath=None,
    marker_codes=["p", "D", "s"],
    sim_lines=None,
    exp_lines=None,
    line_kwargs=None,
    cropy_expkymo=None,
    titlepad=6,
):
    """
    Create a 4-panel comparison plot between simulation and experiment.

    Parameters
    ----------
    key : tuple
        Parameter key used to index D.
    D : dict
        Data dictionary.
    lifetimes_plot : callable
        Function that plots monomer lifetimes into a given axis.
    fct : module or namespace
        Must provide fct.kymo.imshow_kymograph,
        fct.kymo.normalized_kymo_roughness,
        fct.kymo.load_kymo_stds.
    exp_images : list
        Experimental kymograph images.
    marker_codes : list
        Marker codes used for legend placeholders.
    t0 : int, default 400
        Start time index for kymograph slice.
    marker_size : int
        Marker size for placeholder scatter.
    skip : int
        Spatial downsampling for simulated kymograph.
    scale : float
        Overall figure scaling factor.
    cmap : str
        Colormap for kymographs.
    aspect_sim : float
        Aspect ratio for simulated kymograph.
    aspect_exp : float
        Aspect ratio for experimental kymograph.
    kymo_window : int
        Number of time points shown in kymograph.
    ylim_std : float
        Upper y-limit for std histogram.
    figsize_base : tuple
        Base figure size before scaling.
    show : bool
        Whether to call plt.show().
    savepath : str or None
        If given, saves figure to this path.
    sim_lines / exp_lines : list of ((x0, y0), (x1, y1))
        Coordinates must be in *imshow extent coordinates*.

plot_sim_exp_comparison(
    key=highlight_keys[1],
    D=D,
    lifetimes_plot=lifetimes_plot,
    fct=fct,
    exp_images=exp_images,
    marker_codes=marker_codes,
    t0=800,
)
    """

    if line_kwargs is None:
        line_kwargs = dict(color="k", ls="--", lw=1.5)

    fig, ax = plt.subplots(
        1, 4,
        figsize=(scale * figsize_base[0], scale * figsize_base[1]),
        # constrained_layout=True
    )

    # =========================
    # Panel 0: monomer lifetimes
    # =========================
    ax[0].set_title("Monomer lifetimes", pad=titlepad)
    lifetimes_plot(key, ax[0], D, show_means=False, expcolor="C1")
    ax[0].set_ylabel("Probability")

    ax[0].scatter(
        # -5,0.15,
        -0.4, 0.5,                     # x < 0 → left of axis
        transform=ax[0].transAxes,      # axes (0–1) coordinates
        marker=marker_codes[0],
        s=marker_size,
        edgecolor="k",
        facecolors="none",
        clip_on=False,  
    )
    
    # =========================
    # Panel 1: simulated kymograph
    # =========================
    sim_extent = (0, 2 * 5 * 242, 0, kymo_window)

    ax[1].imshow(
        D[key]["center_xcounts"][t0:t0 + kymo_window, ::skip],
        interpolation="bicubic",
        aspect=aspect_sim,
        cmap=cmap,
        extent=sim_extent,
        origin="lower",
    )

    ax[1].set_title("Simulated kymograph", pad=titlepad)
    ax[1].set_ylabel("Time (s)")
    ax[1].set_xlabel("Circumference (nm)")

    if sim_lines is not None:
        for (x0, y0), (x1, y1) in sim_lines:
            ax[1].plot([x0, x1], [y0, y1], **line_kwargs)
            slope = (y1 - y0) / (x1 - x0)
            # print(f"[Simulation] slope = {slope:.4g} (Δy/Δx)")
            print(f"[Simulation] speed = {1/slope:.4g} (nm/s)")

    # =========================
    # Panel 2: experimental kymograph
    # =========================
    exp_extent = fct.kymo.imshow_kymograph(
        ax[2],
        exp_images[2],
        cmap=cmap,
        aspect=aspect_exp,
        cropy=cropy_expkymo,
        return_extent=True,   # ⬅ small change needed (see note below)
    )

    ax[2].set_title("Experimental kymograph", pad=titlepad)
    ax[2].set_ylabel("Time (s)")
    ax[2].set_xlabel("Circumference (nm)")

    if exp_lines is not None:
        for (x0, y0), (x1, y1) in exp_lines:
            ax[2].plot([x0, x1], [y0, y1], **line_kwargs)
            slope = (y1 - y0) / (x1 - x0)
            # print(f"[Experiment] slope = {slope:.4g} (Δy/Δx)")
            print(f"[Experiment] speed = {1/slope:.4g} (nm/s)")


    # =========================
    # Panel 3: intensity variation
    # =========================
    img = fct.kymo.normalized_kymo_roughness(key, D)
    std_sim = img.std(axis=1)

    kymo_stds = fct.kymo.load_kymo_stds()
    std_exp = kymo_stds["Constricting"]

    for data, label in zip(
        [std_sim, std_exp],
        ["Simulation", "Experiment"]
    ):
        ax[3].hist(
            data,
            bins=40,
            histtype="step",
            density=True,
            lw=2,
            label=label,
        )

    ax[3].legend()
    ax[3].set_xlabel(r"$\mathrm{std}(\tilde{I}(t))$")
    ax[3].set_ylabel("Density")
    ax[3].set_title("Intensity variation", pad=titlepad)
    ax[3].set_ylim(top=ylim_std)

    
    fig.tight_layout()
    # fig.subplots_adjust(left=0.72)

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight", pad_inches=0.5)

    if show:
        plt.show()

    return fig, ax


def lifetimes_plot(key, ax, D, overlay=None, convert_to_probab=True,
        expcolor="k", color=None, show_means=True, marker="o",
        sim_lw=1,
        leg_title=None
        ):
        exp = np.genfromtxt("/nfs/scistore26/saricgrp/fhorvath/0__treadmilling/D__hydr/0__ring/D__wsynth/B__switchtime/data/exp_lifetimes.dat", delimiter=",")
        shortest_time = exp[:,0][0]
        longest = exp[:,0][-1]
        bins =   exp[:,0][:-1] - np.diff(exp[:,0])/2

        if "monomer_lifetimes" in D[key]:
            lt = D[key]["monomer_lifetimes"] #/ dict(key)["tscale"]
            # choose the same bin edges you used for the density version
            edges = bins                       # e.g. np.linspace(0, longest, 30)
            
            # plain counts

    
            counts, edges = np.histogram(
                lt[(lt > 3) & (lt < longest)].values,
                bins=edges,
                density=False                 # <- important: keep raw counts
            )
            
            # convert counts → probabilities so they sum to 1
            probs = counts / counts.sum()
            
            # plot at the *centres* of the bins so the shape lines up nicely
            centres = (edges[:-1] + edges[1:]) / 2
            

            lines = ax.plot(centres, probs, color=color, #color="tab:blue", 
                            lw=sim_lw,
                    marker=marker, 
                    label="Simulation" if overlay is None else overlay)
            ax.plot(*exp.T, color=expcolor,
                    label="Experiment" if overlay is None else None, marker="o")
            if show_means:
                ax.axvline(x=np.mean(lt[(lt > 3) & (lt < longest)].values), 
                        color=lines[0].get_color()
                        )
                ax.axvline(x=8.2, color=expcolor,
                        ls="--")
            ax.set_xlabel("Lifetime (s)")
           # ax.set_ylabel("Relative frequency")

            if leg_title is None:
                ax.legend()
            else:
                ax.legend(title=leg_title)


